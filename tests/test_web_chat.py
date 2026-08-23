import json
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import threading
from urllib.request import Request, urlopen

import torch

from pcm.planner.interactive_runtimes import GenerationResult
from pcm.planner.interactive_session import PersonalityManager, SessionRecorder
from pcm.planner.representation import FactorizedStateRepresentation
from pcm.planner.web_chat import PlannerWebGateway, newest_user_message


class FakeRecorder:
    def __init__(self):
        self.recent_events = []

    def event(self, event, **fields):
        self.recent_events.append({"event": event, **fields})


class FakeState:
    def snapshot(self):
        return [{"entity": "silver key", "relation": "location", "value": "desk"}]


class FakeSession:
    def __init__(self):
        self.runtime = SimpleNamespace(
            model_id="fake-model", model_path=Path("/models/fake"),
            max_new_tokens=32, temperature=0.7, top_p=0.9,
        )
        self.state = FakeState()
        self.recorder = FakeRecorder()
        self.messages = []
        self.raw_messages = []
        self.reviews = 0
        self.command_calls = []

    def chat(self, message, *, raw_messages=None):
        self.messages.append(message)
        self.raw_messages.append(raw_messages)
        return GenerationResult("memory-aware answer", 0.1, 4, 3, {})

    def command(self, command, **kwargs):
        self.command_calls.append((command, kwargs))
        if command == "/personality":
            return {
                "promoted": [],
                "page": {
                    "total_active": 0, "returned": 0, "limit": 100,
                    "offset": 0, "truncated": False,
                },
                "last_retrieval": None,
            }
        return None

    def complete_turn_review(self):
        self.reviews += 1


def ui(tmp_path):
    root = tmp_path / "ui"
    root.mkdir()
    (root / "index.html").write_text("<html>llama.cpp UI</html>")
    return root


def post_json(url, value):
    request = Request(
        url, data=json.dumps(value).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urlopen(request, timeout=5)


def test_newest_user_message_supports_text_parts():
    assert newest_user_message({
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": [
                {"type": "text", "text": "new"},
                {"type": "image_url", "image_url": "ignored"},
            ]},
        ]
    }) == "new"


def test_gateway_serves_llama_ui_and_routes_chat_through_session(tmp_path):
    session = FakeSession()
    gateway = PlannerWebGateway(session, ui(tmp_path))
    gateway.start()
    try:
        assert "llama.cpp UI" in urlopen(gateway.url, timeout=5).read().decode()
        with post_json(gateway.url + "v1/chat/completions", {
            "messages": [{"role": "user", "content": "Where is the key?"}],
            "stream": False,
        }) as response:
            payload = json.load(response)
        assert session.messages == ["Where is the key?"]
        assert session.raw_messages == [[
            {"role": "user", "content": "Where is the key?"}
        ]]
        assert payload["choices"][0]["message"]["content"] == "memory-aware answer"
        assert payload["usage"]["total_tokens"] == 7
        assert session.reviews == 1
    finally:
        gateway.close()


def test_gateway_preserves_system_user_assistant_structure_as_side_channel(tmp_path):
    session = FakeSession()
    gateway = PlannerWebGateway(session, ui(tmp_path))
    original = [
        {"role": "system", "content": "Keep punctuation: [A]  β."},
        {"role": "user", "content": "First message", "name": "operator"},
        {"role": "assistant", "content": "Exact prior reply\nline two"},
        {"role": "user", "content": [
            {"type": "text", "text": "Where is the silver key?"},
        ]},
    ]
    gateway.start()
    try:
        with post_json(gateway.url + "v1/chat/completions", {
            "messages": original, "stream": False,
        }):
            pass
        assert session.messages == ["Where is the silver key?"]
        assert session.raw_messages == [original]
        assert session.raw_messages[0] is not original
    finally:
        gateway.close()


def test_gateway_streams_openai_chunks_after_planner_cache_turn(tmp_path):
    session = FakeSession()
    gateway = PlannerWebGateway(session, ui(tmp_path))
    gateway.start()
    try:
        with post_json(gateway.url + "v1/chat/completions", {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }) as response:
            body = response.read().decode()
        assert response.headers["Content-Type"] == "text/event-stream"
        assert '"content": "memory-aware answer"' in body
        assert "data: [DONE]" in body
        assert session.messages == ["Hello"]
    finally:
        gateway.close()


def test_gateway_exposes_debug_state_without_mutating_chat(tmp_path):
    session = FakeSession()
    gateway = PlannerWebGateway(session, ui(tmp_path))
    gateway.start()
    try:
        with urlopen(gateway.url + "planner-cache/state", timeout=5) as response:
            state = json.load(response)
        assert state[0]["value"] == "desk"
        assert session.messages == []
    finally:
        gateway.close()


def test_gateway_personality_debug_view_is_paginated(tmp_path):
    session = FakeSession()
    gateway = PlannerWebGateway(session, ui(tmp_path))
    gateway.start()
    try:
        with urlopen(
            gateway.url + "planner-cache/personality?limit=17&offset=34",
            timeout=5,
        ) as response:
            value = json.load(response)
        assert value["promoted"] == []
        assert value["page"]["limit"] == 100
        assert session.command_calls == [(
            "/personality",
            {
                "source": "web", "personality_limit": 17,
                "personality_offset": 34,
            },
        )]
    finally:
        gateway.close()


def test_ppkg_query_can_run_on_serialized_gateway_worker(tmp_path):
    torch.manual_seed(29)
    representation = FactorizedStateRepresentation(24, 3, 36)
    manager = PersonalityManager(tmp_path / "personality.ppkg", representation)
    recorder = SessionRecorder(tmp_path / "sessions", "web-test", {})
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(manager.query, "Hello", recorder).result()
        assert result is None
    finally:
        manager.close()
        recorder.finalize({"active_p_cache_entries": []}, reason="test")


def test_visible_response_precedes_review_and_next_turn_waits(tmp_path):
    class BlockingReviewSession(FakeSession):
        def __init__(self):
            super().__init__()
            self.review_started = threading.Event()
            self.release_review = threading.Event()

        def complete_turn_review(self):
            self.review_started.set()
            assert self.release_review.wait(timeout=5)
            super().complete_turn_review()

    session = BlockingReviewSession()
    gateway = PlannerWebGateway(session, ui(tmp_path))
    gateway.start()

    def request(message):
        with post_json(gateway.url + "v1/chat/completions", {
            "messages": [{"role": "user", "content": message}],
            "stream": False,
        }) as response:
            return json.load(response)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(request, "first")
            assert session.review_started.wait(timeout=3)
            assert first.result(timeout=3)["choices"][0]["message"]["content"]
            second = executor.submit(request, "second")
            assert session.messages == ["first"]
            session.release_review.set()
            assert second.result(timeout=5)["choices"][0]["message"]["content"]
        assert session.messages == ["first", "second"]
        assert session.reviews == 2
    finally:
        session.release_review.set()
        gateway.close()
