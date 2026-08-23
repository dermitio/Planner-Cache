import argparse
import json
from pathlib import Path
import subprocess

import pytest
import torch

from pcm.planner.chat_cli import run
from pcm.planner.compatibility import TensorTranslationLayer
from pcm.planner.canonical import CanonicalPStore
from pcm.planner.interactive_runtimes import (
    GenerationResult,
    LlamaServerProcess,
    canonical_route,
)
from pcm.planner.interactive_session import (
    CanonicalQueryIntent,
    CanonicalStateManager,
    PersonalityManager,
    SessionRecorder,
    utc_now,
)
from pcm.planner.representation import FactorizedStateRepresentation
from pcm.planner.personality import synthetic_entry
from pcm.planner.split_translator import (
    ByteEntityEncoder,
    CanonicalPRouter,
    SplitTranslateConfig,
)


def representation():
    torch.manual_seed(19)
    return FactorizedStateRepresentation(24, 3, 36)


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_session_directory_jsonl_order_and_final_state(tmp_path):
    recorder = SessionRecorder(
        tmp_path, "test", {"model": "deterministic"},
        started_at="2026-08-22T12:34:56+00:00",
    )
    recorder.turn = 1
    recorder.transcript(role="user", text="hello", model="test", runtime="fake")
    recorder.event("P_STATE_BEFORE", source="p_cache", entries=[])
    recorder.event("P_IGNORE", source="p_cache", reason="no state")
    recorder.event("MODEL_GENERATION_START", source="model")
    recorder.event("MODEL_GENERATION_END", source="model")
    recorder.transcript(
        role="assistant", text="hi", model="test", runtime="fake",
        latency_seconds=0.1, input_tokens=1, output_tokens=1,
    )
    recorder.finalize({"active_p_cache_entries": []}, reason="quit")

    assert recorder.directory.name == "2026-08-22T123456p0000-test"
    transcript = jsonl(recorder.transcript_path)
    events = jsonl(recorder.events_path)
    assert [row["role"] for row in transcript] == ["user", "assistant"]
    assert [row["event"] for row in events] == [
        "SESSION_START", "P_STATE_BEFORE", "P_IGNORE",
        "MODEL_GENERATION_START", "MODEL_GENERATION_END", "SESSION_END",
    ]
    final = json.loads(recorder.final_state_path.read_text())
    assert final["total_turns"] == 1
    assert final["event_counts"]["P_IGNORE"] == 1


def test_canonical_mutations_and_logging_do_not_change_decisions(tmp_path):
    model = representation()
    logged = CanonicalStateManager(model, slots=4)
    silent = CanonicalStateManager(model, slots=4)
    recorder = SessionRecorder(tmp_path / "logged", "test", {}, enabled=True)
    silent_recorder = SessionRecorder(tmp_path / "silent", "test", {}, enabled=False)
    messages = (
        "The silver key belongs to Alice",
        "Alice owns the silver key",
        "The silver key belongs to Bob",
    )
    for message in messages:
        logged.apply(logged.extract_mutations(message), recorder)
        silent.apply(silent.extract_mutations(message), silent_recorder)
    assert logged.snapshot() == silent.snapshot()
    assert [row["event"] for row in jsonl(recorder.events_path)][1:] == [
        "P_CREATE", "P_KEEP", "P_MODIFY",
    ]

    before = logged.snapshot()
    router = CanonicalPRouter()
    query = CanonicalQueryIntent("silver key", 0, "owner", "test")
    route, candidates = canonical_route(router, ByteEntityEncoder(), logged.store, query)
    assert route is not None and candidates
    assert logged.snapshot() == before
    recorder.finalize({"active_p_cache_entries": before}, reason="test")
    silent_recorder.finalize({"active_p_cache_entries": before}, reason="test")


def test_invalidation_and_unsupported_text_are_observational(tmp_path):
    state = CanonicalStateManager(representation(), slots=4)
    recorder = SessionRecorder(tmp_path, "test", {})
    state.apply(state.extract_mutations("remember: silver key.owner=Alice"), recorder)
    assert len(state.snapshot()) == 1
    assert state.extract_mutations("The silver key belongs to Zephyr") == []
    state.apply(state.extract_mutations("invalidate: silver key.owner"), recorder)
    assert state.snapshot() == []
    recorder.finalize({"active_p_cache_entries": []}, reason="test")
    assert [row["event"] for row in jsonl(recorder.events_path)][-2] == "P_INVALIDATE"


def test_personality_promotion_query_and_contradiction_events(tmp_path):
    recorder = SessionRecorder(tmp_path / "sessions", "test", {})
    manager = PersonalityManager(tmp_path / "personality.ppkg", representation())
    first = manager.extract_evidence(
        "I prefer concise responses", turn=1, timestamp=utc_now()
    )
    assert first is not None
    manager.ingest(first, recorder)
    selected = manager.query("hello", recorder)
    assert selected is None
    second = manager.extract_evidence(
        "I prefer detailed responses", turn=2, timestamp=utc_now()
    )
    assert second is not None
    manager.ingest(second, recorder)
    manager.checkpoint()
    manager.close()
    recorder.finalize({"active_p_cache_entries": []}, reason="test")
    names = [row["event"] for row in jsonl(recorder.events_path)]
    assert names.count("PPKG_PROMOTION") == 2
    assert "PPKG_QUERY" in names
    assert "PPKG_CANDIDATES" in names
    assert "PPKG_LOAD" in names
    assert "PPKG_CONTRADICTION" in names


def test_personality_debug_page_never_hydrates_the_full_package(tmp_path, monkeypatch):
    manager = PersonalityManager(tmp_path / "personality.ppkg", representation())
    try:
        manager.package.bulk_insert_entries(
            (synthetic_entry(index) for index in range(250)),
            updated_at="2026-01-01T00:00:00+00:00",
        )
        calls = []
        original = manager.package.entries

        def observed_entries(**kwargs):
            calls.append(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(manager.package, "entries", observed_entries)
        page = manager.visible_entry_page(limit=40, offset=80)
        assert page["total_active"] == 250
        assert page["returned"] == 40
        assert page["truncated"]
        assert page["entries"][0]["id"] == "personality-00000080"
        assert calls == [{
            "status": "active", "limit": 40, "offset": 80,
        }]
        assert len(manager.visible_entries()) == 250
        with pytest.raises(ValueError, match="between 1 and 200"):
            manager.visible_entry_page(limit=201)
        with pytest.raises(ValueError, match="cannot be negative"):
            manager.visible_entry_page(offset=-1)
    finally:
        manager.close()


def test_free_form_location_is_retained_but_inert_when_translator_unsupported(tmp_path):
    state = CanonicalStateManager(representation(), slots=4)
    recorder = SessionRecorder(tmp_path, "test", {})
    message = "My character left the silver key on the desk."
    intents = state.extract_mutations(message)
    assert len(intents) == 1 and intents[0].value == "desk"
    assert not intents[0].translator_compatible
    state.apply(intents, recorder)
    assert state.snapshot()[0]["value"] == "desk"
    assert not state.snapshot()[0]["translator_compatible"]
    query = state.infer_query("Where did I leave the key?")
    assert query.entity == "silver key"
    assert query.relation == "location"
    assert state.translation_store().cache.occupied == 0
    open_store = state.translation_store(include_open_values=True)
    assert open_store.cache.occupied == 1
    assert open_store._pcm_value_surfaces[0] == "desk"
    snapshot = tmp_path / "p-cache.safetensors"
    metadata = tmp_path / "p-cache-runtime.json"
    state.store.save(snapshot)
    state.save_runtime_metadata(metadata)
    restored = CanonicalStateManager(
        representation(), store=CanonicalPStore.load(snapshot, dtype=torch.float32)
    )
    restored.load_runtime_metadata(metadata)
    assert restored.snapshot()[0]["value"] == "desk"
    assert not restored.snapshot()[0]["translator_compatible"]
    recorder.finalize({"active_p_cache_entries": state.snapshot()}, reason="test")


class FakeProcess:
    def __init__(self):
        self.pid = 424242
        self.returncode = None
        self.waited = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.waited = True
        self.returncode = -15
        return self.returncode


def test_gemma_child_server_cleanup(monkeypatch, tmp_path):
    process = FakeProcess()
    killed = []
    monkeypatch.setattr("os.killpg", lambda pid, sig: killed.append((pid, sig)))
    manager = LlamaServerProcess(
        binary=tmp_path / "llama-server", model=tmp_path / "model.gguf",
        control_vector=tmp_path / "direction.gguf", attachment_layer=41,
        context_tokens=512, gpu_layers=0, threads=1,
        pid_file=tmp_path / "llama.pid",
    )
    manager.process = process
    manager.port = 12345
    manager.strength = 2.0
    manager.pid_file.write_text(str(process.pid))
    manager.stop()
    assert killed and killed[0][0] == process.pid
    assert process.waited
    assert manager.process is None
    assert not manager.pid_file.exists()


def test_gemma_interactive_server_disables_reasoning_trace(tmp_path):
    manager = LlamaServerProcess(
        binary=tmp_path / "llama-server", model=tmp_path / "model.gguf",
        control_vector=tmp_path / "direction.gguf", attachment_layer=41,
        context_tokens=512, gpu_layers=0, threads=1,
    )
    command = manager.command(port=12345, strength=0.0)
    reasoning_index = command.index("--reasoning")
    assert command[reasoning_index + 1] == "off"


class FakeRuntime:
    model_id = "fake-model"
    runtime = "fake-runtime"

    def __init__(self):
        self.closed = False
        self.calls = []
        self.router = CanonicalPRouter()
        self.encoder = ByteEntityEncoder()

    def metadata(self):
        return {
            "model": self.model_id, "runtime": self.runtime,
            "compatibility_layer": "ttl", "adapter_artifact": "fake.ttl",
            "adapter_checksum": "fake",
            "router_artifact": "fake.router", "router_format": "fake",
        }

    def generate(self, *args, **kwargs):
        captured = list(args)
        captured[1] = list(captured[1])
        self.calls.append((tuple(captured), kwargs))
        return GenerationResult("answer", 0.1, 2, 1, {})

    def review_memory(self, _prompt, _schema):
        return '{"operations":[]}'

    def close(self):
        self.closed = True


def cli_args(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    adapter = tmp_path / "fake.ttl"
    router = tmp_path / "fake.router"
    TensorTranslationLayer(SplitTranslateConfig(
        model_id="fake-model", model_hidden_width=8,
        canonical_width=24, attachment_layers=(1,),
    )).save(adapter)
    router.write_bytes(b"router")
    return argparse.Namespace(
        runtime="pythia", repo_root=Path(__file__).resolve().parents[1],
        model=model, adapter=adapter, router=router, llama_cpp_dir=None,
        tokenizer_bundle=None,
        ppkg=None, session_root=tmp_path / "sessions", p_cache=None,
        slots=4, context_tokens=32, max_new_tokens=2, temperature=0.0,
        top_p=1.0, seed=1, gpu_layers=0, threads=1,
        llama_pid_file=None, no_logging=False,
    )


def test_ctrl_c_closes_runtime_and_writes_final_state(monkeypatch, tmp_path):
    fake = FakeRuntime()
    model = representation()
    monkeypatch.setattr(
        "pcm.planner.chat_cli.train_and_probe_representation",
        lambda: (model, object(), {"p_only_state_recovery": 1.0}),
    )
    monkeypatch.setattr("pcm.planner.chat_cli.build_runtime", lambda _args: fake)
    monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert run(cli_args(tmp_path)) == 0
    assert fake.closed
    directories = list((tmp_path / "sessions").iterdir())
    assert len(directories) == 1
    final = json.loads((directories[0] / "final-state.json").read_text())
    assert final["end_reason"] == "ctrl-c"
    assert (directories[0] / "p-cache.safetensors").is_file()
    assert (directories[0] / "p-cache-runtime.json").is_file()


def test_default_terminal_path_is_chat_first(monkeypatch, tmp_path, capsys):
    fake = FakeRuntime()
    model = representation()
    monkeypatch.setattr(
        "pcm.planner.chat_cli.train_and_probe_representation",
        lambda: (model, object(), {"p_only_state_recovery": 1.0}),
    )
    monkeypatch.setattr("pcm.planner.chat_cli.build_runtime", lambda _args: fake)
    messages = iter(("Hello!", "How are you?", "/quit"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(messages))
    assert run(cli_args(tmp_path)) == 0
    assert len(fake.calls) == 2
    assert fake.calls[0][0][0] == "Hello!"
    assert fake.calls[1][0][1] == [("Hello!", "answer")]
    output = capsys.readouterr().out
    assert "Planner Cache — Pythia-1.4B" in output
    assert output.count("Assistant: answer") == 2


@pytest.mark.parametrize(
    ("script", "environment", "expected"),
    (
        ("run-pythia.sh", {"PYTHIA_MODEL": "/definitely/missing-pythia"}, "Pythia model not found"),
        ("run-gemma.sh", {"GEMMA_MODEL": "/definitely/missing-gemma.gguf"}, "Gemma GGUF not found"),
    ),
)
def test_shell_environment_overrides_and_missing_path_errors(script, environment, expected):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [str(root / script)], cwd=Path("/"), env={**os_environ(), **environment},
        text=True, capture_output=True,
    )
    assert completed.returncode == 2
    assert expected in completed.stderr


def os_environ():
    import os
    return os.environ.copy()


def test_missing_llama_binary_has_clear_shell_error(tmp_path):
    root = Path(__file__).resolve().parents[1]
    fake_model = tmp_path / "model.gguf"
    fake_model.write_bytes(b"not loaded because preflight fails")
    completed = subprocess.run(
        [str(root / "run-gemma.sh")], cwd=Path("/"),
        env={
            **os_environ(), "GEMMA_MODEL": str(fake_model),
            "LLAMA_CPP_DIR": str(tmp_path / "missing-llama"),
        },
        text=True, capture_output=True,
    )
    assert completed.returncode == 2
    assert "llama-server not found" in completed.stderr
