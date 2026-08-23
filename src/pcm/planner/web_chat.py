"""llama.cpp Web UI gateway for live Planner Cache conversations.

The browser is presentation only. Every chat request terminates here and passes
through :class:`PlannerChatSession` before the selected frozen-model runtime is
called. This prevents the llama.cpp UI from bypassing canonical memory.
"""

from __future__ import annotations

import argparse
import copy
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit
import webbrowser

from pcm.planner.chat_cli import PlannerChatSession, display_json, parser as chat_parser


WEB_HELP = """Chat in the browser using the llama.cpp Web UI.

Terminal commands:
  /help         show this help
  /state        show active canonical P-cache entries
  /personality  show promoted personality entries and the last retrieval
  /events       show recent Planner Cache events
  /save         checkpoint P-cache and P-package state
  /quit         save and stop the Web UI
"""


def parser() -> argparse.ArgumentParser:
    result = chat_parser()
    result.description = "Planner Cache through the llama.cpp Web UI"
    result.add_argument("--web-host", default="127.0.0.1")
    result.add_argument("--web-port", type=int, default=0)
    result.add_argument("--web-ui-path", type=Path, required=True)
    result.add_argument("--no-browser", action="store_true")
    return result


def _text_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(content or "")


def newest_user_message(payload: dict[str, object]) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be an array")
    for row in reversed(messages):
        if isinstance(row, dict) and row.get("role") == "user":
            message = _text_content(row.get("content")).strip()
            if message:
                return message
    raise ValueError("a non-empty user message is required")


def native_messages(payload: dict[str, object]) -> list[dict[str, object]]:
    """Return an opaque structural copy for the model's native chat template."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be an array")
    if not all(isinstance(row, dict) for row in messages):
        raise ValueError("every message must be an object")
    return copy.deepcopy(messages)


def completion_payload(
    *, model: str, text: str, input_tokens: int | None, output_tokens: int | None,
) -> dict[str, object]:
    return {
        "id": "chatcmpl-planner-cache",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": input_tokens or 0,
            "completion_tokens": output_tokens or 0,
            "total_tokens": (input_tokens or 0) + (output_tokens or 0),
        },
    }


class PlannerWebGateway:
    """Own the exact llama.cpp UI assets and the Planner Cache API boundary."""

    def __init__(
        self,
        session: PlannerChatSession,
        ui_path: Path,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        if not (ui_path / "index.html").is_file():
            raise FileNotFoundError(f"llama.cpp Web UI not found: {ui_path}")
        self.session = session
        self.ui_path = ui_path.resolve()
        self.lock = threading.Lock()
        self.httpd = ThreadingHTTPServer((host, port), self._handler())
        self.thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.httpd.server_address[:2]
        return str(host), int(port)

    @property
    def url(self) -> str:
        host, port = self.address
        visible_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        return f"http://{visible_host}:{port}/"

    def _handler(self):
        gateway = self

        class Handler(SimpleHTTPRequestHandler):
            server_version = "PlannerCacheWeb/1"

            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, directory=str(gateway.ui_path), **kwargs)

            def log_message(self, _format: str, *args: object) -> None:
                return

            def _json(self, value: object, status: int = 200) -> None:
                body = json.dumps(value, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()

            def _read_json(self) -> dict[str, object]:
                length = int(self.headers.get("Content-Length", "0"))
                value = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(value, dict):
                    raise ValueError("JSON request must be an object")
                return value

            def do_GET(self) -> None:
                path = urlsplit(self.path).path
                if path == "/health":
                    self._json({"status": "ok"})
                    return
                if path in {"/v1/models", "/models"}:
                    self._json({
                        "object": "list",
                        "data": [{
                            "id": gateway.session.runtime.model_id,
                            "object": "model",
                            "created": 0,
                            "owned_by": "planner-cache",
                        }],
                    })
                    return
                if path == "/props":
                    runtime = gateway.session.runtime
                    self._json({
                        "default_generation_settings": {
                            "params": {
                                "n_predict": runtime.max_new_tokens,
                                "max_tokens": runtime.max_new_tokens,
                                "temperature": runtime.temperature,
                                "top_p": runtime.top_p,
                                "stream": True,
                            }
                        },
                        "total_slots": 1,
                        "model_path": str(runtime.model_path),
                        "chat_template": "Planner Cache runtime-managed chat",
                        "chat_template_caps": {},
                        "modalities": {"vision": False},
                        "build_info": "planner-cache-gateway",
                        "is_sleeping": False,
                    })
                    return
                if path in {"/slots", "/tools", "/mcp-servers"}:
                    self._json([])
                    return
                if path == "/planner-cache/state":
                    with gateway.lock:
                        value = gateway.session.state.snapshot()
                    self._json(value)
                    return
                if path == "/planner-cache/events":
                    with gateway.lock:
                        value = list(gateway.session.recorder.recent_events)
                    self._json(value)
                    return
                if path == "/planner-cache/personality":
                    try:
                        query = parse_qs(urlsplit(self.path).query)
                        limit = int(query.get("limit", ["100"])[0])
                        offset = int(query.get("offset", ["0"])[0])
                        with gateway.lock:
                            value = gateway.session.command(
                                "/personality", source="web",
                                personality_limit=limit,
                                personality_offset=offset,
                            )
                    except (TypeError, ValueError) as error:
                        self._json({"error": {"message": str(error)}}, 400)
                        return
                    self._json(value)
                    return
                super().do_GET()

            def do_POST(self) -> None:
                path = urlsplit(self.path).path
                if path not in {"/v1/chat/completions", "/chat/completions"}:
                    self._json({"error": {"message": "unsupported gateway endpoint"}}, 404)
                    return
                try:
                    request = self._read_json()
                    message = newest_user_message(request)
                    messages = native_messages(request)
                    gateway.lock.acquire()
                    try:
                        result = gateway.session.chat(
                            message, raw_messages=messages,
                        )
                        payload = completion_payload(
                            model=gateway.session.runtime.model_id,
                            text=result.text,
                            input_tokens=result.input_tokens,
                            output_tokens=result.output_tokens,
                        )
                        try:
                            if bool(request.get("stream")):
                                self._stream(payload)
                            else:
                                self._json(payload)
                        finally:
                            gateway.session.complete_turn_review()
                    finally:
                        gateway.lock.release()
                except Exception as error:
                    gateway.session.recorder.event(
                        "ERROR", source="web_gateway",
                        error_type=type(error).__name__, message=str(error),
                    )
                    self._json({"error": {"message": str(error)}}, 500)

            def _stream(self, payload: dict[str, object]) -> None:
                choice = payload["choices"][0]
                message = choice["message"]
                chunk = {
                    "id": payload["id"],
                    "object": "chat.completion.chunk",
                    "created": payload["created"],
                    "model": payload["model"],
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant", "content": message["content"],
                        },
                        "finish_reason": None,
                    }],
                }
                finish = {
                    **chunk,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": payload["usage"],
                }
                rows = (
                    f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    f"data: {json.dumps(finish, ensure_ascii=False)}\n\n"
                    "data: [DONE]\n\n"
                ).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(rows)
                self.wfile.flush()

        return Handler

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self.httpd.serve_forever,
            name="planner-cache-web-ui",
            daemon=True,
        )
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)
            self.thread = None


def run(args: argparse.Namespace) -> int:
    session: PlannerChatSession | None = None
    gateway: PlannerWebGateway | None = None
    reason = "normal"
    try:
        session = PlannerChatSession(args)
        gateway = PlannerWebGateway(
            session, args.web_ui_path, host=args.web_host, port=args.web_port,
        )
        gateway.start()
        print(f"\n{session.title}")
        print(f"Web UI: {gateway.url}")
        print(f"Session records: {session.recorder.directory}")
        print("Chat in the browser. Terminal commands: /help /state /personality /events /save /quit\n")
        if not args.no_browser:
            webbrowser.open(gateway.url, new=2)
        while True:
            try:
                command = input("Debug: ").strip()
            except EOFError:
                # A non-interactive launcher should keep serving until signalled.
                while True:
                    time.sleep(1)
            if not command:
                continue
            if not command.startswith("/"):
                print("Chat in the Web UI. Terminal input accepts slash commands only.")
                continue
            with gateway.lock:
                value = session.command(command)
            if command.casefold() == "/help":
                value = WEB_HELP
            if command.casefold() == "/quit":
                reason = "quit"
                break
            if isinstance(value, str):
                print(value)
            else:
                display_json(value)
    except KeyboardInterrupt:
        reason = "ctrl-c"
        print("\nStopping.")
    finally:
        if gateway is not None:
            gateway.close()
        if session is not None:
            session.close(reason=reason)
    return 0


def main() -> None:
    try:
        raise SystemExit(run(parser().parse_args()))
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        print(f"Planner Cache startup failed: {error}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
