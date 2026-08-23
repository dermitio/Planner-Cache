import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from pcm.planner.interactive_runtimes import (
    LlamaServerProcess,
    gemma_chat_request_body,
)


LLAMA_ROOT = Path(os.environ.get("LLAMA_CPP_DIR", "/unavailable/llama.cpp"))
GEMMA_GGUF = Path(os.environ.get("GEMMA_MODEL", "/unavailable/model.gguf"))


def post(port, path, payload):
    request = Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        return json.load(response)


@pytest.mark.slow_cuda
@pytest.mark.skipif(
    not (LLAMA_ROOT / "build/bin/llama-server").is_file()
    or not GEMMA_GGUF.is_file(),
    reason="exact Gemma GGUF and llama.cpp runtime are unavailable",
)
def test_inactive_gateway_and_raw_llama_have_identical_native_prompt_tokens(tmp_path):
    server = LlamaServerProcess(
        binary=LLAMA_ROOT / "build/bin/llama-server",
        model=GEMMA_GGUF,
        control_vector=tmp_path / "unused.gguf",
        attachment_layer=0,
        context_tokens=512,
        gpu_layers=12,
        threads=8,
        startup_timeout=180,
    )
    messages = [
        {"role": "system", "content": "Answer exactly as yourself. Keep β intact."},
        {"role": "user", "content": "Describe  two spaces and this newline:\nnow."},
    ]
    try:
        server.ensure(0.0)
        raw_prompt = post(server.port, "/apply-template", {"messages": messages})["prompt"]
        inactive = gemma_chat_request_body(
            messages, max_tokens=8, temperature=0.0, top_p=1.0, seed=7,
            target_token_ids=(),
        )
        assert "logit_bias" not in inactive
        assert inactive["messages"] == messages
        gateway_prompt = post(
            server.port, "/apply-template", {"messages": inactive["messages"]}
        )["prompt"]
        raw_tokens = post(
            server.port, "/tokenize",
            {"content": raw_prompt, "add_special": False, "parse_special": True},
        )["tokens"]
        gateway_tokens = post(
            server.port, "/tokenize",
            {"content": gateway_prompt, "add_special": False, "parse_special": True},
        )["tokens"]
        assert gateway_prompt == raw_prompt
        assert gateway_tokens == raw_tokens
    finally:
        server.stop()
