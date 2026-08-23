#!/usr/bin/env python3
"""Record exact inactive Planner Cache versus native llama.cpp prompt identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import Request, urlopen

from pcm.planner.interactive_runtimes import (
    LlamaServerProcess,
    gemma_chat_request_body,
)
from pcm.planner.interactive_session import file_sha256


def post(port: int, path: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError("llama.cpp returned a non-object response")
    return result


def digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-layers", type=int, default=12)
    args = parser.parse_args()

    binary = args.llama_root / "build/bin/llama-server"
    cli = args.llama_root / "build/bin/llama-cli"
    version = subprocess.run(
        [str(cli), "--version"], check=True, capture_output=True, text=True,
    )
    llama_version = "\n".join(
        part.strip() for part in (version.stdout, version.stderr) if part.strip()
    )
    messages = [
        {"role": "system", "content": "Answer exactly as yourself. Keep β intact."},
        {"role": "user", "content": "Describe  two spaces and this newline:\nnow."},
    ]
    request = gemma_chat_request_body(
        messages, max_tokens=8, temperature=0.0, top_p=1.0, seed=7,
        target_token_ids=(),
    )
    server = LlamaServerProcess(
        binary=binary,
        model=args.model,
        control_vector=args.output.with_suffix(".unused.gguf"),
        attachment_layer=0,
        context_tokens=512,
        gpu_layers=args.gpu_layers,
        threads=8,
        startup_timeout=180,
    )
    try:
        server.ensure(0.0)
        raw_prompt = post(server.port, "/apply-template", {"messages": messages})["prompt"]
        gateway_prompt = post(
            server.port, "/apply-template", {"messages": request["messages"]}
        )["prompt"]
        raw_tokens = post(
            server.port, "/tokenize",
            {"content": raw_prompt, "add_special": False, "parse_special": True},
        )["tokens"]
        gateway_tokens = post(
            server.port, "/tokenize",
            {"content": gateway_prompt, "add_special": False, "parse_special": True},
        )["tokens"]
    finally:
        server.stop()

    result = {
        "format": "planner-cache-native-prompt-equivalence-v1",
        "model": str(args.model),
        "model_sha256": file_sha256(args.model),
        "llama_cpp_version": llama_version,
        "conditions": {
            "p_active": False,
            "ltl_logit_bias_present": "logit_bias" in request,
            "message_structure_equal": request["messages"] == messages,
        },
        "raw": {
            "message_sha256": digest_json(messages),
            "prompt_sha256": hashlib.sha256(str(raw_prompt).encode("utf-8")).hexdigest(),
            "token_ids_sha256": digest_json(raw_tokens),
            "token_count": len(raw_tokens),
        },
        "gateway_inactive": {
            "message_sha256": digest_json(request["messages"]),
            "prompt_sha256": hashlib.sha256(
                str(gateway_prompt).encode("utf-8")
            ).hexdigest(),
            "token_ids_sha256": digest_json(gateway_tokens),
            "token_count": len(gateway_tokens),
        },
        "assertions": {
            "message_structure_exact": request["messages"] == messages,
            "rendered_prompt_exact": gateway_prompt == raw_prompt,
            "token_ids_exact": gateway_tokens == raw_tokens,
            "inactive_ltl_has_no_output_control": "logit_bias" not in request,
        },
    }
    if not all(result["assertions"].values()):
        raise RuntimeError("native prompt equivalence failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    print(json.dumps(result["assertions"], sort_keys=True))


if __name__ == "__main__":
    main()
