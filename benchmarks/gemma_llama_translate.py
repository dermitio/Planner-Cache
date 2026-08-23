"""Build and evaluate the Gemma4 Q8 llama.cpp `.translate` compatibility proof."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen

import numpy as np
import torch

from pcm.planner.canonical import CanonicalPConfig, CanonicalPStore
from pcm.planner.llama_gguf_translate import (
    LlamaGGUFTranslateConfig,
    LlamaGGUFTranslatePackage,
)
from pcm.planner.representation import HISTORICAL, train_and_probe_representation
from pcm.planner.split_translator import (
    ByteEntityEncoder,
    CanonicalPRouter,
    FactorizedCanonicalQuery,
)


PROMPT = "The silver key currently belongs to"
NATURAL_RP_PROMPT = "Rain tapped the observatory windows while the old astronomer adjusted the brass lens. Guest:"
ALICE_TOKEN = 32858
BOB_TOKEN = 15943
ALICE_VALUE = 0
BOB_VALUE = 1
OWNER_RELATION = 0
LOCATION_RELATION = 1
CURRENT_METADATA = 0
ATTACHMENT_LAYER = 41
ALICE_STRENGTH = 2.0
BOB_STRENGTH = -5.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def llama_version(binary: Path) -> str:
    completed = subprocess.run(
        [str(binary), "--version"], check=True, text=True, capture_output=True
    )
    return "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )


def load_gguf(llama_root: Path):
    gguf_python = llama_root / "gguf-py"
    if not gguf_python.is_dir():
        raise FileNotFoundError(f"llama.cpp gguf-py not found at {gguf_python}")
    sys.path.insert(0, str(gguf_python))
    import gguf
    return gguf


def gguf_metadata_and_direction(
    gguf_module,
    model_path: Path,
) -> tuple[dict[str, object], np.ndarray]:
    reader = gguf_module.GGUFReader(str(model_path), "r")

    def field(name: str):
        value = reader.get_field(name)
        if value is None:
            raise ValueError(f"required GGUF metadata field is absent: {name}")
        return value.contents()

    metadata = {
        "architecture": field("general.architecture"),
        "name": field("general.name"),
        "size_label": field("general.size_label"),
        "block_count": int(field("gemma4.block_count")),
        "context_length": int(field("gemma4.context_length")),
        "embedding_length": int(field("gemma4.embedding_length")),
        "file_type": int(field("general.file_type")),
        "tensor_count": len(reader.tensors),
    }
    embedding = next(
        (tensor for tensor in reader.tensors if tensor.name == "token_embd.weight"),
        None,
    )
    if embedding is None:
        raise ValueError("GGUF token embedding tensor is absent")
    alice = gguf_module.dequantize(
        np.asarray(embedding.data[ALICE_TOKEN:ALICE_TOKEN + 1]), embedding.tensor_type
    ).reshape(-1)
    bob = gguf_module.dequantize(
        np.asarray(embedding.data[BOB_TOKEN:BOB_TOKEN + 1]), embedding.tensor_type
    ).reshape(-1)
    direction = (alice - bob).astype(np.float32)
    direction /= np.linalg.norm(direction)
    return metadata, direction


def write_control_vector(
    gguf_module,
    path: Path,
    direction: np.ndarray,
    *,
    model_hint: str,
) -> None:
    writer = gguf_module.GGUFWriter(str(path), "controlvector")
    writer.add_string("controlvector.model_hint", model_hint)
    writer.add_int32("controlvector.layer_count", 1)
    writer.add_tensor(f"direction.{ATTACHMENT_LAYER}", direction.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def compile_runner(llama_root: Path, output: Path) -> dict[str, object]:
    source = Path(__file__).with_name("llama_cpp_causal_runner.cpp")
    binary_dir = llama_root / "build" / "bin"
    command = [
        "c++", "-std=c++17", "-O2", str(source),
        f"-I{llama_root / 'include'}",
        f"-I{llama_root / 'ggml' / 'include'}",
        str(binary_dir / "libllama.so"),
        str(binary_dir / "libggml.so"),
        f"-Wl,-rpath,{binary_dir}",
        "-o", str(output),
    ]
    started = time.perf_counter()
    subprocess.run(command, check=True)
    return {
        "command": command,
        "compile_seconds": time.perf_counter() - started,
        "source_sha256": file_sha256(source),
        "binary_sha256": file_sha256(output),
    }


def canonical_vector(representation, value_id: int) -> torch.Tensor:
    with torch.inference_mode():
        return representation.encode(
            torch.tensor([0]),
            torch.tensor([OWNER_RELATION]),
            torch.tensor([value_id]),
            torch.tensor([CURRENT_METADATA]),
        )[0].float()


def build_adapter(
    *,
    model_path: Path,
    model_sha256: str,
    model_metadata: dict[str, object],
    model_direction: np.ndarray,
    llama_build: str,
    output: Path,
) -> tuple[LlamaGGUFTranslatePackage, object, dict[str, object]]:
    representation, _, representation_probe = train_and_probe_representation()
    alice = canonical_vector(representation, ALICE_VALUE)
    bob = canonical_vector(representation, BOB_VALUE)
    config = LlamaGGUFTranslateConfig(
        model_id=str(model_metadata["name"]),
        model_architecture=str(model_metadata["architecture"]),
        model_hidden_width=int(model_metadata["embedding_length"]),
        model_layer_count=int(model_metadata["block_count"]),
        attachment_layer=ATTACHMENT_LAYER,
        model_sha256=model_sha256,
        llama_build=llama_build,
        supported_value_ids=(ALICE_VALUE, BOB_VALUE),
        target_token_ids=(ALICE_TOKEN, BOB_TOKEN),
    )
    package = LlamaGGUFTranslatePackage(config)
    package.fit_pair(
        alice,
        bob,
        torch.from_numpy(model_direction.copy()),
        strength_a=ALICE_STRENGTH,
        strength_b=BOB_STRENGTH,
    )
    package.save(output)
    restored = LlamaGGUFTranslatePackage.load(output)
    torch.testing.assert_close(
        package.state_dict()["model_direction"],
        restored.state_dict()["model_direction"],
    )
    strengths = restored(
        torch.stack((alice, bob)),
        torch.tensor((ALICE_VALUE, BOB_VALUE)),
    ).strength
    torch.testing.assert_close(
        strengths,
        torch.tensor((ALICE_STRENGTH, BOB_STRENGTH)),
        atol=1e-5,
        rtol=0,
    )
    return restored, representation, {
        "canonical_probe": representation_probe,
        "fit_target_strengths": {
            "Alice": ALICE_STRENGTH,
            "Bob": BOB_STRENGTH,
        },
        "fit_observed_strengths": {
            "Alice": float(strengths[0].detach()),
            "Bob": float(strengths[1].detach()),
        },
        "training_method": "analytic minimum-norm affine fit to frozen token-row direction",
        "base_model_optimization": False,
        "adapter_parameter_count": restored.parameter_count,
    }


def query() -> FactorizedCanonicalQuery:
    return FactorizedCanonicalQuery(
        entity=ByteEntityEncoder(128)(["silver key"]),
        relation_logits=torch.tensor([[8.0, -8.0, -8.0]]),
        metadata_logits=torch.tensor([[8.0, -8.0, -8.0, -8.0]]),
    )


def make_store(
    representation,
    *,
    value_id: int,
    label: str = "silver key",
    relation_id: int = OWNER_RELATION,
    metadata_id: int = CURRENT_METADATA,
    invalidate: bool = False,
) -> CanonicalPStore:
    store = CanonicalPStore(CanonicalPConfig(
        slots=1, width=512, dtype=torch.float32, merge_similarity=1.0
    ))
    with torch.inference_mode():
        vector = representation.encode(
            torch.tensor([0]),
            torch.tensor([relation_id]),
            torch.tensor([value_id]),
            torch.tensor([metadata_id]),
        )[0]
    slot, _ = store.create(
        vector,
        entity_id=0,
        relation_id=relation_id,
        value_id=value_id,
        metadata_id=metadata_id,
        label=label,
    )
    if invalidate:
        store.invalidate(slot)
    return store


def route_and_translate(
    router: CanonicalPRouter,
    package: LlamaGGUFTranslatePackage,
    representation,
    condition: str,
) -> dict[str, object]:
    specifications = {
        "correct_alice": dict(value_id=ALICE_VALUE),
        "correct_bob": dict(value_id=BOB_VALUE),
        "wrong_entity": dict(value_id=ALICE_VALUE, label="gold key"),
        "wrong_relation": dict(value_id=ALICE_VALUE, relation_id=LOCATION_RELATION),
        "historical": dict(value_id=ALICE_VALUE, metadata_id=HISTORICAL),
        "invalidated": dict(value_id=ALICE_VALUE, invalidate=True),
        "translator_disabled": dict(value_id=ALICE_VALUE),
    }
    if condition in {"p_disabled", "router_disabled"}:
        return {
            "selected_state": None,
            "router_score": None,
            "router_accepted": False,
            "gate": 0.0,
            "strength": 0.0,
        }
    store = make_store(representation, **specifications[condition])
    index = router.build_index(store, ByteEntityEncoder(128), device="cpu")
    route = router.route(query(), index, top_k=1)
    accepted = bool(route.accepted[0]) if route.accepted.numel() else False
    score = float(route.scores[0, 0].detach()) if route.has_valid else None
    selected_state = None
    gate = 0.0
    strength = 0.0
    if route.has_valid:
        selected_index = int(route.indices[0, 0])
        selected_state = {
            "label": store.cache.labels[selected_index],
            "relation_id": int(store.relation_id[selected_index]),
            "value_id": int(store.value_id[selected_index]),
            "metadata_id": int(store.canonical_metadata_id[selected_index]),
        }
        if condition != "translator_disabled":
            translated = package(
                store.canonical_values[selected_index:selected_index + 1],
                store.value_id[selected_index:selected_index + 1],
                route_accepted=accepted,
            )
            gate = float(translated.gate[0])
            strength = float(translated.strength[0].detach())
    return {
        "selected_state": selected_state,
        "router_score": score,
        "router_accepted": accepted,
        "gate": gate,
        "strength": strength,
    }


def run_causal_runner(
    binary: Path,
    model_path: Path,
    direction_path: Path,
    scales: list[float],
    *,
    prompt: str,
) -> dict[str, object]:
    command = [
        str(binary),
        "--model", str(model_path),
        "--direction", str(direction_path),
        "--prompt", prompt,
        "--scales", ",".join(str(value) for value in scales),
        "--layer", str(ATTACHMENT_LAYER),
        "--alice-token", str(ALICE_TOKEN),
        "--bob-token", str(BOB_TOKEN),
        "--gpu-layers", "12",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return {"command": command, "result": json.loads(completed.stdout)}


def run_server_completion(
    llama_server: Path,
    model_path: Path,
    control_vector: Path,
    strength: float,
) -> dict[str, object]:
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    command = [
        str(llama_server),
        "--model", str(model_path),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--ctx-size", "512",
        "--parallel", "1",
        "--threads", "8",
        "--threads-batch", "8",
        "--gpu-layers", "12",
        "--no-warmup",
        "--no-context-shift",
        "--log-disable",
        "--control-vector-scaled", f"{control_vector}:{strength}",
        "--control-vector-layer-range", str(ATTACHMENT_LAYER), str(ATTACHMENT_LAYER),
    ]
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 90
        while True:
            if process.poll() is not None:
                raise RuntimeError("llama-server exited before becoming ready")
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                    break
            except Exception:
                if time.monotonic() >= deadline:
                    raise TimeoutError("llama-server did not become ready")
                time.sleep(0.25)
        body = json.dumps({
            "prompt": PROMPT,
            "n_predict": 1,
            "temperature": 0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "cache_prompt": False,
            "seed": 1234,
        }).encode()
        request = Request(
            f"http://127.0.0.1:{port}/completion",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
        return {
            "command": command,
            "content": result["content"],
            "prompt_tokens": result["tokens_evaluated"],
            "prompt_ms": result["timings"]["prompt_ms"],
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("artifacts/gemma4-e4b-q8-llama.translate"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gemma4-e4b-q8-causal.json"),
    )
    args = parser.parse_args()
    llama_root = args.llama_root.resolve()
    model_path = args.model.resolve()
    llama_cli = llama_root / "build" / "bin" / "llama-cli"
    llama_server = llama_root / "build" / "bin" / "llama-server"
    if not llama_cli.is_file() or not llama_server.is_file() or not model_path.is_file():
        raise FileNotFoundError("llama.cpp binaries or Gemma GGUF are unavailable")

    original_size = model_path.stat().st_size
    original_mtime = model_path.stat().st_mtime_ns
    hash_started = time.perf_counter()
    model_sha256 = file_sha256(model_path)
    hash_seconds = time.perf_counter() - hash_started
    gguf_module = load_gguf(llama_root)
    model_metadata, direction = gguf_metadata_and_direction(gguf_module, model_path)
    build = llama_version(llama_cli)
    args.adapter.parent.mkdir(parents=True, exist_ok=True)
    package, representation, training = build_adapter(
        model_path=model_path,
        model_sha256=model_sha256,
        model_metadata=model_metadata,
        model_direction=direction,
        llama_build=build,
        output=args.adapter,
    )
    package.validate_compatibility(
        model_id=str(model_metadata["name"]),
        model_architecture=str(model_metadata["architecture"]),
        model_hidden_width=int(model_metadata["embedding_length"]),
        model_layer_count=int(model_metadata["block_count"]),
        model_sha256=model_sha256,
    )

    router = CanonicalPRouter.load(
        Path(__file__).resolve().parents[1] / "artifacts" / "canonical-p-v1.router"
    )
    condition_names = (
        "p_disabled",
        "correct_alice",
        "correct_bob",
        "wrong_entity",
        "wrong_relation",
        "historical",
        "invalidated",
        "router_disabled",
        "translator_disabled",
    )
    routes = {
        name: route_and_translate(router, package, representation, name)
        for name in condition_names
    }
    scales = [float(routes[name]["strength"]) for name in condition_names]

    with tempfile.TemporaryDirectory(prefix="planner-cache-gemma-") as directory:
        workspace = Path(directory)
        runner = workspace / "llama_cpp_causal_runner"
        compilation = compile_runner(llama_root, runner)
        raw_direction = workspace / "gemma-direction.f32"
        package.model_direction.detach().cpu().numpy().astype(np.float32).tofile(raw_direction)
        control_vector = workspace / "gemma-direction.gguf"
        write_control_vector(
            gguf_module,
            control_vector,
            package.model_direction.detach().cpu().numpy(),
            model_hint=package.config.model_id,
        )
        causal_run = run_causal_runner(
            runner, model_path, raw_direction, scales, prompt=PROMPT
        )
        natural_run = run_causal_runner(
            runner, model_path, raw_direction, [0.0, 0.0, 0.0], prompt=NATURAL_RP_PROMPT
        )
        server_verification = {
            "alice": run_server_completion(
                llama_server, model_path, control_vector,
                float(routes["correct_alice"]["strength"])
            ),
            "bob": run_server_completion(
                llama_server, model_path, control_vector,
                float(routes["correct_bob"]["strength"])
            ),
        }

    measured = causal_run["result"]["conditions"]
    conditions = {}
    for name, route, measurement in zip(condition_names, routes.values(), measured):
        conditions[name] = {**route, **measurement}
    baseline = conditions["p_disabled"]
    inactive = (
        "wrong_entity", "wrong_relation", "historical", "invalidated",
        "router_disabled", "translator_disabled",
    )
    assertions = {
        "alice_generation": conditions["correct_alice"]["generated_token_id"] == ALICE_TOKEN,
        "bob_generation": conditions["correct_bob"]["generated_token_id"] == BOB_TOKEN,
        "alice_logit_lift": conditions["correct_alice"]["alice_logit"] > baseline["alice_logit"],
        "bob_logit_lift": conditions["correct_bob"]["bob_logit"] > baseline["bob_logit"],
        "all_inactive_paths_exact": all(
            conditions[name]["max_abs_logit_difference_from_base"] == 0
            and conditions[name]["generated_token_id"] == baseline["generated_token_id"]
            for name in inactive
        ),
        "wrong_entity_rejected": not conditions["wrong_entity"]["router_accepted"],
        "wrong_relation_rejected": not conditions["wrong_relation"]["router_accepted"],
        "historical_rejected": not conditions["historical"]["router_accepted"],
        "invalidated_rejected": not conditions["invalidated"]["router_accepted"],
        "natural_irrelevant_exact": all(
            item["max_abs_logit_difference_from_base"] == 0
            for item in natural_run["result"]["conditions"]
        ),
        "server_alice_generation": server_verification["alice"]["content"] == " Alice",
        "server_bob_generation": server_verification["bob"]["content"] == " Bob",
    }
    if not all(assertions.values()):
        raise AssertionError(json.dumps(assertions, indent=2, sort_keys=True))
    if model_path.stat().st_size != original_size or model_path.stat().st_mtime_ns != original_mtime:
        raise AssertionError("base GGUF changed during adapter construction or evaluation")

    artifact = {
        "experiment": "gemma4-e4b-q8-llama-translate-causal-v1",
        "status": "passed",
        "model": {
            **model_metadata,
            "path": str(model_path),
            "size_bytes": original_size,
            "sha256": model_sha256,
            "sha256_seconds": hash_seconds,
            "quantized_gguf": True,
            "base_modified": False,
        },
        "llama_cpp": {
            "root": str(llama_root),
            "cli": str(llama_cli),
            "server": str(llama_server),
            "version": build,
            "control_vector_generator_failure": (
                "stock generator asserted because Gemma4 did not expose n_layers minus one callback tensors"
            ),
            "runtime_attachment": "public llama_set_adapter_cvec API and llama-server control-vector path",
        },
        "adapter": {
            "path": str(args.adapter.resolve()),
            "size_bytes": args.adapter.stat().st_size,
            "sha256": file_sha256(args.adapter),
            "config": asdict(package.config),
            **training,
            "active_control_buffer_bytes": (
                (package.config.model_layer_count - 1)
                * package.config.model_hidden_width
                * 4
            ),
            "bytes_copied_per_relevant_activation": (
                package.config.model_hidden_width * 4
            ),
            "inactive_vram_bytes": 0,
            "extra_prompt_tokens": 0,
            "recent_kv_source_token_count": 0,
            "model_hidden_query_projection": "not available through llama-server and not claimed",
            "gate_basis": "universal canonical route acceptance plus supported canonical value",
        },
        "prompt": PROMPT,
        "prompt_tokens": causal_run["result"]["prompt_token_count"],
        "conditions": conditions,
        "natural_rp": {
            "prompt": NATURAL_RP_PROMPT,
            "conditions": natural_run["result"]["conditions"],
        },
        "server_verification": server_verification,
        "runner": {
            "compilation": compilation,
            "causal_command": causal_run["command"],
            "natural_command": natural_run["command"],
        },
        "assertions": assertions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
