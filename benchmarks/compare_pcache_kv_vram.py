"""Matched CUDA VRAM comparison for P-cache, retained KV, and their combination."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pcm.planner.cache import Freshness
from pcm.planner.canonical import CanonicalPConfig, CanonicalPStore
from pcm.planner.compatibility import TensorTranslationLayer
from pcm.planner.pythia_split_translate import PythiaSplitTranslatedModel
from pcm.planner.split_translator import ByteEntityEncoder, CanonicalPRouter


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Count unique tensor payload bytes in cache containers."""
    seen = set() if seen is None else seen
    if isinstance(value, torch.Tensor):
        pointer = value.untyped_storage().data_ptr()
        if pointer in seen:
            return 0
        seen.add(pointer)
        return value.numel() * value.element_size()
    if hasattr(value, "to_legacy_cache"):
        return tensor_bytes(value.to_legacy_cache(), seen)
    if hasattr(value, "layers"):
        return tensor_bytes(value.layers, seen)
    if hasattr(value, "keys") and hasattr(value, "values"):
        return tensor_bytes((value.keys, value.values), seen)
    if isinstance(value, dict):
        return sum(tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item, seen) for item in value)
    return 0


def canonical_store(slots: int, *, seed: int) -> CanonicalPStore:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    store = CanonicalPStore(CanonicalPConfig(
        slots=slots, width=512, dtype=torch.float32, device="cpu",
        merge_similarity=1.0,
    ))
    values = torch.randn(slots, 512, generator=generator)
    store.cache.values.copy_(torch.nn.functional.normalize(values, dim=-1))
    store.cache.valid.fill_(True)
    store.cache.labels[:] = [f"entity-{index}" for index in range(slots)]
    store.entity_id.copy_(torch.arange(slots) % 24)
    store.relation_id.copy_(torch.arange(slots) % 3)
    store.value_id.copy_(torch.arange(slots) % 36)
    store.cache.freshness.fill_(int(Freshness.FRESH))
    return store


def canonical_store_bytes(store: CanonicalPStore) -> int:
    tensors = (
        store.cache.values, store.cache.valid, store.cache.slot_type,
        store.cache.confidence, store.cache.importance, store.cache.freshness,
        store.cache.persistence, store.cache.last_updated, store.cache.source,
        store.entity_id, store.relation_id, store.value_id,
        store.canonical_metadata_id,
    )
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def prompt_ids(tokenizer, length: int, device: torch.device) -> torch.Tensor:
    seed = tokenizer(
        "The current state remains relevant.", add_special_tokens=False,
    ).input_ids
    if not seed:
        seed = [tokenizer.eos_token_id]
    repeated = (seed * ((length + len(seed) - 1) // len(seed)))[:length]
    return torch.tensor([repeated], dtype=torch.long, device=device)


def generate(
    model: PythiaSplitTranslatedModel,
    input_ids: torch.Tensor,
    *,
    p_store: CanonicalPStore | None,
    use_cache: bool,
    generated_tokens: int,
) -> tuple[int, int, list[int]]:
    running = input_ids
    past = None
    generated: list[int] = []
    with torch.inference_mode():
        for _step in range(generated_tokens):
            if use_cache and past is not None:
                current = running[:, -1:]
                attention_mask = torch.ones(
                    (1, running.shape[1]), dtype=torch.long, device=running.device,
                )
            else:
                current = running
                attention_mask = torch.ones_like(current)
            output = model(
                input_ids=current,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=use_cache,
                p_store=p_store,
                query_entity_surfaces=("entity-0",) if p_store is not None else None,
            )
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(int(token.item()))
            if use_cache:
                past = output.past_key_values
            running = torch.cat((running, token), dim=1)
            del output, token, current, attention_mask
    return tensor_bytes(past), running.shape[1], generated


def run_condition(
    model: PythiaSplitTranslatedModel,
    tokenizer,
    *,
    condition: str,
    prompt_length: int,
    slots: int,
    generated_tokens: int,
    seed: int,
) -> dict[str, object]:
    use_p = condition in {"p_cache_only", "p_cache_plus_kv"}
    use_cache = condition in {"kv_only", "p_cache_plus_kv"}
    store = canonical_store(slots, seed=seed) if use_p else None
    p_bytes = 0 if store is None else canonical_store_bytes(store)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    failure = None
    kv_bytes = 0
    final_length = prompt_length
    generated: list[int] = []
    try:
        inputs = prompt_ids(tokenizer, prompt_length, torch.device("cuda"))
        kv_bytes, final_length, generated = generate(
            model, inputs, p_store=store, use_cache=use_cache,
            generated_tokens=generated_tokens,
        )
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as error:
        failure = {"type": "CUDAOutOfMemoryError", "message": str(error)}
        torch.cuda.synchronize()
    duration = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    result = {
        "condition": condition,
        "prompt_tokens": prompt_length,
        "requested_generated_tokens": generated_tokens,
        "generated_tokens": len(generated),
        "final_sequence_tokens": final_length,
        "p_cache_slots": slots if use_p else 0,
        "p_cache_canonical_bytes": p_bytes,
        "retained_kv_cache_bytes": kv_bytes,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_allocated_bytes": peak_allocated - baseline_allocated,
        "incremental_peak_reserved_bytes": peak_reserved - baseline_reserved,
        "runtime_seconds": duration,
        "batch_size": 1,
        "precision": "float16 base and float32 TTL",
        "generation": "greedy argmax",
        "retained_kv_enabled": use_cache,
        "p_cache_enabled": use_p,
        "failure": failure,
        "fallback": None,
    }
    del store
    gc.collect()
    torch.cuda.empty_cache()
    return result


def system_details() -> dict[str, object]:
    try:
        driver = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=driver_version",
                "--format=csv,noheader", "--id=0",
            ], check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        driver = None
    properties = torch.cuda.get_device_properties(0)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": driver,
        "gpu": properties.name,
        "gpu_total_bytes": properties.total_memory,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ttl", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workloads", default="64,256,1024")
    parser.add_argument("--generated-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=317)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the VRAM comparison")

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    ttl = TensorTranslationLayer.load(
        args.ttl, device="cuda", dtype=torch.float32,
    ).eval()
    router = CanonicalPRouter.load(args.router, device="cuda").eval()
    wrapper = PythiaSplitTranslatedModel(
        base, ttl, router, ByteEntityEncoder(128),
    ).to("cuda").eval()

    # Materialize lazy CUDA library, attention, router, TTL, and cache state
    # before recording any condition baseline. The warm-up allocations are
    # released so every measured row starts from the same loaded-stack state.
    warm_store = canonical_store(1, seed=args.seed)
    warm_input = prompt_ids(tokenizer, 8, torch.device("cuda"))
    generate(
        wrapper, warm_input, p_store=warm_store, use_cache=True,
        generated_tokens=1,
    )
    del warm_store, warm_input
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    workloads = tuple(int(value) for value in args.workloads.split(","))
    results = []
    for size in workloads:
        for condition in ("p_cache_only", "kv_only", "p_cache_plus_kv"):
            results.append(run_condition(
                wrapper, tokenizer, condition=condition,
                prompt_length=size, slots=size,
                generated_tokens=args.generated_tokens, seed=args.seed + size,
            ))
    output = {
        "experiment": "planner-cache-matched-vram-comparison-v1",
        "method": {
            "model_load": "one frozen model, TTL, and router reused for every condition",
            "baseline": "CUDA allocation after loaded stack and empty_cache",
            "warmup": "all measured mechanisms exercised once before baselines",
            "peak": "torch.cuda reset_peak_memory_stats and synchronized measurement",
            "p_cache_only": "retained KV disabled while P-cache and TTL are active",
            "kv_only": "retained KV active while P-cache and TTL injection are disabled",
            "combined": "retained KV and P-cache TTL path active together",
            "attention_working_memory": (
                "present in every condition and distinct from retained KV cache"
            ),
        },
        "environment": system_details(),
        "model": {
            "identifier": args.model.name,
            "config_sha256": sha256(args.model / "config.json"),
            "ttl_filename": args.ttl.name,
            "ttl_sha256": sha256(args.ttl),
            "router_filename": args.router.name,
            "router_sha256": sha256(args.router),
        },
        "shared_configuration": {
            "batch_size": 1,
            "precision": "float16 base and float32 TTL",
            "generated_tokens": args.generated_tokens,
            "generation": "greedy argmax",
            "seed": args.seed,
            "workload_prompt_and_slot_sizes": list(workloads),
        },
        "results": results,
        "completion": {
            "conditions": len(results),
            "successful": sum(row["failure"] is None for row in results),
            "failed": sum(row["failure"] is not None for row in results),
            "oom_events": sum(
                row["failure"] is not None
                and row["failure"]["type"] == "CUDAOutOfMemoryError"
                for row in results
            ),
            "estimated_values": 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["completion"], sort_keys=True))


if __name__ == "__main__":
    main()
