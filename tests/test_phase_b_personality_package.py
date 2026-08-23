from pathlib import Path

import pytest
import torch

from pcm.planner.personality_eval import (
    build_proof_package,
    cuda_ttl_benchmark,
)
from pcm.planner.representation import train_and_probe_representation


pytestmark = [
    pytest.mark.slow_cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="exact `.ppkg` proof requires CUDA"),
]


def test_exact_ppkg_cold_selected_ttl_path(tmp_path):
    root = Path(__file__).parents[1]
    model_path = Path("pythia-1.4b")
    ttl_path = Path("artifacts/pythia-1.4b-final-layer.ttl")
    router_path = Path("artifacts/canonical-p-v1.router")
    if not all((root / path).exists() for path in (model_path, ttl_path, router_path)):
        pytest.skip("local Pythia/TTL proof artifacts are unavailable")
    package_path = tmp_path / "durable-personality.ppkg"
    build_proof_package(package_path)
    representation, _, _ = train_and_probe_representation()
    result = cuda_ttl_benchmark(
        model_path=model_path,
        ttl_path=ttl_path,
        router_path=router_path,
        proof_package_path=package_path,
        representation=representation,
        workdir=tmp_path,
    )
    candidate = result["candidate_accuracy"]
    assert candidate["package_a_alice"] == 1
    assert candidate["package_b_bob"] == 1
    assert candidate["context_technical_alice"] == 1
    assert candidate["context_creative_bob"] == 1
    assert candidate["irrelevant_matches_base"] == 1
    assert result["inactive_vram_delta_bytes"] == 0
    assert result["base_parameters_with_grad"] == 0
    assert not result["full_package_uploaded_to_cuda"]
    assert result["source_tokens_in_recent_kv"] == 0
    assert result["extra_prompt_tokens"] == 0
    for label, memory in result["active_memory"].items():
        assert memory["loaded_entries"] <= 1, label
        assert memory["canonical_store_bytes"] < memory["package_disk_bytes"]
    base_rp = result["natural_interaction"]["frozen_base"]
    for label in (
        "p_cache_only", "p_package_irrelevant",
        "p_package_contradictory_low_confidence",
    ):
        assert result["natural_interaction"][label]["loss"] == base_rp["loss"]
