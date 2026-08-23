from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from pcm.planner.representation import train_and_probe_representation
from pcm.planner.split_translator import CanonicalPRouter, SplitPTranslatePackage
from pcm.planner.split_translator_eval import run_split_translator_experiment


pytestmark = [
    pytest.mark.slow_cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="exact split translator requires CUDA"),
]


def test_exact_split_router_translator_causal_portable_path(tmp_path):
    model_path = Path(__file__).parents[1] / "pythia-1.4b"
    if not model_path.exists():
        pytest.skip("local Pythia-1.4B checkpoint is unavailable")
    representation, _, _ = train_and_probe_representation(
        steps=180, evaluation_limit=48
    )
    package_path = tmp_path / "pythia-split.translate"
    router_path = tmp_path / "canonical.router"
    result = run_split_translator_experiment(
        model_path,
        representation,
        attachment_count=1,
        query_steps=50,
        router_steps=100,
        value_steps=50,
        causal_steps=40,
        package_path=package_path,
        router_path=router_path,
    )

    SplitPTranslatePackage.load(package_path)
    CanonicalPRouter.load(router_path)
    for path in (package_path, router_path):
        assert not any(
            forbidden in name
            for name in load_file(path)
            for forbidden in ("base_model", "canonical_values", "conversation", "p_cache")
        )
    assert result["query_projector"]["byte_surface_anchor_approach"]["entity_accuracy"] == 1
    hard = result["router"]["hard_negative_metrics"]
    assert hard["wrong_entity_false_positive_rate"] == 0
    assert hard["wrong_relation_false_positive_rate"] == 0
    assert hard["historical_false_positive_rate"] == 0
    assert hard["invalidated_false_positive_rate"] == 0
    assert result["router"]["scaling"]["128"]["top1_accuracy"] > 0.9
    assert result["router"]["scaling"]["512"]["top4_recall"] >= 0.75
    counter = result["counterfactual"]
    assert counter["p1_silver_alice"]["alice_logit"] > counter["p1_silver_alice"]["bob_logit"]
    assert counter["p2_silver_bob"]["bob_logit"] > counter["p2_silver_bob"]["alice_logit"]
    assert counter["p3_gold_alice"]["alice_logit"] == counter["disabled"]["alice_logit"]
    assert counter["p4_silver_historical"]["alice_logit"] == counter["disabled"]["alice_logit"]
    assert result["invalidated_logit_difference"] == 0
    assert result["mutation_chain"]["latest_state_accuracy"] == 1
    assert result["mutation_chain"]["invalidated_max_logit_difference"] == 0
    assert result["base_parameters_with_grad"] == 0
    assert result["package_roundtrip_max_difference"] == 0
    assert result["router_roundtrip_max_difference"] == 0
