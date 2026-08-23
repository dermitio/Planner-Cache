"""Run the first deterministic disk-resident P-package proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcm.planner.personality_eval import run_personality_package_experiment
from pcm.planner.representation import train_and_probe_representation


def completion_gate(result: dict[str, object]) -> tuple[bool, list[str]]:
    mechanical = result["mechanical"]
    growth = result["growth"]
    durability = result["durability"]
    cuda = result["cuda_ttl"]
    checks = {
        "one-event non-promotion": not mechanical["one_event_promoted"],
        "repeated promotion": mechanical["repeated_promoted"],
        "connectivity": mechanical["connectivity_outscores_narrow"],
        "authority": not mechanical["unsupported_model_claim_promoted"],
        "explicit correction": mechanical["explicit_correction_promoted"],
        "context routing": mechanical["technical_context_correct"] and mechanical["creative_context_correct"],
        "relationship routing": mechanical["relationship_context_correct"],
        "irrelevant rejection": mechanical["irrelevant_loaded_entries"] == 0,
        "durability": durability["checksum_valid"] and not durability["conversation_replay_required"],
        "deterministic serialization": result["deterministic_serialization"]["byte_identical"],
        "growth is disk resident": all(row["inactive_vram_bytes"] == 0 for row in growth.values()),
        "growth active cost is top-k bounded": all(
            row["loaded_entries"] <= 4
            and row["active_canonical_bytes"] == row["loaded_entries"] * 1077
            and not row["full_package_loaded"]
            for row in growth.values()
        ),
        "CUDA benchmark present": cuda is not None,
    }
    if cuda is not None:
        candidate = cuda["candidate_accuracy"]
        base_rp = cuda["natural_interaction"]["frozen_base"]
        irrelevant_rp = cuda["natural_interaction"]["p_package_irrelevant"]
        checks.update({
            "P-only causal A/B": candidate["package_a_alice"] == 1 and candidate["package_b_bob"] == 1,
            "contextual causal routing": candidate["context_technical_alice"] == 1 and candidate["context_creative_bob"] == 1,
            "irrelevant causal identity": candidate["irrelevant_matches_base"] == 1,
            "irrelevant RP preservation": abs(irrelevant_rp["loss"] - base_rp["loss"]) / base_rp["loss"] < 0.05,
            "frozen base": cuda["base_parameters_with_grad"] == 0,
            "inactive VRAM": cuda["inactive_vram_delta_bytes"] == 0,
            "selected-only CUDA": not cuda["full_package_uploaded_to_cuda"],
            "relevant personality improves target": (
                cuda["relevant_personality_chat"]["target_accuracy"] == 1
                and cuda["relevant_personality_chat"]["package_target_loss"]
                < cuda["relevant_personality_chat"]["base_target_loss"]
            ),
        })
    failures = [label for label, passed in checks.items() if not passed]
    return not failures, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("pythia-1.4b"))
    parser.add_argument(
        "--ttl", type=Path,
        default=Path("artifacts/pythia-1.4b-final-layer.ttl"),
    )
    parser.add_argument(
        "--router", type=Path, default=Path("artifacts/canonical-p-v1.router")
    )
    parser.add_argument(
        "--package", type=Path, default=Path("artifacts/personality-proof.ppkg")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/phase-b-personality-package.json")
    )
    parser.add_argument(
        "--growth-counts", default="100,1000,10000,100000",
        help="comma-separated package sizes",
    )
    parser.add_argument("--skip-cuda", action="store_true")
    args = parser.parse_args()
    representation = None
    probe = None
    if not args.skip_cuda:
        representation, _, probe = train_and_probe_representation()
    result = run_personality_package_experiment(
        output_package=args.package,
        model_path=None if args.skip_cuda else args.model,
        ttl_path=None if args.skip_cuda else args.ttl,
        router_path=None if args.skip_cuda else args.router,
        representation=representation,
        growth_counts=tuple(int(value) for value in args.growth_counts.split(",")),
    )
    result["canonical_representation_probe"] = probe
    passed, failures = completion_gate(result)
    result["completion_gate"] = "passed" if passed else "failed"
    result["completion_failures"] = failures
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output), "package": str(args.package),
        "completion_gate": result["completion_gate"], "failures": failures,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
