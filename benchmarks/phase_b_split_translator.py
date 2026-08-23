"""Run the matched Phase-B split-router/translator layer sweep."""

import argparse
import json
from pathlib import Path
import shutil

from pcm.planner.representation import train_and_probe_representation
from pcm.planner.split_translator_eval import run_split_translator_experiment


def rejected_translator_baseline():
    """Immutable baseline copied from the archived rejected-translator artifact."""
    return {
        "final_layer": {
            "global_20_fact_retrieval": 0.6500000357627869,
            "p_to_model_value_accuracy": 1.0,
            "exact_generated_token_accuracy": 0.45000001788139343,
            "unseen_name_accuracy": 0.25,
            "wrong_entity_gate": 0.5172024965286255,
            "rp_loss": 4.656857490539551,
        },
        "upper_2_layers": {
            "global_20_fact_retrieval": 0.6500000357627869,
            "p_to_model_value_accuracy": 1.0,
            "exact_generated_token_accuracy": 0.550000011920929,
            "unseen_name_accuracy": 0.0,
            "wrong_entity_gate": 0.6912492662668228,
            "rp_loss": 6.086124897003174,
        },
        "upper_4_layers": {
            "global_20_fact_retrieval": 0.699999988079071,
            "p_to_model_value_accuracy": 1.0,
            "exact_generated_token_accuracy": 0.6500000357627869,
            "unseen_name_accuracy": 0.25,
            "wrong_entity_gate": 0.7020304128527641,
            "rp_loss": 5.704916477203369,
        },
    }


def passes(result):
    scale = result["router"]["scaling"]["128"]
    hard = result["router"]["hard_negative_metrics"]
    full = result["ablations"]["full_system_with_preservation"]
    counter = result["counterfactual"]
    base = counter["disabled"]
    p1 = counter["p1_silver_alice"]
    p2 = counter["p2_silver_bob"]
    p3 = counter["p3_gold_alice"]
    historical = counter["p4_silver_historical"]
    rp = result["natural_rp"]["conditions"]
    rp_degradation = (
        rp["irrelevant"]["loss"] - rp["base"]["loss"]
    ) / rp["base"]["loss"]
    return all((
        scale["top1_accuracy"] > 0.9,
        result["query_projector"]["byte_surface_anchor_approach"]["entity_accuracy"] > 0.8,
        hard["wrong_entity_false_positive_rate"] < 0.05,
        hard["wrong_relation_false_positive_rate"] < 0.05,
        hard["historical_false_positive_rate"] < 0.05,
        hard["invalidated_false_positive_rate"] == 0,
        full["state_candidate_accuracy"] > 0.8,
        full["full_token_accuracy"] > 0.8,
        p1["alice_logit"] > p1["bob_logit"],
        p2["bob_logit"] > p2["alice_logit"],
        p3["alice_logit"] == base["alice_logit"],
        historical["alice_logit"] == base["alice_logit"],
        result["invalidated_logit_difference"] == 0,
        result["mutation_chain"]["latest_state_accuracy"] == 1,
        result["mutation_chain"]["invalidated_max_logit_difference"] == 0,
        rp_degradation <= 0.05,
        result["base_parameters_with_grad"] == 0,
        result["package_roundtrip_max_difference"] == 0,
        result["router_roundtrip_max_difference"] == 0,
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("pythia-1.4b"))
    parser.add_argument("--query-steps", type=int, default=400)
    parser.add_argument("--router-steps", type=int, default=400)
    parser.add_argument("--value-steps", type=int, default=400)
    parser.add_argument("--causal-steps", type=int, default=256)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    representation, _, probe = train_and_probe_representation()
    variants = {}
    for count, label in ((1, "final_layer"), (2, "upper_2_layers"), (4, "upper_4_layers")):
        variants[label] = run_split_translator_experiment(
            args.model,
            representation,
            attachment_count=count,
            query_steps=args.query_steps,
            router_steps=args.router_steps,
            value_steps=args.value_steps,
            causal_steps=args.causal_steps,
            package_path=args.artifact_dir / f"pythia-1.4b-split-{label}.translate",
            router_path=args.artifact_dir / f"canonical-{label}.router",
        )
    passing = [label for label, result in variants.items() if passes(result)]
    selected = min(passing, key=lambda label: variants[label]["attachment_count"]) if passing else None
    diagnostic = selected or max(
        variants,
        key=lambda label: (
            variants[label]["router"]["scaling"]["128"]["top1_accuracy"],
            variants[label]["ablations"]["full_system_with_preservation"]["full_token_accuracy"],
            -variants[label]["attachment_count"],
        ),
    )
    if selected is not None:
        shutil.copyfile(
            args.artifact_dir / f"canonical-{selected}.router",
            args.artifact_dir / "canonical-p-v1.router",
        )
    payload = {
        "experiment": {
            "seed": 307,
            "query_steps": args.query_steps,
            "router_steps": args.router_steps,
            "value_steps": args.value_steps,
            "causal_steps": args.causal_steps,
            "base_frozen": True,
            "lora_used": False,
            "phase_c_started": False,
            "source_tokens_in_recent_kv": 0,
        },
        "canonical_p_probe": probe,
        "immutable_rejected_translator_baseline": rejected_translator_baseline(),
        "layer_sweep": variants,
        "selected_variant": selected,
        "diagnostic_variant": diagnostic,
        "completion_gate": "passed" if selected else "failed_reject_split_router_translator",
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n")
        print(json.dumps({
            "output": str(args.output), "selected_variant": selected,
            "completion_gate": payload["completion_gate"],
        }, sort_keys=True))
    else:
        print(encoded)


if __name__ == "__main__":
    main()
