"""Matched CUDA failure-attribution run for the active P-cache/.ppkg stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from pcm.planner.personality_eval import build_proof_package, cuda_ttl_benchmark
from pcm.planner.representation import train_and_probe_representation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/active-system-cuda-attribution.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    representation, _, probe = train_and_probe_representation()
    with tempfile.TemporaryDirectory(prefix="planner-cuda-audit-") as directory:
        workdir = Path(directory)
        proof = workdir / "personality.ppkg"
        build_proof_package(proof)
        result = cuda_ttl_benchmark(
            model_path=root / "pythia-1.4b",
            ttl_path=root / "artifacts/pythia-1.4b-final-layer.ttl",
            router_path=root / "artifacts/canonical-p-v1.router",
            proof_package_path=proof,
            representation=representation,
            workdir=workdir,
        )
    artifact = {
        "experiment": "active-system-cuda-failure-attribution-v1",
        "base_model": "pythia-1.4b",
        "base_training_performed": False,
        "adapter_training_performed": False,
        "canonical_representation_reconstructed_from_fixed_existing_recipe": True,
        "canonical_probe": probe,
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
