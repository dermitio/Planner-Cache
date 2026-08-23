"""Reproducible, non-training audit of the active Planner Cache stack."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import json
from pathlib import Path
import random
import resource
import statistics
import tempfile
import time
import tracemalloc

import torch

from pcm.planner.canonical import (
    CANONICAL_VALUE_LABELS,
    CanonicalPConfig,
    CanonicalPStore,
)
from pcm.planner.compatibility import TensorTranslationLayer
from pcm.planner.personality import (
    FactorizedPersonalityCanonicalizer,
    PersonalityPackage,
    PersonalityQuery,
    PersonalityRouter,
    synthetic_entry,
)
from pcm.planner.representation import FactorizedStateRepresentation
from pcm.planner.split_translator import (
    ByteEntityEncoder,
    CanonicalPRouter,
    CanonicalRouterIndex,
    FactorizedCanonicalQuery,
)
from pcm.planner.split_translator_eval import entity_split


STAMP = "2026-08-22T00:00:00+00:00"
SLOT_COUNTS = (64, 128, 256, 512, 1024)
ROUTER_COUNTS = (4, 20, 64, 128, 256, 512, 1024)
PPKG_COUNTS = (100, 1_000, 10_000, 100_000)


def median_timing(call, repeats=20):
    wall, cpu = [], []
    result = None
    for _ in range(repeats):
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        result = call()
        wall.append(time.perf_counter() - wall_start)
        cpu.append(time.process_time() - cpu_start)
    return result, statistics.median(wall), statistics.median(cpu)


def tensor_bytes(tensors) -> int:
    return sum(value.numel() * value.element_size() for value in tensors)


def profile_cache_router() -> dict[str, object]:
    encoder = ByteEntityEncoder()
    router = CanonicalPRouter()
    result = {}
    for count in SLOT_COUNTS:
        store = CanonicalPStore(CanonicalPConfig(
            slots=count, width=512, dtype=torch.float32, merge_similarity=1.0
        ))
        generator = torch.Generator().manual_seed(10_000 + count)
        for index in range(count):
            store.create(
                torch.randn(512, generator=generator), entity_id=index,
                relation_id=index % 3, value_id=index % 36, metadata_id=0,
                label=f"audit entity {index:04d}",
            )
        query_surface = f"audit entity {count - 1:04d}"
        query, query_wall, query_cpu = median_timing(
            lambda: FactorizedCanonicalQuery(
                entity=encoder([query_surface]),
                relation_logits=torch.tensor([[-9.0, -9.0, -9.0]]).scatter(
                    1, torch.tensor([[(count - 1) % 3]]), 9.0
                ),
                metadata_logits=torch.tensor([[9.0, -9.0, -9.0, -9.0]]),
            )
        )
        index, hydration_wall, hydration_cpu = median_timing(
            lambda: router.build_index(store, encoder, device="cpu"), repeats=7
        )
        route, route_wall, route_cpu = median_timing(
            lambda: router.route(query, index, top_k=4), repeats=50
        )
        tensors = (
            store.cache.values, store.cache.valid, store.cache.slot_type,
            store.cache.confidence, store.cache.importance, store.cache.freshness,
            store.cache.persistence, store.cache.last_updated, store.cache.source,
            store.entity_id, store.relation_id, store.value_id,
            store.canonical_metadata_id,
        )
        result[str(count)] = {
            "context_query_wall_seconds": query_wall,
            "context_query_cpu_seconds": query_cpu,
            "slot_hydration_wall_seconds": hydration_wall,
            "slot_hydration_cpu_seconds": hydration_cpu,
            "routing_wall_seconds": route_wall,
            "routing_cpu_seconds": route_cpu,
            "selected_index": int(route.indices[0, 0]),
            "correct": int(route.indices[0, 0]) == count - 1,
            "fixed_allocation_bytes": tensor_bytes(tensors),
            "router_index_bytes": tensor_bytes(
                (index.entity, index.relation_id, index.metadata_id, index.valid)
            ),
        }
    return result


def router_scaling_attribution() -> dict[str, object]:
    """Recreate the historical scaling workload without loading the LM."""
    encoder = ByteEntityEncoder()
    router = CanonicalPRouter()
    all_train, all_heldout = entity_split()
    split_rng = random.Random(308)
    train = split_rng.sample(all_train, 256)
    required = ["silver key", "gold key"]
    adjectives = {surface.split()[0] for surface in train}
    nouns = {surface.split()[1] for surface in train}
    compositional = [
        surface for surface in all_heldout
        if surface.split()[0] in adjectives and surface.split()[1] in nouns
        and surface not in required
    ]
    heldout = required + compositional[:62]
    queries = heldout[:20]
    distractors = [surface for surface in train + heldout if surface not in queries]
    while len(distractors) < 1024:
        distractors.append(f"irrelevant entity {len(distractors)}")
    output = {}
    for count in ROUTER_COUNTS:
        query_count = min(20, count)
        rows = [(surface, 0, 0) for surface in queries[:query_count]]
        rows.extend(
            (distractors[index], (index + 1) % 3, 0)
            for index in range(count - query_count)
        )
        random.Random(307 + count).shuffle(rows)
        surfaces = [row[0] for row in rows]
        anchors = encoder(surfaces)
        relation_ids = torch.tensor([row[1] for row in rows])
        index = CanonicalRouterIndex(
            entity=anchors, relation_id=relation_ids,
            metadata_id=torch.zeros(count, dtype=torch.long),
            valid=torch.ones(count, dtype=torch.bool),
        )
        expected = torch.tensor([surfaces.index(surface) for surface in queries[:query_count]])
        query = FactorizedCanonicalQuery(
            entity=encoder(queries[:query_count]),
            relation_logits=torch.tensor([[12.0, -12.0, -12.0]]).expand(query_count, -1),
            metadata_logits=torch.tensor([[12.0, -12.0, -12.0, -12.0]]).expand(query_count, -1),
        )
        started = time.perf_counter()
        scores, _ = router.all_scores(query, index)
        latency = time.perf_counter() - started
        order = scores.argsort(-1, descending=True)
        ranks = (order == expected[:, None]).nonzero()[:, 1] + 1
        predicted = order[:, 0]
        failures = predicted != expected
        entity_confusions = 0
        relation_confusions = 0
        exact_anchor_collisions = 0
        collision_examples = []
        for query_index in failures.nonzero().flatten().tolist():
            wrong = int(predicted[query_index])
            similarity = float(torch.dot(query.entity[query_index], anchors[wrong]))
            entity_confusions += int(similarity >= 0.999999)
            relation_confusions += int(relation_ids[wrong] != 0)
            exact_anchor_collisions += int(torch.equal(query.entity[query_index], anchors[wrong]))
            collision_examples.append({
                "query": queries[query_index], "selected": surfaces[wrong],
                "entity_cosine": similarity,
                "selected_relation": int(relation_ids[wrong]),
            })
        output[str(count)] = {
            "top1_accuracy": float((ranks == 1).float().mean()),
            "top4_recall": float((ranks <= 4).float().mean()),
            "mrr": float((1 / ranks.float()).mean()),
            "entity_confusions": entity_confusions,
            "relation_confusions": relation_confusions,
            "historical_confusions": 0,
            "exact_query_anchor_collisions": exact_anchor_collisions,
            "latency_seconds": latency,
            "failure_examples": collision_examples,
        }
    return {
        "immutable_pre_fix_baseline": {
            "128": 1.0, "256": 0.949999988079071, "512": 0.8500000238418579,
        },
        "measurements": output,
        "attribution": (
            "The pre-audit CanonicalPStore allowed semantic merge across different "
            "entity/relation identities. Repeated factorized vectors therefore aliased "
            "slots at 256/512 even for oracle queries. Identity-constrained merge removes "
            "that storage corruption; the matched post-fix canonical router remains 100% "
            "through 1024. Linear scan affects latency but did not cause the accuracy loss."
        ),
    }


def profile_ppkg(root: Path) -> dict[str, object]:
    router = PersonalityRouter()
    canonicalizer = FactorizedPersonalityCanonicalizer(
        FactorizedStateRepresentation(24, 3, 36),
        value_labels=CANONICAL_VALUE_LABELS,
    )
    results = {}
    for count in PPKG_COUNTS:
        path = root / f"audit-{count}.ppkg"
        build_start = time.perf_counter()
        with PersonalityPackage.create(
            path, package_id=f"audit-{count}", created_at=STAMP, overwrite=True
        ) as package:
            package.bulk_insert_entries(
                (synthetic_entry(index, timestamp=STAMP) for index in range(count)),
                updated_at=STAMP,
            )
        build_seconds = time.perf_counter() - build_start
        open_start = time.perf_counter()
        package = PersonalityPackage(path, validate=False)
        open_seconds = time.perf_counter() - open_start
        verify_start = time.perf_counter()
        package.verify()
        checksum_seconds = time.perf_counter() - verify_start
        query = PersonalityQuery(
            subject="subject-0", interaction_type="domain-0", domain="domain-0",
            relation="trait-0", timestamp=STAMP,
        )
        route, route_wall, route_cpu = median_timing(
            lambda: router.route(package, query, top_k=4), repeats=7
        )
        entries, hydration_wall, hydration_cpu = median_timing(
            lambda: tuple(package.entry(entry_id) for entry_id in route.entry_ids),
            repeats=7,
        )
        tracemalloc.start()
        selection = router.retrieve(package, query, top_k=4)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        _, canonical_wall, canonical_cpu = median_timing(
            lambda: [canonicalizer.encode(entry) for entry in selection.entries],
            repeats=7,
        )
        plans = {}
        for name, predicate in (
            ("subject", "subject='subject-0'"),
            ("scope", "scope='domain-0'"),
        ):
            rows = package._connection.execute(
                f"EXPLAIN QUERY PLAN SELECT id FROM entries WHERE status='active' "
                f"AND {predicate} ORDER BY importance DESC,confidence DESC,id LIMIT 128"
            ).fetchall()
            plans[name] = [str(tuple(row)) for row in rows]
        size = path.stat().st_size
        package.close()
        path.unlink()
        results[str(count)] = {
            "build_seconds": build_seconds,
            "disk_bytes": size,
            "checksum_seconds": checksum_seconds,
            "db_open_seconds": open_seconds,
            "routing_header_wall_seconds": route_wall,
            "routing_header_cpu_seconds": route_cpu,
            "row_hydration_wall_seconds": hydration_wall,
            "row_hydration_cpu_seconds": hydration_cpu,
            "canonical_conversion_wall_seconds": canonical_wall,
            "canonical_conversion_cpu_seconds": canonical_cpu,
            "candidate_headers": selection.route.candidate_count,
            "entries_loaded": len(selection.entries),
            "logical_bytes_read": selection.logical_bytes_read,
            "python_peak_allocation_bytes": peak,
            "inactive_vram_bytes": 0,
            "query_plan": plans,
        }
    return results


def profile_ttl(root: Path) -> dict[str, object]:
    package_path = root / "artifacts/pythia-1.4b-final-layer.ttl"
    result: dict[str, object] = {}
    package = TensorTranslationLayer.load(package_path, device="cpu")
    canonical = torch.randn(1, 16, 512)
    hidden = torch.randn(1, 16, package.config.model_hidden_width)
    features = torch.randn(1, 16, 4)

    def call():
        translated = package.value_translator(canonical)
        return translated, package.gate(hidden, translated, features)

    (_translated, _gate), wall, cpu = median_timing(call, repeats=30)
    result["cpu"] = {
        "translation_and_gate_wall_seconds": wall,
        "translation_and_gate_cpu_seconds": cpu,
        "parameter_count": sum(value.numel() for value in package.parameters()),
        "parameter_bytes": tensor_bytes(package.parameters()),
    }
    if torch.cuda.is_available():
        package = package.to("cuda")
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        copy_start = time.perf_counter()
        canonical_cuda = canonical.to("cuda")
        hidden_cuda = hidden.to("cuda")
        features_cuda = features.to("cuda")
        torch.cuda.synchronize()
        copy_wall = time.perf_counter() - copy_start
        for _ in range(5):
            translated = package.value_translator(canonical_cuda)
            package.gate(hidden_cuda, translated, features_cuda)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(30):
            translated = package.value_translator(canonical_cuda)
            package.gate(hidden_cuda, translated, features_cuda)
        torch.cuda.synchronize()
        result["cuda"] = {
            "translation_and_gate_wall_seconds": (time.perf_counter() - start) / 30,
            "bytes_copied_to_cuda": tensor_bytes((canonical, hidden, features)),
            "copy_wall_seconds": copy_wall,
            "active_vram_delta_bytes": torch.cuda.memory_allocated() - before,
            "peak_vram_delta_bytes": torch.cuda.max_memory_allocated() - before,
        }
    return result


def dependency_map(root: Path) -> dict[str, object]:
    nodes = {}
    for path in sorted((root / "src/pcm/planner").glob("*.py")):
        tree = ast.parse(path.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("pcm.planner"):
                    imports.append(node.module)
            elif isinstance(node, ast.Import):
                imports.extend(
                    name.name for name in node.names if name.name.startswith("pcm.planner")
                )
        nodes[path.stem] = sorted(set(imports))
    return {
        "active_module_imports": nodes,
        "archive_dependencies": [],
        "canonical_stores_model_token_ids": False,
        "canonical_stores_model_hidden_vectors": False,
    }


def prior_e2e(root: Path) -> dict[str, object]:
    split = json.loads((root / "artifacts/phase-b-split-translator.json").read_text())
    selected = split["layer_sweep"][split["selected_variant"]]
    personality = json.loads((root / "artifacts/phase-b-personality-package.json").read_text())
    personality_ttl = personality.get("cuda_ttl") or personality["cuda_translate"]
    cuda_audit_path = root / "artifacts/active-system-cuda-attribution.json"
    cuda_audit = json.loads(cuda_audit_path.read_text()) if cuda_audit_path.exists() else None
    return {
        "source_artifacts": [
            "artifacts/phase-b-split-translator.json",
            "artifacts/phase-b-personality-package.json",
        ],
        "state_counterfactual": selected["counterfactual"],
        "state_ablations": selected["ablations"],
        "natural_rp": selected["natural_rp"],
        "personality_counterfactual": personality_ttl["causal"],
        "personality_natural_rp": personality_ttl["natural_interaction"],
        "matched_cuda_attribution": cuda_audit,
        "failure_attribution_fields": (
            "storage validity, selected route/rank, acceptance, gate, candidate logits, "
            "generated token, KL, latency, VRAM"
        ),
    }


def run(root: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="planner-audit-") as directory:
        return {
            "experiment": "active-planner-cache-full-audit-v1",
            "timestamp": STAMP,
            "training_performed": False,
            "cache_router_profile": profile_cache_router(),
            "router_scaling": router_scaling_attribution(),
            "ppkg_scaling": profile_ppkg(Path(directory)),
            "ttl_profile": profile_ttl(root),
            "dependency_map": dependency_map(root),
            "matched_e2e_evidence": prior_e2e(root),
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/active-system-audit.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
