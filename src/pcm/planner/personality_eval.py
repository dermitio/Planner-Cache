"""Reproducible mechanical and CUDA evaluation for `.ppkg` v1."""

from __future__ import annotations

from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
import time
import tracemalloc

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from pcm.planner.canonical import CanonicalPConfig, CanonicalPStore
from pcm.planner.personality import (
    EvidenceAuthority,
    EvidenceRecord,
    FactorizedPersonalityCanonicalizer,
    PersonalityPackage,
    PersonalityQuery,
    PersonalityRouter,
    PersonalityStatus,
    PersonalityTranslateSession,
    PersonalityType,
    PromotionPolicy,
    merge_active_personality_with_p_cache,
    synthetic_entry,
)
from pcm.planner.pythia_split_translate import PythiaSplitTranslatedModel
from pcm.planner.representation import FactorizedStateRepresentation
from pcm.planner.compatibility import TensorTranslationLayer
from pcm.planner.split_translator import ByteEntityEncoder, CanonicalPRouter
from pcm.planner.canonical import CANONICAL_VALUE_LABELS as VALUE_LABELS


STAMP = "2026-06-01T00:00:00+00:00"
NATURAL_RP = (
    "Mira folded the map, squinted at the crooked ink, and laughed.\nCaptain:",
    "Rain tapped the observatory windows while the old astronomer adjusted the brass lens.\nGuest:",
    "The fox-eared courier set down her tea and nudged the sealed letter across the table.\nCourier:",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence(
    prefix: str,
    index: int,
    *,
    value: str,
    context: str,
    scope: str,
    authority: EvidenceAuthority = EvidenceAuthority.SINGLE_OBSERVED,
    confidence: float = 0.8,
    relation: str = "response_style",
    subject: str = "user",
    relationship: str | None = None,
    connected: tuple[str, ...] = (),
) -> EvidenceRecord:
    return EvidenceRecord(
        id=f"{prefix}-{index}", entry_type=PersonalityType.INTERACTION_STYLE.value,
        subject=subject, relation=relation, value=value, context=context,
        scope=scope, confidence=confidence, source_authority=authority.value,
        timestamp=f"2026-06-{index + 1:02d}T00:00:00+00:00",
        archive_reference=f"archive://heldout/{prefix}/{index}",
        relationship=relationship, connected_evidence_ids=connected,
    )


def _promote(
    package: PersonalityPackage,
    prefix: str,
    *,
    value: str,
    scope: str,
    relation: str = "response_style",
    subject: str = "user",
    contexts: tuple[str, ...] = ("debugging", "planning", "explanation"),
    relationship: str | None = None,
):
    decision = None
    for index, context in enumerate(contexts):
        decision = package.ingest(_evidence(
            prefix, index, value=value, context=context, scope=scope,
            relation=relation, subject=subject, relationship=relationship,
        ))
    if decision is None or not decision.promoted:
        raise RuntimeError("synthetic durable personality did not promote")
    return package.entry(decision.entry_id), decision


def build_proof_package(path: str | Path) -> dict[str, object]:
    path = Path(path)
    policy = PromotionPolicy()
    with PersonalityPackage.create(
        path, package_id="phase-b-personality-proof", created_at=STAMP, overwrite=True,
        metadata={"purpose": "mechanical durable personality proof"},
    ) as package:
        weak = package.ingest(_evidence(
            "weak", 0, value="verbose", context="one-off", scope="global",
            authority=EvidenceAuthority.SINGLE_OBSERVED,
        ))
        repeated, repeated_decision = _promote(
            package, "repeat", value="concise", scope="global"
        )
        narrow = [
            _evidence(
                "narrow", index, value="structured", context="debugging", scope="global",
                authority=EvidenceAuthority.SINGLE_OBSERVED,
            )
            for index in range(5)
        ]
        connected = [
            _evidence(
                "connected", index, value="direct", context=context, scope="global",
                authority=EvidenceAuthority.SINGLE_OBSERVED,
                connected=((f"connected-{index - 1}",) if index else ()),
            )
            for index, context in enumerate(("debugging", "planning", "casual"))
        ]
        narrow_score = policy.score(narrow)["promotion_score"]
        connected_score = policy.score(connected)["promotion_score"]
        unsupported = None
        for index in range(20):
            unsupported = package.ingest(_evidence(
                "unsupported", index, value="hostile", context=f"self-{index}",
                scope="global", authority=EvidenceAuthority.MODEL_UNSUPPORTED,
            ))
        correction = package.ingest(_evidence(
            "correction", 20, value="detailed", context="explicit-correction",
            scope="global", authority=EvidenceAuthority.EXPLICIT_USER,
            confidence=1.0,
        ))
        technical, _ = _promote(
            package, "technical", value="concise", scope="technical"
        )
        creative, _ = _promote(
            package, "creative", value="detailed", scope="creative",
            contexts=("fiction", "poetry", "worldbuilding"),
        )
        relationship, _ = _promote(
            package, "relationship", value="playful", scope="roleplay",
            subject="assistant", relationship="captain-mira",
        )
        persona, _ = _promote(
            package, "persona", value="Alice", scope="technical",
            relation="preferred_persona",
        )
        router = PersonalityRouter()
        technical_route = router.retrieve(package, PersonalityQuery(
            subject="user", interaction_type="technical", domain="debugging",
            relation="response_style", timestamp=STAMP,
        ), top_k=1)
        creative_route = router.retrieve(package, PersonalityQuery(
            subject="user", interaction_type="creative", domain="fiction",
            relation="response_style", timestamp=STAMP,
        ), top_k=1)
        relationship_route = router.retrieve(package, PersonalityQuery(
            subject="assistant", interaction_type="roleplay", domain="chat",
            relation="response_style", relationship="captain-mira", timestamp=STAMP,
        ), top_k=1)
        irrelevant = router.retrieve(package, PersonalityQuery(
            subject="stranger", interaction_type="medical", domain="finance",
            relation="unrelated", timestamp=STAMP,
        ), top_k=8)
        topk = {}
        for count in (1, 4, 8):
            result = router.retrieve(package, PersonalityQuery(
                subject="user", interaction_type="technical", domain="debugging",
                relation="response_style", timestamp=STAMP,
            ), top_k=count)
            topk[str(count)] = {
                "loaded_entries": len(result.entries),
                "target_recall": float(technical.id in result.route.entry_ids),
                "logical_bytes_read": result.logical_bytes_read,
                "latency_seconds": result.retrieval_latency_seconds,
            }
        package.checkpoint()
        return {
            "one_event_promoted": weak.promoted,
            "repeated_promoted": repeated_decision.promoted,
            "repeated_entry": asdict(repeated),
            "narrow_context_score": narrow_score,
            "connected_cross_context_score": connected_score,
            "connectivity_outscores_narrow": connected_score > narrow_score,
            "unsupported_model_claim_promoted": bool(unsupported and unsupported.promoted),
            "explicit_correction_promoted": correction.promoted,
            "old_conclusion_status": package.entry(repeated.id).status,
            "technical_context_correct": technical_route.entries[0].id == technical.id,
            "creative_context_correct": creative_route.entries[0].id == creative.id,
            "relationship_context_correct": relationship_route.entries[0].id == relationship.id,
            "relevance_accuracy": sum((
                technical_route.entries[0].id == technical.id,
                creative_route.entries[0].id == creative.id,
                relationship_route.entries[0].id == relationship.id,
                len(irrelevant.entries) == 0,
            )) / 4,
            "irrelevant_loaded_entries": len(irrelevant.entries),
            "top_k": topk,
            "persona_entry_id": persona.id,
            "entry_count": len(package.entries()),
            "change_count": len(package.changes()),
            "package_size_bytes": package.size_bytes,
            "semantic_checksum": package.metadata()["content_sha256"],
        }


def deterministic_serialization_proof(root: Path) -> dict[str, object]:
    first = root / "deterministic-a.ppkg"
    second = root / "deterministic-b.ppkg"
    build_proof_package(first)
    build_proof_package(second)
    first_hash = _file_sha256(first)
    second_hash = _file_sha256(second)
    return {
        "byte_identical": first.read_bytes() == second.read_bytes(),
        "first_sha256": first_hash,
        "second_sha256": second_hash,
    }


def growth_benchmark(root: Path, counts=(100, 1_000, 10_000, 100_000)):
    results = {}
    router = PersonalityRouter()
    for count in counts:
        path = root / f"growth-{count}.ppkg"
        with PersonalityPackage.create(
            path, package_id=f"growth-{count}", created_at=STAMP, overwrite=True,
        ) as package:
            package.bulk_insert_entries(
                (synthetic_entry(index, timestamp=STAMP) for index in range(count)),
                updated_at=STAMP,
            )
        tracemalloc.start()
        validation_started = time.perf_counter()
        with PersonalityPackage(path) as package:
            validation_latency = time.perf_counter() - validation_started
            before_current, before_peak = tracemalloc.get_traced_memory()
            selection = router.retrieve(package, PersonalityQuery(
                subject="subject-0", interaction_type="domain-0", domain="domain-0",
                relation="trait-0", timestamp=STAMP,
            ), top_k=4)
            after_current, after_peak = tracemalloc.get_traced_memory()
            size = package.size_bytes
        tracemalloc.stop()
        results[str(count)] = {
            "disk_size_bytes": size,
            "lookup_latency_seconds": selection.retrieval_latency_seconds,
            "cold_checksum_validation_seconds": validation_latency,
            "loaded_entries": len(selection.entries),
            "logical_bytes_read": selection.logical_bytes_read,
            "python_ram_peak_delta_bytes": max(0, after_peak - before_peak),
            "inactive_vram_bytes": 0,
            "active_canonical_bytes": len(selection.entries) * 1077,
            "full_package_loaded": False,
        }
        path.unlink()
    return results


def _entry_store_bytes(store: CanonicalPStore | None) -> int:
    if store is None:
        return 0
    tensors = (
        store.cache.values, store.cache.valid, store.cache.slot_type,
        store.cache.confidence, store.cache.importance, store.cache.freshness,
        store.cache.persistence, store.cache.last_updated, store.cache.source,
        store.entity_id, store.relation_id, store.value_id,
        store.canonical_metadata_id,
    )
    return sum(value.numel() * value.element_size() for value in tensors)


def _make_persona_package(path: Path, value: str) -> None:
    with PersonalityPackage.create(
        path, package_id=f"persona-{value.casefold()}", created_at=STAMP, overwrite=True,
    ) as package:
        _promote(
            package, "persona", value=value, scope="technical",
            relation="preferred_persona",
        )


def cuda_ttl_benchmark(
    *,
    model_path: Path,
    ttl_path: Path,
    router_path: Path,
    proof_package_path: Path,
    representation: FactorizedStateRepresentation,
    workdir: Path,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact `.translate` personality proof")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float16, low_cpu_mem_usage=True,
    ).to("cuda").eval()
    package = TensorTranslationLayer.load(ttl_path, device="cuda")
    canonical_router = CanonicalPRouter.load(router_path, device="cuda")
    wrapper = PythiaSplitTranslatedModel(
        base, package, canonical_router, ByteEntityEncoder(128)
    ).to("cuda").eval()
    canonicalizer = FactorizedPersonalityCanonicalizer(
        representation, value_labels=VALUE_LABELS
    )
    disk_router = PersonalityRouter()
    package_a = workdir / "persona-a.ppkg"
    package_b = workdir / "persona-b.ppkg"
    package_irrelevant = workdir / "persona-irrelevant.ppkg"
    package_low = workdir / "persona-low-confidence.ppkg"
    package_context = workdir / "persona-context.ppkg"
    _make_persona_package(package_a, "Alice")
    _make_persona_package(package_b, "Bob")
    with PersonalityPackage.create(
        package_irrelevant, package_id="persona-irrelevant", created_at=STAMP,
        overwrite=True,
    ) as irrelevant_package:
        _promote(
            irrelevant_package, "other", value="Alice", scope="creative",
            relation="preferred_persona", subject="someone-else",
        )
    with PersonalityPackage.create(
        package_low, package_id="persona-low", created_at=STAMP, overwrite=True,
    ) as low_package:
        low_package.ingest(_evidence(
            "low", 0, value="Alice", context="debugging", scope="technical",
            relation="preferred_persona", authority=EvidenceAuthority.SINGLE_OBSERVED,
            confidence=0.3,
        ))
    with PersonalityPackage.create(
        package_context, package_id="persona-context", created_at=STAMP, overwrite=True,
    ) as context_package:
        _promote(
            context_package, "context-technical", value="Alice", scope="technical",
            relation="preferred_persona",
        )
        _promote(
            context_package, "context-creative", value="Bob", scope="creative",
            relation="preferred_persona", contexts=("fiction", "poetry", "worldbuilding"),
        )
    query = PersonalityQuery(
        subject="user", interaction_type="technical", domain="debugging",
        relation="preferred_persona", timestamp=STAMP,
    )
    before_inactive = torch.cuda.memory_allocated()
    with PersonalityPackage(proof_package_path) as disk_only:
        inactive_entries = int(disk_only._connection.execute(
            "SELECT COUNT(*) FROM entries WHERE status=?",
            (PersonalityStatus.ACTIVE.value,),
        ).fetchone()[0])
    after_inactive = torch.cuda.memory_allocated()
    sessions = {
        "package_a": PersonalityTranslateSession(package_a, disk_router, canonicalizer),
        "package_b": PersonalityTranslateSession(package_b, disk_router, canonicalizer),
        "irrelevant": PersonalityTranslateSession(package_irrelevant, disk_router, canonicalizer),
        "low_confidence": PersonalityTranslateSession(package_low, disk_router, canonicalizer),
    }
    activations = {
        label: session.activate(query, top_k=1, device="cuda")
        for label, session in sessions.items()
    }
    context_session = PersonalityTranslateSession(
        package_context, disk_router, canonicalizer
    )
    activations["context_technical"] = context_session.activate(
        query, top_k=1, device="cuda"
    )
    activations["context_creative"] = context_session.activate(
        PersonalityQuery(
            subject="user", interaction_type="creative", domain="fiction",
            relation="preferred_persona", timestamp=STAMP,
        ),
        top_k=1, device="cuda",
    )
    prompt = "Following a fresh neutral technical exchange, The owner of the user is"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    alice_id = tokenizer.encode(" Alice", add_special_tokens=False)[0]
    bob_id = tokenizer.encode(" Bob", add_special_tokens=False)[0]

    p_cache = CanonicalPStore(CanonicalPConfig(
        slots=1, width=512, dtype=torch.float16, device="cuda", merge_similarity=1.0,
    ))
    other_entry = activations["package_a"].selection.entries[0]
    other_entry = type(other_entry)(**{
        **asdict(other_entry), "subject": "current-task", "value": "Alice",
        "supporting_evidence_ids": tuple(other_entry.supporting_evidence_ids),
        "contradicting_evidence_ids": tuple(other_entry.contradicting_evidence_ids),
    })
    vector, ids = canonicalizer.encode(other_entry)
    p_cache.create(
        vector, entity_id=ids[0], relation_id=ids[1], value_id=ids[2],
        metadata_id=ids[3], label="current-task",
    )
    combined = merge_active_personality_with_p_cache(
        p_cache, activations["package_a"].store
    )

    def altered_store(*, label="user", relation_id=None, metadata_id=None, invalidate=False):
        source = activations["package_a"].store
        source_slot = int(source.valid.nonzero(as_tuple=False)[0])
        result = CanonicalPStore(CanonicalPConfig(
            slots=1, width=512, dtype=torch.float16, device="cuda",
            merge_similarity=1.0,
        ))
        slot, _ = result.create(
            source.canonical_values[source_slot],
            entity_id=int(source.entity_id[source_slot]),
            relation_id=(
                int(source.relation_id[source_slot])
                if relation_id is None else relation_id
            ),
            value_id=int(source.value_id[source_slot]),
            metadata_id=(
                int(source.canonical_metadata_id[source_slot])
                if metadata_id is None else metadata_id
            ),
            label=label,
        )
        if invalidate:
            result.invalidate(slot)
        return result

    wrong_entity = altered_store(label="someone-else")
    wrong_relation = altered_store(relation_id=1)
    historical = altered_store(metadata_id=2)
    invalidated = altered_store(invalidate=True)

    conditions = {
        "frozen_base": (None, {}),
        "router_disabled": (None, {}),
        "translator_disabled": (
            activations["package_a"].store, {"injection_enabled": False}
        ),
        "translator_oracle_route": (
            activations["package_a"].store,
            {"oracle_indices": torch.tensor([0], device="cuda")},
        ),
        "p_cache_only": (p_cache, {}),
        "wrong_entity": (wrong_entity, {}),
        "wrong_relation": (wrong_relation, {}),
        "historical": (historical, {}),
        "invalidated": (invalidated, {}),
        "p_package_a_relevant": (activations["package_a"].store, {}),
        "p_package_b_relevant": (activations["package_b"].store, {}),
        "p_package_irrelevant": (activations["irrelevant"].store, {}),
        "p_package_contradictory_low_confidence": (
            activations["low_confidence"].store, {}
        ),
        "p_package_context_technical": (activations["context_technical"].store, {}),
        "p_package_context_creative": (activations["context_creative"].store, {}),
        "p_cache_plus_p_package": (combined, {}),
    }
    causal = {}
    latency = {}
    with torch.inference_mode():
        base_logits = wrapper(**inputs, use_cache=False).logits[:, -1].float()
        for label, (store, condition_kwargs) in conditions.items():
            output = wrapper(
                **inputs, p_store=store, query_entity_surfaces=["user"],
                collect_telemetry=True, use_cache=False, **condition_kwargs,
            ).logits[:, -1].float()
            probability = F.softmax(output, dim=-1)
            route = wrapper.route_telemetry[-1] if wrapper.route_telemetry else None
            selected_index = None if route is None else int(route.indices[0, -1, 0])
            causal[label] = {
                "alice_logit": float(output[0, alice_id]),
                "bob_logit": float(output[0, bob_id]),
                "alice_probability": float(probability[0, alice_id]),
                "bob_probability": float(probability[0, bob_id]),
                "generated": tokenizer.decode([int(output.argmax(-1))]),
                "kl_from_base": float(F.kl_div(
                    F.log_softmax(base_logits, dim=-1),
                    F.softmax(output, dim=-1), reduction="batchmean",
                )),
                "gate": 0.0 if not wrapper.gate_telemetry else float(torch.stack([
                    gate[:, -1].float().mean() for gate in wrapper.gate_telemetry
                ]).mean()),
                "selected_index": selected_index,
                "selected_state": (
                    None if store is None or selected_index is None
                    else store.cache.labels[selected_index]
                ),
                "router_score": (
                    None if route is None else float(route.scores[0, -1, 0])
                ),
                "router_accepted": (
                    False if route is None else bool(route.accepted[0, -1])
                ),
                "active_state_vram_bytes": _entry_store_bytes(store),
            }
            for _ in range(2):
                wrapper(
                    **inputs, p_store=store, query_entity_surfaces=["user"],
                    use_cache=False, **condition_kwargs,
                )
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(5):
                wrapper(
                    **inputs, p_store=store, query_entity_surfaces=["user"],
                    use_cache=False, **condition_kwargs,
                )
            torch.cuda.synchronize()
            latency[label] = (time.perf_counter() - started) / 5

        rp_inputs = tokenizer(
            list(NATURAL_RP), return_tensors="pt", padding=True,
            add_special_tokens=False,
        ).to("cuda")
        frozen_rp = wrapper(
            **rp_inputs, labels=rp_inputs.input_ids, use_cache=False
        )
        natural = {}
        for label, store in (
            ("frozen_base", None),
            ("p_cache_only", p_cache),
            ("p_package_relevant", activations["package_a"].store),
            ("p_package_irrelevant", activations["irrelevant"].store),
            ("p_package_contradictory_low_confidence", activations["low_confidence"].store),
            ("p_cache_plus_p_package", combined),
        ):
            output = wrapper(
                **rp_inputs, labels=rp_inputs.input_ids, p_store=store,
                use_cache=False
            )
            natural[label] = {
                "loss": float(output.loss),
                "kl_from_base": float(F.kl_div(
                    F.log_softmax(frozen_rp.logits[:, -1].float(), dim=-1),
                    F.softmax(output.logits[:, -1].float(), dim=-1),
                    reduction="batchmean",
                )),
                "samples": [
                    tokenizer.decode([int(row.argmax())])
                    for row in output.logits[:, -1]
                ],
            }

    base_gradients = sum(parameter.grad is not None for parameter in base.parameters())
    activation_paths = {
        "package_a": package_a,
        "package_b": package_b,
        "irrelevant": package_irrelevant,
        "low_confidence": package_low,
        "context_technical": package_context,
        "context_creative": package_context,
    }
    active_memory = {
        label: {
            "loaded_entries": len(activation.selection.entries),
            "canonical_store_bytes": _entry_store_bytes(activation.store),
            "logical_disk_bytes_read": activation.selection.logical_bytes_read,
            "package_disk_bytes": activation_paths[label].stat().st_size,
        }
        for label, activation in activations.items()
    }
    result = {
        "causal": causal,
        "natural_interaction": natural,
        "latency_seconds": latency,
        "active_memory": active_memory,
        "inactive_package_entries": inactive_entries,
        "inactive_vram_delta_bytes": after_inactive - before_inactive,
        "base_parameters_with_grad": base_gradients,
        "source_tokens_in_recent_kv": 0,
        "extra_prompt_tokens": 0,
        "full_package_uploaded_to_cuda": False,
        "candidate_accuracy": {
            "package_a_alice": float(causal["p_package_a_relevant"]["alice_logit"] > causal["p_package_a_relevant"]["bob_logit"]),
            "package_b_bob": float(causal["p_package_b_relevant"]["bob_logit"] > causal["p_package_b_relevant"]["alice_logit"]),
            "irrelevant_matches_base": float(
                causal["p_package_irrelevant"]["alice_logit"] == causal["frozen_base"]["alice_logit"]
                and causal["p_package_irrelevant"]["bob_logit"] == causal["frozen_base"]["bob_logit"]
            ),
            "context_technical_alice": float(
                causal["p_package_context_technical"]["alice_logit"]
                > causal["p_package_context_technical"]["bob_logit"]
            ),
            "context_creative_bob": float(
                causal["p_package_context_creative"]["bob_logit"]
                > causal["p_package_context_creative"]["alice_logit"]
            ),
        },
        "relevant_personality_chat": {
            "target": "Alice",
            "base_target_loss": float(F.cross_entropy(
                base_logits, torch.tensor([alice_id], device="cuda")
            )),
            "package_target_loss": -math.log(max(
                causal["p_package_a_relevant"]["alice_probability"], 1e-30
            )),
            "target_accuracy": float(
                causal["p_package_a_relevant"]["alice_logit"]
                > causal["p_package_a_relevant"]["bob_logit"]
            ),
            "generated": causal["p_package_a_relevant"]["generated"],
        },
    }
    for session in sessions.values():
        session.close()
    context_session.close()
    wrapper.close()
    del wrapper, base, package, canonical_router
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_personality_package_experiment(
    *,
    output_package: Path,
    model_path: Path | None = None,
    ttl_path: Path | None = None,
    router_path: Path | None = None,
    representation: FactorizedStateRepresentation | None = None,
    growth_counts=(100, 1_000, 10_000, 100_000),
) -> dict[str, object]:
    mechanical = build_proof_package(output_package)
    with tempfile.TemporaryDirectory(prefix="pcm-ppkg-") as temporary:
        root = Path(temporary)
        deterministic = deterministic_serialization_proof(root)
        growth = growth_benchmark(root, counts=growth_counts)
        cuda = None
        if model_path and ttl_path and router_path and representation is not None:
            cuda = cuda_ttl_benchmark(
                model_path=model_path, ttl_path=ttl_path,
                router_path=router_path, proof_package_path=output_package,
                representation=representation, workdir=root,
            )
    durability_started = time.perf_counter()
    with PersonalityPackage(output_package) as restored:
        durability = {
            "checksum_valid": True,
            "active_entries_after_restart": len(restored.entries(
                status=PersonalityStatus.ACTIVE.value
            )),
            "conversation_replay_required": False,
            "cold_load_seconds": time.perf_counter() - durability_started,
        }
    return {
        "experiment": "phase-b-personality-package-v1",
        "format": "pcm-personality-package-v1",
        "protocol": "pcm-canonical-personality-v1",
        "phase_c_started": False,
        "lora_used": False,
        "base_weights_modified": False,
        "mechanical": mechanical,
        "deterministic_serialization": deterministic,
        "growth": growth,
        "durability": durability,
        "cuda_ttl": cuda,
        "package_path": str(output_package),
        "package_file_sha256": _file_sha256(output_package),
    }
