"""Adversarial and integrity checks for the active Planner Cache stack."""

from __future__ import annotations

import sqlite3
from threading import Barrier, Thread

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import GPT2Config, GPT2LMHeadModel

from pcm.planner.cache import (
    Freshness,
    Persistence,
    PlannerCache,
    PlannerCacheConfig,
    SlotSource,
    SlotType,
    StateOperation,
)
from pcm.planner.canonical import CanonicalPConfig, CanonicalPStore
from pcm.planner.canonical import model_config_checksum
from pcm.planner.pythia_split_translate import pythia_model_identifier
from pcm.planner.personality import (
    PersonalityPackage,
    PersonalityQuery,
    PersonalityRouter,
    synthetic_entry,
)
from pcm.planner.split_translator import (
    ByteEntityEncoder,
    CanonicalPRouter,
    FactorizedCanonicalQuery,
    SplitPTranslatePackage,
    SplitTranslateConfig,
)


def vector(width: int, index: int) -> torch.Tensor:
    result = torch.zeros(width)
    result[index % width] = 1
    return result


def test_all_operations_and_long_mutation_chain_leave_only_current_state():
    cache = PlannerCache(PlannerCacheConfig(slots=4, width=8, merge_similarity=0.95))
    slot, operation = cache.apply(
        StateOperation.CREATE, value=vector(8, 0), slot_type=SlotType.FACT,
        label="silver_key.owner",
    )
    assert operation == StateOperation.CREATE
    assert cache.apply(StateOperation.KEEP, index=slot) == slot
    cache.apply(StateOperation.MODIFY, index=slot, value=vector(8, 1))
    cache.apply(StateOperation.MODIFY, index=slot, value=vector(8, 2))
    other, _ = cache.create(vector(8, 3), slot_type=SlotType.FACT, importance=0.1)
    target = cache.apply(StateOperation.MERGE, indices=(slot, other), value=vector(8, 4))
    assert cache.occupied == 1
    torch.testing.assert_close(cache.values[target], vector(8, 4).to(cache.values.dtype))
    assert cache.apply(StateOperation.IGNORE) is None
    cache.apply(StateOperation.INVALIDATE, index=target)
    assert cache.occupied == 0
    assert not cache.values.any()


@pytest.mark.parametrize("field,bad", [
    ("confidence", -0.1), ("confidence", 1.1), ("confidence", float("nan")),
    ("importance", -1.0), ("importance", float("inf")),
])
def test_malformed_cache_metadata_is_rejected(field, bad):
    cache = PlannerCache(PlannerCacheConfig(slots=1, width=4))
    with pytest.raises(ValueError):
        cache.create(vector(4, 0), **{field: bad})
    assert cache.occupied == 0


def test_canonical_merge_never_crosses_entity_or_relation_identity():
    store = CanonicalPStore(CanonicalPConfig(
        slots=4, width=8, dtype=torch.float32, merge_similarity=0.9
    ))
    first, _ = store.create(
        vector(8, 0), entity_id=1, relation_id=0, value_id=1,
        metadata_id=0, label="Alice",
    )
    second, operation = store.create(
        vector(8, 0), entity_id=2, relation_id=0, value_id=1,
        metadata_id=0, label="Bob",
    )
    third, operation2 = store.create(
        vector(8, 0), entity_id=1, relation_id=1, value_id=1,
        metadata_id=0, label="Alice",
    )
    assert (first, second, third) == (0, 1, 2)
    assert operation == operation2 == StateOperation.CREATE


def test_canonical_snapshot_determinism_repeated_cycles_and_integrity(tmp_path):
    store = CanonicalPStore(CanonicalPConfig(slots=3, width=8, dtype=torch.float32))
    slot, _ = store.create(
        vector(8, 0), entity_id=5, relation_id=1, value_id=2,
        metadata_id=0, label="𐐀lice / 鍵 / e\u0301" * 1000,
    )
    store.modify(
        slot, vector(8, 1), entity_id=5, relation_id=1, value_id=3,
        metadata_id=0, source=SlotSource.CORRECTION,
    )
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    store.save(first)
    CanonicalPStore.load(first, dtype=torch.float32).save(second)
    with safe_open(str(first), framework="pt", device="cpu") as handle:
        first_checksum = handle.metadata()["content_sha256"]
    with safe_open(str(second), framework="pt", device="cpu") as handle:
        second_checksum = handle.metadata()["content_sha256"]
    assert first_checksum == second_checksum
    for name, tensor in load_file(first).items():
        torch.testing.assert_close(tensor, load_file(second)[name])

    tensors = load_file(first)
    with safe_open(str(first), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    tensors["value_id"][slot] = 99
    save_file(tensors, str(first), metadata=metadata)
    with pytest.raises(ValueError, match="checksum"):
        CanonicalPStore.load(first, dtype=torch.float32)


def test_router_empty_duplicates_history_unicode_and_hard_negatives():
    encoder = ByteEntityEncoder()
    router = CanonicalPRouter()
    empty = CanonicalPStore(CanonicalPConfig(slots=4, width=8, dtype=torch.float32))
    query = FactorizedCanonicalQuery(
        entity=encoder(["🗝️ clé 銀"]),
        relation_logits=torch.tensor([[9.0, -9.0, -9.0]]),
        metadata_logits=torch.tensor([[9.0, -9.0, -9.0, -9.0]]),
    )
    route = router.route(query, router.build_index(empty, encoder, device="cpu"), top_k=4)
    assert not route.has_valid and not route.accepted.any()

    store = CanonicalPStore(CanonicalPConfig(
        slots=6, width=8, dtype=torch.float32, merge_similarity=1.0
    ))
    correct, _ = store.create(vector(8, 0), entity_id=1, relation_id=0,
        value_id=0, metadata_id=0, label="🗝️ clé 銀")
    store.create(vector(8, 1), entity_id=2, relation_id=0,
        value_id=0, metadata_id=0, label="🗝️ clé 金")
    store.create(vector(8, 2), entity_id=1, relation_id=1,
        value_id=0, metadata_id=0, label="🗝️ clé 銀")
    historical, _ = store.create(vector(8, 3), entity_id=1, relation_id=0,
        value_id=1, metadata_id=2, label="🗝️ clé 銀")
    result = router.route(query, router.build_index(store, encoder, device="cpu"), top_k=4)
    assert int(result.indices[0, 0]) == correct
    assert result.indices[0].tolist().index(historical) > 0
    store.invalidate(correct)
    scores, _ = router.all_scores(query, router.build_index(store, encoder, device="cpu"))
    assert torch.isneginf(scores[0, correct])

    stale, _ = store.create(
        vector(8, 4), entity_id=1, relation_id=0, value_id=2,
        metadata_id=0, label="🗝️ clé 銀", freshness=Freshness.STALE,
    )
    scores, _ = router.all_scores(query, router.build_index(store, encoder, device="cpu"))
    assert torch.isneginf(scores[0, stale])


def test_translate_rejects_incompatible_width_protocol_and_corruption(tmp_path):
    package = SplitPTranslatePackage(SplitTranslateConfig(
        model_id="audit-model", model_hidden_width=32, attachment_layers=(1,)
    ))
    with pytest.raises(ValueError, match="hidden width"):
        package.validate_compatibility(model_id="audit-model", model_hidden_width=64)
    with pytest.raises(ValueError, match="canonical protocol"):
        package.validate_compatibility(
            model_id="audit-model", model_hidden_width=32,
            canonical_protocol="future-p-v99",
        )
    path = tmp_path / "audit.translate"
    package.save(path)
    tensors = load_file(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    tensors[next(iter(tensors))].view(-1)[0] += 1
    save_file(tensors, str(path), metadata=metadata)
    with pytest.raises(ValueError, match="checksum"):
        SplitPTranslatePackage.load(path)


def test_canonical_snapshot_protocol_mismatch_is_rejected(tmp_path):
    store = CanonicalPStore(CanonicalPConfig(slots=1, width=8, dtype=torch.float32))
    store.create(
        vector(8, 0), entity_id=0, relation_id=0, value_id=0,
        metadata_id=0, label="entity",
    )
    path = tmp_path / "protocol.safetensors"
    store.save(path)
    tensors = load_file(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    metadata["format"] = "future-canonical-p-v99"
    save_file(tensors, str(path), metadata=metadata)
    with pytest.raises(ValueError, match="unsupported"):
        CanonicalPStore.load(path, dtype=torch.float32)


def test_translate_model_compatibility_is_independent_of_absolute_local_path():
    class Config:
        hidden_size = 32
        model_type = "gpt_neox"

        def __init__(self, name):
            self._name_or_path = name

        def to_dict(self):
            return {"_name_or_path": self._name_or_path, "hidden_size": 32}

    class Model:
        def __init__(self, name):
            self.config = Config(name)

    relative = Model("pythia-1.4b")
    absolute = Model("/different/mount/pythia-1.4b")
    assert pythia_model_identifier(relative) == pythia_model_identifier(absolute)
    assert model_config_checksum(relative.config) == model_config_checksum(absolute.config)


def test_ppkg_interleaved_reader_writer_and_indexed_query_plan(tmp_path):
    path = tmp_path / "concurrent.ppkg"
    with PersonalityPackage.create(path, package_id="concurrent") as package:
        package.bulk_insert_entries(
            (synthetic_entry(index) for index in range(1000)),
            updated_at="2026-01-01T00:00:00+00:00",
        )
    failures: list[BaseException] = []

    barrier = Barrier(2)

    def reader() -> None:
        try:
            with PersonalityPackage(path) as package:
                query = PersonalityQuery(
                    subject="subject-0", interaction_type="domain-0",
                    domain="domain-0", relation="trait-0",
                    timestamp="2026-01-01T00:00:00+00:00",
                )
                barrier.wait()
                for _ in range(20):
                    PersonalityRouter().retrieve(package, query, top_k=4)
        except BaseException as error:  # pragma: no cover - surfaced below
            failures.append(error)

    def writer() -> None:
        try:
            with PersonalityPackage(path) as package:
                barrier.wait()
                for index in range(1000, 1020):
                    package.bulk_insert_entries(
                        (synthetic_entry(index),),
                        updated_at="2026-01-02T00:00:00+00:00",
                    )
        except BaseException as error:  # pragma: no cover - surfaced below
            failures.append(error)

    threads = (Thread(target=reader), Thread(target=writer))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures
    with PersonalityPackage(path) as package:
        plans = package._connection.execute(
            "EXPLAIN QUERY PLAN SELECT id FROM entries INDEXED BY entry_subject_route "
            "WHERE status='active' AND subject='subject-0' ORDER BY importance DESC,"
            "confidence DESC,id LIMIT 128"
        ).fetchall()
        assert any("entry_subject_route" in str(tuple(row)) for row in plans)


def test_second_architecture_structural_portability_with_gpt2(tmp_path):
    """Audit-only proof: GPT-2 needs a hook/translate package, not a P change."""
    base = GPT2LMHeadModel(GPT2Config(
        vocab_size=101, n_embd=32, n_layer=2, n_head=4, n_positions=32,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )).eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    package = SplitPTranslatePackage(SplitTranslateConfig(
        model_id="audit-gpt2", model_hidden_width=32, attachment_layers=(1,),
        canonical_width=16,
    ))
    router = CanonicalPRouter()
    encoder = ByteEntityEncoder()
    store = CanonicalPStore(CanonicalPConfig(
        slots=2, width=16, dtype=torch.float32, merge_similarity=1.0
    ))
    slot, _ = store.create(
        torch.randn(16), entity_id=1, relation_id=0, value_id=2,
        metadata_id=0, label="portable entity",
    )
    index = router.build_index(store, encoder, device="cpu")
    entity = encoder(["portable entity"])
    telemetry = []

    def hook(_module, _inputs, output):
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        query = package.query_projector(
            hidden, entity_anchor=entity[:, None].expand(1, hidden.shape[1], -1)
        )
        route = router.route(query, index, top_k=1)
        canonical = store.canonical_values[route.indices]
        translated = package.value_translator(canonical.squeeze(-2))
        features = route.features.squeeze(-2)
        gate = package.gate(hidden, translated, features) * route.accepted
        telemetry.append(gate.detach())
        injected = hidden + gate.unsqueeze(-1) * translated
        return injected if isinstance(output, torch.Tensor) else (injected, *output[1:])

    tokens = torch.randint(0, 101, (1, 8))
    with torch.no_grad():
        disabled = base(tokens).logits.detach()
    handle = base.transformer.h[1].register_forward_hook(hook)
    enabled = base(tokens).logits
    enabled.square().mean().backward()
    handle.remove()
    assert telemetry and not torch.equal(enabled.detach(), disabled)
    assert all(parameter.grad is None for parameter in base.parameters())
    assert any(parameter.grad is not None for parameter in package.parameters())
    assert store.canonical_values.shape[-1] == 16

    package_path = tmp_path / "gpt2.translate"
    package.save(package_path)
    restored = SplitPTranslatePackage.load(package_path)
    assert restored.config.model_hidden_width == 32
    store.invalidate(slot)
    with torch.no_grad():
        invalidated = base(tokens).logits
    torch.testing.assert_close(invalidated, disabled)
