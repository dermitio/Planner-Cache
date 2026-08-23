import pytest
import torch

from pcm.planner.cache import (
    CacheFullProtectedError,
    Freshness,
    Persistence,
    PlannerCache,
    PlannerCacheConfig,
    SlotSource,
    SlotType,
    StateOperation,
)


def value(width, position):
    result = torch.zeros(width)
    result[position] = 1
    return result


def test_allocation_is_reserved_and_stable_across_mutations():
    cache = PlannerCache(PlannerCacheConfig(slots=3, width=4))
    signature = cache.allocation_signature()
    first, operation = cache.create(value(4, 0), slot_type=SlotType.FACT, label="door")
    assert operation == StateOperation.CREATE
    cache.modify(first, value(4, 1), source=SlotSource.CORRECTION)
    cache.keep(first, confidence=0.9)
    second, _ = cache.create(value(4, 2), slot_type=SlotType.GOAL)
    cache.merge((first, second))
    cache.invalidate(first)
    assert cache.allocation_signature() == signature
    assert cache.occupied == 0


def test_modify_replaces_current_state_instead_of_appending_history():
    cache = PlannerCache(PlannerCacheConfig(slots=2, width=4))
    slot, _ = cache.create(value(4, 0), slot_type=SlotType.FACT, label="door.state")
    cache.modify(slot, value(4, 1), confidence=0.95, source=SlotSource.CORRECTION)
    assert cache.occupied == 1
    torch.testing.assert_close(cache.values[slot], value(4, 1).to(cache.values.dtype))
    assert cache.source[slot].item() == SlotSource.CORRECTION


def test_full_cache_merges_related_state_without_reallocation():
    cache = PlannerCache(PlannerCacheConfig(slots=2, width=4, merge_similarity=0.8))
    cache.create(value(4, 0), slot_type=SlotType.FACT, importance=0.8)
    cache.create(value(4, 1), slot_type=SlotType.GOAL, importance=0.8)
    signature = cache.allocation_signature()
    related = torch.tensor([0.99, 0.01, 0.0, 0.0])
    slot, operation = cache.create(related, slot_type=SlotType.FACT)
    assert operation == StateOperation.MERGE
    assert slot == 0
    assert cache.occupied == 2
    assert cache.allocation_signature() == signature


def test_semantically_related_state_merges_before_capacity_is_full():
    cache = PlannerCache(PlannerCacheConfig(slots=4, width=4, merge_similarity=0.9))
    first, operation = cache.create(
        torch.tensor([1.0, 0.0, 0.0, 0.0]), slot_type=SlotType.FACT
    )
    second, operation = cache.create(
        torch.tensor([0.99, 0.01, 0.0, 0.0]), slot_type=SlotType.FACT
    )
    assert operation == StateOperation.MERGE
    assert second == first
    assert cache.occupied == 1
    _, operation = cache.create(
        torch.tensor([0.0, 1.0, 0.0, 0.0]), slot_type=SlotType.FACT
    )
    assert operation == StateOperation.CREATE
    assert cache.occupied == 2


def test_user_correction_cannot_be_overwritten_by_model_inference():
    cache = PlannerCache(PlannerCacheConfig(slots=2, width=4))
    slot, _ = cache.create(
        value(4, 0), slot_type=SlotType.FACT, source=SlotSource.INFERENCE
    )
    cache.modify(slot, value(4, 1), source=SlotSource.CORRECTION, confidence=1.0)
    cache.modify(slot, value(4, 2), source=SlotSource.INFERENCE, confidence=0.2)
    torch.testing.assert_close(cache.values[slot], value(4, 1).to(cache.values.dtype))
    assert cache.source[slot].item() == SlotSource.CORRECTION
    assert cache.confidence[slot].item() == 1.0


def test_full_cache_evicts_low_value_stale_nonpermanent_slot():
    cache = PlannerCache(PlannerCacheConfig(slots=2, width=4, merge_similarity=0.99))
    stale, _ = cache.create(
        value(4, 0),
        slot_type=SlotType.FACT,
        importance=0.0,
        confidence=0.1,
        freshness=Freshness.STALE,
        persistence=Persistence.VOLATILE,
        label="discardable",
    )
    protected, _ = cache.create(
        value(4, 1),
        slot_type=SlotType.GOAL,
        importance=1.0,
        persistence=Persistence.PERMANENT,
        label="protected",
    )
    replacement, operation = cache.create(value(4, 2), slot_type=SlotType.ENTITY)
    assert operation == StateOperation.CREATE
    assert replacement == stale
    assert cache.labels[replacement] is None
    assert cache.labels[protected] == "protected"


def test_all_permanent_slots_cannot_be_silently_displaced():
    cache = PlannerCache(PlannerCacheConfig(slots=1, width=4, merge_similarity=0.99))
    cache.create(value(4, 0), slot_type=SlotType.GOAL, persistence=Persistence.PERMANENT)
    with pytest.raises(CacheFullProtectedError):
        cache.create(value(4, 1), slot_type=SlotType.FACT)


def test_low_importance_incoming_state_is_rejected_when_full():
    cache = PlannerCache(PlannerCacheConfig(slots=1, width=4, merge_similarity=0.99))
    retained, _ = cache.create(
        value(4, 0),
        slot_type=SlotType.GOAL,
        importance=1.0,
        persistence=Persistence.DURABLE,
        label="critical",
    )
    rejected, operation = cache.create(
        value(4, 1),
        slot_type=SlotType.FACT,
        importance=0.0,
        freshness=Freshness.STALE,
        persistence=Persistence.VOLATILE,
    )
    assert rejected == -1
    assert operation == StateOperation.IGNORE
    assert cache.labels[retained] == "critical"


def test_ignore_has_no_effect():
    cache = PlannerCache(PlannerCacheConfig(slots=2, width=4))
    signature = cache.allocation_signature()
    assert cache.apply(StateOperation.IGNORE) is None
    assert cache.occupied == 0
    assert cache.allocation_signature() == signature
