from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest
import torch

from pcm.planner.canonical import CanonicalPConfig, CanonicalPStore
from pcm.planner.personality import (
    EvidenceAuthority,
    EvidenceRecord,
    FactorizedPersonalityCanonicalizer,
    PersonalityEntry,
    PersonalityPackage,
    PersonalityQuery,
    PersonalityRouter,
    PersonalityStatus,
    PersonalityTranslateSession,
    PersonalityType,
    PromotionPolicy,
    evidence_from_p_cache,
    merge_active_personality_with_p_cache,
)
from pcm.planner.representation import FactorizedStateRepresentation
from pcm.planner.canonical import CANONICAL_VALUE_LABELS as VALUE_LABELS


STAMP = "2026-01-01T00:00:00+00:00"


def evidence(
    index: int,
    *,
    value: str = "concise",
    context: str = "debugging",
    scope: str = "global",
    authority: EvidenceAuthority = EvidenceAuthority.SINGLE_OBSERVED,
    subject: str = "user",
    relation: str = "response_style",
    relationship: str | None = None,
    connected: tuple[str, ...] = (),
    confidence: float = 0.8,
) -> EvidenceRecord:
    return EvidenceRecord(
        id=f"ev-{index}-{value}-{context}-{subject}-{relationship}",
        entry_type=PersonalityType.INTERACTION_STYLE.value,
        subject=subject, relation=relation, value=value, context=context,
        scope=scope, confidence=confidence, source_authority=authority.value,
        timestamp=f"2026-01-{index + 1:02d}T00:00:00+00:00",
        archive_reference=f"archive://turn/{index}", relationship=relationship,
        connected_evidence_ids=connected,
    )


@pytest.fixture
def package(tmp_path):
    value = PersonalityPackage.create(
        tmp_path / "personality.ppkg", package_id="test-personality",
        created_at=STAMP,
    )
    yield value
    value.close()


def promote(package, *, value="concise", scope="global", contexts=None, subject="user", relationship=None):
    decision = None
    contexts = contexts or ("debugging", "planning", "chat")
    for index, context in enumerate(contexts):
        decision = package.ingest(evidence(
            index, value=value, context=context, scope=scope,
            subject=subject, relationship=relationship,
        ))
    assert decision is not None and decision.promoted
    return package.entry(decision.entry_id)


def test_serialization_round_trip_is_deterministic_and_references_archive(package):
    entry = promote(package)
    package.checkpoint()
    checksum = package.metadata()["content_sha256"]
    package.close()
    with PersonalityPackage(package.path) as restored:
        assert restored.metadata()["content_sha256"] == checksum
        assert restored.entry(entry.id) == entry
        assert restored.evidence("ev-0-concise-debugging-user-None").archive_reference == "archive://turn/0"


def test_checksum_corruption_is_rejected(package):
    promote(package)
    package.close()
    connection = sqlite3.connect(package.path)
    connection.execute("UPDATE entries SET value='corrupted'")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="checksum"):
        PersonalityPackage(package.path)


def test_protocol_incompatibility_is_rejected(package):
    package.close()
    connection = sqlite3.connect(package.path)
    connection.execute("UPDATE metadata SET value='future-v99' WHERE key='protocol'")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="protocol"):
        PersonalityPackage(package.path, validate=False)


def test_one_event_does_not_promote_but_repetition_does(package):
    first = package.ingest(evidence(0))
    second = package.ingest(evidence(1, context="planning"))
    third = package.ingest(evidence(2, context="chat"))
    assert not first.promoted
    assert not second.promoted
    assert third.promoted
    assert package.entry(third.entry_id).evidence_count == 3


def test_cross_context_connectivity_outscores_narrow_repetition(package):
    policy = PromotionPolicy()
    narrow = [
        evidence(index, context="debugging", authority=EvidenceAuthority.SINGLE_OBSERVED)
        for index in range(5)
    ]
    diverse = [
        evidence(
            index + 10, context=context, authority=EvidenceAuthority.SINGLE_OBSERVED,
            connected=((f"ev-{index + 9}",) if index else ()),
        )
        for index, context in enumerate(("debugging", "planning", "chat"))
    ]
    assert policy.score(diverse)["promotion_score"] > policy.score(narrow)["promotion_score"]


def test_unsupported_model_claims_cannot_self_promote(package):
    for index in range(20):
        result = package.ingest(evidence(
            index, context=f"context-{index}",
            authority=EvidenceAuthority.MODEL_UNSUPPORTED,
        ))
    assert not result.promoted
    assert package.entries() == []


def test_explicit_user_correction_supersedes_weak_old_conclusion(package):
    old = promote(package, value="concise")
    correction = package.ingest(evidence(
        10, value="detailed", context="correction",
        authority=EvidenceAuthority.EXPLICIT_USER, confidence=1.0,
    ))
    assert correction.promoted
    assert package.entry(old.id).status == PersonalityStatus.SUPERSEDED.value
    new = package.entry(correction.entry_id)
    assert new.status == PersonalityStatus.ACTIVE.value
    assert new.confidence >= 0.9
    assert new.supporting_evidence_ids
    assert package.entry(old.id).contradicting_evidence_ids


def test_updates_are_inspectable_and_reversible(package):
    promote(package)
    before = package.entries()
    correction = package.ingest(evidence(
        10, value="detailed", context="correction",
        authority=EvidenceAuthority.EXPLICIT_USER, confidence=1.0,
    ))
    assert correction.promoted and package.changes()
    package.undo_last(timestamp="2026-02-01T00:00:00+00:00")
    assert package.entries() == before


def test_contextual_traits_route_to_matching_scope(package):
    technical = promote(
        package, value="concise", scope="technical",
        contexts=("debugging", "planning", "explanation"),
    )
    creative = promote(
        package, value="detailed", scope="creative",
        contexts=("poetry", "fiction", "worldbuilding"),
    )
    router = PersonalityRouter()
    selected = router.retrieve(package, PersonalityQuery(
        subject="user", interaction_type="technical", domain="debugging",
        relation="response_style", timestamp=STAMP,
    ), top_k=1)
    assert selected.entries[0].id == technical.id
    selected = router.retrieve(package, PersonalityQuery(
        subject="user", interaction_type="creative", domain="fiction",
        relation="response_style", timestamp=STAMP,
    ), top_k=1)
    assert selected.entries[0].id == creative.id


def test_relationship_specific_retrieval_beats_global(package):
    promote(package, value="formal", scope="global", subject="assistant")
    specific = promote(
        package, value="playful", scope="roleplay", subject="assistant",
        relationship="captain-mira",
    )
    selection = PersonalityRouter().retrieve(package, PersonalityQuery(
        subject="assistant", interaction_type="roleplay", domain="chat",
        relationship="captain-mira", relation="response_style", timestamp=STAMP,
    ), top_k=1)
    assert selection.entries[0].id == specific.id


def test_irrelevant_entry_rejection_and_top_k(package):
    for value, scope in (("concise", "technical"), ("structured", "planning"), ("warm", "social")):
        promote(package, value=value, scope=scope)
    router = PersonalityRouter()
    unrelated = router.retrieve(package, PersonalityQuery(
        subject="different-person", interaction_type="medical", domain="finance",
        relation="unrelated", timestamp=STAMP,
    ), top_k=4)
    assert unrelated.entries == ()
    relevant = router.retrieve(package, PersonalityQuery(
        subject="user", interaction_type="technical", domain="debugging",
        relation="response_style", timestamp=STAMP,
    ), top_k=4)
    assert 1 <= len(relevant.entries) <= 4
    for top_k in (1, 4, 8):
        assert len(router.retrieve(package, PersonalityQuery(
            subject="user", interaction_type="technical", domain="debugging",
            relation="response_style", timestamp=STAMP,
        ), top_k=top_k).entries) <= top_k


def test_p_cache_flow_is_explicitly_behavior_gated():
    store = CanonicalPStore(CanonicalPConfig(slots=2, width=512))
    slot, _ = store.create(
        torch.randn(512), entity_id=1, relation_id=1, value_id=1,
        metadata_id=0, label="user", confidence=0.7,
    )
    arguments = dict(
        evidence_id="p-observation", entry_type=PersonalityType.HABIT.value,
        relation="workflow", value="tests_first", context="debugging",
        scope="technical", timestamp=STAMP, archive_reference="archive://turn/99",
    )
    assert evidence_from_p_cache(store, slot, behavioral=False, **arguments) is None
    result = evidence_from_p_cache(store, slot, behavioral=True, **arguments)
    assert result is not None and result.subject == "user"
    assert result.archive_reference == "archive://turn/99"


def test_selected_entry_only_activation_and_sibling_merge(package):
    selected_entry = promote(package, value="Alice", scope="technical")
    promote(package, value="Bob", scope="creative")
    package.checkpoint()
    representation = FactorizedStateRepresentation(24, 3, 36)
    session = PersonalityTranslateSession(
        package.path, PersonalityRouter(),
        FactorizedPersonalityCanonicalizer(representation, value_labels=VALUE_LABELS),
    )
    activation = session.activate(PersonalityQuery(
        subject="user", interaction_type="technical", domain="debugging",
        relation="response_style", timestamp=STAMP,
    ), top_k=1)
    assert activation.store.cache.occupied == 1
    assert activation.selection.entries[0].id == selected_entry.id
    assert activation.inactive_vram_bytes == 0
    current = CanonicalPStore(CanonicalPConfig(slots=1, width=512))
    current.create(
        torch.randn(512), entity_id=2, relation_id=1, value_id=3,
        metadata_id=0, label="current task",
    )
    merged = merge_active_personality_with_p_cache(current, activation.store)
    assert merged.cache.occupied == 2
    assert current.cache.occupied == 1
    assert activation.store.cache.occupied == 1


def test_translate_session_rejects_package_changed_after_validation(package):
    promote(package, value="Alice", scope="technical")
    package.checkpoint()
    representation = FactorizedStateRepresentation(24, 3, 36)
    session = PersonalityTranslateSession(
        package.path, PersonalityRouter(),
        FactorizedPersonalityCanonicalizer(representation, value_labels=VALUE_LABELS),
    )
    package._writable()
    package._connection.execute("UPDATE entries SET value='tampered'")
    package._connection.commit()
    package._readonly()
    with pytest.raises(ValueError, match="changed after integrity validation"):
        session.activate(PersonalityQuery(
            subject="user", interaction_type="technical", domain="debugging",
            relation="response_style", timestamp=STAMP,
        ), top_k=1)


def test_cold_reload_needs_no_evidence_replay(package):
    promoted = promote(package, value="Alice", scope="technical")
    path = package.path
    package.close()
    with PersonalityPackage(path) as fresh:
        selection = PersonalityRouter().retrieve(fresh, PersonalityQuery(
            subject="user", interaction_type="technical", domain="debugging",
            relation="response_style", timestamp=STAMP,
        ), top_k=1)
        assert selection.entries[0].id == promoted.id
        assert selection.entries[0].supporting_evidence_ids


def test_top_k_and_evidence_push_do_not_recompute_full_checksum(package, monkeypatch):
    promote(package, value="concise", scope="technical")
    package.checkpoint()
    calls = 0
    original = package._semantic_checksum

    def counted_checksum():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(package, "_semantic_checksum", counted_checksum)
    router = PersonalityRouter()
    query = PersonalityQuery(
        subject="user", interaction_type="technical", domain="debugging",
        relation="response_style", timestamp=STAMP,
    )
    for _ in range(5):
        assert router.retrieve(package, query, top_k=4).entries
    assert calls == 0
    package.ingest(evidence(
        20, value="concise", context="new-context", scope="technical"
    ))
    assert calls == 0
    package.checkpoint()
    assert calls == 1


def test_translate_session_validates_once_and_reuses_connection(package, monkeypatch):
    promote(package, value="Alice", scope="technical")
    package.checkpoint()
    representation = FactorizedStateRepresentation(24, 3, 36)
    calls = 0
    original = PersonalityPackage._semantic_checksum

    def counted_checksum(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(PersonalityPackage, "_semantic_checksum", counted_checksum)
    session = PersonalityTranslateSession(
        package.path, PersonalityRouter(),
        FactorizedPersonalityCanonicalizer(representation, value_labels=VALUE_LABELS),
    )
    connection_id = id(session._package._connection)
    query = PersonalityQuery(
        subject="user", interaction_type="technical", domain="debugging",
        relation="response_style", timestamp=STAMP,
    )
    assert calls == 1
    for _ in range(3):
        assert session.activate(query, top_k=1).selection.entries
        assert id(session._package._connection) == connection_id
    assert calls == 1
    session.close()
