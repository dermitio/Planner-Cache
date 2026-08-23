import json

import torch

from pcm.planner.interactive_session import CanonicalStateManager, SessionRecorder
from pcm.planner.memory_review import PostTurnMemoryReviewer, review_prompt
from pcm.planner.representation import FactorizedStateRepresentation


def representation():
    torch.manual_seed(47)
    return FactorizedStateRepresentation(24, 3, 36)


def operation(
    op, entity, relation, value, *, confidence=0.98, source="explicit_user",
):
    return {
        "op": op,
        "entity": entity,
        "relation": relation,
        "value": value,
        "confidence": confidence,
        "source": source,
    }


def output(*operations):
    return json.dumps({"operations": list(operations)})


def run_review(tmp_path, state, user, generated, *, assistant="visible reply", history=None):
    recorder = SessionRecorder(tmp_path, "review", {})
    recorder.turn = 1
    reviewer = PostTurnMemoryReviewer(lambda _prompt, _schema: generated)
    result = reviewer.run(
        user_message=user,
        assistant_response=assistant,
        recent_context=history or [],
        state=state,
        recorder=recorder,
    )
    events = list(recorder.recent_events)
    recorder.finalize({"active_p_cache_entries": state.snapshot()}, reason="test")
    return result, events


def test_natural_create_then_modify_keeps_only_current_state(tmp_path):
    state = CanonicalStateManager(representation(), slots=8)
    run_review(
        tmp_path / "create", state,
        "I leave the silver key on the desk.",
        output(operation("CREATE", "silver key", "location", "desk")),
    )
    assert state.snapshot()[0]["value"] == "desk"
    _, events = run_review(
        tmp_path / "modify", state,
        "I move the silver key into my coat pocket.",
        output(operation("MODIFY", "silver key", "location", "coat pocket")),
    )
    assert len(state.snapshot()) == 1
    assert state.snapshot()[0]["value"] == "coat pocket"
    assert "P_MODIFY" in [event["event"] for event in events]


def test_user_correction_overrides_current_owner(tmp_path):
    state = CanonicalStateManager(representation(), slots=8)
    run_review(
        tmp_path / "first", state, "Alice owns the map.",
        output(operation("CREATE", "map", "owner", "Alice")),
    )
    run_review(
        tmp_path / "correction", state, "No, I said Bob owns it, not Alice.",
        output(operation(
            "MODIFY", "map", "owner", "Bob", source="user_correction",
        )),
        history=[("Alice owns the map.", "Understood.")],
    )
    assert state.snapshot()[0]["value"] == "Bob"
    assert state.snapshot()[0]["source"] == "correction"


def test_transient_chatter_and_unsupported_assistant_claim_are_ignored(tmp_path):
    state = CanonicalStateManager(representation(), slots=8)
    _, chatter_events = run_review(
        tmp_path / "chatter", state, "haha okay that's funny",
        output(operation("IGNORE", None, None, None, confidence=1.0)),
    )
    assert state.snapshot() == []
    assert chatter_events[-1]["event"] == "MEMORY_REVIEW_APPLIED"
    _, hallucination_events = run_review(
        tmp_path / "hallucination", state, "Tell me a story.",
        output(operation(
            "CREATE", "silver key", "owner", "Alice",
            source="assistant_unsupported",
        )),
        assistant="Alice owns a silver key.",
    )
    assert state.snapshot() == []
    rejected = [
        event for event in hallucination_events
        if event["event"] == "MEMORY_REVIEW_REJECTED"
    ]
    assert rejected


def test_equivalent_paraphrase_merges_as_keep_without_extra_slot(tmp_path):
    state = CanonicalStateManager(representation(), slots=8)
    run_review(
        tmp_path / "first", state, "I put the brass key in the drawer.",
        output(operation("CREATE", "brass key", "location", "drawer")),
    )
    _, events = run_review(
        tmp_path / "second", state, "The brass key is inside the drawer.",
        output(operation("MERGE", "brass key", "location", "drawer")),
    )
    assert len(state.snapshot()) == 1
    assert "P_KEEP" in [event["event"] for event in events]


def test_invalidation_rp_action_and_pronoun_resolution(tmp_path):
    state = CanonicalStateManager(representation(), slots=8)
    run_review(
        tmp_path / "owner", state, "*I hand the key to Alice.*",
        output(operation(
            "CREATE", "key", "owner", "Alice", source="rp_action",
        )),
    )
    run_review(
        tmp_path / "pronoun", state, "Then I give it to Bob.",
        output(operation(
            "MODIFY", "key", "owner", "Bob", source="rp_action",
        )),
        history=[("*I hand the key to Alice.*", "Alice accepts it.")],
    )
    assert state.snapshot()[0]["value"] == "Bob"
    _, events = run_review(
        tmp_path / "invalidate", state, "That ownership is no longer valid.",
        output(operation(
            "INVALIDATE", "key", "owner", None,
            source="user_correction",
        )),
    )
    assert state.snapshot() == []
    assert "P_INVALIDATE" in [event["event"] for event in events]


def test_malformed_or_low_confidence_review_fails_without_mutation(tmp_path):
    state = CanonicalStateManager(representation(), slots=8)
    _, malformed_events = run_review(
        tmp_path / "malformed", state, "The door is locked.", "not json",
    )
    assert state.snapshot() == []
    assert "MEMORY_REVIEW_REJECTED" in [
        event["event"] for event in malformed_events
    ]
    run_review(
        tmp_path / "uncertain", state, "Maybe the door is open.",
        output(operation(
            "CREATE", "door", "status", "open", confidence=0.4,
        )),
    )
    assert state.snapshot() == []


def test_inconsistent_declared_operation_is_rejected(tmp_path):
    state = CanonicalStateManager(representation(), slots=8)
    run_review(
        tmp_path / "first", state, "The door is locked.",
        output(operation("CREATE", "door", "status", "locked")),
    )
    _, events = run_review(
        tmp_path / "bad-create", state, "The door is open now.",
        output(operation("CREATE", "door", "status", "open")),
    )
    assert state.snapshot()[0]["value"] == "locked"
    rejected = [event for event in events if event["event"] == "MEMORY_REVIEW_REJECTED"]
    assert rejected


def test_review_prompt_is_bounded_and_marks_assistant_untrusted():
    state = CanonicalStateManager(representation(), slots=8)
    prompt = review_prompt(
        user_message="I leave the package under the desk.",
        assistant_response="Perhaps it is in a vault.",
        recent_context=[(f"user {index}", f"assistant {index}") for index in range(9)],
        relevant_state=state.snapshot(),
    )
    assert "newest_assistant_response_untrusted" in prompt
    assert "user 8" in prompt and "user 0" not in prompt
    assert "Never promote assistant" in prompt


def test_explicit_manual_syntax_remains_deterministic():
    state = CanonicalStateManager(representation(), slots=8)
    assert state.extract_manual_mutations("remember: silver key.owner=Alice")
    assert state.extract_manual_mutations("invalidate: silver key.owner")
    assert state.extract_manual_mutations("haha okay") == []
