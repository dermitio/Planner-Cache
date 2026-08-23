"""Fail-closed post-turn semantic review for the active canonical P-cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
import time
from typing import Callable

from pcm.planner.interactive_session import (
    CanonicalStateManager,
    MutationIntent,
    RELATION_NAMES,
    SessionRecorder,
)


REVIEW_FORMAT = "planner-cache-memory-review-v1"
REVIEW_CONFIDENCE_FLOOR = 0.80
MAX_REVIEW_OPERATIONS = 8

REVIEW_SOURCES = (
    "explicit_user",
    "user_correction",
    "rp_action",
    "tool_verified",
    "assistant_inference",
    "assistant_unsupported",
)
AUTHORITATIVE_REVIEW_SOURCES = {
    "explicit_user", "user_correction", "rp_action",
}

MEMORY_REVIEW_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["operations"],
    "properties": {
        "operations": {
            "type": "array",
            "maxItems": MAX_REVIEW_OPERATIONS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "op", "entity", "relation", "value", "confidence", "source",
                ],
                "properties": {
                    "op": {"enum": [
                        "CREATE", "MODIFY", "KEEP", "MERGE",
                        "INVALIDATE", "IGNORE",
                    ]},
                    "entity": {"type": ["string", "null"], "maxLength": 160},
                    "relation": {
                        "type": ["string", "null"],
                        "enum": ["owner", "location", "status", None],
                    },
                    "value": {"type": ["string", "null"], "maxLength": 160},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "source": {"enum": list(REVIEW_SOURCES)},
                },
            },
        },
    },
}


@dataclass(frozen=True)
class ReviewedOperation:
    op: str
    entity: str | None
    relation: str | None
    value: str | None
    confidence: float
    source: str


@dataclass(frozen=True)
class ValidatedReview:
    proposed: tuple[ReviewedOperation, ...]
    accepted: tuple[ReviewedOperation, ...]
    intents: tuple[MutationIntent, ...]
    rejected: tuple[dict[str, object], ...]


def review_prompt(
    *,
    user_message: str,
    assistant_response: str,
    recent_context: list[tuple[str, str]],
    relevant_state: list[dict[str, object]],
) -> str:
    rules = """You are the hidden Planner Cache memory reviewer.
Return one JSON object matching the supplied schema and no prose.
P-cache stores current useful semantic state, never transcript summaries.
Allowed relations are owner, location, and status.
CREATE a new current fact. MODIFY a changed current value. KEEP an unchanged
current fact. MERGE only equivalent duplicate state. INVALIDATE a fact the user
explicitly says is no longer valid. IGNORE transient chat and unsupported claims.
Treat explicit user corrections as highest authority. RP actions directly written
by the user are authoritative current events. Never promote assistant inventions,
inferences, suggestions, jokes, or unsupported generated claims. Use source
assistant_inference or assistant_unsupported for such candidates so validation can
reject them. Resolve pronouns only when recent context or current P makes the
referent unambiguous. Prefer IGNORE when uncertain. Historical values must not
remain active beside their corrected current value.
The source value must be exactly one of: explicit_user, user_correction,
rp_action, tool_verified, assistant_inference, assistant_unsupported."""
    payload = {
        "format": REVIEW_FORMAT,
        "newest_user_message": user_message[-2000:],
        "newest_assistant_response_untrusted": assistant_response[-1000:],
        "limited_recent_context": [
            {"user": user[-600:], "assistant": assistant[-600:]}
            for user, assistant in recent_context[-2:]
        ],
        "relevant_current_p": relevant_state,
        "required_output": {
            "operations": [{
                "op": "CREATE|MODIFY|KEEP|MERGE|INVALIDATE|IGNORE",
                "entity": "string|null",
                "relation": "owner|location|status|null",
                "value": "string|null",
                "confidence": "number 0..1",
                "source": "one allowed authority label from the rules",
            }],
        },
    }
    return rules + "\n\nINPUT:\n" + json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
    )


def parse_review(raw: str) -> tuple[ReviewedOperation, ...]:
    if raw != raw.strip():
        raw = raw.strip()
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"operations"}:
        raise ValueError("review must contain only operations")
    operations = value["operations"]
    if not isinstance(operations, list) or len(operations) > MAX_REVIEW_OPERATIONS:
        raise ValueError("review operations must be a bounded array")
    parsed = []
    required = {"op", "entity", "relation", "value", "confidence", "source"}
    for item in operations:
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError("review operation has an invalid shape")
        op = str(item["op"]).upper()
        if op not in {"CREATE", "MODIFY", "KEEP", "MERGE", "INVALIDATE", "IGNORE"}:
            raise ValueError("unsupported review operation")
        source = str(item["source"])
        if source not in REVIEW_SOURCES:
            raise ValueError("unsupported review source")
        confidence = float(item["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError("review confidence must be between zero and one")
        entity = item["entity"]
        relation = item["relation"]
        candidate_value = item["value"]
        if entity is not None and not isinstance(entity, str):
            raise ValueError("entity must be a string or null")
        if relation is not None and relation not in RELATION_NAMES:
            raise ValueError("unsupported canonical relation")
        if candidate_value is not None and not isinstance(candidate_value, str):
            raise ValueError("value must be a string or null")
        parsed.append(ReviewedOperation(
            op=op,
            entity=None if entity is None else entity.strip()[:160],
            relation=relation,
            value=None if candidate_value is None else candidate_value.strip()[:160],
            confidence=confidence,
            source=source,
        ))
    return tuple(parsed)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[\w'-]+", text.casefold()))


def relevant_state_for_review(
    state: CanonicalStateManager,
    user_message: str,
    recent_context: list[tuple[str, str]],
    *,
    limit: int = 12,
) -> list[dict[str, object]]:
    context = " ".join(
        [user_message]
        + [part for turn in recent_context[-2:] for part in turn]
    )
    context_words = _words(context)
    entries = state.snapshot()
    ranked = sorted(
        entries,
        key=lambda entry: (
            bool(_words(str(entry.get("entity", ""))) & context_words),
            int(entry.get("slot_id", -1)),
        ),
        reverse=True,
    )
    relevant = [
        entry for entry in ranked
        if _words(str(entry.get("entity", ""))) & context_words
    ]
    if not relevant:
        relevant = ranked[:4]
    return [
        {
            key: entry.get(key)
            for key in ("slot_id", "entity", "relation", "value", "confidence", "source")
        }
        for entry in relevant[:limit]
    ]


def validate_review(
    operations: tuple[ReviewedOperation, ...],
    state: CanonicalStateManager,
) -> ValidatedReview:
    accepted = []
    intents = []
    rejected = []
    for operation in operations:
        reason = None
        if operation.op == "IGNORE":
            continue
        if operation.source not in AUTHORITATIVE_REVIEW_SOURCES:
            reason = "assistant or unsupported evidence cannot mutate P"
        elif operation.confidence < REVIEW_CONFIDENCE_FLOOR:
            reason = "confidence below conservative review threshold"
        elif not operation.entity or not operation.relation:
            reason = "entity and canonical relation are required"
        elif operation.op != "INVALIDATE" and not operation.value:
            reason = "value is required for current-state mutation"
        if reason is not None:
            rejected.append({"operation": asdict(operation), "reason": reason})
            continue
        relation_id = RELATION_NAMES.index(operation.relation)
        matches = state._matching_slots(operation.entity, relation_id)
        if operation.op == "INVALIDATE":
            if not matches:
                rejected.append({
                    "operation": asdict(operation),
                    "reason": "no matching active state to invalidate",
                })
                continue
            intent = MutationIntent("invalidate", operation.entity, relation_id)
        else:
            value_id, surface, compatible = state._canonical_value(operation.value or "")
            exact = any(
                int(state.store.value_id[slot]) == value_id
                and state.surface_values.get(slot, "").casefold() == surface.casefold()
                for slot in matches
            )
            consistency_reason = None
            if operation.op == "CREATE" and matches:
                consistency_reason = "CREATE cannot replace existing current state"
            elif operation.op == "MODIFY" and (not matches or exact):
                consistency_reason = "MODIFY requires a changed existing current value"
            elif operation.op in {"KEEP", "MERGE"} and not exact:
                consistency_reason = (
                    f"{operation.op} requires equivalent existing current state"
                )
            if consistency_reason is not None:
                rejected.append({
                    "operation": asdict(operation), "reason": consistency_reason,
                })
                continue
            intent = MutationIntent(
                "upsert", operation.entity, relation_id,
                value_id, surface, compatible,
            )
        accepted.append(operation)
        intents.append(intent)
    return ValidatedReview(
        proposed=operations,
        accepted=tuple(accepted),
        intents=tuple(intents),
        rejected=tuple(rejected),
    )


class PostTurnMemoryReviewer:
    """Run one serialized same-model review and apply only validated user state."""

    def __init__(self, generate: Callable[[str, dict[str, object]], str]) -> None:
        self.generate = generate

    def run(
        self,
        *,
        user_message: str,
        assistant_response: str,
        recent_context: list[tuple[str, str]],
        state: CanonicalStateManager,
        recorder: SessionRecorder,
    ) -> ValidatedReview | None:
        relevant = relevant_state_for_review(state, user_message, recent_context)
        prompt = review_prompt(
            user_message=user_message,
            assistant_response=assistant_response,
            recent_context=recent_context,
            relevant_state=relevant,
        )
        started = time.perf_counter()
        recorder.event(
            "MEMORY_REVIEW_START", source="memory_review",
            source_turn=recorder.turn, relevant_state=relevant,
            recent_context_turns=min(2, len(recent_context)),
        )
        before = state.snapshot()
        raw = ""
        try:
            raw = self.generate(prompt, MEMORY_REVIEW_SCHEMA)
            operations = parse_review(raw)
            validated = validate_review(operations, state)
        except Exception as error:
            recorder.event(
                "MEMORY_REVIEW_REJECTED", source="memory_review",
                source_turn=recorder.turn, validation="malformed_or_runtime_error",
                error_type=type(error).__name__, message=str(error),
                raw_output=raw[:4000],
                latency_seconds=time.perf_counter() - started,
            )
            return None
        recorder.event(
            "MEMORY_REVIEW_RESULT", source="memory_review",
            source_turn=recorder.turn,
            proposed_operations=[asdict(item) for item in validated.proposed],
            accepted_operations=[asdict(item) for item in validated.accepted],
            rejected_operations=list(validated.rejected),
            validation="accepted" if validated.intents else "no_mutation",
            latency_seconds=time.perf_counter() - started,
        )
        if validated.rejected:
            recorder.event(
                "MEMORY_REVIEW_REJECTED", source="memory_review",
                source_turn=recorder.turn,
                validation="operation_rejection",
                rejected_operations=list(validated.rejected),
            )
        if validated.intents:
            cache = state.store.cache
            tensor_owners = (
                (cache, (
                    "values", "valid", "slot_type", "confidence", "importance",
                    "freshness", "persistence", "last_updated", "source",
                )),
                (state.store, (
                    "entity_id", "relation_id", "value_id", "canonical_metadata_id",
                )),
            )
            tensors = {
                (id(owner), name): getattr(owner, name).clone()
                for owner, names in tensor_owners for name in names
            }
            labels = list(cache.labels)
            clock = cache._clock
            surfaces = dict(state.surface_values)
            compatibility = dict(state.translator_compatible)
            buffered_events: list[tuple[str, dict[str, object]]] = []

            class BufferedRecorder:
                @staticmethod
                def event(event: str, **fields: object) -> None:
                    buffered_events.append((event, fields))

            try:
                state.apply(validated.intents, BufferedRecorder())
            except Exception as error:
                for owner, names in tensor_owners:
                    for name in names:
                        getattr(owner, name).copy_(tensors[(id(owner), name)])
                cache.labels = labels
                cache._clock = clock
                state.surface_values = surfaces
                state.translator_compatible = compatibility
                recorder.event(
                    "MEMORY_REVIEW_REJECTED", source="memory_review",
                    source_turn=recorder.turn,
                    validation="atomic_apply_failed",
                    error_type=type(error).__name__, message=str(error),
                    latency_seconds=time.perf_counter() - started,
                )
                return None
            for event, fields in buffered_events:
                recorder.event(event, **fields)
        after = state.snapshot()
        recorder.event(
            "MEMORY_REVIEW_APPLIED", source="memory_review",
            source_turn=recorder.turn,
            applied_mutations=[asdict(item) for item in validated.intents],
            before=before, after=after,
            changed=before != after,
            latency_seconds=time.perf_counter() - started,
        )
        return validated
