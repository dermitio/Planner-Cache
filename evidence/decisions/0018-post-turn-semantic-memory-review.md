# Post-turn semantic memory review

Date: 2026-08-23

Status: active interactive-runtime behavior

## Decision

Normal browser chat now runs a hidden semantic memory review after each visible assistant response.

```text
unchanged visible browser messages
-> frozen model generation
-> visible assistant response delivered
-> serialized hidden memory review
-> strict validation
-> existing CanonicalStateManager mutation path
-> next user turn may begin
```

The review is a side-channel. It does not rewrite the native browser message array, add prompt tokens to visible generation, route through TTL or LTL, or treat P-cache as a transcript summary.

## Existing policy preserved

The reviewer can propose `CREATE`, `MODIFY`, `KEEP`, `MERGE`, `INVALIDATE`, or `IGNORE`. Accepted proposals become the existing `MutationIntent` form and pass through `CanonicalStateManager.apply`. The manager remains authoritative for fixed-capacity admission, identity-safe merge, in-place modification, invalidation, current-state replacement, and emitted `P_*` events.

The active proof still exposes only the established owner, location, and status relations. This review pass does not expand the canonical protocol or introduce a second state schema.

Deterministic `remember:`, `invalidate:`, `forget:`, and supported `note that` forms remain pre-generation manual overrides. Other natural language is reviewed after the visible response.

## Authority and validation

The strict review source labels are:

- `explicit_user`
- `user_correction`
- `rp_action`
- `tool_verified`
- `assistant_inference`
- `assistant_unsupported`

Interactive review accepts only explicit user statements, user corrections, and directly authored RP actions. Tool evidence is not supplied by the current interactive gateway, so a reviewer cannot claim tool authority. Assistant inference and unsupported assistant claims are always rejected.

The conservative confidence floor is 0.80. The original architecture did not specify a numeric learned-review threshold, so this is the simplest explicit initial choice that preserves fail-closed behavior. It is public in `memory_review.py` and covered by regression tests.

Declared operations must agree with current P. `CREATE` cannot overwrite an existing entity and relation. `MODIFY` requires an existing changed value. `KEEP` and `MERGE` require equivalent current state. `INVALIDATE` requires a matching active state. Malformed JSON, unknown fields, unsupported relations, invalid authority, low confidence, and inconsistent operations leave P unchanged.

Application is atomic. Review mutations write through a buffered event sink. On an application error all canonical tensors, labels, clock state, surface values, and compatibility metadata are restored in place before a rejection event is recorded.

## Bounded review input

Review receives the newest user message, the untrusted newest assistant response, at most two prior turns, and at most twelve relevance-ranked P entries. Per-field character limits prevent this side-channel from becoming another transcript window. Selected P entries include only slot ID, entity, relation, value, confidence, and source.

## Runtime reviewers

Gemma uses the same frozen Gemma4 E4B GGUF through a separate grammar-constrained llama.cpp request. LTL is disabled for review.

The frozen Pythia-1.4B base checkpoint was tested twice as its own reviewer. It copied the instruction or payload rather than returning strict JSON. Both failures were rejected without mutation and remain in session evidence. The active Pythia launcher therefore uses the configured frozen Gemma GGUF as a CPU-only structured reviewer while Pythia remains the visible frozen model and TTL consumer. This avoids exceeding the 4 GB GPU. The review model path and the fact that TTL and LTL are inactive are recorded in session metadata.

No deterministic mutation fallback is used for malformed model output.

## Ordering and logging

The Web UI response body is flushed before review begins. The session lock remains held until review finishes, so a subsequent generation, `/state`, `/events`, or `/personality` read cannot observe an intermediate state or overlap another review.

Every reviewed turn records:

- `MEMORY_REVIEW_START`
- `MEMORY_REVIEW_RESULT` when strict parsing completes
- `MEMORY_REVIEW_REJECTED` for malformed or rejected proposals
- the actual `P_CREATE`, `P_KEEP`, `P_MODIFY`, `P_MERGE`, `P_INVALIDATE`, or `P_IGNORE` events that occur
- `MEMORY_REVIEW_APPLIED` with before and after state and total latency

## Manual acceptance

The exact Gemma interactive path created `brass key.location = kitchen drawer` from ordinary RP with confidence 1.0 and source `rp_action`. A subsequent ordinary RP turn changed the same slot to `coat pocket`, emitted `P_MODIFY`, and left only the current value active.

The Pythia interactive path, using the CPU structured reviewer described above, created the same natural RP state while its visible response continued through frozen Pythia and the existing TTL path.

Measured hidden review latency on this machine was approximately 43 to 48 seconds for Gemma with its configured partial GPU offload and approximately 65 seconds for the CPU reviewer alongside resident Pythia. Visible response delivery precedes this cost, while the next turn waits for deterministic completion.
