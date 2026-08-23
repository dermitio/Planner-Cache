# Architecture update: frozen Pythia-1.4B base

Date: 2026-08-22

Status: Active. Phase B implementation started; not complete.

## Decision

The self-trained base-model plan is retired. The local `pythia-1.4b` checkpoint
is now the frozen general-capability base. Its bundled configuration identifies
a 24-layer GPT-NeoX model with width 2048, 16 attention heads, 2,048 positions,
and a 50,304-token tokenizer vocabulary. The loaded checkpoint contains
1,414,647,808 parameters and is Apache-2.0 licensed.

This changes the base implementation and Phase A gate, but does not change the
authoritative memory architecture: P, recent KV, separate tool-KV, the adaptive
extension, archive search, and cold-resume reconciliation remain distinct.

## P-cache attachment decision

- P is a preallocated runtime store, initially 128 slots x width 512 in FP16.
  Values and every metadata field have fixed physical capacity at session
  creation; logical invalid slots are masked from reads.
- P is not converted to prompt text, prefix tokens, or self-attention KV.
- The frozen GPT-NeoX base reads P through a separately trainable, gated
  cross-attention residual. The proof starts at the final transformer layer to
  minimize trainable memory on the 4GB target GPU; reader layer indices remain
  configurable for later capacity experiments.
- Reader queries treat frozen GPT-NeoX hidden states as fixed features. This
  preserves the same forward behavior and full reader gradients while avoiding
  retention of a backward graph through the frozen 1.4B-parameter base.
- Reader gates initialize to zero, so attaching an empty/untrained planner path
  preserves the frozen checkpoint exactly.
- A separate learned controller predicts KEEP, CREATE, MODIFY, MERGE,
  INVALIDATE, or IGNORE plus slot/value/metadata targets. Explicit cache
  mutation remains outside the base model and is supervised in Phase B.
- When full, CREATE first attempts type-compatible semantic merge above a
  configured cosine threshold, then evicts the lowest-value non-permanent slot.
  Permanent entries cannot be silently displaced.

## Phase B gate

Phase B is not complete until tests and benchmarks show fixed allocation,
correct explicit mutations, mutable-state accuracy after recent-KV eviction,
useful P contribution versus P-disabled and rolling-summary baselines, and a
measurable slot-capacity tradeoff. Passing attachment smoke tests alone is not
completion.

## Phase B test checkpoint: 2026-08-22

The deterministic test workload contains 640 canonical entity states, labeled
MODIFY and INVALIDATE events, and enough irrelevant events to evict every state
event from a 32-event recent window. The comparison keeps alternatives honest:
KV-only retains its capped event window, rolling summary maintains up to 128
canonical facts, and retrieval performs reverse-chronological lookup over the
complete external archive. Retrieval is not artificially degraded.

An exact CUDA smoke test loaded the frozen local Pythia-1.4B checkpoint and
optimized only the final-layer P reader. CREATE recall loss decreased from
15.1557 to 0.00000036; after an in-place MODIFY it decreased from 14.5538 to
0.0. INVALIDATE made the wrapper output exactly equal to the P-disabled output.
This is an optimizer-path smoke test, not a trained checkpoint or evidence of
held-out generalization.

On the 640-state workload, active recall was 0.0% for 32-event KV-only, 19.97%
for 128-slot P, 19.97% for the 128-fact rolling summary, and 100% for archive
retrieval. P update accuracy was 100% at 128 slots and above; invalidation
accuracy was 100% at every tested capacity. The independent P sweep measured:

| P slots | Active recall | FP16 allocation including tensor metadata |
|---:|---:|---:|
| 64 | 9.98% | 66,880 bytes |
| 128 | 19.97% | 133,760 bytes |
| 256 | 39.93% | 267,520 bytes |
| 512 | 80.03% | 535,040 bytes |

Phase B remains incomplete. The capacity tradeoff, mutation mechanics, eviction
recall, and exact Pythia training path pass, but equal-capacity P currently ties
the rolling-summary oracle and loses accuracy to unbounded archive retrieval.
The next gate is held-out learned-controller/reader evaluation mixed with
natural RP, including P-disabled and matched-budget summary/retrieval ablations.

## Final Phase B gate: 2026-08-22

Status: **failed; Phase B remains incomplete.**

Two cache-policy corrections were required by the final tests:

- Semantic merge now runs before free-slot allocation, not only after P is
  full. Equivalent state therefore occupies one slot even when unused capacity
  exists; an unrelated fact still allocates a separate slot.
- Full-cache admission compares the incoming state's value against the weakest
  resident state. A lower-value incoming RP detail produces IGNORE instead of
  evicting a more important goal, constraint, or entity. Explicit
  correction-sourced state also cannot be overwritten by a later lower-authority
  model inference.

The importance-pressure workload used 1,024 simultaneously active candidates:
128 critical goals/constraints, 384 useful entities, and 512 irrelevant RP
details. Accuracy by class was:

| Slots | P critical | Summary critical | P useful | Summary useful | P irrelevant |
|---:|---:|---:|---:|---:|---:|
| 64 | 50.0% | 8.59% | 0.0% | 4.95% | 0.0% |
| 128 | 100% | 11.72% | 0.0% | 10.42% | 0.0% |
| 256 | 100% | 25.78% | 33.33% | 23.18% | 0.0% |
| 512 | 100% | 52.34% | 100% | 48.18% | 0.0% |

Physical P allocation remained unchanged throughout every workload and scaled
only with configuration: 66,880 / 133,760 / 267,520 / 535,040 bytes for
64 / 128 / 256 / 512 FP16 slots. Long CREATE -> MODIFY -> MODIFY -> transfer ->
correction -> INVALIDATE chains retained only the latest canonical value after
recent-KV eviction. Equivalent paraphrases used one slot; adding an unrelated
fact increased usage to two.

The lightweight controller achieved 100% held-out classification for KEEP,
CREATE, MODIFY, MERGE, INVALIDATE, and IGNORE. No physical slot IDs were
provided, and the lightweight reader achieved 95.83% on unseen entities and
values after mutation-shuffled training. However, semantic keys in this test
come from structured events and full end-to-end learned slot selection has not
yet been demonstrated, so this is not sufficient to pass the controller gate.

The exact frozen-Pythia natural-RP pressure test is the blocking result. Source
statements were absent from recent KV and no prompt tokens were added. Reader
training loss improved from 14.0147 to 5.7471, but held-out state candidate
accuracy was only 25%. P-enabled RP language loss worsened from 4.2219 to
5.2978. Mean batch latency increased from 32.72 ms to 33.62 ms. Therefore the
current attachment does not yet show acceptable held-out state-dependent
generation or preserved ordinary RP coherence.

The equal-budget comparison continues to show 0% current-state accuracy for
evicted KV-only, 19.97% for both 128-slot P and a 128-fact rolling summary, and
100% for unbounded archive retrieval. Retrieval's exact-recall win is expected
and is not a failure; it required 756 searches and an estimated 6,048 injected
tokens. The failures are the Pythia generalization/coherence result, the tie with
the equal-capacity rolling summary on the neutral workload, and the absence of
end-to-end learned slot selection.

## Reader-variant comparison: 2026-08-22

Nine reader/training variants were run with the same seed, 64 generation
optimization steps, train and held-out entities/values, candidate set, and zero
source tokens in recent KV. Preservation-training RP and coherence-evaluation
RP are disjoint. The frozen-base held-out RP loss was 5.2913.

| Variant | Trainable params | Final state loss | Held-out state | RP loss | RP KL |
|---|---:|---:|---:|---:|---:|
| Final-layer cross-attention | 2.63M | 5.747 | 25% | 6.475 | 0.658 |
| Upper 2 layers | 5.25M | 6.553 | 25% | 6.030 | 0.543 |
| Upper 4 layers | 10.50M | 5.862 | 25% | 5.284 | 0.0072 |
| Pooled residual | 1.05M | 5.669 | 25% | 6.170 | 0.396 |
| Per-token gate | 2.63M | 7.496 | 25% | 7.366 | 0.757 |
| Reader pretraining | 2.63M | 5.677 | 25% | 6.385 | 0.353 |
| Final layer + preservation | 2.63M | 6.811 | 25% | 5.347 | 0.0039 |
| Upper 4 + preservation | 10.50M | 6.103 | 25% | 5.292 | 0.0036 |
| Per-token gate + preservation | 2.63M | 6.153 | 25% | 5.291 | ~0 |

Reader pretraining used 400 state-selection/reconstruction steps before the
matched generation phase; its auxiliary loss decreased only from 1.0035 to
0.7735 and did not improve held-out accuracy. Explicit preservation used
ordinary-RP language loss plus the requested KL(with-P || frozen-base) from
cached frozen-base logits. It materially reduced drift. A per-token gate with
preservation reproduced the base almost exactly, while upper-four-layer
cross-attention with preservation achieved the best coherence/state-training
compromise.

No variant is adopted as a replacement because all remain at 25% held-out
state accuracy. Final-layer cross-attention remains the reproducible baseline.
Upper-four-layer cross-attention with preservation is the leading candidate for
the next training-design iteration, subject to fixing representation and
held-out state generalization first.

## Factorized representation localization test: 2026-08-22

An experimental factorized P representation separates entity, relation, value,
and metadata into four 128-wide learned fields before projecting to the normal
512-wide P slot. It is trained with tuple-level contrastive loss, current-state
selection loss, and canonical entity/relation/value/metadata decoder losses.
This is an experiment within P, not a replacement for fixed slots or explicit
metadata.

The compositional split contains 2,073 training tuples and 519 held-out
entity/relation/value combinations. Every entity and value appears during
training, but held-out combinations do not. Each probe includes randomized slot
order and hard negatives for wrong value, wrong entity, and historical tense.

The upstream P-only probe passed:

- representation loss: 7.4401 -> 0.0411;
- held-out canonical state recovery: 97.27%;
- canonical decoding: 100% for all four fields;
- wrong-value discrimination: 100%;
- wrong-entity discrimination: 97.27%;
- historical-state discrimination: 100%;
- stability across eight slot permutations per tuple: 100%.

Because the upstream probe passed, the exact frozen-Pythia test ran for 128
matched steps. The final-layer reader reached only 16.67% held-out composition
accuracy and severely damaged held-out RP loss (5.7331 base versus 11.4910 with
P). Upper-four-layer reading with KL preservation kept held-out RP close to base
(5.8346, KL 0.0150) but reached only 4.17% composition accuracy.

This localizes the present failure downstream of P representation: canonical,
contrastive, permutation-stable state is recoverable directly from P, but the
Pythia query/reader interface does not compose it into held-out token
generation. Phase B remains incomplete. The factorized encoder is retained as
an experimental representation candidate; it is not yet adopted as the active
runtime representation.
