# Architecture update: split canonical router and model translator

Date: 2026-08-22

Status: Phase B split-router experiment passed its completion gate. Phase C has
not started and remains out of scope pending an explicit decision.

## Decision

The rejected monolithic translator is replaced by two independent stages:

```text
frozen-model hidden + tokenizer-independent entity surface
    -> model-specific canonical query projector
    -> factorized entity/relation/metadata query
    -> universal canonical router
    -> selected current canonical slot(s)
    -> model-specific value translator and relevance gate
    -> residual injection into the frozen model
```

The router never consumes a Pythia hidden dimension. It ranks only canonical
entity, relation, metadata, and validity/current-state fields. The canonical P
protocol and factorized 512-wide representation are unchanged. P remains
separate from prompts, recent self-attention KV, tool KV, and the archive.

The selected implementation uses a tokenizer-independent byte encoder for the
entity surface present in the current query. This is not an oracle slot ID: it
does not expose the answer, source state, value, or physical slot position. It
provides a canonical entity anchor that can be compared with every candidate
slot. Relation and query-type/metadata fields are predicted from the frozen LM
hidden state. A hidden-only reconstruction ablation reached 12.5% unseen-name
entity accuracy, and the experimental frozen-model lexical anchor reached
78.125%; the portable byte-derived anchor reached 100% and was selected.

The universal router uses four semantically justified features: canonical
entity similarity, relation agreement, query-type agreement, and current-valid
state. Confidence, importance, freshness, persistence, and source are not used
as synthetic-answer shortcuts. The router contains five learned scalar
parameters, supports top-1/top-2/top-4 selection, and has a calibrated rejection
threshold. Rejected, invalidated, historical, and wrong-entity selections are
suppressed before translation and gating.

## Portable formats

`pcm-split-translate-v1` stores only:

- model identifier/revision and base-config checksum;
- model hidden width and attachment layers;
- canonical width and P protocol version;
- query-projector, value-translator, and conditional-gate tensors; and
- architecture and tensor checksums.

It contains no base weights, P contents, conversation state, or model-native
vectors inside canonical P. Compatibility loading rejects wrong model IDs,
hidden widths, attachment layers, canonical widths/protocols, and checksums.

The model-independent router is serialized separately as
`pcm-canonical-router-v1`. The selected files are:

```text
artifacts/pythia-1.4b-split-final_layer.translate
artifacts/canonical-p-v1.router
```

Both deterministic save/load round trips have zero output difference. A future
decoder-only model port therefore needs model-hidden-to-canonical query and
canonical-value-to-model-hidden translation, while retaining canonical P and
the universal router.

## Training order

The benchmark trained each component in the required sequence:

1. query projector pretraining on 256 compositional training names, followed by
   evaluation on 64 disjoint unseen names;
2. canonical-router training with the query projector frozen and aggressive
   wrong-entity, wrong-relation, historical, invalidated, and irrelevant hard
   negatives;
3. oracle-routed value translation toward frozen Pythia LM-head value targets;
4. end-to-end causal training of only the compatibility modules, with natural
   RP KL/LM preservation introduced in the second half.

The required router-only, oracle-translator, router-plus-translator without
gate, gated, and full-preservation ablations are saved in the result artifact.
All base-model parameters remained frozen and zero base parameters received a
gradient. No LoRA was used.

## Exact CUDA results

The exact Pythia-1.4B CUDA experiment used the same architecture and data for
the final-layer, upper-two-layer, and upper-four-layer attachments. All three
reached 100% held-out state candidate and exact generated-token accuracy. The
final layer was selected because it is the smallest attachment with the same
state result and exact irrelevant-RP preservation; deeper attachment was not
selected merely for state performance.

For the selected final-layer package:

- package parameters: 2,707,464; universal router parameters: 5;
- held-out byte-derived entity/relation/metadata query accuracy: 100%;
- oracle-routed value accuracy: 100%, cosine similarity 0.999851;
- hard-negative top-1: 100%;
- wrong-entity, wrong-relation, historical, invalidated, and irrelevant false
  positives: 0%;
- mutation-chain latest-state accuracy: 100%;
- invalidated versus P-disabled maximum logit difference: 0;
- source-state tokens in recent KV: 0; extra prompt tokens: 0; and
- active base-model parameters receiving gradients: 0.

Global routing and generation scaled as follows. Active overhead includes the
translator, canonical P allocation, universal router, and transient canonical
routing index; it remains fixed for each configured capacity and independent of
conversation length.

| P slots | Top-1 | Top-4 recall | State generation | Latency | Active overhead |
|---:|---:|---:|---:|---:|---:|
| 4 | 100% | 100% | 100% | 26.63 ms | 10,836,300 B |
| 20 | 100% | 100% | 100% | 86.69 ms | 10,861,996 B |
| 64 | 100% | 100% | 100% | 92.33 ms | 10,932,660 B |
| 128 | 100% | 100% | 100% | 101.45 ms | 11,035,444 B |
| 256 | 95% | 95% | 95% | 118.11 ms | 11,241,012 B |
| 512 | 85% | 85% | 85% | 164.55 ms | 11,652,148 B |

This substantially exceeds the immutable rejected-translator global 20-fact
result of 65-70%, clears the 128-slot proof target, and degrades rather than
collapses at 256 and 512 slots.

## Causality and preservation

For the identical prompt `The silver key currently belongs to`, only P changed:

| Condition | Alice logit | Bob logit | Gate | Generated token |
|---|---:|---:|---:|---|
| P disabled | 5.5156 | 6.2734 | 0 | ` a` |
| silver_key.owner = Alice | 40.0938 | 14.3125 | 0.9964 | ` Alice` |
| silver_key.owner = Bob | 13.0234 | 39.1562 | 0.9843 | ` Bob` |
| gold_key.owner = Alice | 5.5156 | 6.2734 | 0 | ` a` |
| silver_key historical | 5.5156 | 6.2734 | 0 | ` a` |
| silver_key invalidated | 5.5156 | 6.2734 | 0 | ` a` |

Thus relevant P changes token logits and generation directionally, while wrong
entity, historical, and invalidated state reproduce the disabled path.

On the disjoint natural-RP set, frozen base, irrelevant P, wrong-entity P, and
invalidated P all produced loss 5.5833669 and identical samples. Measured KL was
numerical zero (approximately -4e-8). The relevant state channel nevertheless
remained fully active, so preservation did not collapse it to zero.

The rejected translator numbers in update 0004 and
`Archive/rejected-translators/artifacts/phase-b-translator.json` remain unchanged
and are copied into the new
benchmark artifact as an immutable baseline. Full reproducible results are in
`artifacts/phase-b-split-translator.json`. Passing this experiment does not
authorize Phase C; no Phase C work was performed.
