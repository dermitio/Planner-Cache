# Architecture update: bidirectional canonical-P translator experiment

Date: 2026-08-22

Status: Current translator form rejected. Phase B remains incomplete and Phase C
must not start.

## Scope and invariants

This experiment kept the factorized 512-wide canonical P representation and
`pcm-canonical-p-v1` snapshot protocol unchanged. P remained a fixed-capacity
store separate from prompt tokens, recent self-attention KV, tool KV, and the
archive. Pythia-1.4B remained frozen, no LoRA was installed, source-state tokens
were absent from recent KV, and no source facts were added to prompts.

Only a model-specific compatibility translator was trained. The experiment
therefore tests the boundary:

```text
canonical P state -> model-specific .translate -> frozen decoder LM
```

It does not alter the P controller, canonical representation, base model,
adaptive extension, tool-KV design, or cold-resume reconciliation.

## Portable translator contract

`PTranslator` is the model-agnostic interface. A compatibility implementation
must provide:

- `model_to_p(hidden)`: model hidden width to a normalized canonical P query;
- `p_to_model(canonical)`: canonical width to model hidden width;
- `read(hidden, store)`: content retrieval, translation, relevance gating, and
  a residual with the original model dtype; and
- compatibility validation against model identifier, hidden width, canonical
  width/protocol, attachment layers, and optionally a base-config checksum.

The reference `BidirectionalPTranslator` contains two explicit MLPs:

```text
Model -> P:     LayerNorm(H) -> Linear(H,512) -> GELU -> Linear(512,512)
P -> Model:     LayerNorm(512) -> Linear(512,512) -> GELU -> Linear(512,H)
Joint gate:     [LayerNorm(H), LayerNorm(P_model)] -> Linear(2H,64)
                -> GELU -> Linear(64,1) -> sigmoid
Output:         H + gate * P_model
```

Canonical retrieval uses cosine scores at temperature 0.07 and masks invalid
physical slots. The same translator is shared when attached to multiple upper
layers, so the layer sweep does not multiply its parameter count. The Pythia
adapter uses decoder-block forward hooks; an adapter for another decoder-only
transformer needs only to validate its configuration and apply `read()` to the
configured block outputs. It must not modify `CanonicalPStore`.

### `.translate` format

The extension is conventional; the files are safetensors with format metadata
`pcm-translate-v1`. Metadata contains:

- model identifier and revision;
- model-hidden and canonical widths;
- decoder attachment-layer indices;
- canonical P protocol version;
- translator architecture/version;
- retrieval temperature;
- SHA-256 of the base model configuration; and
- deterministic SHA-256 of the translator tensor set.

Only translator and gate tensors are stored. Base-model weights, P-cache
contents, conversation state, and canonical P values are forbidden. Loading
recomputes the tensor checksum. Compatibility validation rejects the wrong
model identifier, model width, P width/protocol, layer set, or model-config
checksum. The exact CUDA benchmark saved and reloaded every adapter with zero
round-trip difference.

To implement a new adapter, construct `TranslatorConfig` for the frozen model,
train a `PTranslator` implementation against the canonical protocol, save it
independently, and add only the model-specific block-hook wrapper. A canonical P
snapshot can then be loaded without knowing which translator will consume it.

## Training procedure

The factorized representation was trained once with the existing deterministic
split and then frozen for all translator variants.

1. Model-to-P alignment trained only the query MLP against canonical query
   cosine and four-way retrieval loss. Training prompts included question and
   causal-statement forms. Evaluation used disjoint prompts and held-out
   entity/value compositions.
2. P-to-Model alignment trained only the value MLP. Its documented native
   target was the frozen Pythia LM-head embedding of the correct single-token
   person/location label. Loss combined cosine, 36-way value classification,
   and vector-magnitude alignment.
3. Causal readout trained all translator parameters, still with the base frozen.
   The prompt contained only the queried entity and no source value. Changing
   only canonical P changed the teacher-forced target. Training also included
   current versus wrong-entity/historical gate supervision and disjoint natural
   RP preservation.

The Stage 3 objective weights were:

```text
1.0 * state-token CE
+ 0.5 * Model->P query alignment
+ 0.2 * P->Model value alignment
+ 2.0 * relevance-gate BCE
+ 0.05 * ordinary-RP LM loss
+ 2.0 * KL(with-P || frozen-base)
```

Every layer variant used seed 211, 400 Stage 1 steps, 400 Stage 2 steps, and 256
Stage 3 steps. Slot order was randomized and no evaluation slot ID was provided
to the translator.

## Exact Pythia-1.4B results

The translator contains 2,900,609 FP32 parameters (11,602,436 bytes). Including
a 128-slot canonical P store, accounted active overhead is 11,740,292 bytes and
does not depend on attachment count or conversation length.

| Metric | Final layer | Upper 2 | Upper 4 |
|---|---:|---:|---:|
| Held-out four-way Model->P retrieval | 100% | 100% | 100% |
| Global retrieval among 20 current facts | 65% | 65% | 70% |
| Held-out causal-prompt global retrieval | 25% | 35% | 40% |
| Held-out P->Model value accuracy | 100% | 100% | 100% |
| Correct-token logit lift | +10.10 | +16.24 | +16.65 |
| Exact generated-token accuracy | 45% | 55% | 65% |
| A/B state-swap pairwise directionality | 100% | 100% | 100% |
| Historical-state gate | 0.00082 | 0.00089 | 0.00118 |
| Wrong-entity gate | 0.517 | 0.691 | 0.702 |
| Unseen-name state accuracy | 25% | 0% | 25% |
| Mutation-chain latest-state accuracy | 100% | 100% | 100% |
| Invalidated versus P-disabled max logit difference | 0 | 0 | 0 |
| Random-slot-order stability | 100% | 100% | 100% |
| RP loss, disabled -> enabled | 4.6547 -> 4.6569 | 4.6547 -> 6.0861 | 4.6547 -> 5.7049 |
| RP KL(with P || base) | 0.000010 | 0.00352 | 0.01387 |
| Latency, disabled -> enabled | 85.82 -> 86.80 ms | 95.88 -> 98.85 ms | 96.03 -> 102.19 ms |

The identical-prompt causal matrix includes P disabled, canonical state A,
canonical state B, wrong entity, historical state, and invalidated P. It records
both candidate logits/probabilities, unrestricted one-token generation, and gate
activation. Invalidation always reproduced the P-disabled logits exactly.
Changing only P reliably reversed the A/B pairwise preference, proving that the
translator can create a causal state channel without base adaptation.

The prior final-layer cross-attention baseline remains unchanged in
`artifacts/phase-b-reader-variants.json`: 25% held-out candidate accuracy, RP
loss 6.4748, KL 0.6575, 2,625,537 parameters, and 34.17 ms latency on its earlier
four-example fixture. It did not record unrestricted generated-token accuracy or
active allocation, so those comparison cells remain null rather than being
retroactively inferred. Translator latency above uses the new matched 20-example
fixture; its P-disabled measurement is the appropriate within-fixture reference.

## Rejection boundary

The completion gate fails despite decreasing losses and a real causal effect:

- local hard-negative retrieval is strong, but global retrieval with many active
  facts is only 65-70%;
- the held-out causal prompt reduces global retrieval further to 25-40%;
- wrong-entity state still activates strongly and produces a large incorrect
  target-logit lift, even though historical state is rejected;
- unseen-name generation is 0-25%; and
- the higher-layer placements improve exact generation only by sacrificing RP
  preservation.

No attachment is selected. Upper four layers are retained only as the strongest
causal diagnostic, not as an adopted adapter. The current bidirectional-MLP
translator form is explicitly rejected as the Phase B solution. The three
saved `.translate` files in `artifacts/` are reproducibility artifacts for this
failed experiment, not production-compatible adapters.

Full results, including every causal intervention, are in
`artifacts/phase-b-translator.json`. Phase C remains blocked.
