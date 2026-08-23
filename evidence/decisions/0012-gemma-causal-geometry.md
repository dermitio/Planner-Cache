# Gemma4 GGUF causal geometry pass

Date: 2026-08-22

Status: rejected as a general-purpose `.translate`

This pass inspected the frozen Gemma4 E4B Q8 GGUF and replaced the previous
assumption about approximate input-embedding targets with direct evidence from
the actual llama.cpp graph. It did not change P-cache, the canonical router,
P-package, canonical P, or the base model. The bounded Alice and Bob Gemma
adapter remains the selected Gemma baseline and was not modified.

## Frozen runtime identity

- Model: `Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf`
- Model SHA-256: `a4c4177f9fd7e3f56522675afb742f079a53f9226195b7db5e9888c872f053da`
- Model size: 8,133,226,464 bytes
- llama.cpp build: 10276
- llama.cpp commit: `6ea215d17`
- Architecture: Gemma4 E4B
- Layers: 42
- Residual width: 2,560
- Vocabulary: 262,144

## GGUF inventory

The inventory contains 720 tensors. The GGUF omits `output.weight`. The
llama.cpp Gemma4 loader explicitly duplicates `token_embd.weight` as the model
output, which verifies tied input and output weights for this artifact.

Gemma4 also contains `per_layer_token_embd.weight` with a 10,752 by 262,144
logical shape, a shared model projection, and a 256-wide token-derived input for
every block. Each block gates and projects that token-derived state into the
2,560-wide residual stream before the public control-vector attachment point.
The failed vocabulary translator did not model this per-layer lexical path.

Evidence:

- `artifacts/gemma4-gguf-layer-inventory.json`
- `artifacts/gemma4-gguf-layer-inventory.md`

## Layer lexical probe

The unmodified llama.cpp evaluation callback captured all 42 `l_out` tensors
and the final normalized LM-head input for 320 complete strings. The split used
240 training strings and 80 held-out strings across common words, names,
locations, objects, statuses, numbers, punctuation, UTF-8, synthetic names, and
multi-token values. Complete-string overlap was zero.

The best held-out canonical linear probe was 85.0% at layer 1. Output-geometry
retrieval peaked at 71.25% at layers 1 and 34. Layers 30 through 41 retained
roughly 77.5% to 83.75% canonical linear-probe accuracy. Lexical identity is
therefore observable throughout the actual quantized graph.

Evidence:

- `artifacts/gemma4-layer-lexical-probe.json`
- `artifacts/gemma4-layer-lexical-probe.split.json`
- `artifacts/gemma4-layer-lexical-probe.f32`

## Causal layer location

The corrected sweep used token identifiers produced by the actual llama.cpp
tokenizer. This matters because the earlier GGUF teacher's fallback tokenizer
did not recover exact first tokens for some unseen strings.

Exact tied-output rows were tested at layers 1, 16, 24, 32, 38, and 41. At
strength 8, layer 41 reached 12 of 12 first tokens. Layers 1, 16, and 24 reached
none. Layers 32 and 38 reached one of 12. Layer 38 required strength 16 to reach
six of 12 and incurred mean KL 5.156.

Four-step residual-only SPSA optimization was also run at every sampled layer.
Layer 41 reached six of 12 first tokens at strength 4. Every earlier sampled
layer remained between zero and one of 12. This independently localized causal
lexical control to the final block for the public control-vector boundary.

Evidence:

- `artifacts/gemma4-causal-output-geometry-sweep.json`
- `artifacts/gemma4-causal-optimized-layer-sweep.json`
- `artifacts/gemma4-causal-optimized-layer-directions/`

## Causal teacher results

Exact output-row directions at layer 41 and strength 8 reached 256 of 256
correct first tokens. Mean target-logit lift was 18.829 and mean KL from the
base was 10.517. Exact full-string generation was 28 of 256. Those 28 were all
single-token values. Every value requiring two, three, four, five, or eight
tokens failed full-string generation because a static control repeatedly
favored the first token piece.

A separate four-step black-box optimized teacher reached 194 of 256 first
tokens and 13 of 256 full strings with mean KL 5.062. Only the 194 successful
first-token residuals were marked positive in that teacher package.

The exact output-row direction generalized across five neutral contexts. Each
context reached 12 of 12 first tokens and seven of 12 full strings. The same
seven complete strings were single-token values. The residual is not tied to
one magic prompt for first-token control, but it remains unsuitable for static
multi-token readout.

Evidence:

- `artifacts/gemma4-causal-output-geometry-256.json`
- `artifacts/gemma4-causal-output-teacher-layer41.safetensors`
- `artifacts/gemma4-causal-full-string-teacher-layer41.safetensors`
- `artifacts/gemma4-causal-residual-sweep-layer41.json`
- `artifacts/gemma4-causal-residual-teacher-layer41.safetensors`
- `artifacts/gemma4-causal-context-invariance/`

The first-token and full-string success masks are serialized separately. The
first-token teacher has 256 positive controls. The full-string teacher has only
28 positives, all with a one-token target. Failed multi-token controls are not
positive examples in the full-string teacher.

## Compact translator result

A fresh canonical UTF-8 byte encoder was trained against the validated exact
output-row causal targets. Learned residual basis ranks 16, 32, 64, and 128 were
matched on the same strict complete-string split.

Held-out residual retrieval improved from 34.33% at rank 16 to 49.25% at rank
64. Rank 128 reached 47.76%. The best held-out causal result was rank 128 with
42 of 67 correct first tokens and zero of 67 exact full strings. All eight
held-out single-token values failed. The apparent success on longer values was
mostly shared first-piece prediction and did not produce the complete value.

The rank-128 candidate contains 518,657 parameters and is 2,077,372 bytes.
Mean held-out KL was 7.223. It is a rejected research artifact and is not loaded
by the active Gemma runtime.

Evidence:

- `artifacts/gemma4-causal-output.training.json`
- `artifacts/gemma4-causal-output.split.json`
- `artifacts/gemma4-causal-output-rank16.translate`
- `artifacts/gemma4-causal-output-rank32.translate`
- `artifacts/gemma4-causal-output-rank64.translate`
- `artifacts/gemma4-causal-output-rank128.translate`
- `artifacts/gemma4-causal-output-rank128-heldout.json`

## Target comparison

The GGUF input and output matrices are physically tied. The important
difference was not a separate LM-head tensor. It was exact runtime tokenization,
exact row recovery, attachment depth, and sufficient control magnitude.

The rejected approximate byte-to-input-row mapper reached 45.56% retrieval on
a held-out vocabulary probe and zero causal generations on its balanced test.
Exact tied-output rows reached 100% first-token control on 256 values. A compact
byte mapper trained on those proven causal rows reached only 62.69% held-out
first-token generation and zero full strings. The remaining boundary is compact
open-vocabulary translation and sequence-aware readout rather than GGUF
observability or first-token causal controllability.

## Inactive paths

The no-control path was checked on 256 repeated evaluations. Maximum KL and
maximum full-vocabulary logit difference were both zero. Wrong-entity,
wrong-relation, historical, invalidated, router-disabled, and
translator-disabled states continue to use this unchanged inactive path. No
state text or extra prompt tokens were introduced by the probe or adapter.

The active control buffer remains 419,840 bytes for 41 attachable layers by
2,560 float32 values. The adapter consumes zero VRAM while inactive. The
rank-128 float32 parameters would occupy 2,074,628 bytes if kept on CUDA. The
parameters plus the control buffer total 2,494,468 bytes. The measured research
runtime used 2,464 MiB of model VRAM with 12 GPU layers. Base weights remained
frozen and were never placed in an optimizer.

## Decision

The experiment does not pass the general-purpose completion gate. The 1,000
held-out final gate and relevant-RP evaluation were not run after every compact
candidate produced zero held-out full strings. Irrelevant and rejected routes
remain exactly inert.

The limiting boundaries are:

1. multi-token readout from a single static residual
2. compact canonical-byte to exact causal-row generalization
3. high KL at strengths that reliably force exact first tokens

GGUF observability, lexical representation, first-token causal residual
controllability, context transfer, and the actual quantized llama.cpp runtime
are not the limiting boundaries in this pass.

The failed `gemma4-e4b-q8-lexical.*` artifacts and the bounded
`gemma4-e4b-q8-llama.translate` baseline remain unchanged. No LoRA, base-model
training, or Phase C work was introduced.
