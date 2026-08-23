# Gemma4 sequence-aware `.translate` prototype

Date: 2026-08-22

Status: sequence prototype passes controlled and 128-value scale tests. It is
not yet classified as a general-purpose Gemma translator.

This experiment leaves P-cache, the universal router, P-package, canonical P,
the frozen GGUF, and the bounded Alice and Bob adapter unchanged. It replaces
the rejected static open-vocabulary research assumption with a model-specific
token sequence controller.

## Compatibility gate

The exact workspace metadata and tokenizer bundle passed before causal work
started. Bundle and GGUF agree on Gemma4 architecture, 42 layers, hidden width
2,560, vocabulary size 262,144, and special IDs. Every one of the 262,144 token
strings has the same ID. Fourteen deterministic probes matched llama.cpp on two
runs each.

The supplied canonical chat template is newer than the template embedded when
the GGUF was converted. The difference is recorded. It does not alter the exact
vocabulary, tokenizer behavior, special IDs, or model geometry used by this
prototype.

Evidence:

- `artifacts/gemma4-tokenizer-compatibility.json`
- `artifacts/gemma4-tokenizer-compatibility.md`

## Prototype

The model-specific path is:

```text
canonical UTF-8 value
        ↓
verified Gemma tokenizer
        ↓
exact target token sequence
        ↓
normalized tied-output row for token N
        ↓
layer-41 llama.cpp control at strength 8
        ↓
advance only after token N is emitted
        ↓
disable on mismatch or sequence completion
```

This uses the previous causal-geometry finding that exact tied-output rows are
reliable controls at layer 41. It does not use approximate input embeddings.
It does not learn or store a vocabulary table. The `.translate` file stores one
control-strength scalar plus compatibility metadata. Exact rows are hydrated
lazily from the compatible frozen GGUF.

Canonical P still contains only canonical UTF-8 state. Token IDs and residuals
exist only inside the model-specific translation call. No source-state tokens
are added to the prompt.

## Controlled result

The first run used 13 values spanning names, unseen nonsense names, locations,
objects, statuses, numbers, multi-word values, Turkish, CJK, and Arabic. Target
lengths ranged from one to six tokens.

At layer 41 and strength 8:

- first-token accuracy was 13 of 13
- exact full-string accuracy was 13 of 13
- multi-token full-string accuracy was 100 percent
- mean active KL from frozen base was 10.8833
- mean applied residual norm was 8.0
- mean per-value latency was 1,480.35 ms
- control disabled after every complete sequence
- matched post-sequence KL was exactly zero
- matched post-sequence maximum full-vocabulary logit difference was zero

The last two checks decode the completed value into both matched contexts after
control shutdown. They show that the public control path does not leave hidden
control contamination after the sequence. The emitted text remains ordinary
recent context.

## 128-value scale result

The disjoint scale set contains 16 values in each of eight categories. No
complete evaluation string was fitted or stored in the adapter.

- first-token accuracy was 128 of 128
- exact full-string accuracy was 125 of 128
- multi-token full-string accuracy was 97.65625 percent
- mean active KL was 6.6101
- mean applied residual norm was 8.0
- mean per-value latency was 2,866.55 ms

Unseen names, synthetic strings, statuses, objects, UTF-8 values, and the
multi-word object category each reached 16 of 16 exact strings. The failures
were `river workshop 208`, `131679`, and `139598`.

A strength-12 diagnostic fixed all three failures. It is retained as an
ablation and is not selected globally because it was evaluated only on the
observed failure subset.

## Inactive behavior

Wrong entity, wrong relation, historical, invalidated,
router-disabled, translator-disabled, and irrelevant-RP conditions construct no
translation. The llama.cpp control vector remains disabled. The controlled
inactive run measured zero KL and zero maximum full-vocabulary logit difference
from the frozen base.

## Serialization

`pcm-sequence-causal-translate-v1` records model and runtime identity, GGUF
checksum, exact tokenizer-bundle checksum, canonical P protocol, layer,
strength, vocabulary and special IDs, and the tied-output sequence policy. Save
output is deterministic and checksummed. Compatibility rejects a different
model checksum, architecture, width, layer count, or tokenizer bundle.

The adapter contains no base weights, vocabulary table, P state, conversation
state, prompt tokens, optimizer state, or cached controls.

## Decision

Sequence-aware control resolves the measured 100-percent-first-token and
zero-percent-multi-token boundary of the static residual. It is the current
open-vocabulary Gemma research prototype.

It is not promoted to general-purpose status because the 1,000-value final gate
and broad natural-RP evaluation during relevant control were not run. Active KL
also remains high. The bounded Alice and Bob runtime adapter remains immutable
and selected for the existing interactive compatibility path.

Evidence:

- `artifacts/gemma4-e4b-q8-sequence-causal.translate`
- `artifacts/gemma4-sequence-causal-controlled.json`
- `artifacts/gemma4-sequence-heldout-128.manifest.json`
- `artifacts/gemma4-sequence-causal-heldout-128.json`
- `artifacts/gemma4-sequence-causal-failures-strength12.json`
- `artifacts/gemma4-sequence-causal-summary.json`
- `artifacts/gemma4-sequence-causal-summary.md`

The rejected static lexical and causal artifacts remain unchanged as historical
negative evidence.
