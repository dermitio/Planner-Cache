# TTL and LTL compatibility boundary

Date: 2026-08-22

Status: active terminology and file-format migration

## Decision

The former `.translate` umbrella is split into two non-equivalent compatibility classes.

```text
Native P support
-> model or runtime consumes canonical P directly

TTL support
-> Tensor Translation Layer
-> semantic or internal model compatibility

LTL support
-> Lexical Translation Layer
-> lexical or output compatibility
```

The universal router remains upstream and does not depend on the selected compatibility class. P-cache, canonical P, `.router`, `.ppkg`, and the frozen base models are unchanged.

## Tensor Translation Layer

The existing Pythia split semantic adapter is now the reference TTL. Its learned tensors and attachment behavior were not retrained or changed. The weights were migrated from `pythia-1.4b-split-final_layer.translate` into `pythia-1.4b-final-layer.ttl`.

The active format is `planner-cache-ttl-v1`. It stores explicit adapter class and semantic support metadata, canonical protocol identity, model identity, hidden width, attachment layers, architecture configuration, tensors, and a tensor checksum. The serialized file excludes base weights, P contents, conversation state, prompt tokens, attention KV, and optimizer state.

Legacy `pcm-split-translate-v1` files remain loadable through `TensorTranslationLayer.load`. Loading emits a deprecation warning and explicitly classifies the artifact as TTL. Other historical `.translate` formats are not silently promoted.

## Lexical Translation Layer

The Gemma sequence complexity audit established that direct adaptive logit bias was the simplest exact selected-string mechanism. It reached 128 of 128 exact strings with active-path KL 3.2561 and zero learned parameters on the recorded 128-value workload. This path is now the reference Gemma LTL.

The active format is `planner-cache-ltl-v1`. It is a deterministic checksummed JSON envelope containing model and runtime identity, GGUF checksum, tokenizer-bundle checksum, canonical protocol, lexical control strategy, and zero parameter count. It excludes model weights, P contents, conversation state, vocabulary tables, and optimizer state.

An LTL receives a canonical value only after router acceptance. Rejected entity, relation, historical, invalidated, router-disabled, and LTL-disabled paths produce no lexical target. LTL support proves lexical or output compatibility and does not prove internal semantic reasoning over P.

## Active artifacts

| Artifact | Class | Support | SHA-256 |
|---|---|---|---|
| `artifacts/pythia-1.4b-final-layer.ttl` | TTL | semantic or internal | `72ef68d07ee27c37b90432d34d4be5c2c280ae1bcb08236e37a0e458c054d8d7` |
| `artifacts/gemma4-e4b-q8-llama.ltl` | LTL | lexical or output | `7eee07ed813796dd0d25b04b48aec73507b934479ad8902e1b657653c040781a` |

The Gemma LTL binds to GGUF SHA-256 `a4c4177f9fd7e3f56522675afb742f079a53f9226195b7db5e9888c872f053da` and tokenizer bundle SHA-256 `b3033e12af0ed503d8b80390c79d02d6bd9bc372e93e377cc1dd6514b7cd21d6`.

## Historical preservation

`.translate` remains in archived file names, benchmark names, experiment reports, and updates that predate this decision. It should be read as the former umbrella term. Bounded Gemma control-vector artifacts, the failed open-vocabulary adapter, causal residual bases, and the sequence-aware adapter remain research evidence. None is silently relabeled as an active Gemma TTL.

| Legacy artifact group | Modern classification |
|---|---|
| `pythia-1.4b-split-final_layer.translate` | deprecated semantic container, explicitly loadable as TTL |
| `gemma4-e4b-q8-llama.translate` | bounded residual research baseline |
| `gemma4-e4b-q8-lexical*.translate` | failed lexical-mapper research |
| `gemma4-causal-*-rank*.translate` | causal-geometry research |
| `gemma4-e4b-q8-sequence-causal.translate` | lexical-forcing research baseline |

## Runtime and logging

Pythia interactive sessions record `compatibility_layer = ttl` and emit `TTL_ENABLE`, `TTL_DISABLE`, and `TTL_OUTPUT`. Gemma sessions record `compatibility_layer = ltl` and emit `LTL_ENABLE`, `LTL_DISABLE`, `LTL_TOKEN_TARGET`, and `LTL_COMPLETE` when those events occur. The existing state, router, P-package, transcript, session, and model-generation event streams remain intact.

## Migration limits

This change is a terminology, interface, container, runtime-dispatch, and documentation migration. It does not introduce a new model architecture, train a new adapter, alter the universal router, or reinterpret historical results. Gemma remains LTL-only until a separate internal semantic proof passes the TTL gate.
