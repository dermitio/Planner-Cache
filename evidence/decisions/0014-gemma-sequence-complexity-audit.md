# Gemma sequence `.translate` complexity audit

Date: 2026-08-22

Status: audit complete. Prefer direct adaptive logit bias for exact selected
string emission. Preserve the sequence residual as the internal-intervention
baseline. Neither is accepted as learned semantic translation.

This audit did not modify P-cache, the universal router, P-package, canonical P,
the frozen Gemma GGUF, or any previous adapter artifact.

## Question

The sequence-aware prototype reached 125 of 128 exact strings by tokenizing a
canonical value and applying one exact output-row control per target token. The
audit tested whether this was mostly brute-force token forcing rather than a
learned semantic translation interface.

All variants used the same frozen GGUF, prompt, 128 complete held-out values,
and exact tokenizer bundle. Scalar and low-rank calibration used 13 disjoint
complete values. Complete-value overlap was zero.

## Matched results

| Variant | First token | Full string | Active KL | Controlled decode ms | Parameters | Training | Inactive |
|---|---:|---:|---:|---:|---:|---|---:|
| tokenizer plus direct adaptive logit bias | 100% | 100% | 3.2561 | 1,284.71 | 0 | none | exact |
| tokenizer plus static mean output-row residual | 18.75% | 0% | 4.0858 | 323.98 | 0 | none | exact |
| tokenizer plus calibrated shared scalar per token | 100% | 82.03125% | 5.6896 | 1,123.48 | 1 | three-point scalar grid | exact |
| tokenizer plus rank-16 residual projection | 25% | 0% | 5.8226 | 363.06 | 43,520 | deterministic SVD | exact |
| current sequence-aware `.translate` | 100% | 97.65625% | 6.6101 | 1,205.70 | 1 | prior layer and scalar calibration | exact |

The latency column measures controlled model decoding only. Direct bias was
6.55 percent slower than the current residual in this run. This is not a
latency improvement. Instrumented wall time is not used for the design decision
because failing variants terminate sequences early.

## Scalar calibration

The disjoint 13-value calibration results were:

| Strength | First token | Full string | KL |
|---:|---:|---:|---:|
| 4 | 30.7692% | 15.3846% | 2.1020 |
| 6 | 100% | 100% | 8.0897 |
| 8 | 100% | 100% | 10.8833 |

Strength 6 was selected by the calibration threshold but fell to 82.03125
percent exact strings on held-out values. It is not an acceptable replacement
for strength 8.

## Attribution

The measured open-value behavior comes from:

1. the exact Gemma tokenizer turning the selected canonical value into target
   token IDs
2. the frozen model exposing its output geometry or logits for those IDs
3. a per-token schedule that explicitly targets token N and advances only after
   token N is emitted

The adapter does not learn a semantic mapping from canonical values. Its only
serialized parameter is a force magnitude. Static and low-rank residual results
show that output geometry without exact per-token scheduling is insufficient.
Direct logit bias shows that output rows are not required for exact lexical
emission once target IDs are known.

The current sequence-aware path is therefore classified as lexical token
forcing through the residual stream. It is not evidence of learned semantic
translation.

## Decision

For an exact selected-string emission contract, prefer direct adaptive target
logit bias. It reached higher full-string accuracy, reduced active KL by 50.74
percent, removed output-row hydration, required zero fitted parameters, and
required no training. Inactive behavior remained exactly base-equivalent.

Direct bias is an explicit lexical output constraint. It must not be presented
as semantic model-state translation or as evidence that Gemma has integrated P
state into its internal reasoning. The universal router is still responsible
for choosing whether a canonical state is relevant. Rejection applies no bias.

The current sequence-aware `.translate` and its artifacts remain unchanged as
the model-residual baseline. It remains useful only when the experiment
specifically requires internal residual intervention.

Evidence:

- `artifacts/gemma4-sequence-complexity-audit.json`
- `artifacts/gemma4-sequence-complexity-audit.md`
- `benchmarks/audit_sequence_translate_complexity.py`
- `benchmarks/llama_cpp_sequence_runner.cpp`

The exact inactive path, absence of source-state prompt tokens, frozen GGUF,
and deterministic held-out manifest are recorded in the audit JSON.
