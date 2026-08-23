# Gemma sequence `.translate` complexity audit

Status: the current sequence residual is classified as lexical token forcing,
not learned semantic translation.

All variants use the same 128 complete held-out values and the same frozen
Gemma4 E4B Q8 GGUF. Calibration contains 13 disjoint values. No source-state or
extra prompt tokens are present.

| Variant | First token | Full string | Active KL | Controlled decode ms | Parameters | Inactive |
|---|---:|---:|---:|---:|---:|---:|
| tokenizer + direct output/logit bias | 100.00% | 100.00% | 3.2561 | 1284.71 | 0 | exact |
| tokenizer + static output-row residual | 18.75% | 0.00% | 4.0858 | 323.98 | 0 | exact |
| tokenizer + calibrated scalar per token | 100.00% | 82.03% | 5.6896 | 1123.48 | 1 | exact |
| tokenizer + rank-16 residual projection | 25.00% | 0.00% | 5.8226 | 363.06 | 43,520 | exact |
| current sequence-aware .translate | 100.00% | 97.66% | 6.6101 | 1205.70 | 1 | exact |

## Interpretation

Direct adaptive target-logit bias achieved 100.00%
full strings with no fitted parameters and no training. Its KL was
3.2561, a 50.74% reduction from the current
residual path. Controlled decoding was 6.55% slower in this run,
which is not a latency improvement. Both paths decode the same full target
sequences and model decoding dominates their runtime.

The static mean output-row residual and rank-16 projection both produced zero
complete strings. This shows that per-token scheduling is essential. The
calibrated shared scalar selected strength 6 on the small calibration set but
fell to 82.03%
on held-out values. The strength-8 current baseline therefore remains the
better residual configuration.

The current adapter does not learn canonical semantics. It obtains the exact
target token IDs from the Gemma tokenizer, hydrates the corresponding output
rows from the frozen GGUF, and applies them sequentially. Its one scalar only
calibrates force magnitude. Held-out complete-value performance comes from the
tokenizer and frozen output geometry rather than learned translation.

## Decision

For tasks whose contract is to emit an exact selected canonical string, prefer
the direct adaptive logit-bias actuator. It is simpler, needs no training or
output-row hydration, reaches higher measured accuracy, halves active KL, and
retains exact inactive behavior.

Do not present direct logit bias as semantic translation. It is an explicit
lexical output constraint. Retain the current sequence residual unchanged as
the model-residual baseline for research that specifically requires internal
causal intervention. Neither result establishes a general semantic Gemma
`.translate`.

Inactive router rejection remains exact for every variant. No changes were made
to P-cache, the universal router, P-package, canonical P, or base weights.
