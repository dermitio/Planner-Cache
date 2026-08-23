# Gemma4 causal geometry experiment

Status: **rejected as a general-purpose `.translate`**.

The frozen Q8 GGUF is lexically observable and its first-token output geometry
is directly causal at layer 41. The compact canonical-byte translator does not
generalize sufficiently, and a static residual cannot emit multi-token values.

## Key results

| Measurement | Result |
|---|---:|
| GGUF tensors inventoried | 720 |
| Best held-out canonical layer probe | 85.0% at layer 1 |
| Exact output-row first-token control | 100.0% on 256 values |
| Exact output-row full strings | 10.9% on 256 values |
| Best compact held-out first token | 62.7% on 67 values |
| Best compact held-out full strings | 0.0% on 67 values |
| Inactive maximum logit difference | 0.0 |

## Boundary

GGUF observability and first-token causal control pass. Context transfer of exact
controls also passes. Multi-token readout and compact held-out translation fail.
The 1,000-value final gate was not run after the compact adapter produced zero
held-out full strings. The bounded Alice/Bob adapter remains the active baseline.
