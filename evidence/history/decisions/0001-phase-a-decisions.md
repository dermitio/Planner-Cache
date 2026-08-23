# Phase A implementation decisions

Date: 2026-08-17

Status: Superseded on 2026-08-22 by `0002-pythia-base-pivot.md`. Retained as
historical evidence; custom base-model training is no longer project direction.

## Authorized proof-model scale fallback

The initial 15–20M target remains the smallest candidate and must be evaluated
first. If its fixed held-out and multi-prompt evaluation cannot demonstrate
coherent casual chat/RP, the base parameter budget may increase to roughly
100M. A roughly 250M base is authorized only as the worst-case fallback if the
100M candidate also fails. Scaling the base does not authorize replacing or
simplifying P, the adaptive extension, separate tool-KV, archive search, or
cold-resume reconciliation.

Evaluation status on 2026-08-17:

- The 15.7M candidate failed fixed-prompt coherence after 166M training tokens.
- The 98.7M candidate completed 165,974,016 training tokens and reached held-out
  loss 1.674 / perplexity 5.336, but still drifted or repeated on the fixed
  multi-prompt gate. Phase B therefore remains blocked and the authorized 250M
  fallback is active.
- The 248.1M preset requires explicitly selected BF16 parameters and optimizer
  moments on the available 4GB GPU. FP32 remains the default. Capacity probing
  selected batch 4 x 512 (~2.98 GiB); batch 6 exceeded available memory.
- The 250M run uses four gradient-accumulation microbatches, giving an
  effective batch of 16 sequences without increasing peak allocation. Trainer
  checkpoints record microbatch, accumulation, precision, and scale settings.
- Compiled steady-state probing reached roughly 4.6k-7.0k tokens/sec below
  2.75 GiB allocated. The two-pass target is 20,260 optimizer steps / 165,969,920
  sampled tokens, with peak LR 2e-4 decaying to 2e-5.
- The original `reduce-overhead` compile mode was rejected because it enables
  CUDAGraphs and replay-owned gradient buffers were overwritten between
  accumulation microbatches. Compilation now uses Inductor with
  `triton.cudagraphs=False`. On 2026-08-18 the exact 248.1M BF16 regression
  passed: two optimizer steps with four backward passes each, full
  safetensors/optimizer/RNG checkpoint, fresh model/optimizer reconstruction and
  resume, then two further four-backward optimizer steps. The 250M training
  configuration is ready, but the long training run was not started.

The authoritative specification leaves the following Phase A details open.
These choices preserve its required behavior:

- The tokenizer is a deterministic custom byte-level BPE. UTF-8 bytes 0–255
  are permanent base symbols, so arbitrary input round-trips without an
  unknown token. Training uses weighted Unicode-aware pre-token pieces to keep
  merge training practical, but encoding remains lossless at byte level.
- Canonical corpus extraction is streaming and emits provenance, a SHA-256
  content hash, and normalized conversational role markers. Source datasets
  remain immutable. This stage does not claim to replace the crawler or helper
  model sanitizer described in the specification; it consumes the already
  normalized/filtered local datasets listed in `Licences.md`.
- Packed shards use a contiguous little-endian integer stream with EOS between
  documents. `uint16` is used while the vocabulary is below 65,536, as proposed
  by the specification.
- The canonical documents are split reproducibly by content hash before
  tokenizer fitting and packing. This avoids source-order bias and keeps the
  validation set independent of pretraining sampling.
- The default base is a decoder-only transformer with tied input/output
  embeddings, learned positions, 7 layers, width 384, 6 heads, and feed-forward
  width 1536. With the target 8,192-token vocabulary it is within the specified
  15–20M parameter range. Planner inputs are deliberately absent in Phase A.
- Checkpoints separate immutable inference weights (`safetensors`) from
  resumable trainer state (optimizer, CPU/CUDA RNG, step, and configuration).
- Base pretraining supports linear warmup followed by cosine decay to a
  configurable floor. The schedule is derived from absolute checkpoint step,
  so resumed runs do not restart warmup or silently jump the learning rate.
- Coherence evaluation uses a fixed sampling policy: PAD is suppressed, EOS
  ends generation, temperature is 0.8, and top-k is 50. Raw untruncated
  sampling remains available with top-k zero for comparison.

Phase A completion gate: tokenizer lossless/deterministic tests, packed-shard
boundary and dtype tests, causal-model and parameter-budget tests, exact
checkpoint-resume tests, plus recorded tokenizer and training throughput
benchmarks. Coherent chat/RP quality still requires an actual training run and
evaluation; passing unit tests alone does not complete Phase A.
