# Planner Cache historical archive

This directory preserves superseded implementation eras and their provenance. Nothing
under `Archive/` is part of the active Python package, test collection, or benchmark
entry points.

## Groups

- `pre-pythia/` contains the abandoned self-trained transformer, tokenizer, corpus
  pipeline, trainer, CUDA accumulation regression, and Phase A scale benchmarks. The
  Pythia-1.4B pivot superseded this work. Its training tests and metrics remain useful
  only as historical engineering evidence.
- `old-training/` contains the self-trained model checkpoints, prepared corpora,
  tokenizer, logs, and evaluation artifacts. These are retained for provenance and
  are not inputs to the active Planner Cache runtime.
- `rejected-readers/` contains controller, cross-attention/reader, bridge, and learned
  synthetic experiments rejected before the split router/translator architecture.
  Their benchmark JSON files remain immutable comparison baselines.
- `rejected-translators/` contains the monolithic bidirectional translator experiment
  and its model adapters, tests, and results. It was superseded by the explicit
  canonical query projector/router plus value translator in Update 0005. Its reported
  retrieval, causality, and coherence results remain an immutable baseline.
- `deprecated-benchmarks/` contains non-selected split-translator layer-sweep packages
  and their paired routers. The final-layer package is the active selected adapter;
  these files remain useful for reproducing the attachment-layer comparison.
- `deprecated-model-files/` contains a redundant PyTorch-format copy of the active
  Pythia weights and an ad-hoc generation script. The active loader uses the equivalent
  `model.safetensors`; neither archived file is required at runtime.
- `historical-docs/` preserves architectural decisions for superseded eras without
  rewriting their original terminology. Active decisions begin with Update 0005.

## Active successors

- `Sources/Updates/0005-split-router-translator.md`: active canonical router and
  model-specific `.translate` boundary.
- `Sources/Updates/0006-personality-package.md`: active disk-resident `.ppkg` design.
- `Sources/Updates/0007-ppkg-integrity-boundaries.md`: active package integrity and
  hot-path checksum policy.

Archived modules deliberately lack a package-level `__init__.py`; importing them as
active `pcm` modules is unsupported.
