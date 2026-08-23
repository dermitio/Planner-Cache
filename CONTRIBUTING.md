# Contributing

Planner Cache contributions should preserve the separation between canonical state and model-specific compatibility.

## Before changing code

1. Read [ARCHITECTURE.md](ARCHITECTURE.md), [LIMITATIONS.md](LIMITATIONS.md), and the active decisions in [`Sources/Updates`](evidence/decisions/).
2. Check [`Archive/README.md`](evidence/history/README.md) before reviving an older reader, bridge, translator, or training path.
3. Treat historical artifacts as immutable evidence.

## Architectural invariants

- Keep P-cache separate from prompt tokens and self-attention KV.
- Keep canonical P model-independent.
- Do not store token IDs or model-native hidden vectors in canonical files.
- Keep the universal router independent of model hidden width.
- Put model layer names, hidden width, and hook mechanics inside a model adapter.
- Keep `.ppkg` disk resident and hydrate only selected entries.
- Keep evidence authority and contradiction history inspectable.
- Do not add LoRA or base-weight training as a silent fallback.
- Leave archive, tool retrieval, and recent-KV policy to the surrounding runtime.

## Tests

Run the active suite.

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
git diff --check
```

Changes to serialization require deterministic round-trip, corruption, checksum, and incompatible-protocol tests. Changes to model adapters require frozen-base gradient isolation, P-disabled identity, invalidation identity, wrong-state rejection, and held-out causal generation tests.

Performance changes should report before and after wall time, CPU time, RAM, VRAM where applicable, bytes read, bytes copied to CUDA, and configured capacity.

## Benchmark evidence

Every public number must map to a benchmark script, test, raw JSON artifact, model configuration, and seed where available. Do not overwrite rejected baseline artifacts. Create a new artifact and explain the relationship.

Publication tables and figures should be regenerated with [`Publishing/assets/generate_assets.py`](assets/generate_assets.py).

## Documentation style

Use the public terms Planner Cache, P-cache, P-package, canonical router, `.router`, Tensor Translation Layer, `.ttl`, Lexical Translation Layer, `.ltl`, and canonical P state. Retain `.translate` only when citing historical work whose provenance depends on it.

Describe results narrowly. Prefer “restored frozen-base logits in the matched wrong-state benchmark” over “cannot affect irrelevant generations.”

## Pull request checklist

- Active architecture remains independent of `Archive`.
- No model-specific state entered canonical serialization.
- Tests and benchmark artifacts are included.
- Claims cite raw evidence.
- New limitations are documented.
- `git diff --check` passes.
