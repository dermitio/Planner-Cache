# Publication sanitization report

Sanitization date: 2026-08-23

## Scope

The GitHub, Hugging Face, and Research packs were rebuilt from the publication
source set. This pass changed packaging and explanation only. It did not change
the architecture, run new benchmarks, or alter recorded benchmark values.

## Removed

- Nested `.git` repositories, including local commit identity and email data
- Cache directories and temporary build output
- Third-party model weights, GGUF files, tokenizer and configuration bundles,
  datasets, llama.cpp files, and ordinary checkpoint formats
- Random temporary-directory identifiers from publication copies of benchmark JSON
- Concrete interactive session IDs from publication copies of acceptance evidence
- Workstation-specific repository, model, and runtime paths
- Machine-specific launcher defaults for local model, tokenizer, and llama.cpp paths

## Replaced

- Repository paths became `${REPOSITORY_ROOT}` where provenance required a path
- Model paths became `${GEMMA_MODEL}` or neutral `/path/to/model.gguf` examples
- Runtime paths became `${LLAMA_CPP_DIR}` or `/path/to/llama.cpp`
- Temporary run directories became `${TEMP_DIR}/planner-cache-run`
- Concrete session IDs became `benchmark-session-gemma` or
  `benchmark-session-pythia`
- Any detected email address in generated pack text becomes `user@example.com`

Path and identifier normalization changes only non-numerical provenance fields in
the publication copies. The authoritative repository benchmark artifacts remain
unchanged. Pack-specific checksums are regenerated after normalization.

## Readability changes

The main README, benchmark guide, Hugging Face cards, research abstract, paper,
and VRAM guide now state the practical result and its boundary before detailed
tables. P-cache, canonical P, router, TTL, LTL, P-package, retained KV, active
and inactive paths, causal intervention, KL divergence, and incremental VRAM are
defined in plain English at first use in each primary publication entry point.

## Intentionally retained technical metadata

The following fields are useful for reproduction and are not treated as personal
identifiers:

- Model family and architecture identifiers
- Model, adapter, router, and evidence checksums
- llama.cpp build number and commit identifier
- GPU model, VRAM capacity, driver, CUDA, PyTorch, Transformers, Python, kernel,
  and platform versions
- Benchmark names, seeds, layer numbers, dimensions, token counts, timestamps,
  durations, and measured values
- Synthetic test entities, names, state values, and controlled role-play examples

No hostname, account name, private email, personal conversation, or original
session identifier is required for reproduction.

## Validation

The release validator checks every pack for nested repository metadata, cache
directories, private email addresses, personal machine identifiers, concrete
session IDs, random temporary paths, absolute home or removable-media paths,
forbidden third-party payloads, broken links, malformed JSON, syntax errors, and
manifest mismatch.

Validation commands:

```bash
.venv/bin/python Publishing/build_release_packs.py
.venv/bin/python Publishing/validate_release.py
PYTHONPATH=src python Publishing/assets/generate_assets.py
git diff --check
```

All publication sanitization, payload, manifest, link, JSON, syntax, asset, and
diff checks passed in the final run.
