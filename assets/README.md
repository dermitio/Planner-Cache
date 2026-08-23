# Shared publication assets

These assets are shared by the GitHub, Hugging Face, and research publication packs.

| Asset | Purpose | Source |
|---|---|---|
| [`architecture.svg`](architecture.svg) | Repository-friendly current architecture diagram | Publication architecture specification in `generate_assets.py` |
| [`architecture.pdf`](architecture.pdf) | Vector publication architecture figure | Generated from `architecture.svg` |
| [`architecture.mmd`](architecture.mmd) | Compact Mermaid architecture diagram | Current architecture boundaries |
| [`router_scaling.svg`](router_scaling.svg) | Post-audit router accuracy and MRR plot | `active-system-audit.json` |
| [`router_scaling.csv`](router_scaling.csv) | Router plot data | `active-system-audit.json` |
| [`ppkg_lookup.svg`](ppkg_lookup.svg) | P-package lookup latency plot | `active-system-audit.json` |
| [`ppkg_scaling.csv`](ppkg_scaling.csv) | P-package growth and lookup table | `active-system-audit.json` |
| [`causal_conditions.csv`](causal_conditions.csv) | Matched CUDA causal attribution table | `active-system-cuda-attribution.json` |
| [`gemma_causal_conditions.csv`](gemma_causal_conditions.csv) | Gemma 4 Q8 GGUF causal conditions | `gemma4-e4b-q8-causal.json` |
| [`vram_comparison.csv`](vram_comparison.csv) | Matched P-cache, retained-KV, and combined VRAM table | `vram-comparison.json` |
| [`vram_comparison.svg`](vram_comparison.svg) | Matched incremental peak VRAM plot | `vram-comparison.json` |
| [`VRAM_COMPARISON.md`](VRAM_COMPARISON.md) | Measurement boundary and exact results | `vram-comparison.json` |
| [`EVIDENCE_MANIFEST.json`](EVIDENCE_MANIFEST.json) | Source artifact hashes and generator metadata | Recorded benchmark JSON files |

Regenerate the derived assets from the repository root with:

```bash
PYTHONPATH=src .venv/bin/python Publishing/Assets/generate_assets.py
```

Generation requires the `publishing` optional dependency and the `qpdf` command.
The generator removes volatile PDF metadata and uses a deterministic document
identifier so the PDF is byte-reproducible.

The historical visual architecture specification remains unchanged in the source repository. It is intentionally not redistributed in the publication packs. The publication architecture diagram is a new project-authored current-architecture asset.
