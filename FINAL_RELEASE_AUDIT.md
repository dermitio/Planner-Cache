# Planner Cache final release audit

Audit date: 2026-08-23

## Release readiness

The implementation and publication packs are technically validated as a release
candidate. Public redistribution is **blocked** because the repository does not
contain a repository-wide software license grant. Final author metadata and a
public release URL are also missing. No license was invented during this audit.

The publication packs intentionally exclude model weights, GGUF files, tokenizer
and metadata bundles, datasets, llama.cpp files, and other third-party copyrighted
payloads. They contain project-authored implementation, documentation, adapters,
benchmark records, and derived assets only.

## Current architecture

The audited active boundary is:

```text
Recent KV and runtime history
        ↓
frozen model and native chat template

canonical P-cache and selected P-package state
        ↓
universal .router
        ↓
native P support, semantic .ttl, or lexical .ltl
```

Recent KV, archive/history, and tool retrieval remain model or runtime
responsibilities. P-cache is bounded mutable current state. P-package is durable
disk-resident personality state. The hidden post-turn memory review observes the
latest exchange as a side-channel and applies only validated canonical P
operations. It does not rewrite the visible message path.

## Fixed BLOCKER and MAJOR issues

| Severity | Finding | Resolution |
|---|---|---|
| BLOCKER | The first staging pass copied the upstream Gemma tokenizer bundle | Removed from every pack. The builder and validator now reject tokenizer, model, GGUF, and common checkpoint payloads |
| MAJOR | Publication JSON contained workstation-specific absolute paths | Publication copies normalize those paths to portable environment placeholders. Authoritative repository artifacts remain unchanged |
| MAJOR | Active public exports still exposed rejected residual and lexical research APIs | Removed rejected adapters from the active planner package exports. Historical modules and evidence remain available for research regression |
| MAJOR | Launch scripts contained machine-specific model and llama.cpp defaults | Replaced model defaults with required environment inputs and made the llama.cpp default home-relative |
| MAJOR | A clean source checkout could not collect tests without an editable install | Added `src` to the pytest configuration |
| MAJOR | `/personality` hydrated and serialized the complete package | Added bounded inspection with a default 100-entry page. The 100,000-entry case fell from 9.7874 seconds and 173,110,748 peak Python allocation bytes to 0.0218 seconds and 196,288 bytes for the action |
| MAJOR | The VRAM comparison initially included first-use CUDA allocations in one condition | Added a matched warm-up. Every recorded row now begins at the same loaded-stack baseline |
| MAJOR | Publication artifact indexes could diverge after portable path normalization | Pack building now refreshes evidence and artifact checksums after normalization |

## Remaining BLOCKER and MAJOR findings

| Rank | Severity | Finding | Release consequence |
|---:|---|---|---|
| 1 | BLOCKER | No repository-wide software license grant exists | Do not publish or redistribute the staged packs until the rights holder adds a license |
| 2 | BLOCKER | Final authors, affiliations, public repository URL, and release identifier are unset | Citation and preprint metadata remain provisional |
| 3 | MAJOR | Trained semantic TTL support is proven only for Pythia-1.4B | Do not claim universal or multi-model semantic compatibility |
| 4 | MAJOR | Natural memory review is narrow and slow | The controlled reviewer targets owner, location, and status. Recorded review latency was 43.14 to 65.50 seconds |
| 5 | MAJOR | Canonical representation weights are reconstructed rather than shipped as a standalone protocol artifact | Exact third-party reproduction depends on the documented construction path |
| 6 | MAJOR | Pythia router-index hydration is linear on each wrapper call | Controlled routing accuracy is strong through 1,024 slots, but arbitrary-scale latency is not established |

No other BLOCKER or MAJOR correctness issue was found in the release-focused
audit. Nuanced personality learning, broader natural-language extraction, large
debug offsets, multi-seed statistics, and wider model portability remain MINOR,
OPTIMIZATION, or documented research limitations depending on intended use.

## Component scorecard

| Component | Correctness | Integrity | Performance | Status |
|---|---|---|---|---|
| P-cache | Mutation, merge, invalidation, capacity, stale-state, and serialization regressions pass | Canonical snapshots reject corruption and protocol mismatch | Bounded allocation verified | CLEAN |
| Universal `.router` | Controlled top-1, top-4 recall, and MRR are 1.0 through 1,024 slots | Deterministic checksummed artifact | 1,024-slot measured routing was 0.675 ms. Per-call index hydration remains a MAJOR limitation | CLEAN with documented scaling limitation |
| Pythia `.ttl` | Relevant P changes causal logits and tested inactive paths reproduce base candidate logits | Model, width, protocol, type, and checksum checks pass | Frozen base has zero gradients. Active cost is included in the matched VRAM run | CLEAN for the proven Pythia configuration |
| Gemma `.ltl` | Exact routed lexical control is proven for the recorded direct adaptive logit-bias benchmark | Runtime, model, tokenizer checksum, protocol, class, and checksum checks pass | Zero learned parameters. Rejected routes create no lexical target | CLEAN within lexical or output support |
| `.ppkg` | Promotion, authority, contradiction, context, cold reload, and selective hydration tests pass | Checksum work occurs at integrity boundaries, not normal lookup | 100,000 entries use 152 candidate headers and hydrate four rows in the recorded query | CLEAN for the mechanical proof |
| Gateway | Inactive P and LTL preserve exact browser messages, rendered prompt, and token IDs | Session files and event logs are structured and deterministic where required | Review is post-response but must finish before the next turn | CLEAN with review-latency limitation |

## Prompt transparency and inert paths

The native Gemma equivalence artifact records identical structured-message,
rendered-prompt, and token-ID SHA-256 values for the gateway and raw llama-server
when P and LTL are inactive. The prompt contained 33 tokens. No logit bias was
present. Wrong-entity, wrong-relation, historical, invalidated, router-disabled,
and compatibility-disabled paths remain inert in the tested causal regressions.

## Natural memory review

The controlled acceptance run recorded a natural RP CREATE followed by MODIFY:

```text
brass key.location = kitchen drawer
brass key.location = coat pocket
```

The final active state contained only `coat pocket`. The same conceptual review
path ran for Gemma and Pythia. Unsupported assistant claims and malformed review
output remain fail-closed in regression tests. The reviewer does not receive or
alter the visible browser request.

## Exact VRAM comparison

### Command

```bash
PYTHONPATH=src .venv/bin/python benchmarks/compare_pcache_kv_vram.py \
  --model pythia-1.4b \
  --ttl artifacts/pythia-1.4b-final-layer.ttl \
  --router artifacts/canonical-p-v1.router \
  --output artifacts/vram-comparison.json \
  --workloads 64,256,1024 \
  --generated-tokens 8 \
  --seed 317
```

### Matched configuration

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU with 3,950,575,616 bytes
- Driver: 610.57.04
- CUDA runtime: 13.0
- PyTorch: 2.13.0+cu130
- Transformers: 5.15.1
- Model: frozen Pythia-1.4B
- Batch: 1
- Base precision: float16
- TTL precision: float32
- Generation: greedy argmax
- Generated tokens: 8
- Baseline method: one warmed loaded stack followed by CUDA synchronization and peak reset

All memory figures below are MiB. `P bytes` is canonical P tensor allocation.
`KV bytes` is retained model KV tensor storage. CUDA peaks also include transient
attention, router, TTL, output, and allocator work.

| Prompt and slots | Condition | P bytes | KV bytes | Base alloc | Base reserved | Peak alloc | Peak reserved | Increment alloc | Increment reserved | Runtime |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | P-cache only | 0.128 | 0.000 | 2717.183 | 2772.000 | 2724.309 | 2776.000 | 7.125 | 4.000 | 0.2555 s |
| 64 | KV only | 0.000 | 13.312 | 2717.183 | 2772.000 | 2735.575 | 2788.000 | 18.392 | 16.000 | 0.1754 s |
| 64 | P-cache plus KV | 0.128 | 13.312 | 2717.183 | 2772.000 | 2735.608 | 2788.000 | 18.425 | 16.000 | 0.2045 s |
| 256 | P-cache only | 0.513 | 0.000 | 2717.183 | 2772.000 | 2744.347 | 2806.000 | 27.164 | 34.000 | 0.7415 s |
| 256 | KV only | 0.000 | 49.312 | 2717.183 | 2772.000 | 2794.609 | 2852.000 | 77.426 | 80.000 | 0.1937 s |
| 256 | P-cache plus KV | 0.513 | 49.312 | 2717.183 | 2772.000 | 2794.739 | 2852.000 | 77.556 | 80.000 | 0.4156 s |
| 1,024 | P-cache only | 2.052 | 0.000 | 2717.183 | 2772.000 | 2820.674 | 2938.000 | 103.491 | 166.000 | 2.7105 s |
| 1,024 | KV only | 0.000 | 193.312 | 2717.183 | 2772.000 | 3011.449 | 3096.000 | 294.266 | 324.000 | 0.3753 s |
| 1,024 | P-cache plus KV | 2.052 | 193.312 | 2717.183 | 2772.000 | 3011.966 | 3114.000 | 294.783 | 342.000 | 1.2812 s |

All nine conditions succeeded. OOM events, failures, fallbacks, and estimated
values were zero. The raw artifact SHA-256 is
`1b1e266c3f6513a5708711f09879a6519ce45abdaea7ba16d04f2510f5c1fc8d`.
See the [raw JSON](artifacts/vram-comparison.json),
[summary](assets/VRAM_COMPARISON.md), [CSV](assets/vram_comparison.csv), and
[plot](assets/vram_comparison.svg).

The result distinguishes P-cache and KV allocation. It does not imply that
semantic state and exact token-level KV are interchangeable.

## Exact validation commands and results

```bash
GEMMA_MODEL=/path/to/tested-gemma.gguf \
LLAMA_CPP_DIR=/path/to/llama.cpp \
.venv/bin/python -m pytest -q
```

The final result was `126 passed in 285.62 seconds` with the exact local Gemma
runtime enabled. The separate portable no-path run completed with 119 passed and
seven exact-runtime skips. The focused exact Gemma subset completed with 33
passed in 216.16 seconds.

```bash
PYTHONPATH=src python Publishing/assets/generate_assets.py
PYTHONPATH=src python Publishing/assets/generate_assets.py
```

The two runs produced byte-identical SVG and normalized PDF hashes. The current
architecture PDF SHA-256 is
`19fad644f3a1e3086a845f07850beec07e20a2352cad000b461c21b6802a2519`.

```bash
.venv/bin/python Publishing/build_release_packs.py
.venv/bin/python Publishing/validate_release.py
bash -n run-pythia.sh run-gemma.sh
.venv/bin/python -m compileall -q src benchmarks Publishing
git diff --check
```

The publication validator requires all three manifests to match, all local links
to resolve, all JSON to parse, shell and Python syntax to pass, no workstation
absolute paths, and no third-party model or tokenizer payloads.

## Publication folder validation

| Pack | Contents | Independent validation |
|---|---|---|
| GitHub | Developer documentation, active source, launchers, tests, benchmarks, active artifacts, historical result evidence, and assets | Passed manifest, link, syntax, JSON, path, and payload checks |
| Hugging Face | Artifact cards, active compatibility source, active artifacts, benchmark evidence, runtime requirements, and assets | Passed manifest, link, syntax, JSON, path, and payload checks |
| Research | Manuscript, experiments, ablations, reproducibility map, benchmark scripts, active and negative-result evidence, and assets | Passed manifest, link, syntax, JSON, path, and payload checks |

Upstream models, tokenizers, llama.cpp, datasets, and the historical third-party
visual specification are referenced as external prerequisites and are not copied.

## Claims safe to publish

- Planner Cache maintains bounded mutable semantic state independently of retained token-level conversation history.
- The canonical router reached top-1 accuracy and MRR 1.0 through 1,024 slots on the recorded controlled audit.
- The Pythia TTL provides tested internal causal state compatibility with frozen-base gradient isolation.
- The Gemma LTL provides tested lexical output compatibility and does not establish internal semantic reasoning.
- Tested inactive and rejected paths preserve base behavior.
- P-package provides deterministic checksummed persistence, evidence-based promotion, selective loading, and zero inactive VRAM in the recorded proof.
- The indexed 100,000-entry P-package query hydrated four entries from 152 candidate headers.
- The gateway preserves native Gemma messages and tokenization when memory output control is inactive.
- Natural post-turn review can create and modify controlled owner, location, and status state while failing closed.
- The recorded matched VRAM matrix completed without failure and keeps P-cache and KV measurements conceptually separate.

## Claims not safe to publish

- Universal model compatibility
- Trained semantic TTL portability beyond Pythia-1.4B
- Gemma internal semantic reasoning over P
- Replacement of arbitrary long context, archives, or historical retrieval
- Production-ready broad natural-memory extraction
- Production-ready learned personality behavior
- Constant-time routing at arbitrary scale
- Multi-seed statistical generality not present in the artifacts

## Final ranked disposition

1. Add an explicit repository-wide software license before redistribution.
2. Finalize authors, affiliations, repository URL, and release identifier.
3. Keep all semantic portability claims scoped to Pythia until a second trained TTL exists.
4. Present natural memory review as a controlled, narrow, high-latency proof.
5. Publish a standalone canonical representation weight artifact if exact external reconstruction becomes a release requirement.
6. Treat per-call router-index hydration as measured technical debt rather than claiming arbitrary-scale routing.

Subject to the two publication metadata blockers, the code, artifacts, evidence,
and publication packs form a technically clean release candidate.
