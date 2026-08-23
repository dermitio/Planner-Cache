# Matched P-cache and retained-KV VRAM comparison

At the largest measured case, canonical P occupied 2.052 MiB while retained KV
tensors occupied 193.312 MiB. Canonical P was therefore about 94 times smaller
as a stored representation in this matched test. The two stores are not
equivalent. Canonical P keeps structured current facts, while retained KV keeps
recent token-level attention state.

Peak execution memory includes temporary computation. At 1,024 units, the
incremental peak was 103.491 MiB for P-only and 294.266 MiB for KV-only. These
peak values should not be confused with the cache tensor sizes above.

The comparison uses one frozen Pythia-1.4B model on one NVIDIA GeForce RTX
3050 Laptop GPU. Every row uses batch size 1, a float16 base, the same float32
TTL, greedy generation, eight generated tokens, and the same synthetic prompt
tokens at each workload size. All mechanisms are warmed once before measurement.

`P-cache only` disables retained runtime KV while leaving normal attention
working memory and the P-cache TTL path active. `KV only` retains model KV and
disables P-cache. `P-cache plus KV` enables both. This distinguishes the two
memory systems without claiming that their contents or purposes are
interchangeable.

Here, **retained KV** means token-level key-value tensors kept by the runtime.
**Incremental peak VRAM** means the additional maximum allocated GPU memory
above the same warmed model baseline.

| Prompt tokens and P slots | Condition | Canonical P | Retained KV | Baseline allocated | Peak allocated | Peak reserved | Incremental peak allocated | Runtime |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | P-cache only | 0.128 MiB | 0 | 2,717.183 MiB | 2,724.309 MiB | 2,776 MiB | 7.125 MiB | 0.2555 s |
| 64 | KV only | 0 | 13.312 MiB | 2,717.183 MiB | 2,735.575 MiB | 2,788 MiB | 18.392 MiB | 0.1754 s |
| 64 | P-cache plus KV | 0.128 MiB | 13.312 MiB | 2,717.183 MiB | 2,735.608 MiB | 2,788 MiB | 18.425 MiB | 0.2045 s |
| 256 | P-cache only | 0.513 MiB | 0 | 2,717.183 MiB | 2,744.347 MiB | 2,806 MiB | 27.164 MiB | 0.7415 s |
| 256 | KV only | 0 | 49.312 MiB | 2,717.183 MiB | 2,794.609 MiB | 2,852 MiB | 77.426 MiB | 0.1937 s |
| 256 | P-cache plus KV | 0.513 MiB | 49.312 MiB | 2,717.183 MiB | 2,794.739 MiB | 2,852 MiB | 77.556 MiB | 0.4156 s |
| 1,024 | P-cache only | 2.052 MiB | 0 | 2,717.183 MiB | 2,820.674 MiB | 2,938 MiB | 103.491 MiB | 2.7105 s |
| 1,024 | KV only | 0 | 193.312 MiB | 2,717.183 MiB | 3,011.449 MiB | 3,096 MiB | 294.266 MiB | 0.3753 s |
| 1,024 | P-cache plus KV | 2.052 MiB | 193.312 MiB | 2,717.183 MiB | 3,011.966 MiB | 3,114 MiB | 294.783 MiB | 1.2812 s |

All nine conditions completed. There were no OOM events, failures, fallbacks,
or estimated values. CUDA peak allocation includes transient attention, router,
TTL, output, and allocator behavior. Canonical P bytes and retained KV tensor
bytes are therefore reported separately from peak deltas.

Raw measurements are in [`vram-comparison.json`](../artifacts/vram-comparison.json). The generated table is
`vram_comparison.csv`, and the plot is `vram_comparison.svg`.
