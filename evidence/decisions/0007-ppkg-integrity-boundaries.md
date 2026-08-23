# Implementation update: `.ppkg` integrity boundaries and 100k lookup profile

Date: 2026-08-22

Status: Full-package checksum work has been removed from normal routing and
evidence-push paths. SQLite transactional semantics remain unchanged. Phase C
was not started.

## Audit finding

The v1 package had three different call paths:

- `PersonalityPackage(path)` performed a full semantic checksum at every
  default open/load;
- the translation session validated once, but opened and closed another
  unverified SQLite connection for every activation; and
- every `ingest`, `undo_last`, and bulk insert committed its mutation and then
  scanned the entire package to write a new checksum.

The router itself did not call the checksum routine. Nevertheless, callers that
opened a package for each query paid the full validation cost, and every
evidence push unconditionally paid it. At 100k entries the isolated checksum
scan is much more expensive than routing.

## Revised integrity contract

Full semantic hashing now occurs only at integrity boundaries:

- verified package open/load (`validate=True`, the default);
- explicit `verify()`;
- explicit `checkpoint()`;
- `export()`, which checkpoints, uses SQLite backup, and verifies the snapshot;
  and
- `close(checkpoint=True)` when committed mutations remain dirty.

An evidence mutation, reversal, or bulk import now:

1. writes canonical rows and change history;
2. marks package metadata `integrity_state=dirty`; and
3. commits that metadata and the mutation in the same SQLite transaction.

It does not scan the package. A checkpoint uses `BEGIN IMMEDIATE`, marks the
state clean, computes and stores the semantic checksum, and commits atomically.
If checkpoint hashing fails, the transaction is rolled back. A default verified
open rejects dirty/uncheckpointed state. The checksum is therefore a durable
package-seal boundary, while SQLite continues to provide transactional
integrity for each push.

`PersonalityTranslateSession` now performs one verified cold open, records the
file size and nanosecond modification time, and retains one read-only SQLite
connection. Repeated top-k activations reuse that connection and perform no
full checksum. A package changed after validation is rejected before routing.

## 100k latency profile

The matched profile used 100,000 entries, a 37,863,424-byte package, top-4
routing, and seven repeated warm queries. The “before” column models the prior
common path of verified open plus whole-package hash for each lookup. The
“after” column is the median reused-connection query.

| Component | Before per query | After per query |
|---|---:|---:|
| Whole-package checksum | 843.619 ms | 0 ms |
| DB open | 0.352 ms | 0 ms (connection reused) |
| Routing + bounded header query | 255.078 ms | 255.078 ms |
| Selected-row hydration | 0.203 ms | 0.203 ms |
| Canonical conversion, four rows | 0.713 ms | 0.713 ms |
| Total | 1,099.965 ms | 255.994 ms |

The modeled end-to-end speedup is 4.30x. Routing/header scoring is now the
dominant per-query cost and remains the next optimization target.

The cold integrity boundary remains explicit:

- unverified SQLite open: 0.352 ms;
- full checksum verification: 843.619 ms; and
- checksum calls: exactly one.

For an evidence push into the same 100k package:

| Operation | Latency | Full checksum calls |
|---|---:|---:|
| Normal transactional evidence push | 1.729 ms | 0 |
| Prior modeled push plus immediate seal | 831.371 ms | 1 |
| Explicit later checkpoint/seal | 829.641 ms | 1 |

Checkpoint work has been moved rather than weakened: accumulated pushes can be
sealed once at close/checkpoint/export instead of rescanning after each push.

## Regression coverage

The regression suite instruments `_semantic_checksum` directly and proves:

- five normal top-k lookups make zero checksum calls;
- a normal evidence push makes zero checksum calls;
- the following explicit checkpoint makes exactly one checksum call;
- translation-session construction validates exactly once;
- three subsequent activations add zero checksum calls; and
- the SQLite connection identity remains stable across those activations.

The exact Pythia-1.4B CUDA `.ppkg` regression still passes, the frozen base
receives zero gradients, and the Phase B personality-package completion gate
remains passed.

Reproducible results are in `artifacts/ppkg-100k-profile.json`. The profiled
fixture is `artifacts/ppkg-profile-100k.ppkg`. The refreshed durable proof
package is `artifacts/personality-proof.ppkg` with SHA-256
`faa701fb6150738937fc7b756fb5df226638eedb7599127b0d04d83899ea4bf9`.
