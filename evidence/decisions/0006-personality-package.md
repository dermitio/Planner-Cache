# Architecture update: durable personality P-package (`.ppkg`)

Date: 2026-08-22

Status: The deterministic/mechanical `.ppkg` proof passes. No Phase C weight
training, LoRA, base-model modification, or neural personality learning was
performed.

## Decision and hierarchy

Long-term personalization is now represented first as canonical durable state,
not as a continually trained live adapter. The historical adaptive-extension
proposal remains in the original architecture record, but is not the active
implementation direction for this proof.

```text
Recent KV  -> what was just said exactly
P-cache    -> what is currently true or unresolved
P-package  -> what stable behavioral/personality patterns have developed
Tool-KV    -> what evidence was just retrieved
Archive    -> what happened exactly
.translate -> how one frozen model consumes selected canonical state
```

P-cache and P-package are siblings. P-cache retains immediate mutation and
cold-reconciliation semantics unchanged. A behavioral P-cache observation may
become a referenced evidence record, but ordinary current-state facts are not
copied automatically. P-package retrieval creates a temporary canonical P view;
it does not mutate or replace the active P-cache. A combined generation view is
constructed separately when both are relevant.

## Canonical package

`pcm-personality-package-v1` is a single SQLite `.ppkg` file using protocol
`pcm-canonical-personality-v1` and canonical P compatibility protocol
`pcm-canonical-p-v1`. It contains no tensors, model hidden states, model/tokenizer
IDs, base weights, LoRA, `.translate` weights, optimizer state, or transcript.

Normalized tables contain:

- package/protocol/extension metadata and a SHA-256 semantic checksum;
- canonical personality conclusions with type, subject, relation, value, scope,
  relationship, strength, confidence, importance, evidence count, context
  diversity, timestamps, authority, support/contradiction references, and
  status;
- evidence candidates with context, scope, polarity, authority, confidence,
  cross-evidence links, timestamp, and archive/source reference; and
- before/after change records grouped into reversible transactions.

Entry classes are extensible strings validated against the v1 enum. The initial
classes cover traits, preferences, relationships, behavior, interaction style,
terminology, habits, response tendencies, and contextual tendencies. Forward
metadata lives in a canonical extension object.

Save/load is deterministic for identical inputs. Two independently constructed
proof packages were byte-identical with file SHA-256
`74237b7bfbeb2034f336e78ed323eb6898aa4abcd0dab31b4f41a2c010c4e760`.
Loading validates format, personality protocol, canonical P protocol, schema,
and semantic checksum. Corrupt content and incompatible protocols are rejected.

## Evidence and promotion policy

Raw interaction text remains in the archive. `.ppkg` evidence stores only the
candidate conclusion and an `archive_reference`, plus an optional small note.
The evidence flow is:

```text
archive/interaction or explicitly eligible P observation
  -> EvidenceRecord
  -> candidate family and evidence graph
  -> deterministic promotion decision
  -> reversible canonical entry update
```

Authority weights are public constants:

| Source | Weight |
|---|---:|
| Explicit user statement/correction | 1.00 |
| Externally verified observation | 0.90 |
| Repeated observed behavior | 0.70 |
| Single observed behavior | 0.45 |
| Model inference | 0.15 |
| Unsupported model-generated claim | 0.00 |

For supporting evidence `i`:

```text
weighted_support = sum(confidence_i * authority_i)
diversity_factor = 1 + 0.8 * log(1 + max(0, unique_contexts - 1))
connectivity_factor = 1 + 0.25 * log(1 + unique_evidence_links)
promotion_score = weighted_support * diversity_factor * connectivity_factor
net_score = max(0, promotion_score - 0.75 * opposing_weight)
```

Normal promotion requires at least three supporting observations and
`net_score >= 1.25`. A single explicit correction at confidence 0.85 or greater
is the documented exception. Inference-only candidate families cannot promote,
regardless of repetition; unsupported model claims contribute zero.

Confidence uses a 0.5 evidence prior. Strength is bounded by
`net_score / (2 * threshold)`. Conflicting values in the same subject/relation/
scope family lower confidence and record contradiction IDs. A strong explicit
correction supersedes rather than deletes the earlier conclusion. Different
scopes remain distinct, allowing technical/concise and creative/detailed entries
to coexist. Every conclusion mutation records inspectable before/after state;
the last transaction can be reversed without deleting its evidence trail.

## Disk router and activation

The model-independent router accepts canonical subject/relationship identity,
interaction type, domain, optional relation, and time. It reads at most 512
lightweight indexed headers, then scores byte-derived semantic relevance,
context, identity, relationship, relation, confidence, strength, importance,
and slow freshness. It supports top-1/top-4/top-8 and hydrates full evidence
lists only for accepted entries. It does not consume model hidden dimensions or
CUDA state.

After the package is closed, only selected conclusions are converted through
the fixed factorized canonical P representation. The temporary selected store
is passed to the existing split Pythia `.translate`; translated model-space
vectors are discarded with the generation. Empty/irrelevant retrieval creates
no canonical CUDA allocation. Supporting another frozen decoder continues to
require only that model's `.translate`, not a different `.ppkg`.

Semantic checksum validation is a cold-open operation which streams ordered
rows without retaining entry objects. A translation session records the
validated file size and nanosecond modification time; later generation queries
skip the full checksum scan, reject a changed file, and read only bounded
headers plus selected rows. Thus integrity checking does not turn every
generation into a full-package read.

## Mechanical results

- one weak event promoted: no;
- three independent repeated events promoted: yes;
- five narrow-context events scored 1.8000;
- three linked cross-context events scored 2.5865;
- twenty unsupported self-generated claims promoted: no;
- explicit correction promoted and old conclusion status: `superseded`;
- technical, creative, and relationship-specific routing: 100%;
- unrelated retrieval loaded entries: 0; and
- top-1/top-4/top-8 target recall: 100%.

The proof package is 61,440 bytes. Typical top-1 loaded one conclusion and read
1,303 logical bytes in 2.27 ms; top-4 loaded four and read 3,451 bytes in 2.39
ms. These are logical deserialization bytes; SQLite may read/cache whole disk
pages underneath.

## Growth and durability

| Entries | Disk size | Lookup | Headers + selected bytes | Loaded | Active canonical bytes | Inactive VRAM |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 81,920 B | 73.1 ms | 16,070 B | 4 | 4,308 B | 0 |
| 1,000 | 421,888 B | 386.7 ms | 73,991 B | 4 | 4,308 B | 0 |
| 10,000 | 3,805,184 B | 391.4 ms | 73,991 B | 4 | 4,308 B | 0 |
| 100,000 | 37,863,424 B | 448.6 ms | 73,991 B | 4 | 4,308 B | 0 |

The initial SQL router prioritizes bounded active state over minimum latency;
the roughly 0.45 second 100k lookup is an explicit optimization target, not
hidden. Package growth remains on disk, while hydrated state depends on top-k.
Measured Python peak allocation for the 100k lookup was 296,182 bytes. Cold
checksum validation took 5.2 ms / 47.6 ms / 467.9 ms / 4.707 s at 100 / 1k /
10k / 100k entries respectively; it is performed once when creating a trusted
translation session, not per generation.

A fresh process reopened the durable proof package, validated its checksum,
recovered five active conclusions, and retrieved them without conversation
replay. Measured cold load for the small proof package was 1.08 ms.

## Frozen Pythia causal proof

The exact CUDA regression loaded only the selected canonical entry and used
`pythia-1.4b-split-final_layer.translate`. It added no prompt/source-state
tokens. The controlled durable conclusion is a preferred response persona,
which lies inside the already-proven translator's canonical value vocabulary;
this is a mechanical causal-channel proof, not a claim that arbitrary nuanced
style has been neurally learned.

| Identical prompt/KV condition | Alice logit | Bob logit | Gate | Generated |
|---|---:|---:|---:|---|
| Frozen / P-cache-only | 7.0273 | 6.1211 | 0 | ` able` |
| Package A: preferred persona Alice | 40.0312 | 13.8594 | 0.9909 | ` Alice` |
| Package B: preferred persona Bob | 14.0000 | 37.1562 | 0.9605 | ` Bob` |
| Irrelevant package | 7.0273 | 6.1211 | 0 | ` able` |
| Unpromoted low-confidence package | 7.0273 | 6.1211 | 0 | ` able` |
| P-cache + Package A | 40.0312 | 13.8594 | 0.9909 | ` Alice` |

A package containing technical/Alice and creative/Bob conclusions routed and
generated the appropriate value in both contexts. Relevant target loss moved
from 8.3597 to effectively zero. Frozen, P-cache-only, irrelevant package,
low-confidence package, and P-cache-plus-package-while-irrelevant all had exact
ordinary-RP loss 5.47837 and numerical-zero KL with identical samples. Relevant
translation latency was 19.08 ms versus 17.49 ms frozen on the matched fixture.

One selected entry used 1,077 canonical CUDA bytes; zero selected entries used
zero. Opening and validating the complete disk package changed allocated CUDA
memory by zero bytes. No full package was uploaded, and zero frozen-base
parameters received gradients.

Full reproducible results are in
`artifacts/phase-b-personality-package.json`; the durable example is
`artifacts/personality-proof.ppkg`. The first mechanical `.ppkg` completion gate
passes. More sophisticated personality learning and Phase C remain unstarted.
