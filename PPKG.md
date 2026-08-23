# P-package and `.ppkg`

P-package is the disk-resident durable personality sibling of P-cache. It stores stable behavioral conclusions rather than current world state or transcript history.

## What belongs in P-package

- traits
- preferences
- relationship patterns
- behavioral patterns
- interaction style
- terminology
- habits
- response tendencies
- context-specific tendencies

An entry records subject, relation, value, scope, optional relationship, strength, confidence, importance, evidence count, context diversity, timestamps, authority, supporting and contradicting evidence IDs, status, and extension metadata.

Raw conversation text remains in the external archive. Evidence records contain an archive reference and only the canonical candidate conclusion needed for promotion.

## Evidence flow

```text
interaction or archive event
  -> evidence record
  -> candidate family and evidence graph
  -> deterministic promotion decision
  -> reversible P-package update
```

Ordinary P-cache facts are not copied automatically. Only explicitly eligible behavioral observations can enter the evidence layer.

## Authority

| Source | Weight |
|---|---:|
| Explicit user correction or statement | 1.00 |
| Externally verified observation | 0.90 |
| Repeated observed behavior | 0.70 |
| Single observed behavior | 0.45 |
| Model inference | 0.15 |
| Unsupported model-generated claim | 0.00 |

Unsupported model claims cannot bootstrap themselves into durable personality through repetition.

## Promotion policy

```text
weighted_support = sum(confidence_i * authority_i)
diversity_factor = 1 + 0.8 * log(1 + max(0, unique_contexts - 1))
connectivity_factor = 1 + 0.25 * log(1 + unique_evidence_links)
promotion_score = weighted_support * diversity_factor * connectivity_factor
net_score = max(0, promotion_score - 0.75 * opposing_weight)
```

Normal promotion requires at least three supporting observations and a net score of at least 1.25. An explicit user correction with confidence at least 0.85 can override this normal repetition requirement.

## Contradiction and scope

Contradictory evidence lowers confidence and records contradiction IDs. A strong explicit correction marks an older conclusion as superseded rather than deleting it. Changes are stored as reversible before and after transactions.

Scopes let technical, creative, roleplay, planning, and global tendencies coexist. A matching scoped tendency can outrank a weaker global tendency.

## Disk routing

`PersonalityRouter` accepts subject, interaction type, domain, optional relationship, optional relation, and time. The package performs bounded indexed coarse filtering by subject, scope, and relationship. The router then scores semantic relevance, identity, context, relationship, relation, confidence, strength, importance, and slow freshness on CPU.

Top-1, top-4, and top-8 retrieval are supported. Only accepted rows are hydrated with full evidence references.

The `/personality` debug action is separate from generation retrieval. It returns
a deterministic page with total and truncation metadata. Its default and shutdown
views hydrate at most 100 entries, preventing inspection from converting a 100k
package into a 61 MB JSON response.

## Activation through compatibility layers

Selected entries are converted into a temporary canonical P store. The canonical router and compatibility resolver then dispatch to native P support, a TTL, or an LTL. Model-space vectors and lexical controls are temporary and are never written back into `.ppkg`.

When no entry is relevant, activation creates no canonical CUDA allocation. Opening the full package changes allocated CUDA memory by zero bytes in the measured proof.

## Integrity

The format is `pcm-personality-package-v1` and the protocol is `pcm-canonical-personality-v1`. The package records canonical P compatibility with `pcm-canonical-p-v1`.

Full semantic checksum work occurs at verified open, explicit `verify`, export, checkpoint, and dirty close. Normal top-k routing, polling, evidence pushes, and row hydration do not recompute the full package checksum. SQLite remains transactional with full synchronous mode.

## Measured proof

- One weak event did not promote.
- Three independent events promoted.
- Cross-context connected evidence outscored narrow repetition.
- Unsupported model claims did not promote.
- Explicit correction superseded an older conclusion.
- Context and relationship retrieval passed.
- Cold reload required no conversation replay.
- The 100k package loaded four selected entries from 152 candidate headers.
- The 100k indexed header route measured 67.10 ms.
- Inactive VRAM measured zero bytes.
- A selected top-1 canonical activation measured 1,077 bytes.

## Limitations

The causal proof uses controlled preferred-persona values inside the existing compatibility range. It is not evidence of sophisticated neural personality understanding. Canonicalization currently depends on a small learned factorized representation whose weights are reconstructed by a fixed recipe rather than shipped as a versioned canonical artifact. Arbitrary personality values can collide in the small factor vocabulary.
