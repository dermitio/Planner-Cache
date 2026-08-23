# Architecture

Planner Cache separates facts that can change from recent token-level conversation history. **P-cache** is the fixed-capacity current-fact store. **Canonical P** is its model-independent structured representation. The **router** selects relevant canonical state. A **TTL** converts selected state into a model's internal tensors, while an **LTL** provides narrower tokenizer or output control. **P-package** keeps durable personality conclusions on disk. **Recent KV** is the runtime's token-level memory and remains outside the core with archive storage and tool retrieval.

![Planner Cache architecture](assets/architecture.svg)

## Ownership boundary

| Component | Owner | Responsibility |
|---|---|---|
| Recent KV | Model runtime | Exact recent wording and local token continuity |
| P-cache | Planner Cache | Current mutable semantic state |
| Canonical router | Planner Cache | Relevant canonical-state selection |
| `.ttl` | Model compatibility layer | Semantic or internal tensor translation |
| `.ltl` | Model and runtime compatibility layer | Lexical or output control |
| P-package | Planner Cache | Durable personality and behavioral conclusions |
| Archive and history | Application runtime | Exact historical evidence |
| Tool retrieval | Application runtime | Temporary external evidence and tool context |

Planner Cache does not implement or prescribe a transcript database, search engine, tool-KV layout, or recent-KV eviction policy.

## Canonical P protocol

`pcm-canonical-p-v1` defines a model-independent state snapshot. A canonical store contains fixed-width values and metadata for validity, type, confidence, importance, freshness, persistence, update time, source, entity identity, relation identity, value identity, and canonical metadata identity.

The active implementation preallocates every tensor at construction. Logical invalidation changes masks and zeros the value while physical allocation remains stable.

Canonical snapshots contain no base-model weights, model token IDs, prompt text, attention KV, optimizer state, or model-native hidden vectors. Snapshot save and load validate a checksum over tensors and canonical metadata.

## P-cache lifecycle

```text
CREATE -> KEEP or MODIFY -> optional MERGE -> INVALIDATE
```

- `CREATE` admits a new state, performs identity-safe semantic merge, or returns `IGNORE` under capacity pressure.
- `KEEP` refreshes an existing state without reallocating it.
- `MODIFY` replaces the current value in place.
- `MERGE` combines compatible state and invalidates redundant slots.
- `INVALIDATE` removes a state from logical reads and routing.
- `IGNORE` performs no mutation.

Canonical merge is constrained to the same entity and relation. This audit fix prevents vector collisions from aliasing unrelated state.

When the cache is full, permanent entries remain protected. Non-permanent eviction uses importance, confidence, persistence, freshness, and age. An incoming state must exceed the weakest resident admission score or it is ignored.

### Natural post-turn memory review

Interactive chat retains deterministic `remember:` and `invalidate:` overrides
and adds a hidden semantic review after each visible assistant response. The
review receives the newest turn, at most two bounded prior turns for reference
resolution, relevant current P entries, authority labels, and the canonical
operation schema. Strict JSON may propose `CREATE`, `MODIFY`, `KEEP`, `MERGE`,
`INVALIDATE`, or `IGNORE`.

Validation rejects malformed output, low-confidence proposals, unsupported
assistant claims, unknown relations, and operations inconsistent with current
state. Accepted batches apply atomically. The gateway flushes the visible reply
before review but holds the session lock until review completes.

This is an extraction side-channel. It never replaces, canonicalizes, summarizes,
or reconstructs visible `system`, `user`, or `assistant` messages. The visible
Gemma request continues through llama.cpp's native chat template unchanged.

## Universal canonical router

The query path is split deliberately.

```text
frozen-model hidden state
  -> model-specific query projector
  -> canonical entity, relation, and metadata query
  -> universal canonical router
  -> selected current canonical slots
```

The router has no model-hidden dimension. Its active index contains tokenizer-independent byte-derived entity anchors, relation IDs, canonical metadata IDs, and a validity mask that excludes invalidated and stale slots.

The router scores entity similarity, relation agreement, query-type agreement, and current-state status. It applies a calibrated acceptance threshold before translation. Top-1 and small top-k selection are supported.

## Compatibility hierarchy

```text
canonical P
  -> universal .router
  -> native P support, .ttl, or .ltl
```

Native P support means a model or runtime consumes canonical P directly. TTL support means a model-specific Tensor Translation Layer exposes P state to internal hidden or residual representations. LTL support means a model and runtime-specific Lexical Translation Layer converts an accepted value to tokenizer or output controls. These support levels are not equivalent.

## Tensor Translation Layer

The selected Pythia TTL contains three model-specific parts.

1. A query projector maps model hidden width into a factorized canonical query.
2. A value translator maps selected 512-wide canonical values into model hidden width.
3. A gate conditions residual injection on current hidden state, translated value, and canonical route features.

The frozen base model receives no gradients. P remains outside prompt tokens and self-attention KV. P-disabled and empty or invalidated state bypass injection.

The universal router is serialized separately from model-specific `.ttl` weights. See [TTL.md](TTL.md).

## Lexical Translation Layer

The active Gemma4 E4B Q8 compatibility is an LTL. After the universal router accepts a state, the exact canonical UTF-8 value is tokenized with the verified Gemma bundle. llama.cpp applies direct adaptive logit bias to emit the selected token sequence. Rejected routes produce no lexical target and remain inert. This proves lexical or output compatibility. It does not prove that Gemma reasons internally over P. See [LTL.md](LTL.md).

## P-package

P-package uses the same portable canonical philosophy but a different mutation policy. P-cache accepts immediate current-state mutations. P-package promotes durable conclusions only after repeated, diverse, or authoritative evidence.

The `.ppkg` file remains SQLite-backed and disk resident. A query performs indexed coarse filtering, CPU scoring, selected-row hydration, canonical conversion, and temporary activation through the compatibility resolver. The full package is never uploaded to CUDA.

See [PPKG.md](PPKG.md).

## Serialization boundaries

| Format | Contains | Excludes |
|---|---|---|
| Canonical P snapshot | Canonical values and state metadata | Model weights, token IDs, KV, transcript |
| `.router` | Canonical scorer weights and protocol metadata | Model hidden dimensions and P contents |
| `.ttl` | Model metadata, query projector, value translator, gate, attachment metadata | Base weights, P contents, conversation state |
| `.ltl` | Model, tokenizer, runtime identity and lexical control configuration | Base weights, P contents, conversation state |
| `.ppkg` | Canonical personality entries, evidence references, reversible changes | Hidden vectors, base weights, adapters, transcript, optimizer state |

## Integrity boundaries

Canonical snapshots, router files, and translator files validate at load. P-package performs full semantic hashing at verified open, explicit verify, export, checkpoint, and dirty close. Normal routing, evidence pushes, row hydration, and generation do not hash the full package.

## Memory behavior

P-cache and router index memory scale with configured slot capacity. P-package disk size scales with stored entries while active RAM and VRAM depend mainly on the bounded candidate set and selected top-k entries. Conversation length is not an input to P-cache allocation.

The `/personality` inspection action is separately bounded to a deterministic
page of at most 200 entries. Debug pagination does not change top-k generation
routing or durable package contents.
