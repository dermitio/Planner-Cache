# Limitations

## Model portability

The full trained semantic result is currently limited to Pythia-1.4B. A tiny random GPT-2 model passed a structural attachment and serialization test.

A quantized Gemma4 E4B GGUF has LTL support through direct adaptive logit bias. This path provides selected lexical output but does not provide model-hidden query projection or internal semantic reasoning. Universal model compatibility is not a supported claim.

A later vocabulary-wide translator experiment did not close this limitation. Its compact byte mapper retrieved 45.56% of a 4,096-row held-out vocabulary probe but only 0.63% to 13.13% of disjoint complete strings by category. Direct layer-41 causal generation was 0% on a balanced diagnostic.

The causal-geometry follow-up found that this GGUF ties input and output weights. Exact runtime-token rows at layer 41 reached 100% first-token control on 256 values. A rank-128 static mapper reached only 62.69% held-out first-token generation and 0% full-string generation.

A sequence-aware successor reached 100% on 13 controlled values and 97.65625% exact full strings on a disjoint 128-value set. It remains a prototype. Three numeric or numeric-suffix strings failed at strength 8. Mean active KL was 6.6101. The 1,000-value gate and broad natural-RP evaluation under relevant control were not run. The approach also depends on lazily reading exact tied-output rows from the compatible GGUF rather than predicting them with a learned compact lexical mapper.

The complexity audit reclassifies this sequence result as lexical token forcing rather than semantic translation. Direct adaptive logit bias reached 100% exact strings with KL 3.2561 and no training. This mechanism is now the Gemma LTL. It does not demonstrate that Gemma internally represents or reasons over canonical P state.

## Canonical representation portability

The factorized canonical representation has strong held-out probes, but its learned weights are reconstructed from a fixed training recipe rather than shipped as an independent checksummed protocol artifact. P-package canonicalization also hashes arbitrary subjects, relations, and values into small fixed embedding vocabularies of 24 entities, 3 relations, and 36 values. This creates a collision and reproducibility concern for broad open-vocabulary use.

## Router hydration

Canonical ranking is sub-millisecond through 1,024 slots. Rebuilding tokenizer-independent entity anchors for every slot is linear and measured 30.12 ms at 128 slots and 233.00 ms at 1,024 slots. The current Pythia wrapper rebuilds this index for each generation call.

An explicit P-store mutation version and cached router index are not implemented.

## P-package performance

Indexed coarse filtering reduced the 100k header route to 67.10 ms. Python byte-feature scoring remains the dominant bounded-candidate cost. The three covering indexes increase disk use to 58.7 MB for the synthetic 100k package.

The optional one-million-entry scale was not run.

## Personality behavior

The current P-package proof is deterministic and mechanical. It demonstrates evidence promotion, authority, contradiction, context routing, persistence, selective loading, and controlled causal preferred-persona values. It does not demonstrate nuanced learned personality, emotional modeling, clinical traits, or safe autonomous user profiling.

## Context and history

Planner Cache does not replace arbitrary long-context reasoning. Exact old wording is not preserved in P-cache. Archive storage, history search, tool retrieval, and recent-KV policy remain outside this project.

## Natural memory review

Natural chat and RP extraction is a controlled mechanical proof. The interactive
canonical schema currently focuses on owner, location, and status relations.
Malformed or unsupported output is rejected, which protects state but can miss
valid facts. The frozen Pythia base did not reliably produce strict JSON, so the
Pythia launcher uses a configured frozen GGUF reviewer on CPU by default.

Measured post-turn review latency was roughly 43 to 48 seconds on Gemma and 65
seconds on the Pythia interactive path's CPU reviewer. Review follows the visible
reply, but the next turn waits for it. Tests cover natural RP creation and
modification, correction, invalidation, paraphrase merging, transient chatter,
unsupported assistant claims, and malformed output. They do not establish broad
extraction quality across arbitrary domains.

## Debug inspection

`/personality` is paginated and bounded to 100 entries by default, with a maximum
page size of 200. A 100k package took 21.8 ms for count, ordering, and hydration
plus 3.8 ms for JSON in the recorded profile. Large offsets remain linear and are
intended only for manual inspection.

## Benchmark scope

The strongest generation tests use controlled entity, relation, and single-token value prompts. Results do not establish general factuality, instruction following, broad roleplay quality, or production safety.

Natural-RP preservation sets are held out within the project, but they are small research fixtures rather than broad external evaluations.

## Base model

Pythia-1.4B is a research language model with known language, safety, and factuality limitations. Planner Cache does not remove those limitations.

## Release metadata

A repository-wide software license, final author list, public repository URL, release version, and DOI are not present. These are publication readiness items rather than architectural results.
