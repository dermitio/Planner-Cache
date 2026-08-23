# Research history

Planner Cache emerged through a sequence of falsifiable experiments. Archived failures are retained because they localize where state storage, routing, translation, gating, and frozen-model consumption succeed or fail.

## Original motivation

The original specification proposed a hierarchy for recent exact wording, current mutable state, personalization, tool evidence, and exact historical records. The enduring idea was that current semantic state should not require keeping an entire conversation as active token-level KV.

The active publication scope is narrower. Planner Cache owns P-cache, the universal router, the TTL and LTL compatibility protocols, and `.ppkg`. Recent KV, tools, and history systems are external runtime responsibilities.

## Self-trained proof model

The first phase implemented a custom byte BPE, corpus preparation, a decoder-only model, checkpointing, and CUDA training. The 15.7M model failed the coherence gate after 166M training tokens. The 98.7M model reached held-out loss 1.674 and perplexity 5.336 after 165,974,016 tokens, but fixed prompts still drifted or repeated.

A 248.1M BF16 configuration passed the exact compiled four-way accumulation, checkpoint, and resume regression after CUDAGraphs were disabled while Inductor remained enabled. The long 250M run was not started. The self-trained path was then retired.

## Pythia pivot

The local Pythia-1.4B checkpoint became the frozen base. Early P-cache mechanics established fixed allocation, explicit state mutation, invalidation, capacity pressure, importance-aware retention, and comparison against KV-only, rolling-summary, and archive retrieval baselines.

These mechanics passed, but a final-layer cross-attention reader achieved only 25% held-out state accuracy and worsened RP loss.

## Reader variants

Final-layer, upper-layer, pooled residual, per-token gate, reader pretraining, and preservation variants were matched on the same data. Preservation reduced ordinary-RP drift, but every variant remained at 25% held-out state accuracy.

The result ruled out a simple layer-placement or regularization fix.

## Factorized canonical representation

A factorized entity, relation, value, and metadata representation reached 97.27% held-out P-only state recovery and 100% canonical decoding. The downstream reader still failed to produce held-out state tokens.

This localized the problem downstream of state representation.

## Model-agnostic bridge

The bridge separated canonical P from Pythia hidden space and reached 100% hard-negative retrieval and 100% canonical-to-model value accuracy. Bridge-only preserved RP but did not improve held-out generation. A rank-4 LoRA fallback reduced training loss while held-out generation remained 0% and RP quality degraded.

The bridge and LoRA fallback were rejected.

## Monolithic translator

The bidirectional translator established P-only causal directionality, invalidation identity, and value translation. It still degraded to 65% to 70% global retrieval over 20 facts, reached only 0% to 25% unseen-name accuracy, and allowed wrong-entity state too readily. Deeper attachment improved generation while harming RP.

This showed that query projection, routing, translation, and gating needed separate attribution.

## Split router and translator

The successful Phase B architecture separated a model-specific query projector, a universal canonical router, a model-specific value translator, and a conditional gate. A tokenizer-independent byte entity anchor solved unseen-name routing in the controlled split. The selected final-layer Pythia package reached the 128-slot causal generation target while preserving unrelated RP. This package is now classified as a TTL.

## P-package

The original live adaptive-extension idea was replaced for the active proof by a deterministic disk-resident package of durable personality conclusions. P-package uses evidence accumulation, authority ordering, contradiction history, contextual scope, selective loading, and the compatibility resolver.

No Phase C-style weight training was started.

## Active-system audit

The audit isolated historical router scaling loss to cross-identity merge corruption and fixed it. It added canonical snapshot checksums, excluded stale slots, normalized local model paths, indexed `.ppkg` routing, separated historical code under `Archive`, and produced matched CUDA failure attribution.

The remaining research boundaries are documented in [LIMITATIONS.md](LIMITATIONS.md).

## Gemma GGUF portability proof

A later bounded port targeted a frozen Gemma 4 E4B Q8 GGUF through llama.cpp. The stock control-vector generator asserted because the Gemma4 graph did not expose the expected layer callback count. The selected adapter instead used a frozen token-row difference as a model-space direction and fitted a 512-wide canonical affine mapper for the controlled `Alice` and `Bob` values.

Changing only canonical P produced the corresponding exact generated token. Wrong entity, wrong relation, historical, invalidated, and disabled paths reproduced the frozen full-vocabulary logits exactly. This established real quantized-runtime causal consumption without changing P-cache, router, GGUF weights, prompt tokens, or KV. It did not establish a full hidden-state query projection or open-vocabulary second-model semantic port.

## General lexical translator attempt

A later trainer enumerated the complete Gemma GGUF vocabulary and learned a compact UTF-8 byte-to-control mapping from frozen input embeddings. It used strict held-out complete strings, multi-token compositions, state values, and spelling hard negatives. The adapter improved embedding cosine but failed value discrimination and produced 0% causal generation on balanced held-out and familiar-value diagnostics.

The experiment established that its compact approximation of stock input rows was not sufficient for layer-41 control. The later exact-row sweep showed that the tied physical rows are causal when recovered exactly. The trainer and artifacts remain reproducible research infrastructure.

## Gemma causal geometry

The next pass inspected all 720 GGUF tensors and captured all 42 residual layers
from the actual quantized llama.cpp graph. It verified tied input and output
weights and found an additional token-derived lexical input at every Gemma4
block. Held-out lexical identity remained recoverable throughout the network.

Exact llama.cpp-tokenized output rows localized causal control to layer 41 and
reached 256 of 256 first tokens. The same static controls produced only 28 of
256 full strings and failed every multi-token value. Learned residual bases up
to rank 128 reached 42 of 67 held-out first tokens and zero full strings. This
localized the remaining failure to compact open-vocabulary translation and
sequence-aware readout. The experiment was rejected as a general semantic adapter.

## Sequence-aware Gemma control

The next experiment used the exact workspace Gemma tokenizer bundle after a
complete vocabulary and runtime compatibility check. Instead of holding one
residual constant, it scheduled the proven layer-41 tied-output control for
each target token. The cursor advanced only after successful emission and
disabled itself at completion or mismatch.

This changed the measured boundary from 100% first-token and 0% multi-token
success to 100% exact strings on 13 controlled values and 97.65625% on 128
disjoint values. The result is retained as a research prototype and does not erase the negative static experiments.

## Sequence complexity audit

A matched five-way audit then separated tokenizer access, output geometry,
scalar calibration, low-rank approximation, and sequence scheduling. Static
and low-rank residuals produced zero exact strings. Direct adaptive logit bias
produced 128 of 128 and reduced KL by 50.74 percent relative to the sequence
residual.

This localized the apparent generalization to exact tokenizer IDs and explicit
per-token forcing. The sequence residual was reclassified as a lexical actuator
rather than learned semantic translation. The prior artifact and benchmark
remain intact as the internal residual baseline.

The selected direct adaptive logit-bias mechanism is now formalized as Gemma LTL support. Pythia's hidden-state adapter is formalized as TTL support. This distinction prevents lexical output control from being reported as internal semantic compatibility.
