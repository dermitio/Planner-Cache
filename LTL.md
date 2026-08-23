# Lexical Translation Layer

LTL means Lexical Translation Layer. Its extension is `.ltl` and its support level is lexical or output.

```text
canonical P -> universal router -> .ltl -> tokenizer and runtime control -> emitted value
```

The reference Gemma4 E4B Q8 LTL uses the exact verified tokenizer bundle and llama.cpp direct adaptive logit bias. The selected LTL has zero learned parameters. It targets one token at a time, advances after each emitted token, and stops when the complete selected value has been emitted.

The universal router remains responsible for entity, relation, validity, and historical-state rejection. When routing is rejected, the LTL creates no target and changes no runtime logits. LTL support therefore preserves inactive-path exactness while making a narrower promise than TTL support.

`planner-cache-ltl-v1` is a deterministic checksummed JSON envelope. It records model identity, GGUF checksum, runtime identity and version, tokenizer-bundle checksum, canonical protocol, control strategy, and parameter count. It contains no model weights, P contents, conversation state, vocabulary table, or optimizer state.

Gemma LTL support proves exact routed-value output compatibility on the recorded lexical audit. It does not prove model-internal semantic consumption or reasoning over P. Historical control-vector, static residual, lexical mapper, and sequence-aware `.translate` artifacts remain research evidence.
