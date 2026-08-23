# Native message passthrough invariant

Date: 2026-08-23

Status: active correctness fix

## Finding

The first Web UI gateway passed only the newest extracted user string into the Gemma runtime. The runtime then reconstructed user and assistant history from terminal-style tuples. This discarded system messages and could change structured message fields before llama.cpp applied Gemma's native chat template.

That behavior violated the intended side-channel boundary and has been removed from the Web UI path.

## Active invariant

The gateway now maintains two independent paths.

```text
opaque original messages array
-> Gemma runtime
-> llama.cpp native chat template

newest raw user text observation
-> deterministic P extraction
-> P-cache and P-package side-channel
-> canonical router
-> LTL output control only after route acceptance
```

System, user, and assistant message objects are deep-copied without replacement, canonicalization, summarization, truncation, or reconstruction. The copy prevents instrumentation or downstream request assembly from mutating the browser-owned object. The exact structured copy becomes the `messages` field sent to llama-server.

Gemma assistant content returned by llama-server is also no longer stripped. This preserves leading and trailing content if the Web UI includes the assistant response in a later native message array.

P extraction may observe the newest user text. It does not supply replacement text to the model. TTL and LTL remain downstream compatibility mechanisms for selected canonical state. They do not translate raw conversation text into P state and do not rewrite native chat messages.

## Inactive equivalence

When routing produces no active P value, the Gemma request has no `logit_bias` or other LTL control. An exact-runtime regression sends the same system and user message objects through raw llama-server templating and through the inactive gateway request builder. It requires byte-identical rendered prompts and exactly equal token ID sequences from the tested Gemma4 E4B Q8 GGUF llama.cpp runtime.

The terminal-only fallback still constructs a simple terminal history because it has no native structured message input. This fallback is not used by `run-gemma.sh`, whose primary conversation surface is the Web UI.
