# llama.cpp Web UI interactive gateway

Date: 2026-08-22

Status: active runtime interface update

## Decision

`run-pythia.sh` and `run-gemma.sh` now open the installed llama.cpp Web UI rather than using terminal text as the primary conversation surface.

The UI is served by a local Planner Cache gateway. It is not connected directly to the frozen model endpoint. Every browser chat request passes through the same active session controller used by the terminal path.

```text
llama.cpp Web UI
-> Planner Cache gateway
-> automatic P-cache mutation
-> P-package query
-> universal router
-> Pythia TTL or Gemma LTL
-> frozen model runtime
-> session and event recording
```

This boundary is required because a browser connected directly to llama-server would bypass canonical state mutation, routing, compatibility control, personality retrieval, and Planner Cache session logging.

## Runtime behavior

The exact static Web UI bundle from the configured llama.cpp build is reused by both launchers. Gemma generation continues through the managed llama-server child process. Pythia generation continues through the frozen Transformers runtime and semantic TTL. Pythia does not load an unrelated GGUF model merely to serve the UI.

The gateway exposes OpenAI-compatible chat-completion responses to the UI. Streaming requests receive a valid event stream after the Planner Cache turn completes. Generation remains serialized so state mutations and event ordering cannot race.

The SQLite P-package connection permits use by the gateway worker thread. A single session lock still serializes reads, updates, checkpoints, and debug commands. SQLite transaction, journal, durability, and integrity-boundary behavior is unchanged.

## Debug behavior

The terminal remains a secondary control surface for `/help`, `/state`, `/personality`, `/events`, `/save`, and `/quit`. Ordinary terminal text is rejected with a reminder to use the browser. Read-only HTTP debug endpoints are also available for state, personality, and recent events.

## Invariants

- Browser text is not inserted as logging-only prompt context.
- UI requests cannot call the model while bypassing Planner Cache.
- Existing P-cache, router, TTL, LTL, and P-package behavior is unchanged.
- Existing transcript, event, snapshot, and final-state artifacts are preserved.
- Gemma llama-server remains managed and is terminated during gateway shutdown.
