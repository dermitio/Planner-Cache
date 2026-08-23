# Interactive Planner Cache session entry points

Date: 2026-08-22

Status: active runtime interface implemented. No new model or architecture training was started.

The default human-facing mode is free-form conversation. The terminal presents `You:` and `Assistant:` prompts and keeps inspection commands secondary. Ordinary text is always sent to the model with retained recent conversation context.

## Decision

Two executable repository-root entry points now expose the proven model integrations through one shared terminal and session recorder.

- `run-pythia.sh` loads the frozen local Pythia-1.4B checkpoint, the selected split `.translate`, and the universal canonical router.
- `run-gemma.sh` loads the recorded Gemma4 Q8 `.translate`, the same canonical router, and a managed llama.cpp server.
- `pcm.planner.chat_cli` owns the shared conversation loop, commands, checkpoints, and shutdown behavior.
- `pcm.planner.interactive_session` owns canonical mutation extraction, P-package evidence and retrieval, and structured event recording.
- `pcm.planner.interactive_runtimes` contains only the model-specific generation and telemetry adapters.

The shell scripts resolve the repository from their own location. The Gemma script also maintains a PID file and an exit trap as a second cleanup boundary around the Python server manager.

## State extraction boundary

The interface does not ask either language model to invent canonical mutations. A conservative deterministic extractor recognizes explicit ownership, placement, location, status, and invalidation statements. Recognized repetitions perform `KEEP`. Changed values perform `MODIFY`. New facts perform `CREATE` or the store's actual returned operation. Explicit invalidations perform `INVALIDATE`. Other text records `IGNORE` without mutating P.

Free-form value surfaces such as `desk` are retained as inspectable current P state with a stable canonical factor ID and separate model-independent surface metadata. If the selected `.translate` does not support that value, the universal router decision and compatibility rejection are logged, the translator remains disabled, and ordinary recent-context generation continues. Unsupported values are never silently mapped into a different supported model value. The companion `p-cache-runtime.json` file preserves these surface labels and compatibility flags beside the canonical safetensors checkpoint.

The silver-key and gold-key factor identities retain the fixed proof recipe identities used by the active translator benchmarks. Other entity surfaces use a stable model-independent hash for store identity and a bounded factor index for the unchanged canonical representation.

P-package evidence extraction is similarly conservative. It currently admits only explicit user statements about supported response styles. The existing deterministic authority, promotion, contradiction, context routing, checkpoint, and checksum behavior remains unchanged.

## Observational logging

Every logged turn records state before mutation, actual mutation decisions, P-package evidence and retrieval where enabled, canonical router activity, translator activity, model generation, and state after generation. Pythia telemetry comes from the existing model hook and now includes detached factorized query fields. Gemma telemetry records canonical routing, gate status, control-vector strength, attachment layer, llama.cpp version, GGUF checksum, inert status, and measured latency.

Telemetry collection detaches existing values and does not alter prompts, canonical state, route scores, translation values, sampling state, or control-vector decisions. A deterministic regression compares Pythia output with telemetry disabled and enabled. State mutation decisions are also compared with session logging enabled and disabled.

## Persistence

Each session has a filesystem-safe unique directory containing `session.json`, `transcript.jsonl`, `events.jsonl`, `p-cache.safetensors`, `p-cache-runtime.json`, and `final-state.json`. The `/save` command checkpoints both P-cache and P-package state. Normal exit, EOF, and Ctrl-C attempt the same checkpoint and final summary. A P-package checkpoint failure is logged and does not prevent the independent final JSON attempt.

The default durable personality path is `state/personality.ppkg`. Session outputs use `sessions/`. Both runtime directories are ignored by Git.

## Gemma runtime behavior

llama.cpp control vectors are configured when the server starts. The managed runtime therefore reuses the server while translated strength remains unchanged and restarts it when the accepted canonical value changes or the path becomes inert. A zero or rejected gate starts the server without a control-vector argument. The P-cache and `.ppkg` are never copied into the server prompt.

The interactive Gemma server disables its optional reasoning trace so ordinary answers are returned directly in the assistant content field. This affects only presentation and generation mode in the chat runtime. It does not change routing, translation, canonical state, or the recorded compatibility benchmark.

The Gemma limitation from Update 0009 remains unchanged. Only canonical values supported by the recorded two-value adapter can open the control path. Other routed values remain inert. The public llama.cpp server still does not provide model-hidden query projection.

## Verification

Unit coverage includes session creation, JSONL validity and ordering, final-state writing, canonical operation logging, instrumentation equivalence, P-package promotion and contradiction events, Ctrl-C cleanup, child-server termination, environment overrides, and missing-path errors.

Real chat-first smoke sessions passed for both launchers. Pythia accepted two ordinary turns through the shared conversation loop and retained the first turn in the second prompt. Its frozen base response quality remains that of the local non-instruction-tuned Pythia checkpoint.

The Gemma acceptance session used the exact free-form sequence requested for manual testing. It recorded `silver key.location = desk`, retained three prior user and assistant turns, and later answered `You left the silver key on the desk.` The universal router selected the canonical state on the relevant turns. The compatibility precheck kept the control-vector path inert because `desk` is outside the recorded two-value Gemma adapter vocabulary. This demonstrates normal conversation continuity and honest compatibility fallback without inventing a translated value. The final shutdown left no llama-server process or PID file.

The earlier Gemma relevant-state proof remains unchanged. That session created `silver_key.owner = Alice`, opened the layer-41 control path at strength `1.999999761581421`, reused the server on the following query, and answered `Alice owns the silver key.`
