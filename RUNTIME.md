# Runtime instructions

Planner Cache does not distribute base-model weights or llama.cpp. Create the
Python environment, provide local model paths, and use the included launchers.
The exact Gemma tokenizer and metadata bundle is also an external upstream
requirement. Its contents are not redistributed in these publication packs.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev,publishing]'

export LLAMA_CPP_DIR=/path/to/llama.cpp
export GEMMA_MODEL=/path/to/compatible-gemma.gguf
export GEMMA_TOKENIZER_BUNDLE=/path/to/matching-gemma-tokenizer-bundle
./run-gemma.sh

export PYTHIA_MODEL=/path/to/pythia-1.4b
export REVIEW_MODEL="$GEMMA_MODEL"
./run-pythia.sh
```

The launchers resolve the project root from their own location. Optional paths
include `PYTHIA_TTL`, `GEMMA_LTL`, `GEMMA_TOKENIZER_BUNDLE`, `ROUTER_PATH`,
`PPKG_PATH`, `SESSION_ROOT`, `LLAMA_WEB_UI`, `WEB_HOST`, and `WEB_PORT`.

Gemma uses llama.cpp and the active `.ltl`. Pythia uses the semantic `.ttl` and
uses the configured frozen GGUF as a CPU structured reviewer by default. The
review request is separate from visible generation and uses neither TTL nor LTL.

The browser is the primary conversation interface. Terminal commands `/state`,
`/personality`, `/events`, `/save`, and `/quit` are secondary diagnostics.
Every session records transcript, events, metadata, P-cache state, and final
state under `sessions/`.
