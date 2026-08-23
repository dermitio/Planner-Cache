#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
GEMMA_MODEL="${GEMMA_MODEL:-}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-}"
GEMMA_LTL="${GEMMA_LTL:-${REPO_ROOT}/artifacts/gemma4-e4b-q8-llama.ltl}"
GEMMA_TOKENIZER_BUNDLE="${GEMMA_TOKENIZER_BUNDLE:-}"
ROUTER_PATH="${ROUTER_PATH:-${REPO_ROOT}/artifacts/canonical-p-v1.router}"
PPKG_PATH="${PPKG_PATH:-${REPO_ROOT}/state/personality.ppkg}"
SESSION_ROOT="${SESSION_ROOT:-${REPO_ROOT}/sessions}"
LLAMA_WEB_UI="${LLAMA_WEB_UI:-${LLAMA_CPP_DIR}/build/tools/ui/dist}"
WEB_HOST="${WEB_HOST:-127.0.0.1}"
WEB_PORT="${WEB_PORT:-0}"
WEB_BROWSER_ARGS=()
if [[ "${NO_BROWSER:-0}" == "1" ]]; then
  WEB_BROWSER_ARGS+=(--no-browser)
fi
LLAMA_PID_FILE="$(mktemp "${TMPDIR:-/tmp}/planner-cache-llama.XXXXXX.pid")"

cleanup() {
  if [[ -s "${LLAMA_PID_FILE}" ]]; then
    server_pid="$(<"${LLAMA_PID_FILE}")"
    if [[ "${server_pid}" =~ ^[0-9]+$ ]] && kill -0 "${server_pid}" 2>/dev/null; then
      kill -TERM -- "-${server_pid}" 2>/dev/null || true
    fi
  fi
  rm -f -- "${LLAMA_PID_FILE}"
}

trap cleanup EXIT
trap 'exit 130' INT TERM

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Planner Cache startup failed: $2 not found: $1" >&2
    exit 2
  fi
}

require_file "${PYTHON_BIN}" "Python virtual environment"
if [[ -z "${GEMMA_MODEL}" ]]; then
  echo "Planner Cache startup failed: set GEMMA_MODEL to the compatible GGUF" >&2
  exit 2
fi
require_file "${GEMMA_MODEL}" "Gemma GGUF"
if [[ -z "${LLAMA_CPP_DIR}" ]]; then
  echo "Planner Cache startup failed: set LLAMA_CPP_DIR to a llama.cpp build" >&2
  exit 2
fi
if [[ -z "${GEMMA_TOKENIZER_BUNDLE}" ]]; then
  echo "Planner Cache startup failed: set GEMMA_TOKENIZER_BUNDLE to the matching tokenizer bundle" >&2
  exit 2
fi
require_file "${LLAMA_CPP_DIR}/build/bin/llama-server" "llama-server"
require_file "${LLAMA_CPP_DIR}/build/bin/llama-cli" "llama-cli"
require_file "${GEMMA_LTL}" "Gemma .ltl artifact"
require_file "${GEMMA_TOKENIZER_BUNDLE}/tokenizer.json" "Gemma tokenizer bundle"
require_file "${ROUTER_PATH}" "canonical .router artifact"
require_file "${LLAMA_WEB_UI}/index.html" "llama-server Web UI"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m pcm.planner.web_chat gemma \
  --repo-root "${REPO_ROOT}" \
  --model "${GEMMA_MODEL}" \
  --adapter "${GEMMA_LTL}" \
  --tokenizer-bundle "${GEMMA_TOKENIZER_BUNDLE}" \
  --router "${ROUTER_PATH}" \
  --llama-cpp-dir "${LLAMA_CPP_DIR}" \
  --ppkg "${PPKG_PATH}" \
  --session-root "${SESSION_ROOT}" \
  --llama-pid-file "${LLAMA_PID_FILE}" \
  --web-ui-path "${LLAMA_WEB_UI}" \
  --web-host "${WEB_HOST}" \
  --web-port "${WEB_PORT}" \
  "${WEB_BROWSER_ARGS[@]}" \
  --max-new-tokens 128 \
  "$@"
