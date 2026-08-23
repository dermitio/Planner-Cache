#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
PYTHIA_MODEL="${PYTHIA_MODEL:-}"
PYTHIA_TTL="${PYTHIA_TTL:-${REPO_ROOT}/artifacts/pythia-1.4b-final-layer.ttl}"
ROUTER_PATH="${ROUTER_PATH:-${REPO_ROOT}/artifacts/canonical-p-v1.router}"
PPKG_PATH="${PPKG_PATH:-${REPO_ROOT}/state/personality.ppkg}"
SESSION_ROOT="${SESSION_ROOT:-${REPO_ROOT}/sessions}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-}"
LLAMA_WEB_UI="${LLAMA_WEB_UI:-${LLAMA_CPP_DIR}/build/tools/ui/dist}"
WEB_HOST="${WEB_HOST:-127.0.0.1}"
WEB_PORT="${WEB_PORT:-0}"
WEB_BROWSER_ARGS=()
if [[ "${NO_BROWSER:-0}" == "1" ]]; then
  WEB_BROWSER_ARGS+=(--no-browser)
fi
REVIEW_MODEL="${REVIEW_MODEL:-${GEMMA_MODEL:-}}"
REVIEW_GPU_LAYERS="${REVIEW_GPU_LAYERS:-0}"
REVIEW_PID_FILE="$(mktemp "${TMPDIR:-/tmp}/planner-cache-review.XXXXXX.pid")"

cleanup() {
  if [[ -s "${REVIEW_PID_FILE}" ]]; then
    reviewer_pid="$(<"${REVIEW_PID_FILE}")"
    if [[ "${reviewer_pid}" =~ ^[0-9]+$ ]] && kill -0 "${reviewer_pid}" 2>/dev/null; then
      kill -TERM -- "-${reviewer_pid}" 2>/dev/null || true
    fi
  fi
  rm -f -- "${REVIEW_PID_FILE}"
}

trap cleanup EXIT
trap 'exit 130' INT TERM

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Planner Cache startup failed: $2 not found: $1" >&2
    exit 2
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Planner Cache startup failed: $2 not found: $1" >&2
    exit 2
  fi
}

require_file "${PYTHON_BIN}" "Python virtual environment"
require_dir "${PYTHIA_MODEL}" "Pythia model"
if [[ -z "${REVIEW_MODEL}" ]]; then
  echo "Planner Cache startup failed: set REVIEW_MODEL or GEMMA_MODEL to the structured-review GGUF" >&2
  exit 2
fi
if [[ -z "${LLAMA_CPP_DIR}" ]]; then
  echo "Planner Cache startup failed: set LLAMA_CPP_DIR to a llama.cpp build" >&2
  exit 2
fi
require_file "${PYTHIA_TTL}" "Pythia .ttl artifact"
require_file "${ROUTER_PATH}" "canonical .router artifact"
require_file "${LLAMA_WEB_UI}/index.html" "llama-server Web UI"
require_file "${LLAMA_CPP_DIR}/build/bin/llama-server" "review llama-server"
require_file "${REVIEW_MODEL}" "structured review model"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m pcm.planner.web_chat pythia \
  --repo-root "${REPO_ROOT}" \
  --model "${PYTHIA_MODEL}" \
  --adapter "${PYTHIA_TTL}" \
  --router "${ROUTER_PATH}" \
  --ppkg "${PPKG_PATH}" \
  --session-root "${SESSION_ROOT}" \
  --web-ui-path "${LLAMA_WEB_UI}" \
  --web-host "${WEB_HOST}" \
  --web-port "${WEB_PORT}" \
  --review-model "${REVIEW_MODEL}" \
  --review-llama-cpp-dir "${LLAMA_CPP_DIR}" \
  --review-gpu-layers "${REVIEW_GPU_LAYERS}" \
  --review-pid-file "${REVIEW_PID_FILE}" \
  "${WEB_BROWSER_ARGS[@]}" \
  "$@"
