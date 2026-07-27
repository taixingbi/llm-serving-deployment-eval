#!/usr/bin/env bash
# Run one LLMPerf token benchmark against a resolved backend.
#
# Required env (set by resolve_endpoints.sh or caller):
#   OPENAI_API_BASE, OPENAI_API_KEY
#
# Required args via env or flags:
#   BACKEND, EXP_ID, MEAN_IN, MEAN_OUT, CONC
#
# Optional:
#   N_REQ (default 50), TIMEOUT (default 600), MODEL, LLMPERF_DIR, RESULTS_ROOT
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/load_env.sh"

# macOS often has no `python` on PATH; prefer venv then python3.
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV}/bin/python}"
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

BACKEND="${BACKEND:-}"
EXP_ID="${EXP_ID:-}"
MEAN_IN="${MEAN_IN:-550}"
MEAN_OUT="${MEAN_OUT:-150}"
CONC="${CONC:-1}"
N_REQ="${N_REQ:-50}"
TIMEOUT="${TIMEOUT:-600}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/results/raw}"
LLMPERF_DIR="${LLMPERF_DIR:-${ROOT_DIR}/vendor/llmperf}"

usage() {
  cat <<EOF
Usage:
  source scripts/resolve_endpoints.sh <backend>
  BACKEND=... EXP_ID=... MEAN_IN=... MEAN_OUT=... CONC=... ./scripts/run_one.sh

Config:
  Repo-root .env (see .env.example) is loaded automatically.

Env:
  LLMPERF_DIR   Path to cloned ray-project/llmperf (default: vendor/llmperf)
  N_REQ         max completed requests (default: 50)
  TIMEOUT       seconds (default: 600)
  SAMPLE_RESOURCES=1  also start scripts/sample_resources.py in background
EOF
}

if [[ -z "${BACKEND}" || -z "${EXP_ID}" ]]; then
  usage >&2
  exit 1
fi

if [[ -z "${OPENAI_API_BASE:-}" || -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_BASE and OPENAI_API_KEY must be set (source resolve_endpoints.sh)" >&2
  exit 1
fi

if [[ ! -f "${LLMPERF_DIR}/token_benchmark_ray.py" ]]; then
  cat >&2 <<EOF
ERROR: LLMPerf not found at ${LLMPERF_DIR}
Clone it once:
  git clone https://github.com/ray-project/llmperf.git "${ROOT_DIR}/vendor/llmperf"
EOF
  exit 1
fi

OUT_DIR="${RESULTS_ROOT}/${BACKEND}/${EXP_ID}"
mkdir -p "${OUT_DIR}"
export BACKEND EXP_ID MEAN_IN MEAN_OUT CONC N_REQ TIMEOUT MODEL OUT_DIR

SAMPLER_PID=""
cleanup() {
  if [[ -n "${SAMPLER_PID}" ]] && kill -0 "${SAMPLER_PID}" 2>/dev/null; then
    kill "${SAMPLER_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "${SAMPLE_RESOURCES:-0}" == "1" ]]; then
  python3 "${ROOT_DIR}/scripts/sample_resources.py" \
    --out "${OUT_DIR}/resources.jsonl" \
    --interval 1.0 &
  SAMPLER_PID=$!
fi

METADATA="backend=${BACKEND},exp_id=${EXP_ID},concurrency=${CONC}"

echo "Running LLMPerf → ${OUT_DIR}"
echo "  model=${MODEL} mean_in=${MEAN_IN} mean_out=${MEAN_OUT} conc=${CONC} n_req=${N_REQ}"

cd "${LLMPERF_DIR}"
# Prefer src/ on PYTHONPATH so we don't need pip install -e (LLMPerf pins Python <3.11).
export PYTHONPATH="${LLMPERF_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" token_benchmark_ray.py \
  --model "${MODEL}" \
  --mean-input-tokens "${MEAN_IN}" \
  --stddev-input-tokens 0 \
  --mean-output-tokens "${MEAN_OUT}" \
  --stddev-output-tokens 0 \
  --num-concurrent-requests "${CONC}" \
  --max-num-completed-requests "${N_REQ}" \
  --timeout "${TIMEOUT}" \
  --llm-api openai \
  --additional-sampling-params '{"temperature":0,"top_p":1.0}' \
  --metadata "${METADATA}" \
  --results-dir "${OUT_DIR}"

"${PYTHON_BIN}" - <<'PY'
import json, os, time
from pathlib import Path
sidecar = {
    "backend": os.environ["BACKEND"],
    "exp_id": os.environ["EXP_ID"],
    "mean_input_tokens": int(os.environ["MEAN_IN"]),
    "mean_output_tokens": int(os.environ["MEAN_OUT"]),
    "concurrency": int(os.environ["CONC"]),
    "max_num_completed_requests": int(os.environ["N_REQ"]),
    "timeout_s": int(os.environ["TIMEOUT"]),
    "model": os.environ["MODEL"],
    "openai_api_base": os.environ["OPENAI_API_BASE"],
    "finished_at": time.time(),
}
path = Path(os.environ["OUT_DIR"]) / "run.json"
path.write_text(json.dumps(sidecar, indent=2) + "\n")
print("Wrote", path)
PY
