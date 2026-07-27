#!/usr/bin/env bash
# Resolve OPENAI_API_BASE and OPENAI_API_KEY for a backend.
# Usage: source scripts/resolve_endpoints.sh <selfhost|ecs|bedrock>
# Works when sourced from bash or zsh.
# Intentionally avoids `set -u` / `set -e` so sourcing does not alter
# the caller's interactive shell options (e.g. Cursor/VS Code RPROMPT).

# BASH_SOURCE is bash-only; zsh uses ${(%):-%x} when sourced.
if [[ -n "${BASH_VERSION:-}" ]]; then
  _RESOLVE_SCRIPT="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  # shellcheck disable=SC2296
  _RESOLVE_SCRIPT="${(%):-%x}"
else
  _RESOLVE_SCRIPT="$0"
fi
ROOT_DIR="$(cd "$(dirname "${_RESOLVE_SCRIPT}")/.." && pwd)"
unset _RESOLVE_SCRIPT
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/load_env.sh"

BACKEND="${1:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"

if [[ -z "${BACKEND}" ]]; then
  echo "Usage: source scripts/resolve_endpoints.sh <selfhost|ecs|bedrock>" >&2
  return 1 2>/dev/null || exit 1
fi

case "${BACKEND}" in
  selfhost)
    BASE="${SELFHOST_URL:-http://192.168.86.176:30080}"
    BASE="${BASE%/}"
    export OPENAI_API_BASE="${BASE}/v1"
    export OPENAI_API_KEY="${SELFHOST_API_KEY:-EMPTY}"
    ;;
  ecs)
    if [[ -z "${INFERENCE_API_KEY:-}" ]]; then
      echo "ERROR: set INFERENCE_API_KEY in .env (see .env.example)" >&2
      return 1 2>/dev/null || exit 1
    fi
    SERVICE_URL="$(aws cloudformation describe-stacks \
      --region "${AWS_REGION}" \
      --stack-name ecs-inference-mvp \
      --query "Stacks[0].Outputs[?OutputKey=='ServiceUrl'].OutputValue" \
      --output text)" || return 1 2>/dev/null || exit 1
    SERVICE_URL="${SERVICE_URL%/}"
    export OPENAI_API_BASE="${SERVICE_URL}/v1"
    export OPENAI_API_KEY="${INFERENCE_API_KEY}"
    ;;
  bedrock)
    if [[ -z "${INFERENCE_API_KEY:-}" ]]; then
      echo "ERROR: set INFERENCE_API_KEY in .env (see .env.example)" >&2
      return 1 2>/dev/null || exit 1
    fi
    FUNCTION_URL="$(aws cloudformation describe-stacks \
      --region "${AWS_REGION}" \
      --stack-name bedrock-inference-mvp \
      --query "Stacks[0].Outputs[?OutputKey=='InferenceFunctionUrl'].OutputValue" \
      --output text)" || return 1 2>/dev/null || exit 1
    # Function URL already includes trailing slash
    export OPENAI_API_BASE="${FUNCTION_URL}v1"
    export OPENAI_API_KEY="${INFERENCE_API_KEY}"
    ;;
  *)
    echo "ERROR: unknown backend '${BACKEND}' (expected selfhost|ecs|bedrock)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

if [[ "${RESOLVE_QUIET:-0}" != "1" ]]; then
  echo "BACKEND=${BACKEND}"
  echo "OPENAI_API_BASE=${OPENAI_API_BASE}"
  # Do not print the API key
fi
