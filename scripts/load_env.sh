#!/usr/bin/env bash
# Load repo-root .env into the current shell (KEY=VALUE lines).
# Usage: source scripts/load_env.sh
# Works when sourced from bash or zsh.
# Intentionally avoids `set -u` / `set -e` so sourcing does not alter
# the caller's interactive shell options (e.g. Cursor/VS Code RPROMPT).

# BASH_SOURCE is bash-only; zsh uses ${(%):-%x} when sourced.
if [[ -n "${BASH_VERSION:-}" ]]; then
  _LOAD_ENV_SCRIPT="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  # shellcheck disable=SC2296
  _LOAD_ENV_SCRIPT="${(%):-%x}"
else
  _LOAD_ENV_SCRIPT="$0"
fi
_LOAD_ENV_ROOT="$(cd "$(dirname "${_LOAD_ENV_SCRIPT}")/.." && pwd)"
unset _LOAD_ENV_SCRIPT
_LOAD_ENV_FILE="${ENV_FILE:-${_LOAD_ENV_ROOT}/.env}"

if [[ -f "${_LOAD_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${_LOAD_ENV_FILE}"
  set +a
elif [[ "${LOAD_ENV_REQUIRED:-0}" == "1" ]]; then
  echo "ERROR: missing ${_LOAD_ENV_FILE} (copy from .env.example)" >&2
  unset _LOAD_ENV_ROOT _LOAD_ENV_FILE
  return 1 2>/dev/null || exit 1
fi

unset _LOAD_ENV_ROOT _LOAD_ENV_FILE
