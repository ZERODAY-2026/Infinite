#!/usr/bin/env bash
set -euo pipefail
printf 'Python: '; python3 --version
printf '\nOllama: '; command -v ollama || true
if command -v ollama >/dev/null 2>&1; then
  ollama list
fi
printf '\nSource directory default: %s\n' "${LEGAL_SOURCES_DIR:-/home/govardhan/rental-legal-ai/legal_sources/india_central}"
