#!/bin/bash
#
# OpenAI-compatible HTTP server launcher (non-blocking jsocket event loop).
# Exposes POST /v1/chat/completions (plain + streamed SSE) driven by our
# llm_inference real generation.
# Usage: scripts/llm_server.sh [MODEL]   (default model qwen3-0.6b)
#
# Requires the addon installed: ./scripts/install_local.sh --force
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:-qwen3-0.6b}"
JINSTALL="$( "$SCRIPT_DIR/jfind.sh" )"
JCONSOLE="${JCONSOLE:-$JINSTALL/bin/jconsole}"
PORT="${PORT:-8790}"

"$JCONSOLE" "$SCRIPT_DIR/../http/run.ijs" "$MODEL"
