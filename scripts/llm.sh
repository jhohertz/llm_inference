#!/bin/bash
#
# LLM CLI helper — one-shot generation from the command line.
# Usage: scripts/llm.sh MODEL "PROMPT" [MAX_STEPS]
#   MODEL   GGUF file path (any supported arch: gemma3 / llama / qwen2)
#   PROMPT  input text
#   MAX_STEPS  optional cap on generated tokens; without it, generate until EOS
#
# Requires the addon installed: ./scripts/install_local.sh --force

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$1"
if [ -f "$MODEL" ]; then MODEL="$(realpath "$MODEL")"; fi   # local path -> abs; specs pass through
PROMPT="$2"
LIMIT="$3"
CHAT="${4:-0}"
JINSTALL="$( "$SCRIPT_DIR/jfind.sh" )"
JCONSOLE="${JCONSOLE:-$JINSTALL/jconsole.sh}"

"$JCONSOLE" "$SCRIPT_DIR/../llm_cli.ijs" "$MODEL" "$PROMPT" "${LIMIT:-100000}" "$CHAT"
