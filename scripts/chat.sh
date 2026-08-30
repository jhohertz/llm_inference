#!/bin/bash
#
# Chat launcher — load a model and open an interactive J console for the
# persistent console chat (KV-cache resume across turns).
# Usage: scripts/chat.sh MODEL
#
# Requires the addon installed: ./scripts/install_local.sh --force
# After the model loads, type J commands at the prompt:
#   llm chat 'your message'              -> answer (per-arch default params)
#   llm chat_p ('msg' ; <temp;k;p;min_p) -> explicit params
#   chat_reset ''                        -> clear session + KV cache
#   exit ''                              -> leave the console
# The session persists across calls until chat_reset '' (or exit).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$1"
if [ -f "$MODEL" ]; then MODEL="$(realpath "$MODEL")"; fi   # local path -> abs; specs pass through
JINSTALL="$( "$SCRIPT_DIR/jfind.sh" )"
JCONSOLE="${JCONSOLE:-$JINSTALL/jconsole.sh}"

"$JCONSOLE" "$SCRIPT_DIR/../chat_launch.ijs" "$MODEL"
