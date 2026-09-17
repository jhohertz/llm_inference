#!/bin/bash
#
# Chat TUI launcher — a minimal terminal chat UI driven by llm_inference
# directly (j-kvm `vt` raw-mode + key reads). Stateful chat: the session
# (chat_session_g) + KV cache carry across turns via chat_core_stream (one
# batched prefill of the new segment), and the reply streams live.
# Usage: scripts/chat_tui.sh [MODEL]        (default model qwen3-0.6b)
#
# Requires the addon installed: ./scripts/install_local.sh --force
# Controls: type a message + Enter to send; /reset clears the session + KV
# cache; Backspace to edit; Ctrl-C or type `exit` to quit.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:-qwen3-0.6b}"
JINSTALL="$( "$SCRIPT_DIR/jfind.sh" )"
JCONSOLE="${JCONSOLE:-$JINSTALL/bin/jconsole}"

"$JCONSOLE" "$SCRIPT_DIR/../chat_tui.ijs" "$MODEL"
