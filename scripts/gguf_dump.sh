#!/bin/bash
#
# GGUF dump helper — pretty-print any GGUF file's header, KV pairs, and tensors.
# Usage: scripts/gguf_dump.sh MODEL
#
# Standalone: does NOT require the addon installed (gguf_dump.ijs lives in the
# repo). The dump prints arch KVs, per-tensor name/dims/etype/offset, and the
# data-section layout.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$(realpath "$1")"
JINSTALL="$( "$SCRIPT_DIR/jfind.sh" )"
JCONSOLE="${JCONSOLE:-$JINSTALL/jconsole.sh}"

"$JCONSOLE" <<EOSCRIPT
load '$SCRIPT_DIR/../gguf_dump.ijs'
gguf_dump_inference_ '$MODEL'
EOSCRIPT
