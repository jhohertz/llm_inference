#!/bin/bash
#
# lint.sh — jlinter (addons/tmcguire/jlinter, headless debug/lint wrapper)
# over the checkout's runtime .ijs files.
#   lint.sh                     lint all runtime files, print report
# Exits non-zero if any runtime file FAILS TO LOAD (a real syntax/load
# regression). debug/lint's "undefined name" findings are expected for a
# multi-file addon (each file is checked standalone) and are informational;
# the load-probe is the gate.
#
# Requires: addons/tmcguire/jlinter installed (JAL: install 'tmcguire/jlinter')
#           + the addon installed for load-probing: ./scripts/install_local.sh --force

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JINSTALL="$( "$SCRIPT_DIR/jfind.sh" )"
J="${JCONSOLE:-$JINSTALL/bin/jconsole}"
cd "$SCRIPT_DIR/.."

out=$("$J" <<'EOFILE' 2>&1
load './tests/j/lint_all.ijs'
lint_all_z_ ''
EOFILE
)
echo "$out"
code=$(echo "$out" | grep -oP 'lint exit:\s+\K[0-9]+' | tail -1 || true)
code=${code:-1}
exit "$code"
