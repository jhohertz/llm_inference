#!/usr/bin/env bash
# jfind.sh — discover the J runtime install directory.
#
# J installs live in $HOME as ~/j9.x (e.g. ~/j9.6, ~/j9.7, ~/j9.8). This
# picks the highest version that has a console binary and prints its install
# dir to stdout (e.g. /home/me/j9.7). Consumers build paths off it:
#   "$JINSTALL/jconsole.sh"   (console wrapper; cds to the install dir)
#   "$JINSTALL/bin/jconsole"  (raw binary; keeps the caller's cwd)
#   "$JINSTALL/addons"        (addon install target)
#
# Override with $JINSTALL (must contain bin/jconsole or jconsole.sh).
set -euo pipefail

if [ -n "${JINSTALL:-}" ]; then
  if [ -x "$JINSTALL/bin/jconsole" ] || [ -x "$JINSTALL/jconsole.sh" ]; then
    echo "$JINSTALL"
    exit 0
  fi
  echo "jfind: JINSTALL=$JINSTALL has no jconsole binary" >&2
  exit 1
fi

# Dot-version installs: ~/j9.6, ~/j9.7, ~/j9.8 ...
for d in $(ls -d "$HOME"/j9.* 2>/dev/null | sort -V); do
  if [ -x "$d/bin/jconsole" ]; then
    echo "$d"
    exit 0
  fi
done

# Legacy numeric installs: ~/j903, ~/j807 ...
for d in $(ls -d "$HOME"/j9* 2>/dev/null | sort -V); do
  if [ -x "$d/bin/jconsole" ]; then
    echo "$d"
    exit 0
  fi
done

echo "jfind: no J runtime under \$HOME (looked for ~/j9.* and ~/j9* with bin/jconsole)" >&2
exit 1
