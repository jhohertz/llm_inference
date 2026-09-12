#!/usr/bin/env python3
# Copyright 2024 Google LLC (adapted for the J minja port)
#
# Golden generator for the J port of minja (util/minja.ijs).
#
# Oracle = Python's jinja2 (the engine minja is designed to match).
# Reads a JSON spec file of render cases and writes the rendered goldens.
#
# Usage:
#   scripts/minja_goldens.py cases.json goldens.json
#
# cases.json  = array of { "template": str, "bindings": obj, "options": {...} }
#   options may be omitted (defaults) or contain trim_blocks / lstrip_blocks /
#   keep_trailing_newline. bindings omitted => {}.
# goldens.json = array of rendered strings (one per case, in order).
#
# Mirrors reference/minja/scripts/render.py but batches many cases per run so
# the J test suite can bake in precomputed expected strings (project convention:
# oracles verified at dev time, not shelled out at test time).
import sys
import json
from pathlib import Path
from jinja2 import Environment
import jinja2.ext

def render(template, bindings, options):
    env = Environment(**options, extensions=[jinja2.ext.loopcontrols])
    return env.from_string(template).render(bindings or {})

def main():
    input_file, output_file = sys.argv[1:3]
    cases = json.loads(Path(input_file).read_text())
    out = []
    for case in cases:
        opts = case.get("options", {})
        out.append(render(case["template"], case.get("bindings", {}), opts))
    Path(output_file).write_text(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
