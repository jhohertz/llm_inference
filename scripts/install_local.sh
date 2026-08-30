#!/usr/bin/env bash
# install_local.sh — copy the addon's runtime FILES from this checkout into ~addons.
# jlinter-style dev workflow: edit the checkout, re-run this, then load
# 'llm/inference' from any J console.
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JINSTALL="$( "$BASE/scripts/jfind.sh" )"
ADDONS="${JADDONS:-$JINSTALL/addons}"
DEST="$ADDONS/llm/inference"

# Runtime FILES from manifest.ijs (manifest itself is copied too).
# Tests are NOT installed — they live in the checkout and run from there.
ITEMS=(
  manifest.ijs
  README.md
  inference.ijs
  gguf_dump.ijs
  llm_cli.ijs
  chat_launch.ijs
  models/gemma3.ijs
  models/llama.ijs
  models/granite.ijs
  models/qwen2.ijs
  models/qwen3.ijs
  models/qwen35.ijs
  models/ernie.ijs
  models/lfm2.ijs
  tokenizers/tokenizer_llama3.ijs
  tokenizers/tokenizer_gpt2.ijs
  tokenizers/tokenizer_spm.ijs
  kernels/jfloat.ijs
  gguf/gguf.ijs
  gguf/quant.ijs
  gguf/quant_tables.ijs
  util/kv_cache.ijs
  util/llm_core.ijs
  util/sampler.ijs
  util/chat.ijs
  util/models.ijs
  util/llmobj.ijs
)

rm -rf "$DEST"
mkdir -p "$DEST"
for item in "${ITEMS[@]}"; do
  mkdir -p "$DEST/$(dirname "$item")"
  cp "$BASE/$item" "$DEST/$item"
done

echo "installed llm/inference -> $DEST"
