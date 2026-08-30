# LLM in J — Project Guide

Operational guide: file map, how to run, and the hard-won J gotchas.
For architecture/implementation details see **docs/ARCHITECTURE.md**. For
status, roadmap, and the jforc chapter reviews see **PLAN.md**.

## Overview

Generic GGUF-based language model inference in J (J9.7), multi-model.
A model-agnostic GGUF parser loads weights; each architecture has its own
module (`models/gemma3.ijs`, `models/llama.ijs`, `models/granite.ijs`, `models/qwen2.ijs`, `models/qwen3.ijs`) implementing the
forward pass. Inference logits are verified exact vs `llama-cpp-python`.

## Addon structure & locale

The project is structured as a J addon: `FOLDER=: 'llm/inference'` (repo will
be renamed `llm_inference`). All public names live in the **`inference`**
locale (`coclass 'inference'` in every script). Internal deps use
`require 'llm/inference/...'` (no `.ijs` extension), so the addon must be
installed to `~addons` first (see `scripts/install_local.sh`). Call verbs from
any locale with the `_inference_` suffix; after `load 'llm/inference'` you may
also `cocurrent <'inference'` and use simple names. Tests run in the
`inference` locale (each test file starts with `coclass 'inference'`).

## Files

| File | Purpose |
|------|---------|
| `manifest.ijs` | Addon manifest (`FOLDER=: 'llm/inference'`, FILES list) |
| `inference.ijs` | Entry point — loads the category modules (defaults to gemma3); `load 'llm/inference'` |
| `gguf_dump.ijs` | **Utility**: pretty-print GGUF file info. `load 'gguf_dump.ijs'; gguf_dump_inference_ 'path.gguf'` (shell helper `scripts/gguf_dump.sh`) |
| `llm_cli.ijs` | **CLI**: one-shot generate from the command line. `jconsole llm_cli.ijs MODEL "PROMPT" [MAX_STEPS] [CHAT]` — if PROMPT starts with `@`, the rest is read as a prompt FILE (curl-style). (shell helper `scripts/llm.sh`, which forwards the 4th arg as chat mode) |
| `chat_launch.ijs` | **Chat console**: load a model and stay in an interactive J REPL (`llm_z_`, `chat_z_`, `chat_p_z_`, `chat_reset_z_` exposed in the GLOBAL/base locale so `llm chat 'msg'` works as-is at the prompt — a script-file `cocurrent` does NOT persist to the REPL). `jconsole chat_launch.ijs MODEL` (shell helper `scripts/chat.sh`) |
| `gguf/gguf.ijs` | GGUF parser, tensor loading (F32/F16/BF16), KV pair extraction |
| `gguf/quant.ijs` / `gguf/quant_tables.ijs` | Quant decoders + block tables — aligned with the loader (packed-quant handling is part of gguf today) |
| `kernels/jfloat.ijs` | Float kernels: matmul, `linear`, gelu, silu, swiglu, rms_norm, rope (`rope_apply2`/`rope_apply2_neox`/`rope_apply2_t`), softcap |
| `util/kv_cache.ijs` | Persistent KV cache: pre-allocated flat arrays (`k_cache_g`/`v_cache_g` = `(n_layers*eff_seq, n_kv*hd)`, one per kind), in-place `kv_create`/`kv_write`/`kv_write_rows`/`kv_read`/`kv_reset` + `kv_meta`/`kv_pos_g`/`kv_max_seq_g` (low-memory context override); session-persistent (alloc once, reset between generations); monadic, no threading |
| `util/llm_core.ijs` | Generic helpers: llm accessors (incl. `llm_arch`), `get_tensor_cached_d`, `embed_tokens`, `output_head`, `sample_from`, `infer_args`, `gen_args` |
| `models/gemma3.ijs` | Gemma3 270M: attention+KV, FFN, blocks, `gem3_infer`/`gem3_generate`, `gem3_load` |
| `models/llama.ijs` | **Generic llama arch** (SmolLM2 + Llama-3.2): standard decoder, GQA, SwiGLU, interleaved RoPE, dims read from GGUF; tokenizer/chat dispatch on `tokenizer.ggml.pre` (`llama_load`/`llama_infer`/`llama_generate`) |
| `models/granite.ijs` | Granite-4.0-350m (granite arch): standard decoder + Granite 4.0 scaling (embed*12, residual*0.263 on attn/ffn outputs, scores*0.015625, logits/4), tied embeddings, dbrx pre (= llama3 regex), granite chat template (`granite_load`/`granite_infer`/`granite_generate`) |
| `models/qwen2.ijs` | Qwen2.5-Coder (qwen2 arch): standard decoder, GQA, SwiGLU, NEOX RoPE, Q/K/V biases, `qw2_load` |
| `models/qwen3.ijs` | Qwen3-0.6B (qwen3 arch): qwen2 + per-head Q/K RMSNorm before RoPE, NO QKV biases, `qw3_load` |
| `models/qwen35.ijs` | Qwen3.5-0.8B (qwen35 arch): hybrid — 6 full-attention (il+1%4==0) + 18 gated-delta-net SSM layers; fused Q+GATE, sigmoid gate, conv1d, L2-norm q/k, sequential delta-net recurrence, `rs_*` recurrent-state cache, MTP blk.24 out of scope (block_count = block_count − nextn_predict_layers), `qw35_load` |
| `models/ernie.ijs` | ERNIE-4.5-0.3B (ernie4_5 arch): standard decoder byte-for-byte the llama arch (GQA 16:2, SwiGLU, interleaved RoPE, 1/sqrt(hd), TIED embeddings, NO scaling KVs) — reuses llama.ijs forward verbs (`ernie_run_blocks`/`_b` aliases); SPM tokenizer, ERNIE chat template (`<|begin_of_sentence|>` cls, User:/Assistant:/system, `'Assistant: '` gen prompt), stop on `</s>`(2)+`<|end_of_sentence|>`(100272), `ernie_load` |
| `models/lfm2.ijs` | LFM2-350M/700M/1.2B (lfm2 arch): hybrid — 6 attention + 10 shortconv layers, new conv component + `lfm2` pre-tokenizer (`lfm2_load`/`lfm2_infer`) |
| `tokenizers/tokenizer_llama3.ijs` / `tokenizers/tokenizer_gpt2.ijs` | BPE tokenizers (llama3-style; gpt2 byte-level) |
| `tokenizers/tokenizer_spm.ijs` | **SentencePiece tokenizer** (llama.cpp `llm_tokenizer_spm` bigram-merge): max-heap over token scores, `▁`-escape, add_space_prefix prepend, `<0xXX>` byte fallback; used by ERNIE (model='llama' → SPM) |
| `util/sampler.ijs` | Temperature / top-k / top-p / min-p sampling |
| `util/chat.ijs` | **Chat-template inference**: per-arch dispatch, `chat_generate`, persistent console chat (`chat`/`chat_p` + `chat_session_g`), prompt-token stripping, `chat_msg` helper |
| `util/models.ijs` | **Model catalog + spec resolution + downloader**: `model_path`/`model_download`/`model_target`/`model_list`/`model_roles`/`model_file`; registers `~models` as `~user/models` (per-user, NOT the install dir); downloads via `web/gethttp` |
| `util/llmobj.ijs` | OOP proof: wrap a loaded LLM in a J object (`conew 'llmobj'`, `infer__obj`) |
| `scripts/jfind.sh` | Discover the J runtime dir in `$HOME` (`~/j9.x`, e.g. `~/j9.7`); prints the install dir |
 | `scripts/install_local.sh` | Dev workflow: copy FILES into the discovered J runtime's addons dir (`$JINSTALL/addons/llm/inference/`, jlinter-style) |
 | `scripts/lint.sh` | **jlinter wrapper**: lint the checkout's runtime `.ijs` files via `tmcguire/jlinter` (headless `debug/lint`). Exits non-zero if a runtime file FAILS TO LOAD (real syntax/load regression — the load-probe gate). Requires `addons/tmcguire/jlinter` installed (JAL) + `install_local.sh` for load-probing. |
 | `tests/j/lint_all.ijs` | jlinter driver: lint + load-probe the runtime files; `lint_all_z_ ''` prints the report and sets `LINT_EXIT_z_`. Test files excluded (debug/lint loads them, and they run suites on load); `llm_cli.ijs`/`chat_launch.ijs` are entry-points (load=0 expected, not gated). |
 | `tests/j/` | Test suites + `run_all_tests.sh` master runner (runs a final **J LINT CHECK** via `scripts/lint.sh`) + `pm_fixture.ijs` (jpm profiling) |

## How to Run

```bash
# Install the addon once (re-run after editing checkout files)
./scripts/install_local.sh --force

# Quick test (run from the checkout root; jfind.sh discovers the J runtime)
J="$(./scripts/jfind.sh)"   # e.g. /home/me/j9.7
"$J/bin/jconsole" << 'EOFILE'
load 'llm/inference'
llm =. load_gguf_to_llm_inference_ 'gemma-3-270m-it'
result =. llm infer_simple_inference_ 'hello world'
echo result
generated =. llm generate_simple_inference_ ('hello world' ; 20)
echo generated
EOFILE
```

**ALWAYS re-run `./scripts/install_local.sh --force` after editing any checkout file before running tests** — `tests/j/*.ijs` load the checkout `inference.ijs` but its `require 'llm/inference/...'` resolves to the INSTALLED addon (`~addons`), not the checkout. Tests exercise the installed copy; stale installs make tests look wrong.

`scripts/jfind.sh` discovers the J runtime under `$HOME` (`~/j9.x`, e.g.
`~/j9.7`) — override with `$JINSTALL`. The console wrapper
`$JINSTALL/jconsole.sh` `cd`s to the J install dir, so run from the checkout
root with `$JINSTALL/bin/jconsole` for `./`-relative paths; `require`/`load`
with addon names (no `.ijs` extension) for library scripts.
`scripts/llm.sh` / `chat.sh` / `gguf_dump.sh` derive paths from the checkout
and discover J automatically.

# Lint (jlinter / debug/lint)
```bash
# Requires addons/tmcguire/jlinter (JAL: install 'tmcguire/jlinter') + the addon installed.
./scripts/lint.sh        # lint the runtime .ijs files; exit non-zero if one fails to LOAD
# run_all_tests.sh runs this as its final "J LINT CHECK" step.
```
debug/lint checks each file STANDALONE, so cross-file "undefined name" findings
are expected (multi-file addon) and are informational; the LOAD-PROBE is the
gate — a runtime file that fails to load is a real syntax/load regression.
Test files are not linted (debug/lint loads them, and they run suites on load).


## Inference / Generation Interface (one canonical form)

Each architecture module exposes dyadic verbs in a **single canonical boxed
form** plus a `_simple` wrapper for default params. From any locale the verbs
carry the `_inference_` suffix (e.g. `infer_inference_`); inside the
`inference` locale simple names work:

 - `llm <name>_infer (text ; <temp;k;p;min_p>)` → `<tokens; pred_tok; decoded; logits>`
 - `llm <name>_infer_simple text` → greedy defaults (`0 0 0.95 0.0`)
 - `llm <name>_generate (text ; max_steps ; <temp;k;p;min_p>)` → decoded answer
 - `llm <name>_generate_simple (text ; max_steps)` → greedy defaults

**Single-shot generation frames the prompt** — `*_generate` wraps the text in
the arch's chat template (single user message) so instruct models emit their
stop tokens, and stops on the arch's `*_stop_tokens` list (not raw EOS). It
returns the answer only (prompt tokens dropped). `infer` stays raw (single
forward pass, no template) for logits verification.

**EOS / stop handling**: `gen_loop_core` stops when the sampled token is in the
stop list — gemma3 `tokenizer_eos` (106) + token 1 (`gem3_stop_tokens`);
qwen2/smollm2/llama32 `tokenizer_eos_g` (`<|im_end|>` 151645 / 2 /
`<|eot_id|>` 128009). The llama3 tokenizer stores eos at index 4
(`tokenizer_eos`); the gpt2 tokenizer at index 2 (`tokenizer_eos_g`). Pass a
large `max_steps` (e.g. 100000, the CLI default) to generate "until the model
decides it's done". Note: gemma3 under sampling (temp 1.0) sometimes emits a
newline instead of `<end_of_turn>` (106) — a "natural stop" is then missed
(small-model behavior; llama.cpp is identical).

**Chat interface (Phase 1.1)** — per-arch chat-template inference:
- `llm chat_generate (messages ; max_steps ; <temp;k;p;min_p>)` → answer text.
  `messages` = boxed list of message boxes; each = `<role ; content>` built
  with `(<role) , <content` (or `role chat_msg content`). Multi-turn renders the
  full history each call (stateless).
- **Crude console chat (persistent, option B)**: `llm chat 'next message'` →
  answer text with per-arch default params; `llm chat_p ('msg' ; <temp;k;p;min_p>)`
  for explicit params. The session (`chat_session_g`) + KV cache persist across
  calls — the next turn re-renders the history ONLY to tokenize the new segment,
  verifies the prefix matches the stored token stream, then **resumes from the
  KV cache** (ONE batched prefill of the new segment through `*_run_blocks_b`).
  `chat_reset ''` clears the session + cache. Session = `<arch; messages;
  total_tokens; cur_pos; max_steps; params>` (never holds a cache ref — a second
  ref would defeat the in-place amend). If the tokenizer round-trip drifts, it
  falls back to a full fresh re-render (correct, slower).
- `llm chat_generate_simple (messages ; max_steps)` → per-arch default params.
- Arch verbs: `gem3_chat_prompt` / `qw2_chat_prompt` / `llama_chat_prompt`
  (render), `gem3_default_params` (`1.0 64 0.95 0.001`) / `qw2_default_params`
  (`0 0 0.95 0.0`) / `llama_default_params`, `gem3_stop_tokens` (eos + 1) /
  `qw2_stop_tokens` / `llama_stop_tokens`.
- **Generation is ONE unified `gen_loop_core`** (llm_core.ijs), replacing the
  four `*_gen_loop` copies: dispatches scale (`%: emb_len` gemma3, else 1) +
  `*_run_blocks`/`*_run_blocks_b` by `llm_arch`. y = `<tokens; start_pos;
  max_steps; temp; k; p; min_p; stop_list>`; `start_pos=''` → fresh
  (kv_create + batched prefill at 0), `start_pos=<n>` → resume (cache exists;
  ONE batched prefill at n — `*_attention_b` is cache-prefix aware: RoPE at
  `start_pos+i.L`, prepends the prefix rows via `kv_read`, writes the batch
  K/V at `start_pos`, and scores over the combined keys). The cache is
  session-persistent: `kv_create` allocates once per session and reuses the
  buffer on repeat calls (reset `kv_pos_g`); generation stops at `eff_seq`
  (`min(model ctx, kv_max_seq_g)`). Thin `*_generate`
  wrappers call it fresh.
- **Batched generation** — `gen_loop_batch` (llm_core.ijs) runs ONE forward
  per decode step over B sequences; the cache gets a B-axis (`kv_batch_g`,
  `kv_seq_g`; base = `(layer*kv_batch_g + seq)*eff_seq`). Each arch adds
  `*_attention_bd`/`*_block_forward_bd`/`*_run_blocks_bd` + `*_generate_batch`
  (`llm <name>_generate_batch (prompts ; max_steps ; <temp;k;p;min_p>)` →
  boxed answers; prompts = boxed list). Step 0 predicts from the prefill-last
  hidden and does NOT advance `cur_pos` (mirror `gen_loop_core` — re-embedding
  at pos L duplicated the last prompt K/V and shifted outputs); later steps
  embed the previous token and run `rb_bd` at `cur_pos`. lfm2 conv / qwen35
  delta-net recurrent caches are per-sequence (`kv_batch_g` dimension) and
  force-reset between generations via `lf2_conv_reset`/`rs_reset` dispatched
  in the fresh paths — a guarded-but-unreset cache made batch-after-single
  inherit stale conv state and diverge. Verified batch==single for all 8
  arches (`tests/j/test_batched.ijs`); ~1.4-2.8x per-seq throughput at small
  ctx. NOTE: big-ctx models (qwen3.5 ctx=262144 → ~51GB/seq KV at full ctx)
  need `kv_max_seq_g` bounded for batched tests.
- **`*_run_blocks` (single-token) must return `<row>` (boxed), like the other
  arches** — `gen_loop_core`'s resume step does `hidden =. > 0 { result`.
  `qw35_run_blocks` originally returned the (emb,) row unboxed, so
  `> 0 { result` took the FIRST ELEMENT (a scalar) — the resume step fed a
  scalar to output_head and every generated token after the first was garbage
  ("The His His His..." instead of "The capital of France is Paris."). Fixed
  to `(< > 0 { state)`; `qw35_infer` opens it (`> 0 { result`).
- **`kv_write_rows` base must be `((a * eff_seq) + start)`, NOT `a * eff_seq
  + start`** (precedence gotcha: the latter is `a * (eff_seq + start)`).
  Harmless at start=0 (all pre-resume callers), but resume prefill writes at
  start>0 and would clobber the cache prefix.
- CLI chat mode: `jconsole llm_cli.ijs MODEL "PROMPT" MAX_STEPS 1` (single-turn;
  the console REPL chat is `llm chat '...'`).
- **BOS is per-arch**: llama3 tokenizer prepends bos (gemma chat includes it);
  gpt2 tokenizer prepends none (qwen2 chat omits it; smollm2's `<|im_start|>`=1
  doubles as bos). Llama-3.2 (`add_bos_token` is true in the GGUF): the llama
  module's `llama_tokenize` prepends bos (128000) for `pre='llama-bpe'` models,
  and the llama3 chat template renders NO `<|begin_of_text|>` marker — so the
  stream matches llama.cpp exactly (bos once). Raw infer gets the same bos.
- **Llama-3.2 tokenizer is GPT-2-style, not SentencePiece**: the llama32 vocab
  marks spaces with `Ġ` (U+0120, the GPT-2 byte-encoding), NOT ▁ (U+2581), and
  needs real BPE merges. `tokenizer_gpt2.ijs` stores `tokenizer.ggml.pre` at
  container index 9 (`tokenizer_pre_g`) and `gpt2_tokenize` dispatches the
  pre-tokenizer (`llama3_pre_tokenize` for 'llama-bpe', else `gpt2_pre_tokenize`).
  The old direct-lookup + ▁-replacement path (gemma) byte-falls-back on llama32.
- **Llama-3.2 chat template ALWAYS emits a system block** even with no system
  message (`Cutting Knowledge Date: December 2023\nToday Date: <date>\n\n` +
  `<|eot_id|>`), and the date is dynamic — llama-cpp-python injects
  `strftime('%d %b %Y')`. `llama32_chat_date_g` pins it for stable oracles;
  `llama32_today_date` formats J's `6!:0 'DD-MM-YYYY'` via a month table.
  `llama_chat_prompt` dispatches on `llama_tokenizer_pre_g` (set by `llama_load`).
- **Granite 4.0 scaling is data-driven, not llama-standard**: `granite.ijs`
  reads `embedding_scale` 12 (input embeddings *12), `residual_scale` 0.263
  (per layer: attn_out*0.263 + input, then ffn_out*0.263 + that), `attention.scale`
  0.015625 (scores, NOT 1/sqrt(hd)), `logit_scale` 4 (lm_head logits /4 — so
  `gen_loop_core` has a per-arch `logit_div` applied after `output_head`).
  Stored in mi at indices 12..15 (`granite_mi_*`). `head_count_kv` is an ARRAY
  KV (all 4s) — take `{.` of `kv_array`, not `kv_uint` (returns _1). Tied
  embeddings: the GGUF has NO `output.weight`; lm_head = `token_embd.weight`.
  `granite.rope.freq_base` is 1e7 = 10,000,000 (not 1e8 — a 1e7/1e8 confusion
  broke the test assert). RoPE is NORM/interleaved (llama.cpp
  `llama_model_rope_type` GRANITE -> NORM). dbrx pre-tokenizer is "same as
  llama3" — `gpt2_tokenize` maps 'dbrx' to `llama3_pre_tokenize`. No BOS
  (add_bos_token=false; bos=eos=100257 `<|end_of_text|>`).
- **llama-cpp-python `eval` + `logits_all=True` accumulates KV across cases**:
  sequential `eval(ids)` calls append to the cache, so only the FIRST case's
  argmax is trustworthy. For argmax pins use fresh-context greedy
  completions (create_completion per case, max_tokens=1, temp=0) — the
  generated single token IS the argmax. (granite oracles: hello 11, The 11,
  Paris 315, hello world 11, france 41958, don't stop 3156.)
- **gemma detokenize renders ▁ as spaces** (`sp_replace` in tokenizer_llama3.ijs;
  ▁ is the 3-char UTF-8 E2 96 81 in J, so it's replaced as a sequence, not a
  char) — chat answers read naturally, and the space-text re-tokenizes to the
  same ▁-pieces, so gemma sessions can resume.
- **`tokenizer.ggml.model='llama'` → SPM bigram-merge, NOT a unigram Viterbi**:
  llama.cpp `llm_tokenizer_spm` (src/llama-vocab.cpp ~110-239): split into
  UTF-8 codepoints, seed a max-heap with every adjacent 2-codepoint pair that
  is a vocab token (priority = token score), repeatedly merge the HIGHEST-
  scoring valid pair, then look up each final symbol. The score-Viterbi is the
  WRONG algorithm (we went down that path first — ERNIE oracles disproved it).
  UGM (score-Viterbi) is only for `tokenizer.ggml.model='t5'`.
- **SPM `add_space_prefix` is a BOOL KV (vt=7)** — `kv_uint` returns _1 for it
  (only handles vt 4/5/10). ERNIE has NO such KV → SPM default true (leading ▁,
  llama-vocab.cpp:2371); gemma3 has it explicitly FALSE (first byte 0x00 at vo).
  `spm_tokenize` hardcodes the prepend (ERNIE-only today).
- **SPM byte fallback is `<0xXX>` lookup, NOT `65536+byte`**: llama.cpp
  `byte_to_token` (SPM/UGM, llama-vocab.cpp:3844) builds the literal
  `'<0x' , hex , '>'` string (e.g. `\n` → `<0x0A>` → token 23) and looks it up
  in the vocab, falling back to the raw 1-char byte string. The `65536+byte`
  clamp is the BPE path.
- **ERNIE chat session falls back to full re-render** (resume_count 0): the
  generation prompt `'Assistant: '` ends with a space (▁), so the model emits
  the answer's first word RAW (`The` 700, no ▁); detokenize → `'The capital...'`
  loses the leading ▁, and the re-render `'Assistant: The...'` re-tokenizes to
  `▁The` (526) — prefix mismatch → designed fallback (correct, slower).
  Granite resumes because its gen prompt ends with no space (`'...assistant<|end_of_role|>'`),
  so the answer keeps its leading ▁ and round-trips.

`inference.ijs` defines the generic `load_gguf_to_llm`: it first resolves the
arg via `model_path` (accepts catalog ids, HF paths, URLs, `~models/...` jpaths,
or filesystem paths — downloading to `~user/models` if needed), reads the model's
`general.architecture` KV once (`detect_arch`), hands off to the arch-specific
loader (`gem3_load`/`llama_load`/`qw2_load`), and maps generic `infer`/`generate`
onto that arch's verbs — playsound-style, so the inference path has no per-call
dispatch. Generic `infer`/`generate`/`infer_simple`/`generate_simple` follow the
**last-loaded** model (one current arch per session); per-arch verbs (`gem3_*`,
`llama_*`, `qw2_*`) are the precise entry points. The llm noun carries its arch at
index 9 (`llm_arch`). e.g.
`llm infer_simple_inference_ 'hello'`,
`llm qw2_infer_inference_ ('hello' ; <0 0 0.95 0.0)`.

## GGUF Parser API

`parse_hdr path` → `<magic; version; tensor_count; kv_count>`

`parse_kv_pairs path` → `<kvs_flat; raw_bytes; count; kv_end_offset>`
- `kvs_flat`: `<key0; vt0; vo0; next0; key1; vt1; vo1; next1; ...>` (4 items per KV)
- Value lookup takes the full result: `'general.architecture' kv_string res`
  (`kv_uint`/`kv_float`/`kv_string`/`kv_array` each take `<kvs; raw>` = `0 1 { res`)

`detect_arch path` → arch string (`'gemma3'|'llama'|'qwen2'|'qwen3'`)

`(<raw_bytes; kv_end; n_tensors) parse_tensor_infos` → flat boxed list of tensor infos
- Each tensor info: `<name; n_dims; dims; etype; data_off; hdr_sz; next>`

`(<path; info_flat; tensor_data_start) load_tensor_data` → tensor as float array

## Debug Tips

1. **Check shapes**: `echo $var` to verify tensor shapes
2. **Check value ranges**: `echo <./ arr` and `echo >./ arr` to catch NaN/inf
3. **Step through blocks**: comment out later blocks to isolate failures
4. **GGUF dump**: `load 'gguf_dump.ijs'; gguf_dump 'path.gguf'`
5. **Build formatted strings first**: `s =. ": i; echo s , ' more text'`
   (never `echo ":i , ' more text'`) — and parenthesize mixed `*`/`+` expressions
6. **Type check**: `3!:0 arr` shows type (2=char, 4=int, 8=float, 32=boxed)


## J Knowledge & Idioms — where the deep material lives

The verbose jforc chapter-by-chapter idiom reviews (Ch 22-43) and the full
J/numeric gotchas live in **@docs/J-KNOWLEDGE.md**.

**Load @docs/J-KNOWLEDGE.md when writing, editing, or debugging J code**
(any `.ijs` file, kernels, tokenizers, quant decode, or numeric-representation
issues). It is the distilled J lore that keeps J's special qualities (rank,
tacit style, no precedence, right-to-left) in sight as we code — read it
before non-trivial J work rather than relying on remembered idioms. Do NOT
load it preemptively for pure planning/reading tasks where J is not involved;
read it on a need-to-know basis for the J-specific task at hand.

AGENTS.md keeps only the operational essentials (file map, how to run,
interface, parser API, debug). Deferred ideas about applying an idiom to this
codebase are surfaced in **PLAN.md** ("Deferred J-idiom applications").
- Ch 30 modifiers (locale-leak, soporific, playsound dispatch) → J-KNOWLEDGE.md
- Ch 35 performance & measurement → J-KNOWLEDGE.md
- Ch 36 tacit accessors `>@(n&{)` → J-KNOWLEDGE.md (used for `llm_*`/`mi_*`/`ti_*`
  and per-arch `gem3_bd_*`/`qw2_bd_*`/`llama_bd_*` — block_data accessors are arch
  prefixed, never shared `bd_*`, or loading a second arch module clobbers the
  first's layout)
- Ch 37/39/40 forks, hooks, `@`/`@:`, `V V N`, dyadic-hook `*_simple` wrappers → J-KNOWLEDGE.md
- Ch 41 `f.` (flatten, special-code recognition) → J-KNOWLEDGE.md
- Ch 42 `13 :` explicit→tacit converter (limits: conjunction-operand x/y,
  assignments `y [ s =. ...`, modifier-operand routing, noun-on-right → manual
  right-bond `({&n)`) → J-KNOWLEDGE.md
- Dictionary verbs (`|:` transpose, `$` reshape, `m}` amend, `+/ .*`,
  `,:` itemize) → ARCHITECTURE.md (RoPE, KV cache, attention) + J-KNOWLEDGE.md

## Key Reference

- llama.cpp: `llama.cpp/` checkout (`src/models/*.cpp`, `src/llama-graph.cpp`)
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- J for C Programmers (JfC): https://www.jsoftware.com/help/jforc/
- J Primer "Precedence": https://www.jsoftware.com/help/primer/precedence.htm
