# HISTORICAL.md — Done Work & Legacy

Completed milestones, resolved limitations, and accumulated learnings from
building this LLM-in-J inference engine. This is **not** plan material (see
**PLAN.md** for planned work) and not architecture (see
**docs/ARCHITECTURE.md**); it preserves the record so nothing is lost.

## Origin Story

This project began as an experiment: build a GGUF-based language model
inference engine in J (J9.7) for educational purposes — simplicity over speed.
J inherently handles dimensioned arrays well and has efficient operators for
working with them, so the loader/inference path is a study in applying J's
special qualities (rank, tacit style, no precedence) to real numerical work.
The original one-page prompt (the project's SCOPE.md) framed the goal: load a
small GGUF into native J structures, then build an inference engine that can
parse a prompt and generate tokens. J's interactive "labs" (`~/j9.x/addons/labs/`)
were the learning companion. It was expected to be an eye-opening experience —
and it has been, layer by layer.

## Current Status — Historical Snapshot

At the point the plan was split, the suite stood at **448 checks, 448 pass,
0 fail**. Below is the Done table and the full-suite breakdown at that time.

| Model / Component | Status |
|---|---|
| Generic GGUF parser | Header, KV pairs, tensor infos, alignment padding; F32/F16/BF16 tensor loading |
| Tokenizers | Llama3-style BPE (`tokenizer_llama3.ijs`) + GPT-2 byte-level BPE (`tokenizer_gpt2.ijs`) |
| Kernels | matmul, linear, RMSNorm, GELU, SiLU, SwiGLU, RoPE (interleaved + NEOX), softcap |
| KV cache | Single persistent global boxed array (`kv_cache_g` + `kv_meta`); monadic `kv_write`/`kv_write_rows` amend one row in place — no threading, no layer copy (details in docs/ARCHITECTURE.md §KV Cache) |
| Unified generation + chat sessions | ONE `gen_loop_core` (llm_core.ijs) replaces the four `*_gen_loop` copies (fresh + resume modes); persistent console chat `llm chat 'msg'` resumes the KV cache across turns (`chat_session_g`); 12/12 Chat Session suite |
| Gemma3 270M | `models/gemma3.ijs` — verified exact match vs llama.cpp; 39/39 tests |
| SmolLM2 360M | `models/llama.ijs` — llama arch, verified vs llama.cpp; 24/24 tests |
| Qwen2.5-Coder 0.5B | `models/qwen2.ijs` — qwen2 arch (NEOX RoPE, Q/K/V biases), verified vs llama.cpp; 27/27 tests |
| Qwen3-0.6B | `models/qwen3.ijs` — qwen3 arch (per-head Q/K RMSNorm, no QKV biases), verified vs llama.cpp; 26/26 tests |
| Qwen3.5-0.8B | `models/qwen35.ijs` — qwen35 arch (hybrid: 6 full-attention + 18 gated-delta-net SSM layers), verified vs llama.cpp; 29/29 tests |
| GGUF dump | `gguf_dump.ijs` — pretty-print any GGUF file |
| Master test runner | `tests/j/run_all_tests.sh` — fresh J process per suite, unified report; 0-test suites flagged as failures (a crash before the summary no longer shows "OK 0/0") |
| Mapped loader | `load_gguf_to_llm` mmaps once, parses from mapped raw, unmaps after load; hybrid slice (`tail < ne*18` → take-of-drop). Loads: qwen2 4.36s, gemma 2.16s, smollm2 2.55s |
| CLI + EOS | `llm_cli.ijs` + `scripts/llm.sh`; `generate` stops at per-model `eos_token_id` (gemma3-it 106, qwen2 151645, smollm2 2), not hardcoded 2 |

**Full suite (historical): 448 checks, 448 pass, 0 fail** (Kernels 26, KV
Cache 16, Tokenizer 15, GGUF Parser 121, Model Catalog 16, Chat Template 4,
Chat Session 12, Sampler 32, Multi-Arch 3, Gemma3 39, Llama Arch 24,
Llama-3.2 1B 19, Qwen2 27, Qwen3 26, Qwen3.5 29, Granite 350M 23, SWA
Boundary 16). All model logits/argmax verified against `llama-cpp-python` on
the same GGUF files. Qwen3 suite 107.8s → 46.1s after the KV cache rewrite
(single persistent global, in-place amend). The harness `assert_test`
double-count fix makes the totals real (Chat Template was reporting 6/3 = its
3 checks counted twice); the runner now flags 0-test suites as failures
instead of "OK 0/0".

## Resolved Limitations & Gotchas

These were once "known limitations" and are now resolved or documented:

- **Chat templates / special tokens**: tokenizers encode chat-template markers
  as their token IDs (sub-phase 1.1a); the per-arch `chat_prompt`/`chat_generate`
  path (`llama-cli -st` behavior — EOS after answering) is **DONE** (1.1b–f) and
  console chat with KV persistence is **DONE** (1.1g). The only remaining
  chat-side limitation: gemma3-270m-it under sampling (temp 1.0) sometimes
  emits a newline instead of `<end_of_turn>` — a "natural stop" is then missed
  (small-model behavior; llama.cpp is identical).
- **One arch module per session**: the arch modules used to redefine shared `bd_*`
  accessors with different indices (gemma3 13/14/15, smollm2 9/10/11, qwen2
  12/13/14), so loading a second arch module clobbered the first's accessors.
  **DONE** — accessors are per-arch prefixed (`gem3_bd_*`/`qw2_bd_*`/`llama_bd_*`),
  no shared names remain; `test_multiach.ijs` loads gemma3 then qwen2 and
  verifies gem3 accessors still read the gemma layout. Generic
  `load_gguf_to_llm` self-requires the right module; multi-model sessions can
  now use per-arch verbs side-by-side.
- **Performance**: interpreted J — single-token inference ~1s; large models load
  in seconds (qwen2 4.36s) after the mmap + hybrid-slice pass. The KV cache is
  now a single persistent boxed array amended in place (no per-layer copy);
  remaining load hot spots (decode_bf16/f16 gather, tensor reshape copies) are
  near J's primitive limits. Efficiency pass deferred — see Phase 3.
- **Single sequence**: no batching of independent prompts.
- **Greedy generation degenerates**: 270M-class models loop/repeat on open-ended
  prompts when sampled greedily (temp=0); llama.cpp's "correct" outputs use
  sampling (temp=1.0, top_p=0.95, top_k=64, min_p=0.001). Logits are exact.
- **`test_gguf.ijs`** **DONE** — re-enabled in the runner as "GGUF Parser"
  suite (121 checks across 8 models: gemma3/granite/qwen3/granitehybrid/
  qwen35/ernie4_5/qwen2/llama). Fixed: shared counters to `=:` (script-level
  `=.` names are invisible inside explicit defs), `ns` to `=:`, `load_tdata`
  to the 5-box `<path; info; tds; name; raw>` form, shape-consistency check
  accounts for the `(|. dims)` 2D reshape, and dropped the 2 model dirs no
  longer present (functiongemma, gemma3-qat).

## Phase 1 — Core correctness & foundation (DONE)

### Phase 1 — Core correctness & foundation (unlocks the rest)

1. **Chat templates + special-token encoding** (user priority). Tokenizers
   split special markers (`<start_of_turn>user`, `<|im_start|>`) into bytes —
   no special-token encoding. Needed so instruct models emit EOS after
   answering (`llama-cli -st` behavior): apply the model's `tokenizer.chat_template`
   for a single user turn, encode markers as their token IDs (gemma3
   `<start_of_turn>`=105/`<end_of_turn>`=106=EOS; qwen2 151644/151645; smollm2
   1/2), stop at EOS. Verify vs llama-cpp-python `create_chat_completion`
   ("The capital of France is Paris." then stop, even greedy).

   **Status — sub-phase 1.1a DONE (tokenizer special-token encoding):**
   - `util/llm_core.ijs`: shared `template_markers` (extract `<...>` markers from
     `tokenizer.chat_template`) + `split_specials` (split text on markers).
   - `tokenizers/tokenizer_llama3.ijs`: container now 6 items `[vocab; bos; tk_len; sym;
     eos; specials]` (`tokenizer_specials`); `llama3_tokenize` splits segments and
     splices marker IDs.
   - `tokenizers/tokenizer_gpt2.ijs`: container 9 items (specials at index 8,
     `tokenizer_specials_g`); `gpt2_tokenize` splits similarly.
   - **Gemma chat prompt tokens verified EXACT vs llama.cpp `_input_ids`**:
     `2 105 2364 107 818 5279 529 7001 563 106 107 105 4368 107`.
   - **Three tokenizer bug fixes found during verification:**
     1. Section 2 condition `-. (b = 13) *. ...` — the leading `-.` applied to the
        WHOLE product (`-. 0` = 1) so letters-section entered for LF/digits and
        merged `\nThe` → fixed to `(-. (b = 13)) *. (-. (b = 10)) *. (-. (is_digit b))`.
     2. Section 3 `-. i - start < 3` parsed `i - (start < 3)` = `i - 1`, so `-.` =
        `2 - i` → broke at i=0 → empty piece → **infinite loop** (masked by #1's
        always-true elseif; caused the tokenizer suite hang) → fixed to `-. (i - start) < 3`.
     3. `split_specials` dropped `(# best_s) }. text` from the front instead of
        through the marker → fragment segments → fixed to `(best_pos + (# best_s)) }. text`.
   - Suite green 177/177 after fixes (85.7s).
   - **1.1b–f DONE:**
     - Per-arch `chat_prompt`/`default_params`/`stop_tokens` in the arch modules
       (gemma3: `gem3_chat_prompt`, `1.0 64 0.95 0.001`, stops 106+1; qwen2:
       `qw2_chat_prompt`, `0 0 0.95 0.0`, stop eos; smollm2: `llama_chat_prompt`,
       `0 0 0.95 0.0`, stop eos). Rendered templates verified EXACT vs
       llama-cpp-python `_input_ids` for all 4 arches (gemma3, qwen2, smollm2,
       qwen35).
       BOS behavior is per-arch (gemma llama3-tokenizer adds bos; qwen2/smollm2
       gpt2-tokenizer add none — matches llama.cpp).
      - `util/chat.ijs`: dispatch (`chat_prompt`/`chat_tokenize`/`chat_detokenize`/
        `chat_default_params`/`chat_stop_tokens`), `chat_args`,
        `chat_generate` (single-turn + rudimentary multi-turn — re-render full
        history), `chat_generate_simple`, `chat_msg` helper.
        (Superseded by 1.1g: `chat_gen_loop` → unified `gen_loop_core`;
        multi-turn now persists via `chat_session_g` + the KV cache.)
     - Arch generate loops refactored: `*_gen_loop` (takes a stop-list; stop
       token NOT appended) + thin `*_generate` wrapper (eos stop-list).
     - CLI chat mode (`jconsole llm_cli.ijs MODEL PROMPT MAX_STEPS 1`); single-turn.
     - Tests: `tests/j/test_chat.ijs` (prompt-token oracle, all 4 arches) +
       gemma chat generation mechanism check in `test_gemma3_all.ijs`. Suite
       green (185 total, 0 fail).
   - **NOTE — gemma argmax discrepancy (RESOLVED — our bug, not llama.cpp's):** our
     J implementation predicted "Paris" (50429) for the chat prompt while
     llama-cpp-python / llama-cli / HF transformers all predicted "The" (818).
     Investigation (HF transformers gold reference via the unsloth mirror
     `unsloth/gemma-3-270m-it` safetensors) proved llama.cpp was correct and
     our forward pass had TWO bugs, both now fixed:
     1. **Per-layer RoPE freq_base**: gemma3 SWA layers (all except 5,11,17 —
        pattern `il%6<5`) use rope_freq_base=10000, the dense layers use 1e6.
        Our code used 1e6 for every layer. Fixed by building per-layer cos/sin
        tables in gem3_build_block (bd_cos_tab/bd_sin_tab/bd_swa_l, block_data
        now 24 boxes).
      2. **Byte-token clamp corrupted real high-vocab tokens**: gemma3's
         SentencePiece tokenizer has a 262144 real-token vocab (no byte-token
         scheme), but the forward pass clamped token ids ≥ 65536 (`tok-65536`),
         corrupting tokens like "?" (236881) → wrong embedding → nonsense
         generation (e.g. "What is AI?" → garbage). Fixed by using token ids
         directly in gem3_infer/gem3_gen_loop.
      - **The same clamp was wrong for qwen2/smollm2 too** — removed from
        qwen2.ijs/llama.ijs. gpt2 byte-level tokenizers map byte tokens to
        their REAL vocab ids (qwen2 0x01=189, 'A'=32; smollm2 0x01=191), so
        llama.cpp embeds them directly. The clamp corrupted every token ≥ 65536,
        including qwen2's `<|im_start|>`/`<|im_end|>` (151644/151645 → 86108/
        86109), which made qwen2 chat generation garbage ("heim heim") while
        llama-cli -st gave "The capital of France is Paris.". The raw oracle
        tests never exercised the clamp (all input tokens < 65536).
      - After the fixes, per-position logits match HF exactly (all positions)
        and chat generation matches llama.cpp word-for-word ("What is AI?" →
        "AI stands for Artificial Intelligence..."; qwen2 greedy chat →
        "The capital of France is Paris.").
   - **Single-shot `generate` now frames the prompt (stop tokens on single shot):**
     the whole reason for the chat-template work was to frame the prompt so
     instruct models emit their stop tokens. Previously `*_generate` tokenized
     the raw string (no template) and stopped on EOS only — so gemma's raw
     one-shot CLI ran a coherent answer then spilled into garbage ("Please
     provide an example..." → nonsense) because the model never entered chat
     mode. `*_generate` now wraps the text in the arch's chat template (single
     user message), stops on `*_stop_tokens`, and returns the answer only
     (prompt tokens dropped). CLI default mode is now chat-framed; the CLI chat
     flag (4th arg=1) is effectively redundant for single turn. Regression pins
     (greedy, temp=0 — deterministic given exact logits):
     `test_gemma3_all.ijs` greedy chat "The capital of France is" →
     `The capital of France is Paris.` (HF + llama.cpp gold; gemma's
     detokenizer now renders ▁ as spaces);
     `test_qwen2.ijs` `qw2_generate` greedy → `The capital of France is Paris.`
     (llama-cli -st gold — pins the clamp removal);
     `test_llama.ijs` `llama_generate` greedy → `I'm sorry for any confusion,
     but` (llama.cpp --temp 0 gold).
   - **Gemma stop under sampling is hit-or-miss (known limitation, matches
     llama.cpp):** gemma-3-270m-it under temp 1.0 sometimes emits a newline
     (107) instead of `<end_of_turn>` (106) after its answer, so the stop list
     misses the "natural stop" and generation continues. Greedy (temp=0) emits
     106 reliably. This is small-model sampling behavior, not a bug.
   - **Gemma spill ROOT CAUSE — sampler top_k zeroing corruption (FIXED):** the
     "hit-or-miss" spill was actually a sampler bug. `sampler_topk` set excluded
     logits to 0; softmax computed `2^(0−max)` for every excluded token, flooding
     the denominator when logits are low. At gemma's after-newline position
     `<end_of_turn>` (106) is the dominant argmax (8.97 vs next −11) but prob
     became ≈ 0.002, so the stop token was never sampled → spill into garbage.
     Fixed to mask excluded logits with `_1e30`. The forward pass was verified
     exact vs HF at every position (batched AND incremental); the sampler now
     picks 106 200/200 there. For "The capital of France is", the after-Paris
     position is genuinely tied (106/107 ~50/50 per HF) — sampling emits the
     newline first then stops at 106, so the CLI now stops after the answer.
   - **1.1g DONE (crude console chat + KV persistence, option B):** no stdin loop —
      the J console REPL is the chat UI: `llm chat 'next message'` (per-arch default
      params) or `llm chat_p ('msg' ; <temp;k;p;min_p>)` for explicit. The session
      (`chat_session_g`) + `kv_cache_g` persist across calls: each turn re-renders
      history only to tokenize the new segment, verifies the prefix against the
      stored token stream, then resumes from the cache (incremental prefill of the
      segment through `*_run_blocks` at prev_len — batched attention sees only its
      batch, so resume reuses the verified single-token path). `chat_reset ''`
      clears session + cache. On tokenizer round-trip drift it falls back to a full
      fresh re-render (correct, slower). Verified: turn-by-turn chat == stateless
      `chat_generate` (full re-render) for qwen2 AND gemma3 (12/12 Chat Session
      suite). The four per-arch `*_gen_loop` copies were replaced by ONE unified
      `gen_loop_core` (llm_core.ijs) that dispatches embedding scale + 
      `*_run_blocks`/`*_run_blocks_b` by `llm_arch`, with `start_pos=''` (fresh:
      kv_create + batched prefill) / `start_pos=<n>` (resume). Gemma's detokenizer
      now renders ▁ as spaces (`sp_replace` in tokenizer_llama3.ijs) — presentation
      AND persistence: the space-text re-tokenizes to the same ▁-pieces, so gemma
      sessions resume.
2. **Multi-arch session safety** (`bd_*` accessor clobber). The arch modules
   used to redefine shared `bd_*` names with different indices (gemma3 13/14/15,
   smollm2 9/10/11, qwen2 12/13/14), breaking the first's attention on a second
   load. **DONE** — per-arch accessor prefixes (`gem3_bd_*`/`qw2_bd_*`/`llama_bd_*`)
   + `tests/j/test_multiach.ijs` regression (gemma3 then qwen2 module load).
   Note: locales/OOP are the deeper architecture direction (each model as an
   object locale holding its arch/block_data/tokenizer, `llmobj.ijs` already
   seeds the OOP path); the rename was chosen as the contained fix to keep the
   public `inference`-locale contract unchanged. Enables side-by-side
   multi-model testing (Phase 2 verification) and chat-template tests across
   arches.
3. **Test infra**: **DONE** — re-enabled `test_gguf.ijs` (parser suite
   135/135, 9 models); add chat-template oracle tests for the three models.


## Phase 2 — Architecture coverage (DONE models)

4. **Qwen3-0.6B** — **DONE** (`models/qwen3.ijs`, arch registered in
   `load_gguf_to_llm`, `test_qwen3.ijs` 26/26). Arch vs qwen2: per-head
   RMSNorm on Q and K (`attn_q_norm`/`attn_k_norm`, shared weights
   size=head_dim) BEFORE RoPE; NO Q/K/V biases; `qwen3.*` KV prefix; chat
   template adds no default system. Same NEOX RoPE, gpt2 tokenizer, tied
   embeddings. Single/multi-token argmax pins vs llama-cpp-python
   (hello 14582, The 15846, Paris 38297; hello world 198, ... 220, don't
   stop 279). Note: the batched prefill q/k-norm reshape hit the J
   `L * n_heads, head_dim` parse gotcha (→ `L * (n_heads, head_dim)` =
   `(32,256)` shape, silently cycling data); fixed with
   `((L * n_heads) , head_dim)`. `'a'` is a 0.004 BF16 near-tie
   (61832/21806) — F32-vs-double rounding flips it, not pinned. Qwen3 is a
   thinking model: greedy chat emits the reasoning block before answering.
5. **Qwen3.5-0.8B** — **DONE** (`models/qwen35.ijs`, arch registered in
    `load_gguf_to_llm` + chat dispatchers, `test_qwen35.ijs` 29/29; full suite
    403/403/0). Hybrid arch: 6 full-attention layers (il+1 % 4 == 0 →
    3,7,11,15,19,23) + 18 gated-delta-net (SSM) layers. Fused Q+GATE
    projection (`attn_q` 4096 = 2*head_dim*n_head; Q first 256, gate second),
    per-head Q/K RMSNorm BEFORE RoPE, sigmoid gate, conv1d (kernel 4),
    L2-norm q/k, sequential delta-net recurrence (`s *= exp(gate)`,
    `sk = kᵀ·s`, `delta = (v-sk)*β`, `s += k⊗delta`, `o = qᵀ·s`), norm-gated
    output, NEOX RoPE over first n_rot=64 dims (pairs (i,i+32), freq_base 1e7),
    head_dim=256, GQA 8:2, tied embeddings. SSM recurrent state cached
    (`rs_*`): per-layer `<conv_state (3,6144); s_state (16,128,128)>`, block
    index→ordinal via `qw35_ssm_layers`. MTP blk.24 NOT in scope: block_count
    = `qwen35.block_count` (25) − `nextn_predict_layers` (1) = 24; blk.24
    tensors skipped at load. Single/multi-token argmax pins vs
    llama-cpp-python (hello 11, The 2614, Paris 11; hello world 271, ... 39509,
    don't stop 279).
    **Gotchas hit:** `rs_write` originally stored `(conv , s)` (J `,` on
    (3,6144) + (16,128,128) concatenates to (17,128,6144)) instead of the
    `<conv ; s>` 2-box — corrupted the cache on resume (generation step 1
    read (128,6144)); fixed to `(<((<conv) , <s))`. `gen_loop_core` needed a
    `qwen35` case (scale 1, `qw35_run_blocks`/`qw35_run_blocks_b`).
    **Chat template gotcha:** the real GGUF jinja renders the generation prompt
    with angle-bracket tags (` thinking LF LF response LF LF`), not the plain
    `' thinking'`/`' response'` markers. The stand-in `qw35_chat_prompt` used the
    space form; the turtle-prompt greedy argmax then flipped EOS(248046)→9175
    ('Yes') — CLI chat exited empty while llama.cpp chat completion answered.
    Fixed `qw35_chat_prompt` to match the real template byte-for-byte (tags,
    content trim, merged-system, last_query_index, assistant reasoning split on
    ` response`); added a qwen35 prompt-token oracle to `test_chat.ijs`.
    **Deferred:** `chat_template_kwargs` / `enable_thinking=true` (generation
    prompt ` thinking LF`) not yet plumbed through `chat_generate`/`chat_core` —
    defaults only.
 6. **Llama-3.2-1B-Instruct** — **DONE** (`models/llama.ijs` — the generic llama
     arch module, renamed from smollm2.ijs; `test_llama32.ijs` 19/19). 16
     blocks, emb 2048, GQA 32:8, head_dim 64, SwiGLU 8192, interleaved RoPE
     (freq_base 5e5), vocab 128256, context 131072. **No clone needed**: the
     llama forward pass reads all dims from the GGUF, so Llama-3.2 is handled
     by the same module as SmolLM2. The differences are data-driven:
     - Tokenizer: `tokenizer.ggml.pre` 'llama-bpe' (vs 'smollm') — the llama32
       vocab is GPT-2-style (space marker U+0120 Ġ, not SentencePiece ▁) and
       needs real BPE merges. `tokenizer_gpt2.ijs` now carries `pre` at index 9
       and `gpt2_tokenize` dispatches the pre-tokenizer (`llama3_pre_tokenize`
       for llama-bpe, `gpt2_pre_tokenize` else). Verified: "hello"→15339,
       "The"→791, "hello world"→15339 1917, chat prompt tokens EXACT vs
       llama.cpp (system block + date).
     - BOS: llama32 `add_bos_token` is true — `llama_tokenize` prepends bos
       (128000) for llama-bpe; the chat template omits `<|begin_of_text|>`, so
       the stream matches llama.cpp exactly (bos once).
     - Chat template: `llama_chat_prompt` dispatches on the tokenizer pre —
       llama3 template (ALWAYS-emitted system block `Cutting Knowledge Date:
       December 2023\nToday Date: <date>\n\n` + `<|eot_id|>` + messages) vs
       SmolLM2 `<|im_start|>`. The date is dynamic (llama-cpp-python injects
       `%d %b %Y`); `llama32_chat_date_g` pins it for stable oracles.
     - Stop: eos 128009 (`<|eot_id|>`); `llama_stop_tokens` reads `eos_g`.
     Infer argmax pins (hello 11, The 2768, Paris 11, hello world 271, The
     capital of france is 12366, don't stop 2888); generation "The capital of
     France is Paris."; chat session resumes (resume_count 1, fallback 0).
 7. **granite-4.0-350m (base)** — **DONE** (`models/granite.ijs`, `test_granite.ijs`
     23/23). Pure attention standard decoder, arch `granite`: 28 blocks, emb
     1024, GQA 16:4 (`head_count_kv` = all 4s — an ARRAY KV, so take `{.` of
     `kv_array`, not `kv_uint`), head_dim 64, SwiGLU 2048, RoPE freq_base 1e7
     (10 million — not 1e8), vocab 100352, context 32768. **No SSM tensors**
     despite `granite.ssm.*` KVs (metadata leftover). Tied embeddings — the
     GGUF has NO `output.weight`; lm_head = `token_embd.weight`. The Granite
     4.0 scaling scheme (read from KVs, stored in mi at indices 12..15):
     `embedding_scale` 12 (input embeddings *12), `residual_scale` 0.263
     (per layer: attn_out*0.263 + input, then ffn_out*0.263 + that),
     `attention.scale` 0.015625 (Q*K^T scores, NOT 1/sqrt(hd)), `logit_scale`
     4 (lm_head logits /4). RoPE is NORM/interleaved (llama.cpp
     `llama_model_rope_type` GRANITE → NORM), full head_dim rotary.
     Tokenizer: gpt2 byte-level BPE with `dbrx` pre — the dbrx regex is
     "same as llama3" (llama.cpp `LLAMA_VOCAB_PRE_TYPE_DBRX`), so
     `tokenizer_gpt2.ijs` maps 'dbrx' to `llama3_pre_tokenize`. No BOS
     (add_bos_token=false; bos=eos=100257 `<|end_of_text|>`).
     Chat template: granite 4.0 (always-emitted default system message
     "You are a helpful assistant. Please ensure responses are professional,
     accurate, and safe.", `<|start_of_role|>`100264 /
     `<|end_of_role|>`100265 / `<|end_of_text|>`100257). Verified vs
     llama-cpp-python: infer argmax pins (hello 11, The 11, Paris 315,
     hello world 11, france 41958, don't stop 3156 — fresh-context greedy
     completions; NOTE `logits_all=True` eval accumulates KV across cases,
     contaminating all but the first), chat prompt tokens EXACT (35 ids),
     generation "The capital of France is Paris.", chat session resumes
     (resume_count 1, fallback 0). gen_loop_core gained a per-arch
     `logit_div` (granite divides logits by logit_scale after output_head).
 8. **ERNIE-4.5-0.3B-PT** — **DONE** (`models/ernie.ijs`, `tokenizers/tokenizer_spm.ijs`,
    `test_ernie.ijs` 26/26). Standard attention decoder, arch `ernie4_5`: 18
    blocks, emb 1024, GQA 16:2, head_dim 128 (key/value_length 128), SwiGLU
    3072, RoPE freq_base 5e5, vocab 103424 (from token_embd dims — no
    vocab_size KV), context 131072. Forward pass is byte-for-byte the generic
    llama arch (llama.cpp `src/models/ernie4-5.cpp`: no scaling KVs, Q
    1/sqrt(head_dim), NORM/interleaved RoPE, TIED embeddings — no output.weight)
    so it REUSES llama.ijs forward verbs (`llama_run_blocks`/`_b` aliased as
    `ernie_*`); only loader/tokenizer/chat are new. **Tokenizer**: the
    analysis-first "unigram Viterbi" was the WRONG path — `tokenizer.ggml.model`
    `'llama'` maps to `LLAMA_VOCAB_TYPE_SPM` → llama.cpp `llm_tokenizer_spm`,
    the SentencePiece BIGRAM-MERGE (max-heap over token scores), not a
    score-Viterbi and not UGM (UGM is only model='t5'). ERNIE has no
    add_space_prefix KV → SPM default true → leading ▁; byte fallback is the
    `<0x0A>`-style `byte_to_token` lookup (not 65536+). Chat template: cls
    `<|begin_of_sentence|>`; User:/Assistant:/system messages; generation
    prompt `'Assistant: '` (trailing space — the answer starts with a raw
    token, so chat sessions fall back to full re-render, correct); stop on
    `</s>` (2) + `<|end_of_sentence|>` (100272). Verified vs llama-cpp-python:
    raw tokenize (hello 23013, The 526, Paris 11855, hello world 23013 3135,
    don't stop 1504 93968 93921 4038, 2-leading 269 8143 15075, x 843), chat
    prompt 12 tokens EXACT, argmax pins (hello 290, The 290, Paris 23, hello
    world 93937, The capital of France is 7365 — fresh-context greedy
    completions), generation 'The capital of France is Paris.'. Also fixed
    `kv_array` float-array bug (didn't slice `aoff` — scores read was garbage;
    the SPM tokenizer needs the scores).
 9. **LFM2-350M** — **DONE** (`models/lfm2.ijs`, `test_lfm2.ijs` 22/22, arch
   `lfm2`). Hybrid: 16 blocks, emb 1024, head 16, SwiGLU 4608, RoPE 1e6,
   vocab 65536, context 32768. **6 attention layers** (2,5,8,10,12,14; GQA
   16:8, per-head Q/K RMSNorm qwen3-style) + **10 shortconv layers**
   (0,1,3,4,6,7,9,11,13,15; `shortconv.conv.weight [1024]`, `in_proj
   [1024,3072]`, `out_proj [1024,1024]`) + FFN every layer. New shortconv
   block (conv1d; kernel semantics TBD, `shortconv.l_cache=3`) + new `lfm2`
   pre-tokenizer. Lighter than qwen35 (conv-only, no delta-net recurrence)
   but hybrid pattern dispatch + one new conv component.
   Catalog entries also cover LFM2-700M and LFM2.5-230M/1.2B (same lfm2 arch,
   `test_lfm25.ijs` / `test_lfm2700.ijs` 22/22 each).

## Phase 3 — Performance pass (done items)

### Phase 3 — Performance pass (independent; educational engine, low priority)

8. **softmax·V matmul** — largest single matmul per block; explore fused
   computation. ✅ Done: fused the causal/SWA mask into the scores matmul
   (`gem3_attention_b` applies `mask_2d` directly via `(0 2 1) |:` on the
   outer-product [t,j,g]→[t,g,j], skipping the (L,L,nh,nk) 4D expansion +
   permute + reshape). Logits verified EXACT (SWA Boundary 16/16), but
   time-neutral (prefill ~1.34s vs 1.32s baseline at L=600) — kept for
   simplicity, not speed.
9. **Embedding batch lookup** — batch token→embedding lookups for multi-token
   prompts. ✅ Already batched (`tok_list { emb_w`); removed the dead
   `embed_tokens` helper (stale byte-token clamp) from `llm_core.ijs`.
10. **KV single-pass merge** — fold string-array extraction into the `read_kv_hdr`
    scan (stride 4→5). ✅ Implemented (GGUF suite 135/135), measured slightly
    WORSE (load 2.32 vs 2.28s) — REVERTED (`git checkout gguf/gguf.ijs
    gguf_dump.ijs`); revisit only if load-time matters again.
11. **Resume prefill vectorization** — ✅ Done, uniform across all five arches
    (gemma3/qwen2/qwen3/llama/qwen35). `*_attention_b` now takes `start_pos`: RoPE at
    `(start_pos + i. L)`, prepends the cache prefix via `kv_read ((<layer) , <(start_pos-1))`,
    writes the batch K/V at `start_pos` via `kv_write_rows`, and scores/softmax/
    softmax·V over the combined (prefix+batch) keys. `*_block_forward_b` /
    `*_run_blocks_b` take `start_pos`; `gen_loop_core` uses ONE batched
    `rb_b ((<llm) , <start_pos)` for both fresh (start_pos=0) and resume.
    Chat session resume == stateless re-render verified 12/12 (qwen2 + gemma3);
    SWA Boundary 16/16; full suite 404/404/0 (with qwen35). Measured qwen2 29-token resume
    prefill: 0.115s vs 4.81s serial per-token loop (~42x).
    **Fix en route:** `kv_write_rows` had the precedence gotcha
    `a * max_seq + start` = `a * (max_seq + start)` — harmless at start=0
    (all prior callers), corrupted the cache prefix at start>0 (resume).
    Parenthesized to `((a * max_seq) + start)`.

### Performance-pass completions (2026-08)

These Phase-3 items landed after the plan split; the suite stayed green after
each. (Ordered by commit.)

- **Chunked prefill** (`0ac00b9`) — fresh prompts prefilled in
  `prefill_chunk_sz` chunks; bounds peak memory (the 140KB long prompt no
  longer OOM-kills; ~5GB flat vs 20/80GB sawtooth).
- **Causal mask `-"2`** (`8769bd4`) — no `(n_heads,L,tot)` 3D mask broadcast
  (J only broadcasts scalars).
- **RoPE hoist** (`808257f`) — cos/sin tables + expansions once per chunk,
  threaded through the layer loop (not per-layer).
- **GQA without KV expansion** (`26c4f16`) — group-major batched matmul:
  `Q_g2 (n_heads_kv, n_groups*L, hd)` vs unexpanded `Kp (n_heads_kv, ctx, hd)`
  (K/V stay n_heads_kv — 7x qwen2.5, 4x llama/granite, 2x qwen3); scores stay
  group-major through mask + softmax (no scores re-shape copy); mask hoisted as
  rope-box item 7; the tiled mask uses the `(*/)` broadcast — the cyclic
  boolean reshape `(n_groups,L,ctx)$mask_2d` was ~100x slower (5.4s vs 0.05s).
- **kv_cache per-layer growing arrays** (`90e2666`) — `kv_read` opens one
  contiguous box instead of razing `pos+1` row boxes (was ~0.005s/layer at 8k).
  Generation at 8k: 2.69 -> ~3.5-4 tok/s. (Superseded by the pre-alloc flat
  cache below.)
- **One-pass K transpose** (`594cedc`) — `1 2 0 |:` instead of the
  `1 0 2 |: + |:"2` two-pass (J special-codes axis *swaps* ~0.0002s, but cyclic
  permutations are strided ~0.0013s). Prefill ~15% better.
- **kv_cache pre-allocated flat, session-persistent** — ONE flat array per
  kind `(n_layers*eff_seq, n_kv*hd)`; writes are IN-PLACE amends (scalar/list
  selectors on a refcount-1 noun — measured 1us row, 6us 10-row), so no
  O(used) full-array copy per write (the growing append was O(n^2) total over
  a generation — unusable at 128K). `kv_max_seq_g` override caps `eff_seq`
  for low-memory (llama.cpp-style allocate-once-write-in-place); `kv_create`
  allocates once per session and reuses the buffer on repeat calls (reset
  `kv_pos_g`), matching llama.cpp's per-context cache. Reads gather the window
  (O(pos) — unavoidable, no views). Aligned with llama.cpp's `(stream, kv_size,
  n_embd_k_gqa)` layout except positions-leading (J's in-place amend demands a
  leading-axis selector).
- **K transpose exploration** — the cyclic `1 2 0 |:` (win, n_kv, hd) →
  (n_kv, hd, win) is the dominant remaining cache-path cost: strided ~1GB/s
  vs ~25GB/s contiguous, ~0.07s/token at 8k (qwen2.5-0.5b, load-noisy),
  growing with ctx. Storing K transposed (llama.cpp's layout) is BLOCKED in J:
  per-layer boxes force refcount-2 on unbox → the whole-layer copy per write
  (measured 1.7GB/s — worse than the transpose); a flat all-layers transposed
  array needs a full-column x-build per layer per write (~15ms/token constant)
  AND bulk prefill writes become per-column loops (+60% prefill at 8k). The
  two-swap transpose measured SLOWER than one-pass cyclic. Accepted as the
  J-floor for positions-leading storage (documented in ARCHITECTURE.md §KV Cache).
- **Batched decode** (`f8984a1`..`e6cffb8`) — cache gets a B-axis
  (`kv_batch_g`/`kv_seq_g`; base = `(layer*kv_batch_g + seq)*eff_seq`), each
  arch adds `*_attention_bd`/`*_block_forward_bd`/`*_run_blocks_bd` + a
  `*_generate_batch`, and `gen_loop_batch` runs ONE forward per decode step
  over B sequences (weight matmuls amortized across B; step 0 predicts from
  prefill-last hidden, mirroring `gen_loop_core` — re-embedding at pos L
  duplicated the last prompt K/V and shifted outputs). Verified
  batch==single for all 8 arches (qwen2/qwen3/llama/granite/ernie/gemma3/
  lfm2/qwen35), B=2 and B=3; measured ~1.4-2.8x per-seq throughput at small
  ctx. lfm2/qwen35 recurrent states (conv / delta-net) became per-sequence
  and are force-reset between generations (`lf2_conv_reset`/`rs_reset`).
  New `tests/j/test_batched.ijs` suite.
- **Read-once / mmap loader cleanup** (`2e5e294`) — the loader already maps
  once (`load_gguf_to_llm` passes the mapped raw to `detect_arch` + loaders,
  which preload all tensors via `load_tdata` with raw); the remaining full
  `1!:1` reads were the public `parse_hdr`/`parse_kv_pairs` used by
  `gguf_dump`. `parse_hdr` is now mmap-based (map+parse+unmap) and `gguf_dump`
  maps once + uses `parse_*_raw`. `parse_kv_pairs` stays `1!:1` — returning a
  live mapping is a footgun (`map_jmf_` refuses to remap `'gguf_raw'` while
  the caller's boxed raw ref is alive; run_model_test's `gguf_dump` call
  tripped it).
- **Transposed-canonical embedding/output storage (`emb_canonical`)** — J's
  matvec (m,n)x(n,) is ~3.9x slower than the vector-matrix (n,)x(n,m), so
  `token_embd.weight` is stored TRANSPOSED (emb, vocab) at load: lm_head becomes
  `hidden (+/ .* ) emb_w` (vector-matrix) and embedding lookup becomes column
  access `|: (tok {"1 emb_w)` (~1-4us, negligible). No memory doubling (same
  bytes, layout swapped; no J view/stream — `|:` materializes, so store
  transposed as canonical). output_head 22% -> 7.4% (jpm, qwen2.5), verb
  0.0504s -> 0.0129s. All arches + suites bit-exact.
- **Tacit hot-path conversions (kernels + qwen35, `9f1bfd1`)** — `linear_r`,
  `swiglu`/`geglu` (fork + `f.`), `rms_norm_rows`/`l2norm_rows`/`softplus`/
  `rms_norm` tacit via `>@(N&{)` accessors + At-composition; qwen35 needless
  `input=.x` reassigns dropped, `qw35_block_forward_s_b` aliased, `ssm_recur`
  precomputes `^ gate`. Validated on qwen3.5 first, then full suite 557/557.
- **qwen35 delta-net recurrent-state cache → two FLAT arrays** (`bb04407`) —
  `rs_conv_g`/`rs_s_g` positions-leading replace the boxed per-layer list (the
  boxed amend unboxed a refcount-2 global and COPIED the whole batch state,
  16.8MB at B=8, ~0.0038s/call — the #1 decode cost); the flat in-place
  list-selector amend fires ~78x faster (2.1MB slice, bit-exact). Measured
  qwen35 decode ~3x (0.277s/step vs 0.83, 28.9 vs 9.67 tok/s agg at B=8).
- **lfm2 conv cache → ONE FLAT array** (same bb04407 pattern, applied to
  lfm2) — `lf2_conv_cache_g` was the same boxed-per-layer list (n_conv boxes,
  each (kv_batch_g, d_conv=2, emb)); the boxed amend unboxed a refcount-2
  global and COPIED the whole batch conv state per write. Flattened to
  `(n_conv * kv_batch_g * d_conv * emb)` positions-leading with an in-place
  list-selector amend (`lf2_conv_slice` base `(ord*kv_batch_g + seq)*(d_conv*emb)`),
  mirroring `rs_conv_g`. All lfm2 suites (350M/700M/1.2B + 230M) + test_batched
  bit-exact; lint green.
- **lfm2 transpose hoisting** (`36aacba`, 2026-08) — hoist `|: hidden` once per
  attention layer (was 3×: qv/kv/vv each re-transposed the (L/B,emb) hidden) and
  `|: ffn_in` once per conv-layer FFN (was 2×: gate+up), both single-token
  (`_b`) and batched-decode (`_bd`); cuts redundant transpose copies. All lfm2
  suites + test_batched bit-exact, lint green. (The `$ ,` ravel reshapes stay —
  J's `$` is rank-dependent on rank-2/3 right operands, so flatten-then-reshape
  is required, not removable duplication.)
- **LFM2 decode scan findings** (2026-08, focus LFM2.5-1.2B) — single-token
  decode matmuls are matrix×vector (M=1, no column reuse); J threads them
  poorly (~9GB/s) vs ~28GB/s once B≥4 (matrix×matrix). The 1.2B reads ~4GB of
  weights/token (conv layers: in_proj 6144×2048 + out_proj 2048×6144 + FFN
  8192×2048×2 + 2048×8192, ×10; attention+FFN ×6), pinning decode at ~9GB/s ≈
  33% of memory bandwidth (the observed CPU saturation). Stored-transposed
  vector-matrix is WORSE (3.2GB/s vs 8.2GB/s). The lever is batching
  (`*_generate_batch`/`gen_loop_batch`) — already implemented; single-sequence
  chat/generate can't benefit without batching. Prefill is already batched and
  fast (~102 tok/s at L=56); its dominant cost is `kv_create`'s one-time 8.4GB
  zero-alloc for the 1.2B 128K ctx (2.8s) — bound `kv_max_seq_g` (→ ~256MB at
  eff_seq 4096) for big-ctx models.

**Review candidates — picked off (2026-08), all closed/rejected with reasons:**

- **The M=1 matvec floor is J's matrix-vector path, ~9GB/s — batching is the
  lever.** Generation reads ~4GB of F32 weights per token (lfm2-1.2B), but a
  SINGLE-token matmul `(out,emb) (+/ .*) (emb,1)` is a matrix×vector with no
  column reuse; J threads it poorly — measured ~9GB/s (vs ~28GB/s once B≥4,
  matrix×matrix). So single-sequence decode is pinned at ~9GB/s ≈ 33% of the
  ~28GB/s memory bandwidth (the observed CPU saturation). The stored-transposed
  vector-matrix is WORSE (3.2GB/s vs 8.2GB/s) — the PLAN's rejection stands;
  output_head's `emb_canonical` works only on the huge vocab axis. The only
  lever is BATCHING (`*_generate_batch`/`gen_loop_batch`, done) — it makes the
  matmuls matrix×matrix and threads to ~28GB/s; single-sequence chat/generate
  cannot benefit without batching.
  jpm profile (qwen2.5, tiny ctx): `output_head` 0.057s/token (vocab
  151936x896) + `linear_r` 0.107s/token. The transposed-weight refactor for the
  SMALL per-layer matvecs was investigated: they are FASTER in the current
  (n, emb) matvec form (ffn_gate 0.0017s vs 0.0135s transposed), so it was
  rejected THERE. But output_head's huge (151936x896) matvec is ~3.9x faster
  as a vector-matrix (`hidden (+/ .* ) |: emb_w`) — implemented as transposed-
  canonical storage (see the done list; the old "transposed output_head slower"
  claim was errant/noisy). FFN gate+up fusion halves matvec CALLS but not
  bytes — saves only interpreter overhead (~us/call). Rejected.
- **Non-float weights — NOT APPLICABLE in J (removed 2026-08).** F16/int8
  weight storage assumes a smaller C type to store weights in and matvec over;
  J9.7's native float types are F32/F64 (no F16/int8), so there is no smaller
  type to use. The M=1 matvec floor stays F32 / memory-bound. No longer a
  candidate lever.
- **The long-context single-token attention is near its J-floor.** Remaining
  cost: the K cyclic transpose (`1 2 0 |:`) ~0.03s/token at 8k, inherent to the
  append-friendly cache layout. Storing K transposed `(n_kv, hd, win)` was
  rejected: the last-axis append needs a 2D-reshape at 0.021s/write (~100x);
  `,"2` appends the wrong axis. Softmax/V/output matmuls are fast.
- **Batch the K/V transpose or fuse the per-seq softmax/V loops — CLOSED (no
  merit; measured in the fused-attention exploration).** Batching the strided
  `1 2 0 |:` K transpose saves no bytes — strided throughput (~1GB/s) is
  unchanged whether one window or B windows are transposed (same bytes/op),
  so a batched transpose is no faster than the per-seq loop. Fusing the
  softmax/V loops into one rank-4 pass was measured SLOWER (see the rejected
  fused-attention item below): the stack/pad reshape copies B×win×n_kv×hd per
  layer and outweighs the per-seq savings. The per-seq loop is already near
  its floor; the remaining cost is the inherent strided K transpose + gather.
  Closed as a non-lever.
- **Fusing the batched per-sequence attention loop — REJECTED (measured).**
  Tried replacing the per-seq transpose+scores+softmax+V loop in `*_attention_bd`
  with one rank-4 pass: gather each window, pad to `win_max = >./ pos+1`, stack
  to `(B, win_max, n_kv, hd)`, one `(0 2 3 1) |:` K transpose + `(0 2 1 3) |:` V
  transpose, `Q_g2_b (+/ .*"2) Kp_b` → `(B, n_kv, groups, win_max)` scores, a
  causal mask via `(-. mask_b) * __` (-inf; J: `0*__`=0, `^__`=0), and
  `softmax_b (+/ .*"2) Vp_b`. Correct (batch==single, suites green) but SLOWER:
  layer-0 attention qwen2 B=2, fused/old = 1.15x at win=5, 1.48x at win=1025.
  The stack/pad reshape copies B×win×n_kv×hd per layer outweigh the per-seq
  loop savings, and batching the strided `|:` saves no bytes (same strided
  throughput ~1GB/s). The per-seq loop is already near its floor; the real
  cost is the strided K transpose + per-seq gather, inherent to the
  positions-leading cache (see §K transpose exploration). Reverted.
- **qwen35 conv1d fusion — CLOSED (measured slower; not the bottleneck).**
  The 4-tap conv (`qw35_ssm_forward_b`/`_bd` lines 474-478) is four separate
  `(k + i. L) { input` × broadcast-w-tap terms. The proposed one-gather fuse
  `+/ (w4 * ((0 1 2 3 +/ i. L) { input))` (w4 = tap weights repeated over L)
  is CORRECT but SLOWER: the (4,L,conv_dim) 3D intermediate + strided `+/`
  over the tap axis costs more than the four contiguous (L,conv_dim) terms
  + adds. Measured (qwen35 conv_dim 1536, min-of-reps): prefill L=64 current
  0.00013s vs fused 0.00023s (~1.8x slower); decode L=1 current 1.1e-5s vs
  fused 1.2e-5s. And conv is small vs the per-layer matmuls (one 4-gather per
  SSM layer; wqkv/gate/beta/alpha/out/ffn dominate), so it is never the
  bottleneck. Closed as a non-lever. (J gotchas hit while exploring:
  `+/` reduces the LEADING axis (rank `_ _ _`), `$` on a matrix right operand
  is rank-dependent, `0 1 2 3 +"0 0 i. L` is a length error — see
  J-KNOWLEDGE items 29-30.)
- **Delta-net parallel scan — CLOSED (high-risk; decode gets nothing).**
  (`qw35_ssm_recur` is inherently sequential — the delta term couples
  `s_{t-1}`.) The state-update outer term `k_t ⊗ delta` was ~84% of the loop
  (COMPUTE-BOUND, ~1.6 GFLOP/s) and is **DONE** (`4b9bd40`) as a batched
  matmat — see the next bullet.
  A parallel scan is NOT a plain associative scan: the delta term makes the
  per-token operator `decay_t*I - beta_t*decay_t*k_t k_t^T`, and composing
  such operators blows up symbolically (O(n^2) rank-1 terms). The correct
  parallel form is llama.cpp's `build_delta_net_chunking` (delta-net-base.cpp):
  CS=64 chunks, `decay_mask = exp(cumsum(g)_j - cumsum(g)_i)` lower-triangular,
  a lower-triangular SOLVE (`ggml_solve_tri` on `attn+I`), then cross-chunk
  carry. Note the cross-chunk carry is a SEQUENTIAL loop over chunks
  (delta-net-base.cpp:235), so parallelism is within-chunk only: for prefill
  L < CS=64 (typical prompts) the whole recurrence is ONE parallel chunk
  (potentially ~Lx), but for long prompts the gain is capped by the sequential
  chunk carry. Decode (L=1) gets no chunking benefit — the matmat outer is the
  decode win. High risk: it REORDERS the summation (last-ULP differences), so
  the bit-exact logits oracles would need retuning; and it's a large J rewrite
  (cumsum, decay_mask, triangular solve, cross-chunk carry). Not forced — the
  decode path (the hot one) gains nothing and prefill benefit is capped by the
  sequential carry. Closed as a non-lever.
  The `^ gate` precompute already removed the per-token scalar exp.
- **qwen35 delta-net outer-product → batched matmat** (`4b9bd40`) — the
  state-update outer `k_t ⊗ delta` (COMPUTE-BOUND ~1.6 GFLOP/s, ~84% of the
  loop) reformulated as a batched matmat `(n_v_heads,S_v,1) x (n_v_heads,1,S_v)`
  — J threads the rank-4 matmul over the v-head axis, ~5x on the outer
  (8 GFLOP/s), bit-exact (values match), shared prefill+decode; measured
  qwen35 decode ~1.1x (9.67 vs 8.79 tok/s agg, B=8). The recurrence is 23.6%
  of qwen35 prefill (top prefill cost).
- **qwen35 SSM batch norm-gated silu(z)** (`d2442d2`) — precompute
  `z_silu =. silu z3` once over all B per layer instead of B per-seq silu
  calls (~1152 activation overheads per decode batch); oracles preserved.
  (The conv1d-silu two-pass attempt was correct but slower — reverted.)
- **Struct-layout housekeeping** (`2ff5705`..`fb063fb`, 2026-08) — removed
  dead verbs (`create_kv_cache`, `sampler_seed`, `qw35_rope_b`, `f64_decode_t`,
  `le16`, `read_kv`, `read_tensor_hdr`, `gqa_expand` ×6, `move_axes`) and
  dead struct fields: `kvs_ctx` (llm noun 11→10 slots, `llm_arch` 10→9),
  `ti_row.n_dims` (stride 7→6), gemma3 `attn_k`/`attn_v`/`block_idx`,
  per-arch `rope_freq`/`n_ff`/`block_idx`, qwen35 `ssm_d_conv`. Accessors
  shifted + build lists + layout comments + test element-count asserts
  updated; all arch suites + test_chat/test_batched + lint green.

## Cross-cutting notes (recent learnings)

### Cross-cutting notes (recent learnings)

- Generation `cur_pos` must increment only after a block run (not on step 0,
  which does no KV write) — fixed in `gemma3.ijs`, `llama.ijs`, `qwen2.ijs`.
- Qwen2.5-Coder uses **NEOX RoPE** (not interleaved) and **Q/K/V biases** added
  after projection before RoPE; tokenizer is **gpt2 byte-level** (`pre=qwen2`).
- **Tacit accessor conversions (verified identical)** — gemma3 `mi_swa`/`bd_*`
  and the tokenizer verbs (`tokenizer_*`, `input_*`, `llm_tokenizer`) were
  converted to the `>@(n&{)` tacit form, uniform across modules. One real bug
  surfaced: `>@(N&{})` (errant `}` appended) makes `N&{}` a hook (`12&{}` ≠
  `12&{`) → domain errors in `mi_swa`/`llm_block_data`/`bd_fused_qkv`/`mi_sin_tab`;
  fixed to `>@(N&{)`. Runtime after conversions ~8% faster (full suite 612s vs
  665s). The `>@(N&{})` gotcha itself is general and lives in J-KNOWLEDGE.md.

