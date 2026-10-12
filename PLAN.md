# PLAN.md — LLM Inference in J (J9.8)

Roadmap and task-tracking document, scoped to **planned work only**. All
completed/informational material lives in **docs/HISTORICAL.md** (done work &
legacy); J-language knowledge in **docs/J-KNOWLEDGE.md**; architecture &
implementation details in **docs/ARCHITECTURE.md**; operational guidance (file
map, how to run, interface) in **AGENTS.md**.

## Project

Generic GGUF-based language model inference in J (J9.8), multi-model across
architectures. An educational inference engine; simplicity over speed.

## Current Status

The suite is green (all arch + kv-cache suites bit-exact vs `llama-cpp-python`).
Everything through **Phase 6 is COMPLETE** — minja Jinja port (Phase 5), the
chat-template layer (Phase 5G), GGUF-jinja chat integration (Phase 5H),
tool/typed-content prompts, streaming OpenAI-compatible chat API, the stateful
chat TUI, the network HTTP server, multi-session + batched HTTP generation,
**batched prefill** (all 8 arches), and **batched decode vectorization**
(qwen35 SSM recurrence, lfm2 conv decode+prefill, gemma3 SWA attention — all 8
arches batch==single, `tests/j/test_batched.ijs` 11/11). **Phase 4 (engineering
stretch) is evaluated** — each item was measured/analyzed and closed (kept or
dropped) with a reason. **No prior planned work remains** — the deferred idiom
ideas (Broadcastly, LoopWithInitial, tokenizer obverse) and the **jllama study**
(findings 1-8, all resolved) are assessed/closed in **docs/HISTORICAL.md**
("jllama study — adopted/evaluated").

## Deferred Optimization Opportunities

Found during the 2026-10 boxing/copy-elimination pass (see HISTORICAL.md
"Boxing/copy elimination — generation hot paths (2026-10)"). Measured/assessed,
left for a later pass — each is real but marginal/invasive relative to the
matmul-dominated decode. (The RoPE table+expansion hoist for the batched paths
and the per-layer `mi` dict-lookup hoist for the single-token path are DONE —
see HISTORICAL.md "Redundant computation elimination — generation hot paths
(2026-10)".)
- **Tacit conversion**: the hot kernels (`rms_norm`, `rms_norm_rows`,
  `linear_r`) are already tacit; the batched loops stay explicit `while.`
  because the candidate `u^:v^:_` DoWhile / `u^:n` Power / `m@.v` idioms are
  rank-0 slow for these per-layer/per-token loops — keep explicit unless a
  measured win appears.

## DRY / Dict-ification Refactor (PLANNED)

The 2026-10 dict-ification pass (llm noun → named dict, `catalog_d`, `dict_probe`,
`kv_write` re-box — see HISTORICAL.md "Dict-ification & accessor robustness")
made the per-arch wrappers largely redundant: the llm dict now carries
`tokenizer`, `mi`, `kv_data`, `default_params`, `block_data`, `arch` — so the
per-model verbs/nouns that just dispatch on those are pure aliases/data and can
be shed into ONE generic set that reads everything from the dict and dispatches
the arch-specific forward via `llm_arch`. This is a DRY study (2026-10) —
inventory below; the plan is 3 tiers, executed in order. **Tier 1 items 1-3 and
the Tier 3 per-model-globals members are DONE** (see HISTORICAL.md
"Dict-ification & accessor robustness" — generic `chat_prompt`/`llm_stop_tokens`/
`llm_tokenize`/`llm_detokenize`, and the `ct_tmpl`/`ct_caps`/`ct_tool_ex`/`lf2_conv`
dict members). The remaining tiers/items below are still planned.

### Tier 1 — shed pure aliases/data (low risk, ~40 verbs)

The per-arch wrapper verbs below are identical across all 8 arch modules except
for the arch-specific forward math + tokenizer — fold them into one shared verb
each. The per-arch names stay as thin aliases so the documented "precise entry
points" (and the tests that pin them) keep working.

1. **`*_kv_uint`/`*_kv_float` (16)** — thin wrappers over the gguf `kv_uint`/
   `kv_float`; the `kv_value` vt dispatch already handles them. Fold into
   `kv_value` (or keep one generic `kv_uint`/`kv_float`); alias the 8 pairs.
2. **gemma3 `*_infer` sample call** — gemma3 uses `params sampler_sample logits`
   (gemma3.ijs:991) while all others use `sample_from ((<temp),(<k),(<p),(<min_p),<logits)`
   (llm_core.ijs:297 wraps `sampler_sample`). Fold gemma3 onto `sample_from`.

### Tier 2 — generic infer/generate in llm_core (the big win, ~720 lines)

Collapse the per-arch `*_infer`/`*_generate`/`*_generate_batch`/`*_infer_simple`/
`*_generate_simple` (40 verbs, ~720 lines, ~90% identical) into ONE generic set
in util/llm_core.ijs that reads everything from the dict and dispatches the
forward via `llm_arch`:

- **Generic `infer`**: read tokenizer (`llm_tokenize`/`llm_detokenize` dispatch),
  embed-scale + logit-scale (from the mi dict: `mi_emb_scale`/`mi_logit_scale` —
  gemma3 `%: emb_len`, granite `granite_mi_embed_scale`/`granite_mi_logit_scale`,
  others 1), `kv_create` args (block_count/ctx_len/n_heads_kv/head_dim from mi),
  and dispatch `*_run_blocks`/`*_run_blocks_b` on `llm_arch`.
- **Generic `generate`/`generate_batch`**: frame via `chat_prompt`, tokenize via
  `llm_tokenize`, stop via `llm_stop_tokens`, then the EXISTING unified
  `gen_loop_core`/`gen_loop_batch` (already arch-dispatched).
- **Generic `*_simple`**: the two-line default-params wrappers.

The per-arch `*_infer`/`*_generate`/`*_generate_batch`/`*_infer_simple`/
`*_generate_simple` names become aliases to the generic (they still pin the
loaded arch via `llm_arch` — no behavior change). Each arch module then keeps
ONLY: hparam extraction, block_data build, and the forward math
(`*_run_blocks`/`*_attention`/`*_block_forward`/`*_run_blocks_b`/`_bd`/`_bp`).
Tests (test_gemma3/test_llama/test_qwen2/test_qwen3/test_ernie/test_qwen35/
test_lfm2/test_batched) + CLI/chat/TUI/server all keep working via the aliases.

### Tier 3 — per-model globals into the llm dict

Per-model state that is currently a shared global (set per `*_load`), move into
dict members so each loaded model keeps its own (no clobber, multi-model-ready).
(The chat-template `ct_tmpl`/`ct_caps`/`ct_tool_ex`, the tokenizer-pre, and the
lfm2 `lf2_conv` members are DONE — see the intro + HISTORICAL.md.)

- **KV cache state** (`kv_pos_g`/`kv_max_seq_g`/`kv_batch_g`/`kv_seq_g`/
  `kv_batch_alloc_g` + `k_cache_g`/`v_cache_g`) → the dict's `kv_cache` member.
  **Medium risk**: the kv verbs are monadic (no llm arg) — need dyadic `llm`
  threading. DEFER until Tier 2 settles (it's the only tier needing
  monadic→dyadic verb changes).

### Risk / verification

- Tier 1/2: per-arch names stay as aliases → the documented "precise entry
  points" and all tests keep working. Verify each arch suite + test_batched +
  lint load-probe after each tier.
- Tier 3: data-ification only (no verb signatures change except the KV-cache
  tier, which is deferred); verify the chat/TUI/server suites still pass.

## Key Reference

- llama.cpp: `llama.cpp/` checkout (`src/models/*.cpp`, `src/llama-graph.cpp`)
- minja: `reference/minja/include/minja/minja.hpp` + `chat-template.hpp`, tests `test-syntax.cpp`/`test-polyfills.cpp`/`test-capabilities.cpp`/`test-supported-template.cpp`; golden generator `scripts/minja_goldens.py`
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- J for C Programmers (JfC): https://www.jsoftware.com/help/jforc/
- J Primer "Precedence": https://www.jsoftware.com/help/primer/precedence.htm
