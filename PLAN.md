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
(findings 1-8, all resolved) are assessed/closed in **docs/HISTORICAL.md**.

## jllama Study (reference/jllama) — RESOLVED

The eight candidate-adoption findings from the `reference/jllama` study are all
resolved (done / adopted-in-simpler-form / evaluated-and-kept). They are
recorded in **docs/HISTORICAL.md** ("jllama study — adopted/evaluated"); this
section is kept only as a pointer — no planned work remains here.

## Deferred Optimization Opportunities

Found during the 2026-10 boxing/copy-elimination pass (see HISTORICAL.md
"Boxing/copy elimination — generation hot paths (2026-10)"). Measured/assessed,
left for a later pass — each is real but marginal/invasive relative to the
matmul-dominated decode. The RoPE table+expansion hoist for the batched paths
(`*_run_blocks_bd`/`_bp` → `*_block_forward` → `*_attention`, all 8 arches) is
now DONE — see HISTORICAL.md "Redundant computation elimination — generation
hot paths (2026-10)".

- **Per-layer `mi` dict-lookup hoisting** (single-token path): the
  `mi_rms_eps mi` / `mi_attn_scale mi` / `mi_resid_scale mi` /
  `mi_cos_tab mi` / `mi_sin_tab mi` accessors are hash-looked-up EVERY layer
  inside `*_attention`/`*_block_forward` (~5-7 lookups × n_layers × tokens;
  ≈1-2% of a decode step). **Measured 2026-10**: `mi_rms_eps mi` ≈ 0.38μs (vs
  ~68μs for a 256×768 matmul); only ~2-4 of these lookups fire per layer, so
  the whole hoist saves ~30-50μs per decode step against a step that runs
  milliseconds of matmuls — **≈0.1-0.5%, genuinely marginal**. Fix = compute
  once in `*_run_blocks` and thread the scalars/tables through the `y`
  signatures — invasive (changes `*_attention`/`*_block_forward` arg order
  again, all 8 arches + granite/ernie aliases) AND conflicts with the
  pass-`y`-through landed in the same pass (adding boxes to `y` shifts the
  unpack indices). Recommendation: keep deferred; revisit only if a profile
  shows the lookups as a hotspot. (The `_bd`/`_bp` RoPE-table part — the
  `mi_cos_tab`/`mi_sin_tab` fetches + expansion broadcasts — was eliminated by
  the RoPE hoist above; only the single-token `*_attention`/`*_block_forward`
  path still does the per-layer `mi_*` lookups.)
- **`kv_write` re-box in single-token `*_attention`**: after `'block_data pos
  mi layer' =. y`, `kv_write ((<layer), (<pos), (<K), (<V))` re-boxes layer/pos
  that were already boxed in `y`. Fix = pass `(3{y), (1{y)` (index-based) to
  reuse the existing boxes (2 boxes × n_layers × tokens; readability cost —
  index-based arg selection is opaque vs the named spread).
- **Tacit conversion**: the hot kernels (`rms_norm`, `rms_norm_rows`,
  `linear_r`) are already tacit; the batched loops stay explicit `while.`
  because the candidate `u^:v^:_` DoWhile / `u^:n` Power / `m@.v` idioms are
  rank-0 slow for these per-layer/per-token loops — keep explicit unless a
  measured win appears.

## Key Reference

- llama.cpp: `llama.cpp/` checkout (`src/models/*.cpp`, `src/llama-graph.cpp`)
- minja: `reference/minja/include/minja/minja.hpp` + `chat-template.hpp`, tests `test-syntax.cpp`/`test-polyfills.cpp`/`test-capabilities.cpp`/`test-supported-template.cpp`; golden generator `scripts/minja_goldens.py`
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- J for C Programmers (JfC): https://www.jsoftware.com/help/jforc/
- J Primer "Precedence": https://www.jsoftware.com/help/primer/precedence.htm
