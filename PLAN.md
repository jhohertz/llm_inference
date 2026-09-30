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
chat TUI, the network HTTP server, multi-session + batched HTTP generation, and
**batched prefill** (all 8 arches). **Phase 4 (engineering stretch) is
evaluated** — each item was measured/analyzed and closed (kept or dropped) with
a reason. **No prior planned work remains** — the deferred idiom ideas (Broadcastly,
LoopWithInitial, tokenizer obverse) are assessed/closed in **docs/HISTORICAL.md**.
The **jllama study** below reopened a small set of **candidate adoptions**
(planned, not committed) from the third-party `reference/jllama` engine.

## jllama Study (reference/jllama) — Candidate Adoptions

Deep dive on the third-party `reference/jllama` (tmcguirefl/jllama) engine — a
much smaller, single-skeleton, CLI-only J LLM. Findings in priority order; each
is a **candidate** (planned work — evaluate/adopt), not committed.

1. **Boxing/packing gotcha doc** — **DONE.** Adopted jllama's explicit "Boxing /
   packing rules (critical in J)" block as the canonical section
   **"Boxing / Packing Rules — CRITICAL IN J"** in **docs/J-KNOWLEDGE.md**
   (the 5 rules: `<"_` enclose vs `<`, multiple-assignment spread, `'a b c' =.
   <open_list` does NOT spread — open first, chained `a ; b ; c` RE-BOXES when
   the left is already a box list — pack mixed nested args with
   `(<a) , (<b) , already_boxed_c , (<d)`, pure-numeric `;`). Cross-references
   gotchas 12/14/15/20 + the `;`-nesting corollaries we hit (`max_rounds`
   ordering, `kv_write_rows` precedence, `_run_blocks` boxing, convert/json
   cells, named-box catenation). AGENTS.md now points to it.
2. **F16 decode LUT** — **ALREADY IMPLEMENTED (assessed/closed).** jllama builds
   a 65536-entry LUT once (`F16_LUT =: f16_from_bits_raw i. 65536`) and indexes
   it (`F16_LUT {~ 16bffff (17 b.) , y`). We already do the same in
   `gguf/gguf.ijs`: `f16_table =: f16_build_table ''` (line 107) + the tacit
   `f16_load =: ({&f16_table) @ (0&(3!:4)) @ (>&(1&{))` (line 122, ~0.37s for
   270M values vs ~0.76s explicit) + a per-tensor `f16_decode` fallback. No
   change needed; ours is already vectorized, tacit, and benchmarked.
3. **Clone-and-run bootstrap** — **ASSESSED; adopted in SIMPLER FORM.** jllama's
   `sysutils.ijs` derives `ROOT` from `4!:3 ''` via `setroot` + `jload ROOT ,
   relpath` so the checkout runs directly without `install_local.sh` +
   `require 'llm/inference/...'` resolving to `~addons`. We REJECTED the literal
   ROOT-relative-load rewrite as too invasive (every module uses addon-name
   `require`; the conditional arch `require` in `load_gguf_to_llm` relies on
   `require` idempotency — a `load`-based version re-inits module-scope state per
   call; risk to the addon-install REPL path). Instead adopted the SAME goal
   (no stale-install footgun, no manual install step) with **auto-install**:
   `tests/j/run_all_tests.sh` and `scripts/lint.sh` now run
   `install_local.sh --force` up front, so tests/lint always exercise the
   current checkout.
4. **`sample_cfg_pack` config normalization** — **DONE.** Added
   `sample_params_pack` to **util/sampler.ijs** (jllama `sample_cfg_pack`-style):
   accepts an OPEN numeric list, a `;`-list of boxes, or a (possibly
   double-boxed) scalar box of `<temp;k;p;min_p>`; returns the OPEN flat list
   padded to 4 with defaults (`temp=1.0, k=0, p=0.95, min_p=0.0`) and truncated
   to 4. Refactored the three duplicate inline normalizers to use it —
   `sampler_sample` (util/sampler.ijs), `infer_args` + `gen_args`
   (util/llm_core.ijs). Removes the 3x duplication and makes all params handling
   consistent (short/empty params now fill defaults instead of index-erroring).
   Verified: sampler suite 32/32 pass.
5. **`allclose` (atol/rtol)** — **DONE.** Added the jllama-style
   `x allclose y` = `*./ , (| x - y) <: ATOL + RTOL * | y` to
   **kernels/jfloat.ijs** (`ATOL =: 1e_9`, `RTOL =: 1e_6`, dyadic-only). A
   reusable per-element atol+rtol closeness helper for near-equality checks
   (batched-vs-single, path-vs-path); exact bit-level verification keeps the
   tighter absolute-sum patterns. Refactored the test_swa.ijs single==batched
   logits check (`1e_6 > >./ | ...`) to use `allclose`. Verified: allclose
   exact/tiny/relative/empty cases behave, kernel suite 32/32 pass.
6. **Small kernel idioms — ALREADY IMPLEMENTED (assessed/closed, no change).**
   Verified in the codebase: attention scale `1/sqrt(head_dim)` via
   `Q =. Q % head_dim ^ 0.5` (gemma3/qwen2/qwen3/qwen35/lfm2/ernie; llama via
   `mi_attn_scale`; granite's data-driven `0.015625` is the documented
   exception) — we pre-scale Q (llama.cpp style) instead of jllama's score-scale
   `% %: d` (equivalent); ravel-before-reshape `$ , y` used everywhere with an
   explicit "ravel is REQUIRED" comment (lfm2.ijs:246) plus kv_cache/jfloat/
   tokenizer_gpt2; fused-weight split via `{."1`/`}."1` (gemma3 fused QKV
   lines 271-273, qwen35 conv 501-502). jllama names these (`make2d`,
   `split_heads`); we inline them.
7. **Per-module named locales (tradeoff)** — **ASSESSED; keep single locale.**
   jllama uses `jllamarope`/`jllamaattn`/`jllamablock`/`jllamasample`/`jllamagguf`…
   with explicit `_jllama*_` cross-locale aliases: modular, dependencies visible.
   Our single `inference` locale is deliberate — the global-verb rebinding
   pattern (`gen_cb_g`/`chat_cb_g`/`chat_tool_fn_g`) works because
   verb-assignment aliases the NAME where the verb is CALLED, which a named-locale
   split would break (chat_tui.ijs). Modularity is cleaner; not worth breaking
   the global-verb pattern. No change.
8. **Arch-as-noun + `0!:0` rewiring (tradeoff)** — **ASSESSED; keep per-arch
   modules + dispatch.** jllama defines each arch as a `0 : 0` noun and
   `". '0!:0 ' , detect_arch model` rewires `model_from_gguf` + `block_full`/
   `block_step`/`block_prefill_cached` in place (filename detection). We already
   have the shared-skeleton idea via per-arch modules (`models/*.ijs`) + unified
   `gen_loop_core`/`gen_loop_batch` dispatch by `llm_arch`, with `general.architecture`
   KV detection (more robust than filename substring matching). The noun+rewiring
   is more dynamic but adds indirection and is filename-fragile; keep ours. No change.

**Status:** all eight are candidates (planned); none committed. Items 1, 2, 4, 5
are the strongest low-risk adoptions; 3 removes the stale-install test pain;
6 is verify-only; 7–8 are tradeoff evaluations.

## Key Reference

- llama.cpp: `llama.cpp/` checkout (`src/models/*.cpp`, `src/llama-graph.cpp`)
- minja: `reference/minja/include/minja/minja.hpp` + `chat-template.hpp`, tests `test-syntax.cpp`/`test-polyfills.cpp`/`test-capabilities.cpp`/`test-supported-template.cpp`; golden generator `scripts/minja_goldens.py`
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- J for C Programmers (JfC): https://www.jsoftware.com/help/jforc/
- J Primer "Precedence": https://www.jsoftware.com/help/primer/precedence.htm
