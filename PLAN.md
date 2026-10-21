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

The suite is green (all arch + kv-cache suites bit-exact vs `llama-cpp-python`);
everything through **Phase 6 is COMPLETE** and **Phase 4 is evaluated**; the
DRY/Dict-ification refactor is complete (Tier 1 + Tier 2 done; the Tier 3
KV-cache member assessed/closed — see HISTORICAL.md). See
**docs/HISTORICAL.md** for the completed/evaluated details (phases, minja/jinja
+ GGUF-jinja, HTTP server, batched prefill + decode vectorization, jllama
study, boxing/copy + redundant-computation passes).

## DRY Refactor (Tier 2 follow-up — in progress)

A deep-inspection pass (Oct 2026) found the hot path is already at its
practical peak (memory-bound; mi-hoist, `|: hidden` hoisting, fused QKV,
unified `gen_loop` all done; matvec-vs-vector-matrix measured ~1.0x — NOT a
win). Remaining value is structural DRY. Work through **in this order**:

1. **Batched RoPE kernels** — the NEOX batched RoPE block is inlined ~6×
   (qwen2.ijs:191-200/:410-419/:588-597, gemma3.ijs:261-274/:453-466/:637-648)
   and the interleaved form ~3× (llama.ijs:179-188/:401-410/:522-531). Extract
   `rope_apply2_neox_b` / `rope_apply2_t_b` into kernels/jfloat.ijs (any-rank,
   dim-last-axis; the 4D `B,c,heads,dim` case already works via `"1`).
   **DONE** — both kernels added; all 9 standard inline copies replaced
   (llama/qwen2/qwen3/gemma3/lfm2; qwen35's Qtail variant is genuinely
   different and stays). Verified: lint/load-probe + test_kernels 32/32 +
   rank-generic kernel == inline (bit-exact).
2. **GQA attention scores/softmax/output** — the group-major
   `((n_kv,n_groups,hd)$ ,Q) (+/ .* "2) (1 2 0 |: k_all)` + direct-3D-softmax +
   `(attn_o) linear_r ...` block is identical across llama/qwen2/qwen3/granite.
   Extract `gqa_attn_single` / `gqa_attn_batch` into llm_core.ijs (scope to
   standard-GQA archs; gemma3/qwen35/lfm2 differ). **DONE** — both helpers
   added; single + batched call sites replaced in llama/qwen2/qwen3 (the
   `_bd`/`_bp` per-seq loops are different shapes and stay). Verified:
   lint/load-probe + test_kernels 32/32 + single==reference (bit-exact);
   batched helper runs (correct shape/finiteness).
3. **Loader duplication** — `*_load` in llama/qwen2/gemma3/granite/qwen3/
   qwen35/lfm2 each repeat ~50 lines of parse + tensor-load + llm-noun build.
   Extract `load_gguf_base` (returns `<path; ti; tds; all_tensors; kv_data;
   kvs_ctx; kvres; ct_tmpl>`) with arch hooks composing the rest. **DONE** —
   `load_gguf_base` added to llm_core.ijs; the 6 standard loaders (llama/qwen2/
   qwen3/gemma3/granite/ernie) refactored to use it. qwen35 (skips `blk.24.`
   tensors) and lfm2 (lf2_conv member + conv loop) have genuinely special
   tensor loops — correctly left. Verified: lint/load-probe (all load=1).
4. **`*_run_blocks_b`/`_bd`/`_bp` rope-table build + loop** — the cos_exp/sin_exp
   expansion + block_data_list loop is duplicated ~6×. Extract
   `build_rope_expansions` + `run_blocks_loop`. **PARTIALLY DONE** — the
   rope-expansion subset is DONE: `build_rope_expansions` added to llm_core.ijs
   (NEOX, lead-parameterized; x = <cos_all; sin_all; half; n_heads; n_heads_kv;
   lead>, returns <cos_expq; sin_expq; cos_expk; sin_expk>), replacing the
   duplicated `(0 2 1)|:((lead,half,n_heads)$,(cos_all(*/)(n_heads$1)))` builds
   across qwen2/qwen3/gemma3/lfm2 (b=L, bd=B, bp=B*c-with-reshape). Verified:
   test_qwen2/qwen3/gemma3_all/lfm2/batched (Failed: 0) + lint (exit 0).
   The **loop stays DEFERRED** — the loop cannot be shared: J cannot box a verb
   into a list (noun-verb syntax error), so the per-arch block-forward verb
   can't be threaded as data (the global-verb pattern would be a bigger
   architectural change).
   Low value, high awkwardness — leave.
5. **qwen2/qwen3 forward-pass unification** — normalize block_data to a shared
   17-element layout + flag-driven shared verbs (`qwen_attention`/
   `qwen_block_forward`/`qwen_run_blocks` + `_b`/`_bd`/`_bp`) in llm_core.ijs.
   qwen2 (`has_bias=1/has_qk_norm=0`) vs qwen3 (`has_bias=0/has_qk_norm=1`)
   dispatch on the flag pair; each arch module keeps thin wrappers. **DONE** —
   the per-arch `*_bd_*`/`*_run_blocks*` duplicated bodies are replaced with
   wrappers (the qwen2/qwen3 pair shares; the cross-arch loop stays deferred
   because other archs differ). Verified: test_qwen2/qwen3 (Failed: 0) +
   batch==single (qwen2/qwen3 MATCH) + lint (exit 0).
5. **Dead code cleanup** — `rope_apply`/`rope_apply2`/`rope_apply2_v1`/
   `rope_apply_neox`/`rope_apply2_neox` (per-call-trig RoPE, superseded by the
   table-based `rope_apply2_neox_t`/`_b` + `rope_apply2_t`/`_b`),
   `matmul` (alias of `+/ .*`) and `linear` (boxed wrapper; hot path uses
   `linear_r`/`+/ .*`). **DONE** — removed from kernels/jfloat.ijs and their
   tests from tests/j/test_kernels.ijs. The per-call-trig RoPE kernels compute
   cos/sin trig every call (slow path); the runtime precomputes tables once
   (build_rope_tables) and threads them — no reason to switch back. The
   ggml-oracle NEOX tests were dropped too: the runtime table-based NEOX kernels
   are verified bit-exact vs llama.cpp in the arch suites (stronger than the
   isolated oracle). `rope_apply2_v1` was fully dead (no tests, no runtime).
   Verified: test_kernels (20 pass/0 fail), test_qwen3/test_batched (Failed: 0),
   lint (exit 0).

Each step: implement, then `./scripts/install_local.sh --force`, then run the
affected suites (test_llama/test_qwen2/test_qwen3/test_batched) bit-exact.

## Key Reference

- llama.cpp: `llama.cpp/` checkout (`src/models/*.cpp`, `src/llama-graph.cpp`)
- minja: `reference/minja/include/minja/minja.hpp` + `chat-template.hpp`, tests `test-syntax.cpp`/`test-polyfills.cpp`/`test-capabilities.cpp`/`test-supported-template.cpp`; golden generator `scripts/minja_goldens.py`
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- J for C Programmers (JfC): https://www.jsoftware.com/help/jforc/
- J Primer "Precedence": https://www.jsoftware.com/help/primer/precedence.htm
