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
dropped) with a reason. **No prior planned work remains** — the DRY/Dict-ification refactor is complete
(Tier 1 + Tier 2 done; the Tier 3 KV-cache member is ASSESSED/CLOSED — moving the
cache buffers into the dict breaks the in-place write, measured ~95× slower, so
the current design is kept); the deferred idiom ideas (Broadcastly, LoopWithInitial,
tokenizer obverse) and the **jllama study** (findings 1-8, all resolved) are
assessed/closed in **docs/HISTORICAL.md** ("jllama study — adopted/evaluated").

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
inventory below; the plan is 3 tiers, executed in order. **Tier 1 (all items) and
Tier 2 are DONE** — the generic `infer`/`generate`/`infer_simple`/`generate_simple`/
`generate_batch` live in util/chat.ijs and all 8 arch names are thin aliases to
them (`gem3_infer =: infer`, etc.); the generic `infer` uses `sample_from` for
all arches; the per-arch `*_kv_uint`/`*_kv_float` wrappers are gone (the arch
modules read the `kv_data` dict via `dict_get`). **Tier 3's chat-template/
tokenizer-pre/lfm2-conv members are DONE too** (see HISTORICAL.md
"Dict-ification & accessor robustness"). **The Tier 3 KV-cache member is
ASSESSED/CLOSED** — moving the cache buffers into the dict breaks the in-place
write optimization (O(buffer) write-back, measured ~95× slower); the current
design (config in dict, buffers+state in globals) is kept. **The DRY refactor is
complete — no remaining planned work.**

### Sprint ordering (recommended)

- **Sprint 1 — Tier 1 completion** — **DONE** (the generic `kv_uint`/`kv_float`
  wrappers are gone; gemma3's `*_infer` folds onto `sample_from` via the generic
  `infer`).
- **Sprint 2 — Tier 2** — **DONE** (generic infer/generate in util/chat.ijs,
  per-arch names are aliases, ~720 lines collapsed, no behavior change).
- **Sprint 3 — Tier 3** — **ASSESSED/CLOSED** (KV-cache state → the dict member
  would break the in-place write — O(buffer) write-back, measured ~95× slower;
  the current design is kept — see HISTORICAL.md).

### Tier 1 — shed pure aliases/data (low risk, ~40 verbs)

**DONE** — see the intro + HISTORICAL.md. The generic `infer` uses `sample_from`
for all arches (gemma3 folded); the per-arch `*_kv_uint`/`*_kv_float` wrappers no
longer exist (the arch modules read the `kv_data` dict via `dict_get`).

### Tier 2 — generic infer/generate in llm_core (the big win, ~720 lines)

**DONE** — see the intro + HISTORICAL.md. (Generic `infer`/`generate`/
`infer_simple`/`generate_simple`/`generate_batch` in util/chat.ijs; the 8 arch
names are thin aliases; each arch module keeps only hparam extraction + block_data
build + forward math.)

### Tier 3 — per-model globals into the llm dict

Per-model state that is currently a shared global (set per `*_load`), move into
dict members so each loaded model keeps its own (no clobber, multi-model-ready).
(The chat-template `ct_tmpl`/`ct_caps`/`ct_tool_ex`, the tokenizer-pre, and the
lfm2 `lf2_conv` members are DONE — see the intro + HISTORICAL.md.)

- **KV cache state** (`kv_pos_g`/`kv_max_seq_g`/`kv_batch_g`/`kv_seq_g`/
  `kv_batch_alloc_g` + `k_cache_g`/`v_cache_g`) → the dict's `kv_cache` member.
  **ASSESSED/CLOSED (deferred)** — moving the cache buffers (`k_cache_g`/
  `v_cache_g`) into the dict member breaks the in-place amend: the dict's
  `get__d` hands back a value not shared with the dict's storage, so an amend
  doesn't persist; persisting needs `put__d`, which copies the WHOLE buffer
  (O(buffer) per write — measured ~95× slower on a 0.8MB buffer; ~2ms per write
  for a 100MB cache). The current design (config in the dict, buffers+state in
  globals, in-place amend) is kept — see HISTORICAL.md.

### Risk / verification

- Tier 1/2: per-arch names stay as aliases → the documented "precise entry
  points" and all tests keep working. Verify each arch suite + test_batched +
  lint load-probe after each tier. (Done — all suites green.)
- Tier 3 KV-cache member: assessed/closed — see the Tier 3 section; the current
  design is kept (no verb-signature changes, no regression).

## Key Reference

- llama.cpp: `llama.cpp/` checkout (`src/models/*.cpp`, `src/llama-graph.cpp`)
- minja: `reference/minja/include/minja/minja.hpp` + `chat-template.hpp`, tests `test-syntax.cpp`/`test-polyfills.cpp`/`test-capabilities.cpp`/`test-supported-template.cpp`; golden generator `scripts/minja_goldens.py`
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- J for C Programmers (JfC): https://www.jsoftware.com/help/jforc/
- J Primer "Precedence": https://www.jsoftware.com/help/primer/precedence.htm
