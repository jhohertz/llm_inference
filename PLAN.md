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
**batched prefill** (all 8 arches). Full done-work detail is recorded in
**docs/HISTORICAL.md**; the remaining planned work is below.

## Roadmap — Planned Work

Each item is independently shippable; the suite must stay green after each item.

### Phase 4 — Engineering stretch (deep refactors, high risk / low priority)

11. **`llama3_pre_tokenize` / `gpt2_pre_tokenize` → `;:` state-table** —
    replaces verified-correct `while.` scanners. **Measured 2026-09: keep the
    `while.` scanners.** They're fast (~1.2-1.5M chars/s; 60k chars ≈ 0.04s)
    and NOT the hot path (tokenization runs once per prompt, not per token);
    the `pieces , <piece` append is sublinear in practice (no O(n²) blowup).
    The scanners are sequential byte scans (rank-0, state-dependent — J cannot
    parallelize a sequential scan, so the shape/parallelization consideration
    doesn't apply), and vectorizing the contraction/letter/number/non-word
    state machine is high drift risk against the llama.cpp oracle. Do only if a
    measured tokenizer bottleneck appears.
12. **Generation-loop refactors** — `u^:v^:_` DoWhile / `u^:n` Power / `m@.v`
    agenda candidates (rank-0 slow). **Measured 2026-09: keep the explicit
    loop.** The single-token decode is M=1 (rank-1 vectors) — memory-bound on
    weight reads; J cannot parallelize M=1 matvecs (worker threads help large
    matmuls, not one-row). The loop-construct refactors don't change the
    shapes (Power needs a fixed count, DoWhile threads state through one verb,
    agenda is rank-0 slow) — so no parallelization/zero-copy benefit. The
    efficiency win comes from B-row (batched) shapes: `gen_loop_batch` runs the
    matmuls concurrently across cores (measured ~2.4x total / ~1.19x per-seq on
    lfm2.5-230m at B=2; ARCHITECTURE records 1.4-2.8x per-seq on larger
    models). Keep the explicit loop for the single-token (interactive/streaming)
    path; the shape-aware efficiency path is the batched generator (already
    implemented).
13. **Tokenizer encode/decode mutual obverse** (`u&.:v`) — future.

### Deferred J-idiom applications

Deferred ideas about applying a jforc idiom to *our* code (the general reviews
live in docs/J-KNOWLEDGE.md):

- **`LoopWithInitial` (Ch 36)** — the tool if a small-state fold ever appears
  (e.g. piece accumulation in a tokenizer) where space is not a concern; the
  generation loop stays a `while.` because it carries per-step KV tensors too
  large to materialize looplessly.
- **`Broadcastly` (verb-rank broadcasting) — DONE, available.** Ported into
  `kernels/jfloat.ijs` (see docs/HISTORICAL.md); NOT wired into the hot kernels
  because our per-layer broadcasts are small and J's `$`-replication is
  special-coded. Revisit only if large-batched projections appear.
- **Tokenizer encode/decode mutual obverse (Ch 33)** — item 13 above; defining
  `tokenize =: ... :. detokenize` would enable `u&.:tokenize` round-trips, but
  no current call site needs it.

## Key Reference

- llama.cpp: `llama.cpp/` checkout (`src/models/*.cpp`, `src/llama-graph.cpp`)
- minja: `reference/minja/include/minja/minja.hpp` + `chat-template.hpp`, tests `test-syntax.cpp`/`test-polyfills.cpp`/`test-capabilities.cpp`/`test-supported-template.cpp`; golden generator `scripts/minja_goldens.py`
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- J for C Programmers (JfC): https://www.jsoftware.com/help/jforc/
- J Primer "Precedence": https://www.jsoftware.com/help/primer/precedence.htm
