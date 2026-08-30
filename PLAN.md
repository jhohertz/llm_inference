# PLAN.md — LLM Inference in J (J9.7)

Living roadmap and task-tracking document, scoped to **planned work**. For
done/legacy material see **docs/HISTORICAL.md**; for J-language knowledge see
**docs/J-KNOWLEDGE.md**; for architecture details see **docs/ARCHITECTURE.md**;
for operational guidance (file map, how to run) see **AGENTS.md**.

## Project

Generic GGUF-based language model inference in J (J9.7), multi-model across
architectures. An educational inference engine; simplicity over speed.

## Current Status

The suite is green (all arch + kv-cache suites bit-exact vs `llama-cpp-python`).
The **performance pass (Phase 3) is DONE** — prefill memory is bounded, the
long-context generation overhead is cut, and every Phase-3 review candidate has
been implemented or closed with measured reasons (below). Full done-work detail
is recorded in **docs/HISTORICAL.md**; the remaining planned work is Phase 4
(engineering stretch, low priority).

## Roadmap — Planned Work

Each phase is independently shippable; the suite must stay green after each item.

### Phase 2 — Architecture coverage (DONE)

Items 6-9 (llama clone, standard decoders, LFM2-350M conv hybrid) are DONE;
see HISTORICAL.md. Item 10 — granite-4.0-h-350m (arch `granitehybrid`, the
gated-delta-net hybrid + granite scale KVs) — DROPPED 2026-08: its 1M context
(`context_length` 1048576) and MoE aspects (`expert_count` /
`expert_shared_feed_forward_length`) are out of scope for this project. The
GGUF parser still parses `granitehybrid` (test_gguf), but no inference module
is planned. Phase 2 is therefore complete.

### Phase 3 — Performance pass (DONE; memory + generation speed)

Done items from the pass — chunked prefill, causal-mask drop, RoPE hoist,
GQA-without-expansion, the flat session-persistent KV cache + K-transpose
exploration, batched decode, mmap/read-once loader, transposed-canonical
storage, tacit hot-path kernels, delta-net matmat-outer, the struct-layout
housekeeping cleanups, and the lfm2 conv-cache flatten + transpose-hoist
(2026-08) — are all recorded in **docs/HISTORICAL.md** §Phase 3.
The suite is green after each item. Every Phase-3 review candidate — the M=1
matvec floor, non-float weights, long-context attention, batched K/V-transpose,
fused batched attention, qwen35 conv1d fusion, and the delta-net parallel scan —
was picked off (implemented or closed with measured reasons); all recorded in
**docs/HISTORICAL.md** §Phase 3.


### Phase 4 — Engineering stretch (deep refactors, high risk / low priority)

11. **`llama3_pre_tokenize` / `gpt2_pre_tokenize` → `;:` state-table** —
    replaces verified-correct `while.` scanners; high drift risk, do only if
    tokenizer perf or clarity demands.
12. **Generation-loop refactors** — `u^:v^:_` DoWhile / `u^:n` Power / `m@.v`
    agenda candidates (rank-0 slow); keep explicit loops unless measured.
13. **Tokenizer encode/decode mutual obverse** (`u&.:v`) — future.
    (Item 14 batched decode moved to Phase 3.)

## Deferred J-idiom applications

The general jforc idiom reviews live in **docs/J-KNOWLEDGE.md** (project-agnostic).
Deferred ideas about applying an idiom to *our* code are kept here:

- **`LoopWithInitial` (Ch 36)** — the tool if a small-state fold ever appears
  (e.g. piece accumulation in a tokenizer) where space is not a concern; the
  generation loop stays a `while.` because it carries per-step KV tensors too
  large to materialize looplessly.
- **Tokenizer encode/decode mutual obverse (Ch 33)** — item 13 below; defining
  `tokenize =: ... :. detokenize` would enable `u&.:tokenize` round-trips, but no
  current call site needs it.

## Key Reference

- llama.cpp: `llama.cpp/` checkout (`src/models/*.cpp`, `src/llama-graph.cpp`)
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- J for C Programmers (JfC): https://www.jsoftware.com/help/jforc/
- J Primer "Precedence": https://www.jsoftware.com/help/primer/precedence.htm
