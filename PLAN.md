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
been implemented or closed with measured reasons (below). The **minja Jinja
port (Phase 5), chat-template layer (Phase 5G), and the GGUF-jinja chat
integration (Phase 5H) are DONE** — every architecture's chat rendering now
uses its own `tokenizer.chat_template` from the GGUF, parsed and rendered by
the minja/chat_template port; the bespoke per-arch prompt verbs are removed.
Full done-work detail is recorded in **docs/HISTORICAL.md**; the remaining
planned work is Phase 4 (engineering stretch, low priority) plus a few open
items below.

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

### Phase 5 — Real Jinja template engine (minja port) — DONE

A faithful J port of **minja.hpp** (reference/minja/, the C++ Jinja engine
llama.cpp uses for chat templates), plus the chat-template layer
(`chat-template.hpp`) — `util/minja.ijs` (`coclass 'minja'`) and
`util/chat_template.ijs` (`coclass 'chat_template'`), both standalone and
liftable. The full phasing (5A-5F engine, 5Ga-5Gc chat-template layer, 5H
GGUF-jinja integration) and every completion detail are recorded in
**docs/HISTORICAL.md** §Phase 5. The suite wires in test_minja.ijs,
test_minja_render.ijs, test_chat_template.ijs, and test_chat_template_goldens.ijs.

### Phase 5H — GGUF-fetched jinja chat templates — DONE

Replaced the per-arch hardcoded chat-prompt verbs with ONE GGUF-driven jinja
renderer: each arch loader reads `tokenizer.chat_template` from the GGUF and
renders messages through the minja/chat_template port (`chat_tmpl_render` in
util/chat.ijs). The bespoke per-arch prompt verbs and their helpers were
removed — chat rendering is pure jinja. Template variables (e.g.
`enable_thinking`) are settable via an optional `tmpl_vars` arg to
`chat_generate`; `now` can be pinned via `ct_now_g` for stable test oracles.
Details in **docs/ARCHITECTURE.md** §Chat-template rendering and
**docs/HISTORICAL.md** §Phase 5.

## Open items

- **Tool/typed-content prompts** — the real templates already express tools
  (qwen2/qwen3/qwen35), but the chat path currently renders `tools=null`; wiring
  tool definitions through `chat_generate` (tools as a template input) unlocks
  function-calling prompts.
- **Excluded ToolTest cases** — CommandR7b (HF-gated template), FirefunctionV2
  (upstream marks BROKEN/TODO, not in CI), and test-fuzz.cpp (fuzztest property
  fuzzing, no J equivalent). All are upstream-gated/known-broken/fuzz-only.

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
- minja: `reference/minja/include/minja/minja.hpp` + `chat-template.hpp`, tests `test-syntax.cpp`/`test-polyfills.cpp`/`test-capabilities.cpp`/`test-supported-template.cpp`; golden generator `scripts/minja_goldens.py`
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- J for C Programmers (JfC): https://www.jsoftware.com/help/jforc/
- J Primer "Precedence": https://www.jsoftware.com/help/primer/precedence.htm
