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
**Tool/typed-content prompts are DONE** — `chat_generate` accepts a `tools`
JSON input (threaded via `ct_tools_g` into the template's `tools` input) and
pre-built message Values carrying `tool_calls`/`tool_call_id`/typed content,
so tool-capable templates render real function-calling prompts (verified
against the upstream llama-3.1 tool_use golden).
Full done-work detail is recorded in **docs/HISTORICAL.md**; the remaining
planned work is **Phase 6 (streaming OpenAI-compatible chat API + direct jpi
fork, in progress)**, Phase 4 (engineering stretch, low priority), plus a few
open items below.

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

### Phase 6 — Streaming OpenAI-compatible chat API + own minimal chat TUI (in progress)

**Motivation.** A tool-use loop can't cleanly stand on its own — it *is* a chat
loop (intercept the model's tool-call message, execute, feed the result back,
continue). The console TUI problem is solved with our **own minimal chat UI**
built on **reference/j-kvm** (a J keyboard/video/mouse console library: `vt`
escape-code + raw-mode key reads, `vid` video buffers, `loop` event adverb).
We abandoned the jpi fork (2026-09): jpi carries too much baggage for a
temporary path — its own TUI fd quirks, the HTTP/agent/provider machinery, and
a vendored `vt` that read keys from stdout (fd 1) instead of stdin (fd 0),
which silently broke input. The core formalization is the **streaming
OpenAI-compatible chat-completion contract** on our side (streaming-first,
faithful to the OpenAI shapes), so the same verb later serves a network OpenAI
SSE server.

**Decisions (2026-09).**
- Target models for the first end-to-end proof: **qwen3 / qwen3.5** (their real
  templates already express `tools` + `tool_calls`).
- **Own minimal chat TUI (`chat_tui.ijs`) — stateless now, stateful as the end
  goal.** Each turn re-renders the full history via the shared `chat_generate`;
  the message list persists in-session. KV-cache resume (stateful) is the
  end-goal, not yet built. No markdown/video buffers initially — plain text.
- **HTTP server is deferred** — a separate agent is building J HTTP-server APIs
  and will bring them to us; do not start the network layer until then.

**Work items.**
1. **Streaming-first `chat_completion` verb** (util/chat.ijs, OpenAI-shaped):
   request `messages` + `tools` + params → response `content`/`tool_calls` +
   `finish_reason`. Adds to `gen_loop_core` an optional **per-token callback**
   (llm_core.ijs:326-341 samples token-by-token; currently returns the full list
   at the end with no hook). The callback serves the TUI (per-delta print) and
   the future SSE server (per-delta `data:` lines).
2. **Incremental text detokenizer** — the genuinely new piece. The TUI/SSE want
   *text* deltas (not raw tokens); `chat_detokenize` decodes the whole stream at
   once. Port llama.cpp's streaming detokenizer (accumulate bytes, emit a full
   UTF-8 char at a time).
3. **Stop at message/tool-call terminator + classification** — the generation-side
   "chat loop" gap: stop on the arch's end-of-message marker (not just EOS),
   then classify the message as text vs tool call to set `finish_reason`
   (`'stop'` vs `'tool_calls'`) and extract `tool_calls`.
4. **Stateful TUI** — reuse `chat_core`/`chat_session_g` (KV-cache resume) in
   `chat_tui.ijs` instead of stateless `chat_generate` re-render.

**Progress (2026-09).**
- **jpi checkouts removed** (`reference/jpi`, `jpi_local/`); the fd-fixed
  `vt.ijs` is vendored at `util/vt.ijs` (read stdin fd 0, write stdout fd 1).
- **`chat_tui.ijs` + `scripts/chat_tui.sh`** — a minimal terminal chat UI in a
  `coclass 'chatu'` locale (globals must use `=:`, not local `=.`, to be visible
  across definitions — a J gotcha hit here). Loads the addon + catalog model
  (qwen3-0.6b default), drives the j-kvm `vt` raw loop, calls
  `chat_generate_simple` per turn, prints the reply, and quits on Ctrl-C or
  `exit`. **Stateless text-chat proof works** (pty-verified: input → generation
  → reply → redraw). Added to the addon manifest.
- **Streaming-first `chat_completion` verb (item 1, DONE)** — `chat_completion`
  (util/chat.ijs) renders messages+tools via the shared renderer, generates,
  and returns an OpenAI-shaped `<content ; finish_reason ; tool_calls>`
  response. Streaming is driven by a **per-token callback** added to
  `gen_loop_core` (llm_core.ijs): the monadic verb `gen_cb_g` (gated by the
  noun flag `gen_cb_on_g`) is called on each generated token and may return a
  (possibly replaced) token id — returning a stop token forces a stop
  (interception). Verbs can't be boxed into a list (noun-verb syntax error) and
  can't be distinguished from a noun by `-:`/`3!:0`, so a global verb + noun
  flag gates it. `chat_completion` takes
  `(messages ; tools ; max_steps ; stream ; <params>)` — `<params>` MUST be the
  last operand (a pre-boxed `;` operand that isn't trailing nests). Verified in
  test_qwen3.ijs: 3-element response, finish_reason 'stop'/'length',
  per-token streaming (callback count), non-streaming skip, and interception
  forcing stop (eos read from the GGUF-built tokenizer). finish_reason
  `'tool_calls'` classification is item 3.
- **Streaming text deltas (item 2, DONE)** — `chat_stream_piece` (util/chat.ijs)
  is a port of llama.cpp's streaming incremental detokenizer: it appends each
  token's raw bytes to `st_buf_g`, holds any incomplete trailing UTF-8 sequence
  (`utf8_tail`), and emits only complete characters. `chat_stream_cb` is the
  per-token callback the caller installs as `gen_cb_g` (with `gen_cb_on_g=1`);
  it reads arch/llm from `chat_cb_arch_g`/`chat_cb_llm_g` (set by
  `chat_completion`) and forwards each text delta to `chat_cb_g` (a monadic verb
  on the delta string), returning the token unchanged so the delta never leaks
  into the token stream. Verified streaming == batch detokenize on all tokenizer
  families (greedy; e.g. qwen3 170/170 chars, gemma3 32/32, ernie 31/31); a
  streaming==batch test is in test_qwen3.ijs. **Gotcha:** do NOT capture a
  caller verb via `x =: gen_cb_g` then reassign `gen_cb_g` — J verb assignment
  is a dynamic ALIAS to the name, so rebinding makes the "capture" track the new
  value (infinite recursion). `chat_completion` never overwrites the caller's
  callback.
- **Terminator/tool-call classification (item 3, DONE)** — `chat_completion`
  (util/chat.ijs) classifies the generated content: if it carries a
  `<tool_call>...</tool_call>` region, `finish_reason` becomes `'tool_calls'`,
  the text `content` is nulled (OpenAI convention), and `tool_calls` are
  extracted (OpenAI-shaped minja Values `{type; function:<name; arguments>;
  id}`, `id` = `'call_' , name`). `chat_extract_tool_calls`/`chat_parse_tool_call`
  handle the generation formats: qwen3.5's
  `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>`
  (parsed into a pjson key/value table, `enc`'d to a JSON string), qwen3/granite's
  `<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>` (parsed with
  pjson `dec`, arguments re-`enc`'d), and a **bare OpenAI-style JSON** tool call
  (qwen2.5-coder emits `{"name":..., "arguments":{...}}` without the
  `<tool_call>` wrapper) — `chat_extract_tool_calls` falls back to parsing the
  whole content if it carries `name`+`arguments` keys. The JSON dependency is
  `convert/pjson` (added to DEPENDS — it preserves numbers/bools, unlike
  convert/json's 0/1-as-bool).
  Verified end-to-end (greedy): qwen3.5, qwen2.5-coder-0.5b/1.5b, granite-4.0 —
  `finish_reason='tool_calls'`, `content=''`, one call `get_weather` args
  `{"city":"Paris"}`; test_qwen35.ijs Section 6. Also fixed a latent **gpt2 byte-table bug** (see ARCHITECTURE.md):
  the tables followed OpenAI's bytes_to_unicode (codepoints 0..321) but llama.cpp
  treats bytes 160/173 as control -> codepoints 322/323; qwen3.5 vocab tokens like
  `ł`/`Ń` decode to 0xA0/0xAD. Streaming stop tokens are now suppressed too
  (`chat_cb_stop_g`) so stream==batch holds (qwen3.5 <|im_end|> has a non-empty
  byte-encoded vocab string).

**Open items (Phase 6).**
- **Tool-use loop (item 4, DONE)** — `chat_tool_loop` (util/chat.ijs):
  `llm chat_tool_loop (messages ; tools ; max_steps ; stream ; max_rounds ;
  <params>)` calls `chat_completion`; on `finish_reason='tool_calls'` it
  executes each tool via the global verb `chat_tool_fn_g` (y = `<name ;
  args-JSON>`, returns the result string; mirrors the gen_cb_g global-verb
  pattern), appends the assistant tool_calls message + one `tool` role message
  per result (minja Values), and re-calls until the model stops (cap
  max_rounds). Returns `<content ; finish_reason ; tool_calls_made>`. Verified
  on qwen3.5: model calls get_weather args `{"city":"Paris"}`, the loop executes
  it, feeds back `"The weather in Paris is sunny and 22C."`, and the model then
  answers `"The weather in Paris is sunny and 22°C."` (finish 'stop').
  Streaming re-arms `gen_cb_on_g`/`gen_cb_g` each round, so stream==batch holds
  across rounds. test_qwen35.ijs Section 6. J gotcha: `max_rounds` must come
  BEFORE `<params>` (a pre-boxed `;` operand that isn't trailing nests).
- **Chat TUI streams (ui) — DONE** — `chat_tui.ijs` renders the reply
  token-by-token via `chat_stream_cb` (no more '...thinking...' block): on Enter
  it arms streaming (`chat_stream_start`), rebinds `chat_cb_g` to a `stream_delta`
  consumer that appends each delta to STREAM and redraws live, calls
  `chat_completion` with stream=1, then disarms (`chat_stream_stop`). Locale
  fix: the TUI runs in the inference locale (not a separate chatu locale) — J
  verb assignment aliases the NAME (resolved where the verb is CALLED), so
  rebinding `chat_cb_g` from an external locale made `chat_stream_cb`'s
  `chat_cb_g delta` look up an unresolvable name. pty-verified: qwen3-0.6b
  answers "What is the capital of France?" streamed live, `[user]/[assistant]`
  rows rendered, answer mentions Paris. util/chat.ijs gains
  `chat_stream_start`/`chat_stream_stop` (the gen_cb_on_g/gen_cb_g globals are
  NOUNS/verbs llm_core doesn't export, so external locales can't arm streaming).
- **Cross-arch streaming/tools verification — DONE** — `chat_completion`
  (stream=1, stream==batch) verified on all 8 arches (gemma3/qwen2/llama/
  granite/ernie4_5/lfm2 + qwen3/qwen35): "The capital of France is Paris."
  with finish `stop` on each. `chat_tool_loop` runs cleanly on all arches.
  No arch-specific streaming bugs — `chat_tok_bytes` covers gemma3 (llama3),
  qwen2/qwen3/qwen35/llama/granite/lfm2 (gpt2), ernie4_5 (spm).
- **Tool-call formats by model — DONE** — capability detection
  (`ct_new_chatpl_`) shows tools supported ONLY in qwen3/qwen3.5/granite/qwen2
  (smollm2/llama, gemma3, ernie, lfm2 templates don't support tools — they just
  answer). The supported models emit three formats, all handled by
  `chat_extract_tool_calls`: qwen3.5 `<function=>`, qwen3/granite
  JSON-in-`<tool_call>`, qwen2.5-coder bare `{"name":...,"arguments":{...}}`
  JSON (no wrapper). Streaming during tool-call generation emits the markers
  for each format (verified: qwen3.5 `<function=get_weather>`, granite
  `<tool_call>{"name"...}</tool_call>`, qwen2 bare JSON). Tool-use loop
  executes + re-calls for all 4 (qwen2.5-coder loops calls, capped by
  max_rounds; granite-4.2/4.0 emit + execute).
- **HTTP server** — deferred to the J HTTP-server APIs (separate agent); reuse
  the same `chat_completion` verb behind an OpenAI SSE endpoint.

## Open items

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
