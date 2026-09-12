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

### Phase 5 — Real Jinja template engine (minja port) — NEW (exploration)

A faithful J port of **minja.hpp** (reference/minja/, the C++ Jinja engine
llama.cpp uses for chat templates). NOT a Python-jinja port: we port minja's
own two-header implementation, so our oracle is minja's unit tests
(tests/test-syntax.cpp EXPECT_EQ strings); Python's jinja2
(scripts/minja_goldens.py) is the cross-check oracle for the cases minja
deliberately matches. Kept **independent** of the inference addon — `util/minja.ijs`
uses its own `coclass 'minja'` so it can be lifted out into its own repo.
Chat-template layer comes later (`util/chat_template.ijs`).

**Reference:** `reference/minja/include/minja/minja.hpp` (3099 LoC, the whole
jinja engine in ONE header) + `chat-template.hpp` (569 LoC). Unit tests:
`tests/test-syntax.cpp` (668 LoC, the core render cases), `test-polyfills.cpp`,
`test-capabilities.cpp`, `test-supported-template.cpp`. Golden oracle:
`scripts/render.py` (Python jinja2) — we adapted it to batch mode in
`scripts/minja_goldens.py`.

**Deliverables so far:** `util/minja.ijs` (Phase-A foundation: Python-like
Value model + Context), `scripts/minja_goldens.py`, and the test-case files
`tests/j/test_minja.ijs` + `tests/j/test_minja_render.ijs` (NOT yet wired into
run_all_tests.sh — see note below).

**Phasing** (each phase independently testable; wire test files into
run_all_tests.sh as each phase lands and its cases pass):

- **5A — Value model + Context** (foundation, ~done in util/minja.ijs):
  Python-like values (null/bool/int/float/str/array/object/callable) with
  Python `repr`/`dump` (single-quote strings, `True`/`False`/`None` vs to_json
  `true`/`false`/`null`, `, ` / `: ` separators matching nlohmann + jinja2),
  `to_str`/`to_bool`/`to_int`, equality (numeric-tolerant), `in` membership,
  array/object accessors, scoped Context with parent chain + builtins.
  Validation: Value dump + Context get/set cases in test_minja.ijs.
  NOTE (J boxing): the flat-pair representation (`('arr';<k0;p0;k1;p1;...)`,
  `('obj';<key;value;...>)` with single-boxed items) sidesteps the `;`/`,<`/
  `>`-open rank gotchas (AGENTS.md/J-KNOWLEDGE gotchas 21, 26, 4). `{::` (fetch)
  returns unboxed scalars; numeric payloads come back as 1-lists, so index uses
  `{.` — see the TODO list at the end of util/minja.ijs.
- **5B — Expression grammar (recursive descent)** — port `parseExpression`
  and its helpers (`parseLogicalOr/And/Not/Compare`, `parseMathPow/
  PlusMinus/MulDiv/Unary`, `parseValueExpression`, `parseCallArgs`,
  `parseDictionary`, `parseIdentifier`, filter-expr): literals (numbers,
  strings, true/false/none, arrays, dicts), variable refs `a.b.c`, subscript
  `x[i]` + slices (incl. negative/step), method calls, binary ops (`+ - * / //
  % ** ~ == != < > <= >= and or not in is`), if-expr, unary, `|` filters.
  Validation: the full SimpleCases render set in test_minja_render.ijs.
- **5C — Tokenizer + Template parser (two-phase)** — port `tokenize()`
  (regex scanning `{{ }}` `{% %}` `{# #}` with `-`/`~` whitespace markers,
  comment/expression/block keyword tokens) and `parseTemplate()` (build
  SequenceNode AST from If/For/Set/Macro/Filter/Call/Generation tokens,
  unterminated-token errors). Validation: syntax + error-substring cases.
- **5D — Statements/controls** — IfNode (if/elif/else), ForNode (loop.* incl.
  cycle, destructuring, recursive, if-condition, else), break/continue
  (LoopControlException), SetNode (namespace + destructuring), MacroNode
  (+ default args, fresh arrays per call), CallNode (`caller()`), FilterNode.
- **5E — Whitespace + errors** — `trim_blocks`/`lstrip_blocks`/
  `keep_trailing_newline`, `-`/`~` space-strip/newline handling, the exact
  error message substrings minja throws (e.g. "Unterminated if",
  "break outside of a loop", "pop from empty list").
- **5F — Builtin globals/filters/tests** — port `Context::builtins()`:
  raise_exception, tojson, items, first/last, trim, capitalize, lower/upper,
  default, e/escape, joiner, count, dictsort, join, namespace, equalto,
  length, safe, string, int, list, in, unique, select/reject/selectattr/
  rejectattr, map, indent, range. (SimpleCases exercises nearly all.)

### Phase 5G — Chat-template layer (chat_template.hpp port) — NEW (standalone)

A faithful J port of **reference/minja/include/minja/chat-template.hpp**
(569 LoC) into **`util/chat_template.ijs`** (`coclass 'chat_template'`,
depends on util/minja.ijs's Value model). The HuggingFace-standard
`messages`/`tools` → prompt formatter that wraps a parsed template. It is
**standalone** (not wired into inference.ijs/chat.ijs) and only integrates
with the larger project after Phase 5 (engine) + 5G (this layer) land.
Oracle: minja's own tests — `tests/test-polyfills.cpp` (per-capability
goldens) + `tests/test-capabilities.cpp` (caps flags) +
`tests/test-supported-template.cpp` (e2e goldens via Python-rendered files).

**Reference layout** (chat-template.hpp):
- `chat_template_caps` — 10 flags: supports_tools, supports_tool_calls,
  supports_tool_responses, supports_system_role, supports_parallel_tool_calls,
  supports_tool_call_id, requires_object_arguments, requires_non_null_content,
  requires_non_empty_content, requires_typed_content.
- `chat_template_inputs` — messages, tools, add_generation_prompt,
  extra_context, now (time).
- `chat_template_options` — apply_polyfills + use_bos/eos_token +
  define_strftime_now + per-polyfill toggles.
- `chat_template` class:
  - constructor: `Parser::parse(source, {trim, lstrip, no-keep-newline})`
    (needs engine Phase 5C), then **capability detection** via `try_raw_render`
    probes (rendering dummy messages with `apply_polyfills=false`, fixed date)
    — needs engine Phase 5B+; plus `tool_call_example_` inference from a
    prefix/full render pair.
  - `try_raw_render(messages, tools, add_gen, extra)` — apply w/ no polyfills,
    fixed `now=0`.
  - `apply(inputs, opts)` — compute has_tools/has_tool_calls/has_tool_responses/
    has_string_content; derive polyfill_* flags; if needs_polyfills, normalize
    messages: typed-content conversion, pending_system/flush (system-role
    polyfill), `add_system` (tools polyfill), tool_calls polyfill (string→obj
    arguments, content=dump of {tool_calls:...}), tool_responses polyfill
    (role tool→user, content=dump of {tool_response:...}); build Context with
    messages/add_generation_prompt/bos/eos/strftime_now/tools/extra_context;
    `template_root_->render(context)`.
  - static `add_system(messages, prompt)` — prepend/inject system message.

**Phasing** (standalone; render-dependent parts land with the engine):
- **5Ga — data structures + standalone normalization** (do now): caps/inputs/
  options as boxed structs; `add_system` (prepend/inject system); the
  has_* message-scan and polyfill-flag computation; `add_message` typed-content
  conversion; pending_system/flush_sys accumulation. These operate purely on
  the Value model — no renderer needed. Testable against the polyfill test
  message shapes.
- **5Gb — render-dependent**: capability detection (try_raw_render probes for
  typed_content/system_role/tools/tool_calls/object_arguments/parallel/
  tool_responses/tool_call_id), tool_call_example inference, and `apply`'s
  final `template_root_->render(context)` — gated on engine Phase 5B+5C.
- **5Gc — context binding**: `strftime_now` callable (needs Value::callable,
  Phase 5A callable support), bos/eos token binding, tools/extra_context into
  Context.

**Test strategy:** test_chat_template.ijs documents the polyfill goldens
(ToolCallSupported/ToolCallPolyfill/ToolsPolyfill/ToolPolyfill shapes) +
capability flags from test-polyfills.cpp/test-capabilities.cpp, plus
standalone `add_system`/normalization cases that pass without the renderer.
Not wired into run_all_tests.sh until 5Gb lands.

**Test strategy:** bake Python-jinja2 goldens (scripts/minja_goldens.py) into
test_minja_render.ijs for the cases minja matches; use minja's own hardcoded
EXPECT_EQ strings for the `!USE_JINJA2`-guarded divergences (`{% generation %}`,
`{{ None | trim }}` → `""`, bare `{{ none }}` → empty, error substrings).
Wire files into run_all_tests.sh as their cases pass; keep the suite green.

### Phase 5H — Integration: GGUF-fetched jinja chat templates (swap the bespoke prompts)

**Goal.** Replace the per-arch hardcoded chat-prompt verbs (`gem3_chat_prompt`,
`qw2_chat_prompt`, `smollm2_chat_prompt`/`llama32_chat_prompt`, `granite_chat_prompt`,
`ernie_chat_prompt`, `lf2_chat_prompt`, `qw35_chat_prompt`) with ONE GGUF-driven
renderer: fetch `tokenizer.chat_template` (the jinja) from the GGUF metadata,
parse it once with the minja engine, and render messages/tools through the
chat-template layer. The bespoke verbs are hand-copies of these templates; the
swap makes the model's own template authoritative and removes the per-arch
copies. It also unblocks tool/typed-content/vision prompts (qwen3.5) that the
bespoke verbs cannot express.

**The GGUF already carries the template.** `tokenizer.chat_template` is a
STRING KV (vt=8) — `'tokenizer.chat_template' kv_string kv` (gguf/gguf.ijs:455)
extracts it from `parse_kv_pairs`. llama.cpp reads this same key and renders it
with minja, so rendering it with our minja port reproduces llama.cpp exactly
(that is the oracle guarantee — both run the same engine on the same source).
qwen35.ijs:995 already notes "The real template (GGUF tokenizer.chat_template,
reference/qwen35-chat_template.jinja)" — that file is the reference copy of the
jinja that will now come from the GGUF itself.

**Gating (analysis conclusion).** This CANNOT land yet. The swap needs the
renderer, and the port has only Phase 5A (Value model + Context + minimal
`render_expr`). Real chat templates use `{% for %}`/`{% if %}`/`{% set %}`/
`{% macro %}` loops, `| trim`/`| length` filters, `~` concatenation, and
`strftime` — all of 5B (expression grammar), 5C (tokenizer + template parser),
5D (statements/controls), 5E (whitespace), 5F (builtins), plus chat_template
5Gb (`ct_apply` render + capability detection) and 5Gc (`strftime_now`, bos/eos
binding). Order of work: finish minja 5B→5F, then 5Gb/5Gc, then wire 5H.

**Target shape.**
- **Load**: in each arch loader (or `load_gguf_to_llm`), read
  `'tokenizer.chat_template' kv_string kv` once and store it on the llm noun
  (new index, e.g. `llm_chat_template_g`). If the KV is absent (older GGUFs),
  fall back to the existing bespoke `*_chat_prompt` — keep them as the
  no-template fallback, not the primary path.
- **Render**: `chat_prompt` (util/chat.ijs dispatch) becomes: convert the
  `<role ; content>` message boxes to a minja Value array of `{role, content}`
  objs, `mk_inputs` (add_generation_prompt=1), `ct_apply (inputs ; options)`
  → prompt string. `add_generation_prompt=1` reproduces the bespoke verbs'
  appended gen prompt (`<start_of_turn>model`, `<|im_start|>assistant`, ...),
  which the template itself now emits.
- **Options**: `mk_options` must set `use_bos=0`/`use_eos=0` for every arch.
  BOS is tokenizer-owned (AGENTS.md): llama3/gemma tokenizers prepend bos and
  their templates omit `<|begin_of_text|>` (llama32_chat_prompt explicitly
  omits it) — use_bos=1 would double-bos and break the token-stream oracles.
  The `now`/`define_strftime_now` path (5Gc) replaces the hardcoded
  `llama32_today_date`; pin `now` in tests via the options.
- **Stop tokens stay per-arch** — they come from the vocab, not the template;
  `chat_stop_tokens` is untouched.
- **Persistent chat resume** (chat_session_g) is unaffected: it stores the
  token stream + messages, and only re-renders the NEW segment; the renderer
  runs only on the fresh path (`chat_fresh`) and the first-turn of a resume.

**Oracle/verification.** test_chat.ijs pins prompts against llama-cpp-python;
the minja-rendered GGUF template should match those `_input_ids` EXACTLY (both
render the same template with the same engine). Add a 5H test that: parses the
GGUF, reads `tokenizer.chat_template`, renders a single user message, and
compares to the existing llama-cpp-python references — this is the swap's
correctness gate and doubles as an end-to-end minja+chat_template engine test
(test-supported-template.cpp goldens).

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
