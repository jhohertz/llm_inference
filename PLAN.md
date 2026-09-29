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
a reason. Full done-work detail is recorded in **docs/HISTORICAL.md**; the
remaining planned work is below.

## Roadmap — Planned Work

Each item is independently shippable; the suite must stay green after each item.

### Deferred J-idiom applications

Deferred ideas about applying a jforc idiom to *our* code (the general reviews
live in docs/J-KNOWLEDGE.md):

- **`LoopWithInitial` (Ch 36)** — the tool if a small-state fold ever appears
  (e.g. piece accumulation in a tokenizer) where space is not a concern; the
  generation loop stays a `while.` because it carries per-step KV tensors too
  large to materialize looplessly.

## Key Reference

- llama.cpp: `llama.cpp/` checkout (`src/models/*.cpp`, `src/llama-graph.cpp`)
- minja: `reference/minja/include/minja/minja.hpp` + `chat-template.hpp`, tests `test-syntax.cpp`/`test-polyfills.cpp`/`test-capabilities.cpp`/`test-supported-template.cpp`; golden generator `scripts/minja_goldens.py`
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- J for C Programmers (JfC): https://www.jsoftware.com/help/jforc/
- J Primer "Precedence": https://www.jsoftware.com/help/primer/precedence.htm
