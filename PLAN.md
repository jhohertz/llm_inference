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

## Key Reference

- llama.cpp: `llama.cpp/` checkout (`src/models/*.cpp`, `src/llama-graph.cpp`)
- minja: `reference/minja/include/minja/minja.hpp` + `chat-template.hpp`, tests `test-syntax.cpp`/`test-polyfills.cpp`/`test-capabilities.cpp`/`test-supported-template.cpp`; golden generator `scripts/minja_goldens.py`
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- J for C Programmers (JfC): https://www.jsoftware.com/help/jforc/
- J Primer "Precedence": https://www.jsoftware.com/help/primer/precedence.htm
