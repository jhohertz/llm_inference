# llm_inference — GGUF language model inference in J

**Run small LLMs — Gemma, Llama, Qwen, Granite, ERNIE, LFM2 and more — from
GGUF weights, entirely in J (J9.7).** No native code, no bindings, no
`llama-cpp` under the hood: a model-agnostic GGUF parser loads the weights and
each architecture has its own pure-J forward pass.

This is an **educational / research engine**, not a general end-user inference
server. It is deliberately *simple over fast*: an interpreted-J, single-token
inference loop trades throughput for transparency. It exists to be read, taken
apart, and experimented with — and its logits are verified **exact** against
`llama-cpp-python`, so you can trust the numbers while you explore.

---

## Why this exists

Most inference engines are C/C++/CUDA with a thin scripting wrapper. This one
is the inverse: the whole engine — GGUF parsing, quantization decode, tokenizers,
attention, KV cache, sampling, chat templates — is written in J, the array
language, so that the *entire* forward pass is inspectable line by line.

It is suited to:

- **Learning how LLMs actually work** — read the architecture modules and
  kernels, not a black box.
- **Studying J** — a non-trivial, performance-sensitive, multi-architecture
  codebase that exercises rank, tacit composition, boxing, and numeric
  representation.
- **Prototyping / research** — swap in any supported GGUF and get verified
  logits you can diff against the reference implementation.

It is **not** suited to (see [Limitations](#limitations)): serving a chatbot
at interactive speed, running large models, or production workloads.

---

## Features

- **Model-agnostic GGUF parser** — reads any GGUF: header, KV pairs, tensor
  infos, and weight data. F32 / F16 / BF16 tensors plus the **complete quant
  family** (Q2_K…Q8_K, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, and the IQ variants) decode to
  exact float values.
- **Multi-architecture inference** — each supported model architecture has its
  own module: Gemma3, generic Llama (SmolLM2 / Llama-3.2), Granite, ERNIE-4.5,
  Qwen2.5-Coder, Qwen3, Qwen3.5 (hybrid SSM), and LFM2 (hybrid conv).
- **Verified exact logits** — every architecture's inference output is checked
  token-for-token against `llama-cpp-python`; the test suite runs bit-exact
  oracles.
- **Built-in BPE + SentencePiece tokenizers**, matching llama.cpp's
  pre-tokenization and merge logic.
- **Chat templates** — per-architecture instruct chat rendering with a
  persistent multi-turn session (KV-cache resume across turns).
- **Sampling** — temperature, top-k, top-p, min-p.
- **Model catalog + downloader** — reference a model by id, Hugging Face path,
  or URL and it downloads to a per-user model folder.
- **Three ways to run it** — one-shot CLI, interactive chat console, or the
  J API — plus a GGUF inspector.
- **An OOP wrapper** (`conew`) if you prefer objects over box-of-boxes.

---

## Install

### Path A — from GitHub (J package manager)

Once the repository is published as the J addon **`llm_inference`**, install it
from any J console with the standard addon installer:

```j
load 'pacman'
install 'github:jhohertz/llm_inference'
```

### Path B — from a checkout (local install)

Clone the repository and run the local install script, which copies the addon
into your J runtime's addons folder (`~addons`):

```bash
./scripts/install_local.sh --force
```

`scripts/jfind.sh` discovers your J install under `$HOME` (`~/j9.x`, e.g.
`~/j9.7`); override with `$JINSTALL` if needed. Re-run it after updating a
checkout. Then, from any J console:

```j
require 'llm/inference'
```

---

## Quick start

### One-shot generation from the shell

```bash
./scripts/llm.sh 'gemma-3-270m-it' 'Hello world'            # until stop token
./scripts/llm.sh 'gemma-3-270m-it' 'Hello world' 50         # cap at 50 tokens
./scripts/llm.sh 'gemma-3-270m-it' @prompt.txt              # prompt from a file
```

Models load from the catalog by id (first use downloads them), or pass any
GGUF path / Hugging Face spec:

```bash
./scripts/llm.sh './models/gemma-3-270m-it-F16.gguf' 'Hello world'
./scripts/llm.sh 'unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-BF16.gguf' 'Hello world'
```

### Interactive chat from the shell

```bash
./scripts/chat.sh 'qwen3-0.6b'
```

This loads the model and opens an interactive J console. Type:

```j
llm chat 'your message'              NB. answer with the model's default params
llm chat_p ('msg' ; <temp;k;p;min_p) NB. explicit sampling params
chat_reset ''                        NB. clear the session + KV cache
exit ''                              NB. leave the console
```

The session persists across turns until `chat_reset ''` (or `exit`).

### Interactive chat from J console

```j
load 'llm/inference'
llm =: load_gguf_to_llm_inference_ 'qwen3.5-0.8b'
llm chat_inference_ 'your message'              NB. answer with the model's default params
llm chat_p_inference_ ('msg' ; <temp;k;p;min_p) NB. explicit sampling params
chat_reset_inference_ ''                        NB. clear the session + KV cache
exit ''                              NB. leave the console
```

The session persists across turns until `chat_reset_inference_ ''` (or `exit`).

### GGUF inspector

```bash
./scripts/gguf_dump.sh './models/model.gguf'   # pretty-print arch KVs + tensors
```

Standalone — does not require the addon installed.

---

## Using the J API

All public names live in the `inference` locale. After `require`, call them
with the `_inference_` suffix from any locale:

```j
llm =. load_gguf_to_llm_inference_ 'gemma-3-270m-it'        NB. catalog id (downloads if needed)
llm =. load_gguf_to_llm_inference_ './path/to/model.gguf'   NB. or a local file
```

Generate an answer (framed in the model's chat template, stops at its stop
tokens):

```j
generated =. llm generate_simple_inference_ ('Hello world' ; 20)   NB. greedy defaults
```

Raw single-pass inference (no template, returns logits) — for verification
and research:

```j
result =. llm infer_simple_inference_ 'Hello world'
NB. result = <tokens ; pred_tok ; decoded ; logits>
```

Explicit sampling params `(temp ; k ; p ; min_p)`:

```j
result =. llm infer_inference_ ('Hello world' ; <1.0 64 0.95 0.0)
```

Multi-turn chat via the chat template:

```j
msgs =. (<'user') , <'What is the capital of France?'
answer =. llm chat_generate_simple_inference_ (msgs ; 200)
```

You may also `cocurrent <'inference'` to use plain simple names inside the
addon's locale. For the precise boxed interface, sampling defaults, and
per-architecture verbs, see **AGENTS.md** (interface section).

### Per-architecture entry points

The generic `infer` / `generate` follow the most recently loaded architecture.
To pin a specific one, use its arch verb (all share the same canonical form):

| Architecture | Verbs |
|---|---|
| Gemma3 | `gem3_infer_inference_` / `gem3_generate_inference_` (+ `_simple`) |
| Llama (SmolLM2, Llama-3.2) | `llama_infer_inference_` / `llama_generate_inference_` |
| Granite-4.0 | `granite_infer_inference_` / `granite_generate_inference_` |
| ERNIE-4.5 | `ernie_infer_inference_` / `ernie_generate_inference_` |
| Qwen2.5-Coder | `qw2_infer_inference_` / `qw2_generate_inference_` |
| Qwen3 | `qw3_infer_inference_` / `qw3_generate_inference_` |
| Qwen3.5 | `qw35_infer_inference_` / `qw35_generate_inference_` |
| LFM2 | `lfm2_infer_inference_` / `lfm2_generate_inference_` |

### Object wrapper

Prefer objects? `util/llmobj.ijs` wraps a loaded model so you don't juggle
box-of-boxes:

```j
require 'llm/inference/util/llmobj.ijs'
obj =. './models/model.gguf' conew 'llmobj'
text =. generate__obj ('Hello' ; 20 ; <1.0 64 0.95 0.0)
destroy__obj ''
```

---

## Models

**Models are not bundled.** `load_gguf_to_llm_inference_` accepts any of:

- a **catalog id** — `'gemma-3-270m-it'`, `'qwen3-0.6b'`, `'smollm2-360m'`, …
- a **Hugging Face path** — `'owner/repo[/subdir]/file.gguf'`
- a **full URL** — `'https://huggingface.co/.../resolve/main/...gguf'`
- a **jpath** — `'~models/...'`
- a **filesystem path** — `'./models/model.gguf'` or `/abs/path.gguf`

HF specs and URLs download (if needed) to the per-user model folder `~models`
(`~/j9.x-user/models` — **not** the J install dir), cached by HF layout.
List the catalog with `model_list_inference_ ''`; resolve / download with
`model_path_inference_` / `model_download_inference_`.

Supported architectures and catalog ids:

| Model | Arch module | Catalog ids |
|---|---|---|
| Gemma3 | `gemma3` | `gemma-3-270m-it`, `gemma-3-1b-it` |
| SmolLM2 | `llama` | `smollm2-135m`, `smollm2-360m`, `smollm2-1.7b` |
| Llama-3.2-1B | `llama` | `llama-3.2-1b` |
| Granite-4.0-350m | `granite` | `granite-4.0-350m` |
| ERNIE-4.5-0.3B | `ernie4_5` | `ernie-4.5-0.3b` |
| Qwen2.5-Coder | `qwen2` | `qwen2.5-coder-0.5b/1.5b/3b` |
| Qwen3 | `qwen3` | `qwen3-0.6b`, `qwen3-1.7b` |
| Qwen3.5 | `qwen35` | `qwen3.5-0.8b`, `qwen3.5-2b` |
| LFM2 | `lfm2` | `lfm2-350m`, `lfm2-700m`, `lfm2.5-230m`, `lfm2.5-1.2b-instruct`, `lfm2.5-1.2b-thinking` |

---

## Limitations

Be honest about what this engine is:

- **Slow.** Interpreted J, single-token decode — on the small catalog models,
  roughly **a few tokens per second**, and still orders of magnitude behind
  C/C++ engines (larger models slower still). Long generations are minutes, not
  seconds. This is the price of the whole forward pass being readable J.
- **Memory.** Weights are decoded to J float arrays; a model's F32 footprint is
  its working size. Large models (many GB) need matching RAM, and big-context
  models balloon (a 262144-context model is ~51GB/seq of KV at full context —
  bound it with `kv_max_seq_g`). Practical sweet spot is the small models in the
  catalog. (Actually, these become native J floats, so even bigger)
- **Single-token decode** — generation is inherently sequential; batching exists
  but the interactive path is one sequence.
- **Not a server.** There's no HTTP/gRPC layer, no streaming protocol, no
  concurrency. It's a console / API engine.
- **Not a drop-in `llama-cpp`.** Some architecture quirks (e.g. the Qwen3.5 MTP
  next-token-prediction head, the Granite hybrid) are out of scope; the parser
  still reads the files, inference covers the supported arches above.
- **Models are not bundled** — you download or supply the GGUF files.

---

## Documentation

This README is the entry point. The deeper material lives in the other docs,
which you should read when you start editing or investigating:

- **AGENTS.md** — operational guide: file map, how to run, the exact
  infer/generate/chat interface, GGUF parser API, debug tips, and J gotchas.
- **docs/ARCHITECTURE.md** — architecture & implementation: GGUF layout, weight
  format, quant decode, KV cache, RoPE variants, attention, generation loop,
  chat sessions, per-architecture notes.
- **docs/J-KNOWLEDGE.md** — the J language knowledge base (jforc idiom reviews +
  gotchas), project-agnostic and reusable in any J project. Load it before
  writing or editing J code.
- **docs/HISTORICAL.md** — the origin story, resolved limitations, and the
  performance pass — what was tried, measured, and why.
- **PLAN.md** — roadmap and planned work.

---

## Repository layout (brief)

The addon ships as `llm/inference` (the repo is `llm_inference`).

```
inference.ijs        entry point (loads the architecture modules)
gguf_dump.ijs        GGUF info pretty-printer
llm_cli.ijs          one-shot CLI (scripts/llm.sh)
chat_launch.ijs      interactive chat console (scripts/chat.sh)
models/              per-architecture forward passes (gemma3, llama, granite, ...)
tokenizers/          BPE + SentencePiece tokenizers
kernels/             float kernels (jfloat.ijs)
gguf/                GGUF parser + quant decoders
util/                KV cache, llm core, sampler, chat, model catalog, llmobj
tests/               test suites (run tests/j/run_all_tests.sh from the checkout)
```

For the full file-by-file map and how to run, see **AGENTS.md**.

---

## Special thanks

This project would never have happened, if not for these people/projects:

- Michael Dykman [@mdykman](https://github.com/mdykman), who helped me discover J, and much more.
- [J Software](https://www.jsoftware.com/) / [@jsoftware](https://github.com/jsoftware), for implementing J.
- [Llama.cpp](https://github.com/ggml-org/llama.cpp), for bringing GGUF to the world, and providing a reference.
- [OpenCode](https://opencode.ai/), Which served as the AI harness for this synthesis.
- [DeepSeek](https://huggingface.co/deepseek-ai), for their DeepSeek-4-Flash model, used to drive this.
- [DwarfStar](https://github.com/antirez/ds4), for bringing DeepSeek inference to the homelab.
