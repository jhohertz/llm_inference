# LLM in J — Architecture Reference

This document preserves the architecture and design details of the GGUF-based
language model inference engine written in J (J9.8). It is a reference for
implementing new model architectures and debugging. For operational guidance
(file map, J gotchas, performance idioms) see **AGENTS.md**. For status and
roadmap see **PLAN.md**.

## Overview

Generic GGUF-based language model inference. A model-agnostic GGUF parser
loads weights into native J arrays; each supported architecture has its own
module that consumes the parser and implements the forward pass.

The generic entry `load_gguf_to_llm` reads the model's `general.architecture`
KV once (`detect_arch`) and playsound-style maps `infer`/`generate` onto the
loaded arch's verbs (`gem3_*` / `llama_*` / `qw2_*`) — the check is made at load
time, not at each call. The llm noun carries its arch at index 9 (`llm_arch`).

```
inference.ijs (entry point)   — all code lives in the 'inference' locale
├── gguf/gguf.ijs       — Generic GGUF parser: parse_hdr, parse_kv_pairs, parse_tensor_infos, load_tensor_data
├── util/kv_cache.ijs   — Persistent KV cache: kv_cache_g/kv_meta globals, monadic create/write/write_rows/read/reset (no threading)
├── gguf_dump.ijs      — Utility: pretty-print any GGUF file
 ├── tokenizers/tokenizer_llama3.ijs / tokenizers/tokenizer_gpt2.ijs — BPE tokenizers
 ├── tokenizers/tokenizer_spm.ijs   — SentencePiece tokenizer (llama.cpp llm_tokenizer_spm bigram-merge)
 ├── kernels/jfloat.ijs    — matmul, linear, RMSNorm, GELU, SiLU, SwiGLU, RoPE, softcap
 ├── util/llm_core.ijs   — llm accessors, get_tensor_cached_d, embed_tokens, output_head, sample_from, infer_args, gen_args
 ├── models/gemma3.ijs     — Gemma 3 270M: attention+KV, FFN, blocks, gem3_infer/gem3_generate
 ├── models/llama.ijs      — generic llama arch (SmolLM2 + Llama-3.2): standard decoder, GQA, SwiGLU, interleaved RoPE
  ├── models/granite.ijs    — Granite 4.0/4.1/4.2: standard decoder + Granite scaling (data-driven: embed*12, residual*resid, scores*0.015625, logits/logit_scale), tied embeddings
 ├── models/ernie.ijs      — ERNIE-4.5-0.3B: standard decoder byte-for-byte the llama arch (tied embeddings), reuses llama.ijs forward verbs, SPM tokenizer
 ├── models/qwen2.ijs      — Qwen2.5-Coder: standard decoder, GQA, SwiGLU, NEOX RoPE, Q/K/V biases
├── util/llmobj.ijs     — OOP wrapper: conew 'llmobj', infer__obj/generate__obj/destroy__obj
└── util/sampler.ijs    — temperature / top-k / top-p / min-p sampling
```

## Supported Models

| Model | Module | Blocks | Emb | FFN | Q Heads | KV Heads | Head Dim | SWA | RoPE | Tokenizer |
|-------|--------|--------|-----|-----|---------|----------|----------|-----|------|-----------|
| Gemma 3-270M | `gemma3.ijs` | 18 | 640 | 2048 | 4 | 1 | 256 | 512 | NEOX | llama3 |
| SmolLM2-360M | `llama.ijs` | 32 | 960 | 2560 | 15 | 5 | 64 | none | interleaved | gpt2 |
| Llama-3.2-1B-Instruct | `llama.ijs` | 16 | 2048 | 8192 | 32 | 8 | 64 | none | interleaved | llama-bpe |
| Granite-4.0-350m | `granite.ijs` | 28 | 1024 | 2048 | 16 | 4 | 64 | none | interleaved | dbrx |
| ERNIE-4.5-0.3B | `ernie.ijs` | 18 | 1024 | 3072 | 16 | 2 | 128 | none | interleaved | spm |
| Qwen2.5-Coder-0.5B | `qwen2.ijs` | 24 | 896 | 4864 | 14 | 2 | 64 | none | NEOX | gpt2 |
| Qwen3-0.6B | `qwen3.ijs` | 28 | 1024 | 3072 | 16 | 8 | 128 | none | NEOX | gpt2 |
| Qwen3.5-0.8B | `qwen35.ijs` | 24 | 1024 | 3584 | 8 | 2 | 256 | none | NEOX | gpt2 |
| LFM2-350M | `lfm2.ijs` | 16 | 1024 | 4608 | 16 | 8 | 64 | none | interleaved | gpt2 |
| LFM2.5-230M | `lfm2.ijs` | 14 | 1024 | 2560 | 16 | 8 | 64 | none | interleaved | gpt2 |

Architecture KV prefix: Gemma3 uses `gemma3.*`, SmolLM2 uses `llama.*`, Qwen2
uses `qwen2.*`. Each `*_extract_hparams` reads the model-specific KV keys.

## GGUF File Layout (from spec)

```
[24 bytes header]    magic(4) + version(4) + tensor_count(8) + kv_count(8)
[KV pairs]           sequential: key_len(8) + key + value_type(4) + value
[Tensor infos]       sequential: name_len(8) + name + n_dims(4) + dims + type(4) + data_offset(8)
[Padding]            to ALIGNMENT=32
[Tensor data]        aligned, data_offset relative to this region
```

**Tensor infos come IMMEDIATELY after KV pairs — no scanning needed.**
After parsing all KV pairs, `kv_end` is the offset where tensor infos begin.

Tensor data start (aligned to 32):
```j
tds =. 32 * <. (kv_end + (#ti) + 31) % 32
```

### Element Type Names

| Code | Type | Bytes |
|------|------|-------|
| 0 | F32 | 4 |
| 1 | F16 | 2 |
| 13 | Q8_0 | 2 |
| 30 | BF16 | 2 |

### VT9 Array Element Sizes

| et | Type | Bytes/Item |
|----|------|-----------|
| 0 | uint32 | 4 |
| 1 | float16 | 2 |
| 2 | int16 | 2 |
| 3 | int32 | 4 |
| 4 | uint32 | 4 |
| 5 | int32 | 4 |
| 6 | float32 | 4 |
| 7 | bool | 1 |
| 8 | string | variable |
| 10 | uint64 | 8 |
| 11 | int64 | 8 |
| 12 | float64 | 8 |

## Numeric Representation & F16/BF16 Decode

J9.8 has NO native float32 type; only float64 (8) and complex (16) exist. All
GGUF numeric decoding goes through the native verbs — **never DIY IEEE 754**:

- `3!:5` dyad: `_1(3!:5)` 4 chars → float32, `_2(3!:5)` 8 chars → float64.
- `3!:4` dyad: `_1(3!:4)` 2 chars → **signed** int16 (0xFFFF → `_1`),
  `0(3!:4)` 2 chars → **unsigned** uint16 (0xFFFF → 65535), `_2(3!:4)` 4 chars
  → uint32, `_3(3!:4)` 8 chars → uint64. The unsigned `0` form indexes a
  0..65535 lookup table directly.
- `3!:4`/`3!:5` interpret chars little-endian (last char = MSB) — no byte
  reversal needed for GGUF.

**CRITICAL: `3!:4`/`3!:5` return 1-element arrays, NOT scalars** — extract
with `0 { ... }`. And `_3(3!:4)` parses as `_ 3(3!:4)` (negate applied), so
use an explicit `4 : 0` wrapper for the 64-bit readers:
```j
le64 =: 4 : 0
  chars =. (x+i.8) { y
  0 { _3(3!:4) chars
)
```

### F16 tensors — unsigned read + 65536-entry table

GGUF etype 1 stores float16 as uint16: sign(1bit) | exponent(5bit, bias=15) |
mantissa(10bit). `f16_load` decodes a flat `ne*2`-byte slice in two steps —
bulk unsigned read + table gather (no reshape/ravel, no signed→unsigned pass):
```j
ints =. 0 (3!:4) raw   NB. unsigned uint16 0..65535, directly indexes table
ints { f16_table       NB. precomputed F16->float64 for all 65536 values
```
The old `_1(3!:4)` path returned SIGNED int16 and needed
`ints + 65536 * ints < 0` plus a `(ne,2)$raw` reshape + `,chars` ravel — three
extra full-array passes (~540MB copies) that cost ~0.6s per 270M values.
`0(3!:4)` skips all three (270M values 1.36s → 0.71s; load 4.25s → 3.79s).

`f16_load` is fully tacit (Ch 42) — table right-bond, second-box accessor,
unsigned read; drops runtime lexical handling (~2× on 270M):
```j
f16_load =: ({&f16_table) @ (0&(3!:4)) @ (>&(1&{))
```

### Loader file I/O — memory-mapped (jmf)

`load_gguf_to_llm` maps the GGUF file ONCE (`mmap_gguf` in gguf/gguf.ijs, jmf
addon) instead of `1!: 1 < path` — pages fault in lazily on each slice fetch,
so the full 540MB-1GB file is never materialized. The mapped array is passed
to detect_arch + the arch loader, which parse the header/KVs/tensor infos
from it (`parse_hdr_raw` / `parse_kv_pairs_raw` — no file reads anywhere).
`load_tdata` slices the mapped array with the index-list `(off + i. n) { raw`
(copies only the slice). One model per load: the mapping is unmap'd after
load (`unmap_gguf`), and the llm noun stores `''` at kvs_ctx (load-time only).

Do NOT use `n {. off }. raw` for ALL slices: `{.` (take) on a mapped array is a
view, but `}.` (drop) with a nonzero offset materializes the suffix — a tail
copy per tensor that caused a 20x load regression (off `}.` copies
`file_size - off` bytes; Σ over every tensor ≈ 130GB). The loader uses a
**hybrid** (gguf/gguf.ijs `load_tdata`): take-of-drop is ~memcpy (0.25ns/B)
vs the index-list's n-int alloc + element fetch (~4.6ns/elem), so it picks
take-of-drop when the tail copy is cheaper — `tail < ne * 18` — covering the
big early `token_embd` (tail ≈ file size < ne×18 for a 155M-elem embedding)
and the late small-tail tensors, and index-list for mid-file tensors with
big tails. qwen2 load 4.86s → 4.36s, gemma 2.72s → 2.16s. `require 'jmf'`
switches the current locale to `jmf`, so `cocurrent 'inference'` follows it
in gguf.ijs. Mapped nouns ALIAS: assigning to a noun aliasing a mapped file
writes THROUGH to the file (we corrupted the gemma model file this way
during bring-up) — read-only use only, keep model files out of write paths.

`f16_build_table` vectorizes the sign/exponent/mantissa split over all 65536
uint16 values at once. Bit extraction uses the boolean-verb `17 b.` (bitwise
AND = `1 + 16 b.`) — arithmetic `32 | <. ints % 1024` leaks the sign bit into
the exponent. Masks: `sign = <. ints % 32768` (bit 15), `exponent = (<. ints % 1024) 17 b. 31`
(bits 10-14), `mantissa = ints 17 b. 1023` (bits 0-9). Decode algorithm:
```j
norm =. smult * (2 ^ (exponent - 15)) * (1 + mantissa % 1024)   NB. exponent in 1..30
sub  =. smult * (2 ^ _14) * (mantissa % 1024)                   NB. exponent = 0
spec =. (_ * smult) * (31 = exponent) *. 0 = mantissa           NB. ±inf; 31=exp & 0<mantissa = NaN
value =. norm * (0 < exponent) *. exponent < 31 + sub * (0 = exponent) *. 0 < mantissa + spec * (31 = exponent) *. 0 = mantissa
```
All 65536 uint16→float16 values validated against Python reference (max
relative error 3.77e-09). J display: `_` = +inf, `__` = -inf, `_.` = NaN,
`_0.001` = negative finite.

### BF16 tensors — zero-extend to F32

BF16 is F32 with low 16 bits truncated. `bf16_to_f32` zero-extends each LE
uint16 value with 2 leading zero chars, then `_1(3!:5)` decodes (no reversal;
J's `3!:5` is LE). Tacit (Ch 42, `13 :` output):
```j
bf16_to_f32 =: ((_1) 3!:5 [: , (0 00{a.) ,"1 (2 ,~ 2 %~ #) $ ])
```
`decode_bf16` = `([: bf16_to_f32 ([: (i.) 2 * [: > 0 { ]) { [: > 1 { ]) f.`
— `f.` inlines `bf16_to_f32` at definition (self-contained, no runtime name
lookup). `f32_decode`/`f64_decode` are the converter's tacit forms too.
Measured (100M values): f32/f64 tacit ≈ explicit (±2%), decode_bf16 `f.` no
gain (0.72s) — the name lookup is once per tensor, not per atom; the value is
scheduler flexibility + self-contained definitions.

## Weight Format — 2D GGUF WEIGHTS ARE `[out, in]`, NOT `(in, out)`

GGUF stores each 2D tensor's `dims` field as `[in, out]` (e.g. `attn_q` dims
`640 1024`), but the **actual data is stored reversed** as `[out, in]`
(`(1024, 640)`). The dims field does NOT match the data layout.

```j
NB. CORRECT — reshape every 2D tensor as (|. dims) → [out, in]
result =. dims tensor_reshape flat     NB. dims=(in,out) → reshape (out,in)
```

Then `linear` is **weight-first** (last axis of weight = input size):

```j
result =. weight (+/ .* ) input    NB. (out,in) +/.* (in,) → (out,)
```

**Transposed-canonical embedding/output storage (`emb_canonical`, 2026-08).**
J's matvec `(m,n)x(n,)` is ~3.9x slower than the vector-matrix `(n,)x(n,m)`,
so `token_embd.weight` is stored TRANSPOSED at load (via `emb_canonical`):
`emb_w` is `(emb, vocab)` (same bytes, layout swapped — J has no view/stream,
`|:` materializes, so store transposed as canonical). Embedding lookup is a
column access `|: (tok {"1 emb_w)` (~free); the lm_head is a vector-matrix
`hidden (+/ .* ) emb_w`. The per-layer weights stay `[out, in]` (weight-first
matvec — the transposed form is measured slower there for the small matvecs).

```j
NB. CORRECT (post emb_canonical) — emb_w is (emb, vocab), NOT (vocab, emb)
hidden =. scale * |: (tok {"1 emb_w)   NB. embedding column access
logits =. hidden (+/ .* ) emb_w        NB. lm_head vector-matrix (emb,)x(emb,vocab)
```

**Why this matters**: matmuls with the wrong axis silently produce plausible-but-wrong
numbers, and hidden-state RMS grows monotonically across blocks instead of staying ~1.
Always check the first 2D weight's actual shape against its dims field.

## Implementation idioms (loopless)

These are the loopless J forms the loader and arch modules actually use; each
replaces a `while.` scan/build loop (the general idiom lives in J-KNOWLEDGE.md).

- **Tensor-cache search → boxed `i.` index search.** `get_tensor_cached_d`
  (`util/llm_core.ijs`) and `find_tensor_idx` (`gguf/gguf.ijs`) use
  `names =. (step * i. n) { list; idx =. names i. <name` (boxed `i.` matches
  exact boxes, no padding) instead of a `while.` scan stepping by 4/7.
  Verified: `names i. <'blk.0.attn_q.weight'` → 1, missing → `#names`.
- **Block-data prebuild → `build_block each i. block_count`.** The rank-1
  `*_build_block` verbs (x=llm, y=block idx) replace the `while. b < block_count`
  build loops; each prebuild is `(<llm) build_block each i. block_count`.
- **Boxed-arg unpacking → multiple assignment.** `'a b c' =. y` replaces
  `> N { y` per-item unbox in `rms_norm` (`'eps weight input' =. y`), the
  per-arch `*_block_forward`/`*_attention` (`'block_data pos [swa] mi layer' =. y`),
  and `*_run_blocks` (`'llm pos' =. args`) — fewer box ops in the per-block hot path.

## Tokenizer

Two BPE tokenizers are used, selected by the GGUF's `tokenizer.ggml.model`:
- **Llama3-style** (`tokenizer_llama3.ijs`): regex pre-tokenizer, vocab hash,
  byte fallback. Used by Gemma3 (`tokenizer.ggml.model = llama3`).
- **GPT-2 byte-level** (`tokenizer_gpt2.ijs`): byte-level BPE with
  `gpt2_regex` pre-tokenizer. Used by SmolLM2 and Qwen2.5-Coder
  (`tokenizer.ggml.model = gpt2`, `pre = qwen2`).

Both build from `tokenizer.ggml.tokens` (vocab) and `tokenizer.ggml.merges`.
Byte tokens ≥ 65536 are clamped (`tokens - 65536 * -. tokens < 65536`), not
filtered — only 65536-65791 are byte tokens, 65792+ are vocab.

**Byte-token decode is a vectorized selection, not a per-token loop.** The
detokenizer computes a single `sv_byte` mask once, fetches vocab only for
non-byte tokens (`(tid_list * -. sv_byte) { vocab`), builds byte chars via
`a. {~ tid-65536`, and selects per token with `; sv_byte {"_1 vb ,. bc` — this
removed the per-token `while.` loop + 3-way if/elseif/else (a ~3% runtime win).

## KV Cache

One unshared pair of flat arrays lives in the `inference` locale — refcount 1
nouns, so `m}` amend fires IN PLACE (measured: scalar row ~1us, 10-row list
~6us on a 102MB array). The representation is aligned with llama.cpp's
kv-cache (allocate once per context, write in place, reset between
generations):

- `k_cache_g` / `v_cache_g` = ONE flat `(n_layers * eff_seq, n_heads_kv * head_dim)`
  array per kind; layer stride = `eff_seq`, position stride = `n_kv*hd`.
  Positions are the LEADING axis — J's in-place amend requires a refcount-1
  noun with a simple scalar/list selector, and only a leading-axis selector
  fires in place (per-layer boxes force refcount-2 on unbox → writes copy the
  whole layer, measured 1.7GB/s — rejected).
- **Batched decode adds a B-axis**: `kv_batch_g` parallel sequences (default
  1); `k_cache_g` is `(n_layers * kv_batch_g * eff_seq, n_kv*hd)` and the row
  base is `(layer * kv_batch_g + seq) * eff_seq`. `kv_seq_g` selects the
  current sequence for the 4-item `kv_write`/`kv_read` calls (batched prefill
  sets it per sequence); `*_attention_bd` passes an explicit 5th batch index
  (`b`) instead. `kv_create` reallocates when `kv_batch_g` changes (the
  alloc-batch is tracked in `kv_batch_alloc_g`).
- `kv_meta` = `<n_layers; eff_seq; n_heads_kv; head_dim>`; `kv_pos_g` = the
  used length (all layers write together); `kv_max_seq_g` = a low-memory
  context override (default `_1` = model max); `eff_seq = min(max_seq, kv_max_seq_g)`.

Writes are IN-PLACE amends (no array copy): `kv_write` amends one row
(`(, k_new) (base + pos)} k_cache_g`, scalar selector), `kv_write_rows` amends
L contiguous rows (list selector) — both O(cell). Reads gather the window
rows `(base + i. count) { k_cache_g` (an O(pos) copy — unavoidable in J, no
views) and reshape to `(count, n_kv, hd)` (free on the contiguous data).

Session-persistent: `kv_create` ALLOCATES once per session; a repeat call with
identical dims just resets `kv_pos_g` (no realloc per generate). `kv_reset`
only resets `kv_pos_g` — the buffer is reused; stale rows beyond the used
length are never read (every position is written before it is read).

The API is monadic and the cache is NEVER threaded through the forward pass:

```j
kv_create ((<n_layers) , (<max_seq) , (<n_heads_kv) , (<head_dim))   NB. allocate/reuse flat buffers; eff_seq from kv_max_seq_g
kv_write ((<layer) , (<pos) , (<k_new) , (<v_new))                  NB. in-place row amend (n_kv, hd)
kv_write_rows ((<kind) , (<layer) , (<start) , <rows)               NB. in-place L-row amend (L, n_kv, hd)
result =. kv_read ((<layer) , <pos)        NB. → <k_all; v_all>, each (count, n_kv, hd), count=pos+1
kv_reset ''                                NB. reset kv_pos_g (buffer reused, no realloc)
```

Why the flat refcount-1 form: the old growing per-layer boxed arrays appended
via `,` — an O(used) full-array copy PER WRITE (O(n^2) total over a
generation; unusable at 128K). Per-layer boxes give O(1) reads but a second
ref defeats in-place (writes copy the layer). The flat form keeps writes
O(cell); the read window copy moves to the read (same O(pos) bytes — the
long-context floor is the window materialization, unavoidable in J).

**Recurrent-state caches are flat too (2026-08).** The hybrid arches' recurrent
states are NOT the attention KV cache but the same boxed-per-layer anti-pattern
was eliminated: qwen35's `rs_conv_g`/`rs_s_g` and lfm2's `lf2_conv_cache_g` are
ONE flat positions-leading array `(n_arch * kv_batch_g * slice, ...)` with an
in-place list-selector amend (`rs_conv_g =: (, conv) (base + i. slice)} rs_conv_g`,
`lf2_conv_slice` base `(ord*kv_batch_g + seq)*(d_conv*emb)`). The old boxed amend
unboxed a refcount-2 global and COPIED the whole batch state per write (qwen35
16.8MB at B=8, ~0.0038s/call — the #1 decode cost); the flat amend fires ~78x
faster and cut qwen35 decode ~3x. lfm2 conv state is `(kv_batch_g, d_conv=2, emb)`.
Both are force-reset between generations (`rs_reset`/`lf2_conv_reset`).

**K transpose (1 2 0 |:) remains** — the single-token path must turn the
positions-leading `(win, n_kv, hd)` slice into `(n_kv, hd, win)` for the
scores matmul, and this cyclic permutation is strided (~1GB/s vs ~25GB/s
contiguous — the dominant cache-path cost at 8k+, ~0.07s/token on qwen2.5-0.5b).
Storing K transposed (llama.cpp's layout) is blocked in J: per-layer boxes
copy the whole layer per write (refcount-2, measured 1.7GB/s — worse than the
transpose); a flat all-layers transposed array needs a full-column x-build per
layer per write (~15ms/token) and bulk prefill writes become per-column loops
(+60% prefill at 8k). The two-swap transpose is slower than one-pass cyclic.
The transpose is accepted as the J-floor for positions-leading storage.

`kv_create`/`kv_reset` are called once at the top of each `*_infer` /
`gen_loop_core` fresh mode (kv_create reuses the buffer on repeat calls —
session-persistent); the arch `*_run_blocks`
also create if `kv_meta` is empty (defensive). Chat-session RESUME mode skips
`kv_create` — it continues from the existing cache at `start_pos`.

## RoPE Variants

Two layouts exist, set per-architecture (llama.cpp `llama_model_rope_type`):

- **Interleaved (NORM)** — pairs are consecutive `(i, i+1)`. Used by SmolLM2
  (llama arch). Kernel: `rope_apply2_t`.
- **NEOX** — pairs are offset by `head_dim/2` (`(i, i+half)`). Used by Gemma3
  and Qwen2 (`LLAMA_ROPE_TYPE_NEOX`). Kernel: `rope_apply2_neox_t`.

Both share the same cos/sin tables (`theta_i = pos * freq^(-2i/dim)`); only the
pair selection differs. Tables are precomputed at load for all positions via
`build_rope_tables` (removes trig from the per-token loop).

Gemma3's reference applies RoPE via a reshape-permute-reshape that interleaves
group/half data (essential for correctness — the reference rotates interleaved
pairs, not per-head):
```python
x = x.reshape(b, s, n_grp, 2, d//2)
x = x.permute(0, 1, 4, 2, 3)   # [b, s, d/2, n_grp, 2]
x = x.reshape(b, s, n_grp, d)
rope(x, t)
```

## Attention Forward Pass (single-token, autoregressive)

1. Q projection → `[n_heads, head_dim]`
2. K/V projections → `[n_heads_kv, head_dim]`
3. Apply RMSNorm to Q/K (Gemma3), or Q/K/V biases after projection before RoPE (Qwen2)
4. Apply RoPE to Q and K
5. Scale Q by `1/sqrt(head_dim)`
6. Read all cached K/V: `[seq, n_heads_kv, head_dim]`
7. Scores: `Q (+/ .* ) |: K_all` → per head
8. GQA expansion, softcap, causal/SWA mask, softmax
9. Output: `scores (+/ .* ) V_all` → `[n_heads_kv, head_dim]`
10. Output projection, post-attention norm (if enabled), residual

**Attention mask broadcast — prefix rows, don't cycle.** J broadcasts only when
the vector is a **leading-axis prefix** of the matrix shape. `(n_heads, win) $ max_sf`
cycles the `(n_heads,)` vector wrongly at win≥2; `(n_heads,)` is a prefix of
`(n_heads, win)` so J broadcasts it — keep `max_sf` as the plain vector:
```j
NB. WRONG — (n_heads, win) $ max_sf cycles the (n_heads,) vector wrongly at win≥2
NB. CORRECT — (n_heads,) is a prefix of (n_heads, win); J broadcasts it
exp_sf =. ^ (scores_f - max_sf)
```

**Empty-operand safety.** An operand is empty iff its *frame* (w.r.t. the verb's
cell rank) contains a `0`; J then executes the verb on a **cell of fills**
(`0`/`' '`/`a:`) and returns frame-concat-s` — which is why broadcasting over
empty arrays works: `(0 4 $ 100) % (0 $ 0)` → shape `0 4`, no error. The
`norm_vals` prefix in `rms_norm_rows`, the `max_sf` prefix, and the `$ q_bias`
reshape-cycles stay correct even if `L`/`win`/`n_rows` is 0 (`0 % 0` → `0`).
Empty *cells* (a cell has a `0` in its shape) are handled by the verb, NOT
fill-cell processing (`3 {. ''` → 3-char, `3 {. 0$0` → 3-numeric: 'data'
operands keep type, 'control' info like `|.` count ignores it).

## Generation Loop

The KV cache **persists across all blocks and all generation steps**. Each step:
1. Embed current token → `[emb_len]`
2. Pass through all blocks with persistent KV cache
3. Final output_norm + lm_head projection

**CRITICAL — the prompt's last hidden state already predicts the next token.**
Process ALL prompt tokens once (positions 0..len-1) via batched prefill, then
predict the first generated token from that hidden state WITHOUT re-embedding.
The first generation step must NOT embed/run blocks; subsequent steps embed
the predicted token at `cur_pos`.

**`cur_pos` off-by-one gotcha**: `cur_pos` must increment ONLY inside the
block-run branch (the `else` branch of the generation loop), NOT on step 0 —
step 0 does no KV write (it predicts from the prompt's last hidden), so
incrementing there skips a cache position and corrupts attention at step≥2.

**UNIFIED `gen_loop_core`** (llm_core.ijs) replaces the four per-arch
`*_gen_loop` copies — one implementation for all arches. Per-arch differences
(embedding scale: `%: emb_len` gemma3 else 1; `*_run_blocks`/`*_run_blocks_b`
verbs) are dispatched by `llm_arch`. The arg list is
`<tokens; start_pos; max_steps; temp; k; p; min_p; stop_list>`:
- `start_pos = ''` → FRESH: `kv_create` + batched prefill at 0 (all arches
  bit-exact vs the old per-arch loops, pinned by the model suites).
- `start_pos = <n>` → RESUME (chat sessions): the cache already holds
  positions 0..n-1; the new segment is prefilled with ONE batched pass at n —
  `*_attention_b` is cache-prefix aware: RoPE at `start_pos+i.L`, prepends the
  prefix rows (0..n-1) via `kv_read ((<layer) , <(start_pos-1))`, writes the
  batch K/V at `start_pos` via `kv_write_rows`, and scores/softmax/softmax·V
  over the combined (prefix+batch) keys (causal mask covers q_pos
  `start_pos+i.L` vs key_pos 0..start_pos+L-1). Uniform across all arches.
  Verified: persisted turn-by-turn chat == stateless full re-render (12/12
  Chat Session suite), SWA Boundary 16/16, full suite 374/374/0. qwen2
  29-token resume prefill: 0.115s vs 4.81s serial per-token (~42x).
    NB: `kv_write_rows`' base must be `((a * max_seq) + start)` — `a * max_seq
   + start` is `a * (max_seq + start)` (no precedence), harmless at start=0
   (pre-resume callers) but clobbers the cache prefix at start>0.

**PER-TOKEN STREAMING CALLBACK (Phase 6 item 1)** — `gen_loop_core` calls the
monadic verb `gen_cb_g` (gated by the noun flag `gen_cb_on_g`) on each generated
token after sampling, before the stop check. The callback may return a
(possibly replaced) token id — returning a stop-token id forces a stop — so it
serves both as a streaming hook (per-delta emit) and an interception point.
Verbs can't be boxed into the y list (noun-verb syntax error) and can't be
distinguished from a noun by `-:`/`3!:0`, so it's a global verb + noun flag
rather than an argument. `chat_completion` (util/chat.ijs) is the OpenAI-shaped
 verb that uses it: `llm chat_completion (messages ; tools ; max_steps ; stream ;
 <params>)` → `<content ; finish_reason ; tool_calls>`. `stream=1` uses the
 caller's `gen_cb_g`; `stream=0` skips it. `<params>` MUST be the last `;`
 operand — a pre-boxed `;` operand that isn't trailing nests (J gotcha). 
`finish_reason` is `'stop'`/`'length'`/`'tool_calls'` (classification: Phase 6
  item 3, see below).

**STREAMING TEXT DELTAS (Phase 6 item 2)** — `chat_stream_piece` (util/chat.ijs)
 is a port of llama.cpp's streaming incremental detokenizer: it appends each
 token's raw bytes to `st_buf_g`, holds any incomplete trailing UTF-8 sequence
 (`utf8_tail`), and emits only complete characters. `chat_stream_cb` is the
 per-token callback the caller installs as `gen_cb_g` (with `gen_cb_on_g=1`);
 it reads arch/llm from `chat_cb_arch_g`/`chat_cb_llm_g` (set by
 `chat_completion`) and forwards each text delta to `chat_cb_g` (a monadic verb
 on the delta string), returning the token unchanged so the delta never leaks
 into the token stream. Streaming == batch detokenize on all tokenizer families
 (greedy, e.g. qwen3 170/170 chars); a streaming==batch test is in
  test_qwen3.ijs. Gotcha: J verb assignment `a =. b` is a dynamic ALIAS to the
  name `b`, not a copy — so `x =: gen_cb_g` then `gen_cb_g =: chat_stream_cb`
  would make the "capture" track the new value (infinite recursion). 
`chat_completion` never overwrites the caller's callback. Stop tokens are
  suppressed from the delta stream via `chat_cb_stop_g`: `gen_loop_core` breaks
  WITHOUT appending a stop token to `output`, so the batch detokenize excludes it
  — the streaming callback sees it (it fires before the stop check) and would
  leak its text. qwen3.5's `<|im_end|>` (151645) has a NON-EMPTY byte-encoded
  vocab string, so this is required for stream==batch there (verified 216/216).

**TOOL-CALL CLASSIFICATION (Phase 6 item 3)** — `chat_completion` detects a
  `<tool_call>...</tool_call>` region in the generated content: `finish_reason`
  becomes `'tool_calls'`, `content` is nulled, and `tool_calls` are extracted as
  OpenAI-shaped minja Values (`{type:'function'; function:<name; arguments>;
  id}` with `id = 'call_' , name`, arguments a JSON string). 
  `chat_extract_tool_calls`/`chat_parse_tool_call` handle the two generation
  formats:
  - qwen3.5: `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>` — parsed into a pjson key/value table and `enc`'d to a JSON string.
  - qwen3/granite (JSON): `<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>` — parsed with `dec_pjson_` (convert/pjson, added to DEPENDS; it preserves numbers/bools, unlike convert/json which coerces 0/1 to bool), arguments re-`enc`'d.
  - bare OpenAI-style JSON: qwen2.5-coder emits `{"name":..., "arguments":{...}}` WITHOUT the `<tool_call>` wrapper its template asks for — `chat_extract_tool_calls` falls back to parsing the whole content as JSON if it carries `name`+`arguments` keys (one tool call).
  Verified end-to-end (greedy): qwen3.5, qwen2.5-coder-0.5b/1.5b, granite-4.0 — `get_weather` args `{"city":"Paris"}`; test_qwen35.ijs Section 6.

**TOOL-USE LOOP** — `chat_tool_loop` (util/chat.ijs):
  `llm chat_tool_loop (messages ; tools ; max_steps ; stream ; max_rounds ;
  <params>)` drives the execute-and-recall cycle. It calls `chat_completion`;
  on `finish_reason='tool_calls'` it executes each extracted tool via the
  global verb `chat_tool_fn_g` (y = `<name ; args-JSON-string>`, returns the
  result STRING), builds an assistant message Value carrying the model's
  `tool_calls` (`{role:'assistant'; content:null; tool_calls:[...]}`) plus one
  `tool` role message Value per result (`{role:'tool'; content:result;
  tool_call_id:id}`), appends them to the history, and re-renders/re-calls
  until the model stops calling tools (cap `max_rounds`). Returns
  `<content ; finish_reason ; tool_calls_made>`. `chat_build_tool_call_msg`,
  `chat_build_tool_msg`, `chat_exec_tool` are the builders; `chat_tool_fn_g`
  mirrors the gen_cb_g global-verb pattern (verbs can't be boxed). Streaming
  re-arms `gen_cb_on_g`/`gen_cb_g`/`chat_cb_*_g` each round, so stream==batch
  holds across rounds. Verified on qwen3.5-0.8b: model calls get_weather, the
  loop executes it (`"The weather in Paris is sunny and 22C."`), and the model
  then answers `"The weather in Paris is sunny and 22°C."` (finish 'stop');
  test_qwen35.ijs Section 6. Gotcha: `max_rounds` must come BEFORE `<params>`
  (a pre-boxed `;` operand that isn't trailing nests).

**GPT2 BYTE TABLES (llama.cpp vs OpenAI)** — `gpt2_build_tables`
  (tokenizer_gpt2.ijs) originally followed OpenAI's `bytes_to_unicode`
  (identity 33..126 + 160..255, control 0..32 + 127..159 -> codepoints 256..321).
  llama.cpp's `unicode_utf8_to_byte_map` (unicode.cpp:172) treats byte 160 (0xA0)
  and 173 (0xAD) as CONTROL (not identity): identity 33..126 + 161..172 +
  174..255, control 0..32 -> 256..288, 127..160 -> 289..322, 173 -> 323. So the
  byte-encoded codepoints run 0..323 (not 0..321). The old table broke qwen3.5
  detokenize on tokens whose piece contains codepoints 322/323 (e.g. `ł`/`Ń`
  decode to bytes 0xA0/0xAD — `index error: 323 > 322`). Fixed to match llama.cpp
  (byte_tab length 324); verified tokenize/detokenize vs llama-cpp-python oracle.

**STATEFUL CHAT + STREAMING (`chat_core_stream`)** — `chat_completion` streams
but re-renders the full history each turn; the console `chat`/`chat_p` resumes
the KV cache but never streams. `chat_core_stream` (util/chat.ijs) combines
both: it takes `<msg ; max_steps ; <params>` (msg = the NEW user message only,
the session holds the history), re-renders the history ONLY to tokenize the new
segment, verifies the prefix matches the stored token stream, then **resumes
from the KV cache** (ONE batched prefill of the new segment through
`*_run_blocks_b`) AND arms the per-token streaming callback (mirrors
`chat_completion`'s stream mode). Shared helpers `chat_gen_stream`/
`chat_fresh_stream` do generate+stream+flush+reset. The stateful TUI
(`chat_tui.ijs`) calls it per turn, renders from the session's messages
(`get_msgs`), and `/reset` calls `chat_reset` (clears session + KV cache).
J gotchas: a `;` chain with a boxed operand in the middle nests
(`(messages ; max_steps ; <flat) ; <stop` → length-2) — append the trailing box
with `, <stop`; `chat_stream_stop` rebinds `chat_cb_g` to `]`, so the caller
must re-set its delta consumer each turn.

**BATCHED DECODE (`gen_loop_batch`)** — the way off the M=1 matvec floor:
one forward per decode step over B independent sequences. The KV cache gets a
B-axis: `kv_batch_g` parallel sequences, `kv_seq_g` selects the current
sequence for 4-item calls; the flat base becomes
`(layer * kv_batch_g + seq) * eff_seq` (see §KV Cache). Each arch adds
`*_attention_bd` (per-sequence scores/softmax/output inside, weight matmuls
amortized over B), `*_block_forward_bd`, `*_run_blocks_bd`, and a
`*_generate_batch` wrapper. `gen_loop_batch` prefills each sequence with the
per-arch `*_run_blocks_b` under `kv_seq_g = i`, then decodes:
- Step 0 predicts from the prefill-last hidden WITHOUT re-embedding and does
  NOT advance `cur_pos` — mirrors `gen_loop_core`'s off-by-one rule. (An early
  version re-embedded at pos L every step: it duplicated the last prompt
  token's K/V into the cache and shifted outputs by one; qwen2/llama/granite/
  ernie matched only by argmax coincidence, gemma3 diverged.)
- Steps 1+ embed the previous token and run `rb_bd` at `cur_pos` (the B-vector
  of per-sequence positions); `cur_pos` advances only on real forwards.
- The lm-head is `rms_norm_rows` + `emb_w (+/ .*) |: hidden_n` (tied
  embeddings — same as `output_head` per row), divided by `logit_div` for
  granite.

Recurrent-state arches (lfm2 conv, qwen35 delta-net) keep per-sequence states:
their caches get the `kv_batch_g` dimension and read/write via `kv_seq_g`
(prefill) or explicit `_b` variants (decode). Between generations they are
force-zeroed by `lf2_conv_reset`/`rs_reset` dispatched in the fresh paths —
a guarded-but-unreset cache made batch-after-single inherit stale conv state
and diverge (the KV cache self-cleans because prefill overwrites its rows).

Verified: `tests/j/test_batched.ijs` — batch==single token-identical for all
8 arches at B=2 (plus B=3 for qwen2). Measured per-seq throughput at small
ctx: ~1.6x qwen2, ~2.3x qwen3, ~1.4x llama, ~1.35x granite, ~2.8x ernie,
~2.6x lfm2, ~1.5x qwen35 (gemma3 not re-measured post-fix). Big-ctx models
(qwen3.5 ctx=262144 → ~51GB/seq KV at full ctx) need `kv_max_seq_g` bounded.

## Chat Sessions (persistent multi-turn, option B)

`llm chat 'next message'` (or `llm chat_p ('msg' ; <temp;k;p;min_p>)`) is the
crude console chat. The session state `chat_session_g` =
`<arch; messages; total_tokens; cur_pos; max_steps; params>`:
- `messages` — the full message history (`<role ; content>` boxes; assistant
  answers stored with role 'assistant', gemma3 maps to 'model' internally).
- `total_tokens` — the EXACT token stream processed so far (prompt + generated),
  boxed; `cur_pos` = its length = the KV write frontier.
- The session never holds a cache reference — a second ref would defeat the
  in-place amend; the cache stays in `kv_cache_g` (one active session).

Each turn:
1. Append the new user message and RE-RENDER the full history (through the
   real GGUF jinja template — see §Chat-template rendering below) — only to
   tokenize the new segment.
2. Prefix-check: the re-render's token stream must start with the stored
   `total_tokens`. If it matches, `seg = cur_pos }. new_toks` and
   `gen_loop_core` resumes at `cur_pos` (ONE batched prefill of `seg`).
3. If the tokenizer round-trip drifted (detokenize→tokenize not exact), fall
   back to a full fresh re-render (correct, slower) and reset the session.

**gemma3 ▁→space**: the llama3 detokenizer (`sp_replace`) renders the
SentencePiece space marker ▁ (3-char UTF-8 E2 96 81) as a real space. This is
presentation AND persistence — the space-text re-tokenizes to the same
▁-pieces, so gemma's stored token stream and the re-render prefix agree and
the cache can be resumed. Byte-level BPE (qwen2/smollm2) round-trips exactly
without the conversion.

## Chat-template rendering (real GGUF jinja)

Every architecture renders chat through its own **real `tokenizer.chat_template`**
pulled from the GGUF, parsed and rendered by the minja/chat_template port —
there are no bespoke per-arch prompt verbs anymore.

- **Load**: each arch loader extracts the template once —
  `'tokenizer.chat_template' kv_string (0 1 { kv_result)` (gguf/gguf.ijs:455;
  vt=8 string) into the `ct_tmpl_g` global (`''` if absent). `load_gguf_to_llm`
  resets `ct_tmpl_g` per load. `ct_tmpl_g`/`ct_vars_g`/`ct_now_g` are initialized
  in util/chat.ijs.
- **Render**: `chat_prompt` (util/chat.ijs dispatch) calls each arch's
  `*_chat_prompt`, which is a thin wrapper over `chat_tmpl_render`
  (util/chat.ijs): convert the `<role ; content>` message boxes (or pre-built
  minja message Values carrying `tool_calls`/`tool_call_id`/typed content —
  detected by the `'obj'` kind marker on the unboxed element) to a minja
  Value array of `{role, content}` objs, `mkarr_minja_`, then
  `ct_apply (source ; <msgs; tools; add_generation_prompt=1; extra;
  now; ''; ''> ; caps ; tool_ex ; options)` → prompt string.
  `tools` comes from `ct_tools_g` — a JSON string of tool definitions parsed
  by `ct_parse_json_chatpl_` (`''` → null). Tool-capable templates render the
  tool dump + tool_calls; non-tool templates get the tools polyfill (system
  message) automatically via the layer's `detect_caps`.
  `add_generation_prompt=1` makes the template emit its own gen prompt
  (`<start_of_turn>model`, `<|im_start|>assistant`, ...). If a model has no
  `tokenizer.chat_template`, `chat_tmpl_render` throws a clear error.
- **Template variables**: `ct_vars_g` (a minja obj) is passed as the `extra`
  input — e.g. `enable_thinking` (qwen3: default no thinking block, false →
  ` thinking\n\n response\n\n`; qwen3.5: undefined → ` thinking\n\n response\n\n`,
  true → ` thinking\n`). Settable per-call via the optional 4th `tmpl_vars`
  arg to `chat_generate` (`chat_vars_obj` builds the obj from `<key ; value>`).
- **Tools**: `chat_generate` accepts an optional 5th `tools` arg — a JSON
  string of OpenAI-style function schemas, set into `ct_tools_g` per call
  (mirroring `ct_vars_g`/`ct_now_g`, cleared by the persistent chat path and
  reset on model load). `chat_tmpl_render` parses it to the template's `tools`
  input, so tool-capable templates produce real function-calling prompts;
  messages may be pre-built minja Values to express `tool_calls`/typed content.
- **now**: `chat_tmpl_render` uses the current epoch (Hinnant `days_from_civil`
  date→days inverse of the minja `civil_from_days`) unless `ct_now_g` is
  non-zero (test oracle pinning, e.g. 1721952000 = 26 Jul 2024). The template's
  `strftime_now` callable renders dates (Llama-3.2's "Today Date").
- **BOS/EOS**: `mk_options` defaults keep `use_bos=0`/`use_eos=0` — BOS is
  tokenizer-owned (llama3/gemma prepend bos; their templates omit the
  `<|begin_of_text|>` marker), so the token stream matches llama.cpp exactly.
- **Stop tokens stay per-arch** (from the vocab), untouched by the template.

## HTTP Server (`http/`)

A network OpenAI-compatible server in `http/` — `server.ijs` (driver +
event loop), `protocol.ijs` (pure HTTP/1.1 framing/parse/build, NO socket
calls), `builders.ijs` (OpenAI JSON/SSE body builders). Runs in the
**inference** locale (`+ coinsert 'jsocket'`) so the streaming callback
(`chat_cb_g -> sse_sender`) and `chat_completion` resolve where
`chat_stream_cb` CALLS them (mirrors the TUI locale fix). `scripts/llm_server.sh`
is the launcher; `http/run.ijs` is the entry point.

- **Concurrency model (from the handoff, proven)**: every socket is
  NON-BLOCKING (`sdcheck sdioctl fd,FIONBIO,1`). The event loop is
  `z=: sdcheck sdselect FSET;FSET;FSET;200` — sdcheck drops the status cell →
  `<read;write;error>`; READ fds at cell 0. One op per ready fd per cycle:
  listener → `sdaccept`, conn → `sdrecv` (4096) accumulated in `CBF` until
  `h11_complete`. `sdaccept` → `<0;newfd>` | `<11;''>` EAGAIN; `sdrecv` →
  `<0;DATA>` | `<0;''>` EOF | `<11;''>` EAGAIN. Inspect error codes directly
  (never block).
- **Endpoints**: POST `/v1/chat/completions` (plain JSON via `respbody`, or
  streamed SSE via `chat_stream_cb` → `sse_sender` → `frame_*`), GET `/v1/models`
  (`v1_models`), GET `/` (status line). `finish` sends the response, closes,
  deregisters (`rmconn`).
- **J gotcha — `sdclose` is BROKEN in this jsocket build**: its `0=res
  closesocketJ <y` passes a BOXED arg to the libc close foreign (15!:0), which
  domain-errors, so every `sdclose` crashes the event loop after a response.
  `closefd` (server.ijs) calls the libc close directly with the UNBOXED fd
  (`'"libc.so.6" close i i'&(15!:0) y`), then `rmconn` deregisters separately.
- **J gotcha — `res=:` clobbers jsocket's `res` verb**: `sdselect` uses
  `_1=res q=.selectJ(...)` where `res` is jsocket's global verb (`res=: >@:{.`).
  The server's handlers must NOT use the global name `res` (the old
  `res=: LLM chat_completion ...` overwrote the verb with a noun → the next
  `sdselect` syntax-errors "unexecutable fragment (noun noun)" and the loop
  crashes). The handlers use `cres=:` instead.
- **Builder gotcha**: convert/json needs each value cell EXPLICITLY boxed with
  `<` (`b1=. <MODEL` ... `v=. b1,b2,b3,b4`), NOT a `;`-chained boxed list —
  `MODEL ; 'model'` boxes each cell, then appending a scalar re-boxes the whole
  list (nested), so `enc_json` drops/coerces the last cell (e.g. `owned_by` 0).
  Contract-tested in `tests/j/test_http_server.ijs`.

## Per-Architecture Notes

### Gemma3 270M (`gemma3.ijs`)

```
Token Embedding → scale by sqrt(n_embd)
For each block (18×):
  1. Pre-norm (RMSNorm): attn_in = norm(input)
  2. QKV projection: Q,K,V = linear(attn_in)
  3. Q/K norm + RoPE (NEOX)
  4. Attention: scores = softmax(Q·Kᵀ/sqrt(d))·V  (GQA 4:1 + sliding window 512)
  5. Output proj: attn_out = W_o · attention
  6. Post-norm: attn_out = norm(attn_out)        ENABLED (plain RMSNorm w/ GGUF weights)
  7. Residual: sa_out = attn_out + input
  8. Pre-norm (RMSNorm): ffn_in = norm(sa_out)
  9. FFN (GEGLU): ffn_out = W_down · GEGLU(W_gate · ffn_in, W_up · ffn_in)
     GEGLU(gate, up) = GELU(gate) × up
     GELU(x) = 0.5 × x × (1 + tanh(sqrt(2/π) × (x + 0.044715 × x³)))
 10. Post-norm: ffn_out = norm(ffn_raw)          ENABLED (plain RMSNorm w/ GGUF weights)
 11. Residual: output = ffn_out + sa_out
Final: output = norm(last_output) → matmul(output, tok_embd.T) → softcap(tanh)
```

Key traits: GQA 4:1, sliding window 512, GEGLU (not SwiGLU), token embd scaled
by sqrt(n_embd), final logits softcapped (`tanh(logits/scale) × scale`), NEOX
RoPE, Q/K norm BEFORE RoPE, attention scale `1/sqrt(head_dim)` after RoPE to Q.

### SmolLM2 360M (`llama.ijs`) — llama arch

Standard decoder, GQA (15→5), SwiGLU (`silu(gate) * up` — NOT `gate*silu(gate)*up`),
interleaved RoPE (NORM), separate QKV/O weights, no q/k norm, no post-attention/ffn
norms, no embedding scale. Tokenizer: GPT-2 byte-level BPE. The llama module is
generic: dims are read from the GGUF, and Llama-3.2-1B-Instruct shares it —
llama-bpe tokenizer (llama3 regex pre + gpt2 BPE merges, `Ġ` space marker),
BOS prepended by `llama_tokenize`. Chat is rendered from each model's real
GGUF `tokenizer.chat_template` (SmolLM2's `<|im_start|>` or Llama-3.2's
always-emitted system block + dynamic "Today Date" via `strftime_now`).

### Granite-4.0-350m (`granite.ijs`) — granite arch

Standard decoder, GQA (16→4), SwiGLU, interleaved RoPE (NORM, full head_dim
rotary, freq_base 1e7), separate QKV/O weights, no q/k norm, **tied
  embeddings** (no `output.weight`). The Granite scaling scheme (read from
  KVs, stored in mi at indices 12..15, data-driven per model — 4.0/4.1/4.2):
  input embeddings *12 (`embedding_scale`),
  per layer `attn_out*resid + input` then `ffn_out*resid + that`
  (`residual_scale`; 0.263 on 4.0, 0.22 on 4.1), Q*K^T scores *0.015625
  (`attention.scale` — NOT 1/sqrt(head_dim)), lm_head logits /`logit_scale`
  (4 on 4.0, 10 on 4.1). Tokenizer: gpt2
  byte-level BPE with `dbrx` pre (same regex as llama3), no BOS
(bos=eos=100257). Chat is rendered from the real GGUF `tokenizer.chat_template`
(granite `<|start_of_role|>` format with the always-emitted default system
message).

### Qwen2.5-Coder 0.5B (`qwen2.ijs`) — qwen2 arch

Standard decoder, GQA (14→2), SwiGLU, **NEOX RoPE**, separate QKV/O weights,
no q/k norm, no post-attention/ffn norms. **Q/K/V attention biases**
(`attn_q.bias` = `(n_heads, head_dim)`, `attn_k/v.bias` = `(n_kv_heads, head_dim)`),
added AFTER projection BEFORE RoPE (per llama.cpp `build_qkv`). Q scaled
`1/sqrt(head_dim)`. Tokenizer: GPT-2 byte-level BPE.

Batched bias broadcast: J only broadcasts leading-axis prefixes, so bias is
added via reshape-cycles `(L, n_heads*head_dim) $ bias`, not plain `+`.

### LFM2 (`lfm2.ijs`) — lfm2 hybrid arch

Hybrid: **6 attention layers** (2,5,8,10,12,14; qwen3-style per-head Q/K
RMSNorm, NEOX RoPE, GQA) + **10 shortconv layers** (conv1d with a `shortconv`
in_proj/conv/out_proj, `l_cache=3` → `d_conv=2`) + FFN every layer. Covers
LFM2-350M/700M/1.2B + LFM2.5-230M. Tokenizer: gpt2 byte-level with the `lfm2`
pre (llama3 regex). Recurrent conv state is the flat `lf2_conv_cache_g`
(see §KV Cache recurrent-state note), force-reset between generations
(`lf2_conv_reset`). Decode is dominated by the conv layers' matvecs (in_proj
`3*emb`, out_proj, FFN) — the single-token M=1 matvec floor (~9GB/s in J);
batched decode (`lfm2_generate_batch`) amortizes them to ~28GB/s.

(The qwen3, qwen3.5, and ernie4_5 arches — standard decoders/SSM hybrid, SPM —
are covered in docs/HISTORICAL.md; the Supported Models table lists all dims.)

## Verification Reference

- llama.cpp checkout at `llama.cpp/`; arch sources `src/models/*.cpp`, graph
  `src/llama-graph.cpp`, rope type `llama_model_rope_type` in `src/llama-model.cpp`.
- `llama-cpp-python` (`Llama(model_path, logits_all=True)`) is the oracle for
  logits/argmax; `eval_logits` is a property, `tokenize(bytes, add_bos=False)`.
- Model files in `models/` (one dir per model).
