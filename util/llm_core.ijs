NB. ================================================================
NB. llm_core.ijs — generic reusable helpers shared by all architecture modules
NB. Provides: llm noun layout/accessors, generic tensor lookup, embedding,
NB. output head, KV cache creation, RoPE table build.
NB. Depends on: kernels.ijs, gguf.ijs, kv_cache.ijs, sampler.ijs
NB. ================================================================
coclass 'inference'
require 'llm/inference/kernels/jfloat'
require 'llm/inference/gguf/gguf'
require 'llm/inference/util/kv_cache'
require 'llm/inference/util/sampler'
require 'dict'

NB. ---- Prefill chunk size (tunable) ----
NB. Fresh prompts are prefilled in chunks of this many tokens to bound peak
NB. memory: a one-shot full-batch prefill materializes a quadratic
NB. (n_heads, L, L) scores matrix per layer (OOM on deep contexts), whereas
NB. chunked prefill's peak is (n_heads, chunk, context). Set at runtime:
NB.   prefill_chunk_sz =: 64   NB. smaller = less peak, more (slower) chunks
prefill_chunk_sz =: 256

NB. ---- Optional per-token generation callback (streaming hook) ----
NB. Set gen_cb_on_g = 1 and gen_cb_g to a monadic verb to stream each generated
NB. token; gen_loop_core calls `gen_cb_g pred` after sampling. The callback may
NB. return a (possibly replaced) token id — returning a stop-token id forces a
NB. stop. Verbs can't be boxed into a list (noun-verb syntax error) and can't
NB. be distinguished from a noun by `-:`/`3!:0`, so a noun flag gates it.
gen_cb_on_g =: 0
gen_cb_g =: ]

NB. ---- Optional generation timing (tok/s report) ----
NB. gen_loop_core/gen_loop_batch measure each prefill chunk + decode step with
NB. 6!:2 and call report_timing. Gated by gen_timing_g (default OFF) — the
NB. per-step 6!:2 string-eval is overhead when timing isn't needed (library
NB. use); turn on for the CLI/chat tok/s report or timing tests. NOTE: the
NB. timed string-eval must run INLINE (not via a helper verb) — the string
NB. references the caller's locals (hidden, result_b, emb_seg, llm_box), which
NB. are not visible inside a helper verb's scope.
gen_timing_g =: 0

NB. ---- llm noun layout (shared across architectures) ----
NB. llm is a NAMED DICT (jdict) of the loaded model's parts, keyed as:
NB.   path; ti; default_params; tokenizer; mi; kv_cache; tds; all_tensors;
NB.   block_data; kv_data; arch
NB. The llm_* accessors query by NAME (not index) — flexible: adding a member
NB. never shifts the others. (The old 11-element boxed list is gone; the
NB. loader builds this dict. The per-layer kernels stay on box-of-lists.)

NB. ---- Robust dual-mode probe: dict lookup, else unwrap-and-index ----
NB. x = key name; y = container (dict locale ref | boxed list | 1-element box).
NB. Returns <1; value> if y is a dict holding x, else <0; container> where
NB. container is the (1-element-box-unwrapped) non-dict for index-based access.
NB. Shared by llm_get and the tokenizer accessors. Dict detection is robust
NB. (has__d in try/catch) — a bare # / 3!:0 probe misdetects a 1-element box
NB. as a dict (a box has # = 1, same as a locale ref).
dict_probe =: 4 : 0
  d =. y
  got =. 0
  r =. ''
  try.
    if. has__d (<x) do.
      r =. > get__d (<x)
      got =. 1
    end.
  catch. end.
  if. got do. 1 ; <r return. end.
  if. 1 = # y do.
    y =. 0 {:: y
    d =. y
    try.
      if. has__d (<x) do.
        r =. > get__d (<x)
        got =. 1
      end.
    catch. end.
    if. got do. 1 ; <r return. end.
  end.
  0 ; <y
)

llm_get =: 4 : 0
  'found v' =. x dict_probe y
  if. found do. v return. end.
  idx =. ('path';'ti';'default_params';'tokenizer';'mi';'kv_cache';'tds';'all_tensors';'block_data';'kv_data';'stop_tokens';'ct_tmpl';'arch';'ct_caps';'ct_tool_ex';'lf2_conv') i. <x
  > idx { v
)
llm_path          =: 'path'&llm_get
llm_ti            =: 'ti'&llm_get
llm_default_params =: 'default_params'&llm_get   NB. chat sampling defaults <temp;k;p;min_p>
llm_mi            =: 'mi'&llm_get
llm_kv_cache      =: 'kv_cache'&llm_get   NB. per-model cache config (authoritative; mirror globals are the hot path)
llm_tds           =: 'tds'&llm_get
llm_all_tensors   =: 'all_tensors'&llm_get
llm_block_data    =: 'block_data'&llm_get
llm_kv_data       =: 'kv_data'&llm_get   NB. decoded-KV dict (query via kv_get)
llm_arch          =: 'arch'&llm_get   NB. arch string ('gemma3'|'llama'|'qwen2')
llm_ct_tmpl       =: 'ct_tmpl'&llm_get   NB. chat-template string ('' = none)
llm_ct_caps       =: 'ct_caps'&llm_get   NB. chat-template caps mkobj ('' = not computed yet)
llm_ct_tool_ex    =: 'ct_tool_ex'&llm_get   NB. chat-template tool example ('' = none)
llm_lf2_conv      =: 'lf2_conv'&llm_get   NB. lfm2 conv config ('' = not an lfm2 model)
llm_stop_tokens   =: 'stop_tokens'&llm_get   NB. stop-token id list (boxed/scalar)

NB. ---- Generic tokenize/detokenize (DRY Tier 1 item 3) ----
NB. Dispatch on the arch (from the llm dict); the per-arch *-tokenize/
NB. *-detokenize names are aliases to these. y = <llm; text> (tokenize) or
NB. <llm; tokens> (detokenize). The llama/lfm2 cases inline the bos-prepend
NB. (llama-bpe pre / bos>0) so the aliases don't recurse.
llm_tokenize =: 3 : 0
  llm_data =. input_llm y
  text =. input_text y
  arch =. llm_arch llm_data
  select. arch
  case. 'gemma3' do. llama3_tokenize y
  case. 'qwen2'  do. gpt2_tokenize y
  case. 'qwen3'  do. gpt2_tokenize y
  case. 'qwen35' do. gpt2_tokenize y
  case. 'granite' do. gpt2_tokenize y
  case. 'ernie4_5' do. spm_tokenize y
  case. 'llama' do.
    tokenizer =. llm_tokenizer llm_data
    tokens =. gpt2_tokenize (<llm_data) , <text
    if. 'llama-bpe' -: tokenizer_pre_g tokenizer do.
      bos =. tokenizer_bos_g tokenizer
      if. bos > 0 do. tokens =. (<bos) , tokens end.
    end.
    tokens
  case. 'lfm2' do.
    tokenizer =. llm_tokenizer llm_data
    tokens =. gpt2_tokenize (<llm_data) , <text
    bos =. tokenizer_bos_g tokenizer
    if. bos > 0 do. tokens =. (<bos) , tokens end.
    tokens
  end.
)
llm_detokenize =: 3 : 0
  llm_data =. input_llm y
  arch =. llm_arch llm_data
  select. arch
  case. 'gemma3' do. llama3_detokenize y
  case. 'qwen2'  do. gpt2_detokenize y
  case. 'qwen3'  do. gpt2_detokenize y
  case. 'qwen35' do. gpt2_detokenize y
  case. 'granite' do. gpt2_detokenize y
  case. 'llama'  do. gpt2_detokenize y
  case. 'lfm2'   do. gpt2_detokenize y
  case. 'ernie4_5' do. spm_detokenize y
  end.
)

NB. ---- Count members of the llm dict (tests assert 13) ----
llm_count =: 3 : 0
  d =. y
  count__d ''
)

NB. ---- Query a decoded KV by name from a kv_data dict ----
NB. x = key name; y = kv_data dict.  Returns the decoded value (opened), or a:
NB. (absent).  Used by kv_get and by loaders extracting arch config from kv_data.
dict_get =: 4 : 0
  name =. x
  dict =. y
  if. 0 = # dict do. a: return. end.
  if. has__dict (<name) do.
    > get__dict (<name)
  else.
    a:
  end.
)

NB. ---- Query a decoded KV by name from the llm's kv_data dict ----
NB. x = key name; y = llm.  Returns the decoded value (opened), or a: (absent).
NB. The kv_data dict is built at load time by build_kv_dict (all GGUF KVs).
kv_get =: 4 : 0
  name =. x
  llm =. y
  name dict_get llm_kv_data llm
)

NB. ---- Canonicalize token_embd.weight to transposed (emb, vocab) storage ----
NB. J's matvec (m,n)x(n,) is ~4.7x slower than the vector-matrix (n,)x(n,m), so
NB. the embedding/output weight is stored TRANSPOSED and used as
NB.   hidden (+/ .* ) emb_w      (lm_head — vector-matrix, fast)
NB.   |: (tok {"1 emb_w)          (embedding lookup — column access, ~free)
NB. Called once per loader after all_tensors is built. Same bytes, layout
NB. swapped — no memory doubling. The name stays 'token_embd.weight'.
emb_canonical =: 3 : 0
  at =. y
  if. 0 = # at do. at return. end.
  n =. (# at) % 4
  names =. (4 * i. n) { at
  idx =. names i. <'token_embd.weight'
  if. idx < n do.
    base =. idx * 4
    td =. > (base + 1) { at
    dims =. > (base + 2) { at
    etype =. > (base + 3) { at
    td_t =. |: td
    dims_t =. |. dims
    pre =. base {. at
    post =. (base + 4) }. at
    at =. pre , (<'token_embd.weight') , (<td_t) , (<dims_t) , (<etype) , post
  end.
  at
)

NB. ---- mi accessors (per-arch modules extend this) ----
NB. The mi is a dict of named hparams (built by build_mi_dict + the arch loader's
NB. named puts). Each accessor takes the mi dict ref as y and opens the named value.
mi_block_count =: 3 : 0
  d =. y
  > get__d (<'block_count')
)
mi_context_len =: 3 : 0
  d =. y
  > get__d (<'context_len')
)
mi_emb_len     =: 3 : 0
  d =. y
  > get__d (<'emb_len')
)
mi_n_heads     =: 3 : 0
  d =. y
  > get__d (<'n_heads')
)
mi_n_heads_kv  =: 3 : 0
  d =. y
  > get__d (<'n_heads_kv')
)
mi_head_dim    =: 3 : 0
  d =. y
  > get__d (<'head_dim')
)
mi_rope_freq   =: 3 : 0
  d =. y
  > get__d (<'rope_freq')
)
mi_vocab_size  =: 3 : 0
  d =. y
  > get__d (<'vocab_size')
)
mi_rms_eps     =: 3 : 0
  d =. y
  > get__d (<'rms_eps')
)
mi_n_ff        =: 3 : 0
  d =. y
  > get__d (<'n_ff')
)
mi_cos_tab     =: 3 : 0
  d =. y
  > get__d (<'cos_tab')
)
mi_sin_tab     =: 3 : 0
  d =. y
  > get__d (<'sin_tab')
)
mi_attn_scale  =: 3 : 0
  d =. y
  > get__d (<'attn_scale')
)
mi_resid_scale =: 3 : 0
  d =. y
  > get__d (<'resid_scale')
)

NB. ---- Shared GGUF load boilerplate (DRY: the parse + tensor-load + base) ----
NB. y = <path; raw>.  Returns <path; ti; tds; all_tensors; kv_data; kvs_ctx;
NB. kvres; ct_tmpl> — the common components every arch loader needs.  The
NB. arch-specific parts (extract_hparams + mi puts, tokenizer, stop_tokens,
NB. default_params, block_data) stay in the per-arch *_load.  kvs_ctx =
NB. <kvs; raw> (for extract_hparams/build_kv_dict/kv_string); kvres = the full
NB. 4-item kv_result (for the tokenizer builders); all_tensors is emb_canonical.
load_gguf_base =: 3 : 0
  data =. y   NB. <path; raw>
  path =. > 0 { data
  raw =. > 1 { data
  header =. parse_hdr_raw raw
  n_tensors =. > 2 { header
  kvres =. parse_kv_pairs_raw raw
  kvs =. > 0 { kvres
  kv_end =. > 3 { kvres
  ti =. parse_tensor_infos (<raw) , (<kv_end) , (<n_tensors)
  ti_end_offset =. > ((n_tensors * 6) - 1) { ti
  tds =. 32 * <. (ti_end_offset + 31) % 32
  kvs_ctx =. (<kvs) , (<raw)
  kv_data =. build_kv_dict kvs_ctx
  all_tensors =. ''
  tensor_idx =. 0
  while. tensor_idx < n_tensors do.
    tname =. > (tensor_idx * 6) { ti
    tdata =. (<path) , (<ti) , (<tds) , (<tname) , (<raw)
    td =. load_tdata tdata
    if. 0 < # td do.
      ti_row =. tname get_tensor_info ti
      dims =. ti_dims ti_row
      etype =. ti_etype ti_row
      all_tensors =. all_tensors , (<tname) , (<td) , (<dims) , (<etype)
    end.
    tensor_idx =. tensor_idx + 1
  end.
  all_tensors =. emb_canonical all_tensors
  ct_tmpl =. 'tokenizer.chat_template' kv_string kvs_ctx
  (<path) , (<ti) , (<tds) , (<all_tensors) , (<kv_data) , (<kvs_ctx) , (<kvres) , (<ct_tmpl)
)

NB. ---- Build a named-hparams mi dict from the boxed mi noun (indices 0-9) ----
NB. y = the boxed mi noun from *_extract_hparams:
NB.   <block_count; context_len; emb_len; n_heads; n_heads_kv; head_dim;
NB.    rope_freq; vocab_size; rms_eps; n_ff>.  Returns the dict locale ref.
NB. The arch loader then puts the rope tables + scale fields + arch-specific
NB. fields (named) into it, and the mi_* accessors query by name.
build_mi_dict =: 3 : 0
  mi =. y
  NB. 'hash concurrent': J9.8 runs matmuls on worker threads; concurrent dicts
  NB. are the documented thread-safe index type.  dict_new (gguf.ijs) retries
  NB. the intermittent create deadlock.
  d =. dict_new 'hash concurrent' ,&< ('keytype';'boxed'),:('valuetype';'boxed')
  keys =. 'block_count' ; 'context_len' ; 'emb_len' ; 'n_heads' ; 'n_heads_kv' ; 'head_dim' ; 'rope_freq' ; 'vocab_size' ; 'rms_eps' ; 'n_ff'
  mi put__d keys
  d
)

NB. ---- ti_row accessors ----
ti_dims       =: >@(1&{)
ti_etype      =: >@(2&{)
ti_data_off   =: >@(3&{)

NB. ---- Generic tensor lookup: name get_tensor_cached_d llm ----
NB. Searches llm_all_tensors cache first; falls back to file I/O.
get_tensor_cached_d =: 4 : 0
  name =. x
  llm =. y
  cache =. llm_all_tensors llm
  cache_found =. 0
  result =. $0
  if. 0 < # cache do.
    n_names =. (# cache) % 4
    names =. (4 * i. n_names) { cache
    idx =. names i.!.0 < name
    if. idx < n_names do.
      result =. > ((idx * 4) + 1) { cache
      cache_found =. 1
    end.
  end.
  if. -. cache_found do.
    ti =. llm_ti llm
    tds =. llm_tds llm
    path =. llm_path llm
    ti_row =. name get_tensor_info ti
    if. 0 = # ti_row do. $0 return. end.
    etype =. ti_etype ti_row
    data_off =. ti_data_off ti_row
    dims =. ti_dims ti_row
    ne =. */ dims
    file_off =. tds + data_off
    raw =. 1!: 1 < path
    bpe =. etype_bpe etype
    NB. Hybrid slice — see gguf/gguf.ijs load_tdata: take-of-drop when the
    NB. tail copy is cheaper than the index-list fetch (tail < ne * 18).
    tail =. (# raw) - file_off
    if. tail < ne * 18 do.
      slice =. (ne * bpe) {. file_off }. raw
    else.
      slice =. (file_off + i. ne * bpe) { raw
    end.
    flat =. decode_tensor_flat (<etype) , (<ne) , <slice
    result =. dims tensor_reshape flat
  end.
  result
)

NB. ---- Embed token list -> hidden states ----
NB. y = <emb_w; scale; tok_list>  (emb_w is TRANSPOSED-canonical (emb, vocab); the
NB. embedding is a column access |: (tok {"1 emb_w), NOT tok { emb_w)
NB. ---- Output head: rms_norm + lm_head projection ----
NB. y = <rms_eps; output_norm_w; emb_w_final; hidden>  (hidden is the (emb,) row;
NB. the lm_head weight is TRANSPOSED-canonical (emb, vocab) — hidden (+/ .*) efw).
output_head =: 3 : 0
  NB. y = <rms_eps; output_norm_w; emb_w_final; hidden>.  The caller pre-boxes
  NB. the invariant (<rms_eps), (<output_norm_w), (<emb_w_final); reuse those
  NB. boxes directly (0 1 { y) instead of open+reboxing them for rms_norm.
  hidden =. rms_norm ((0 1 { y) , <(> 3 { y))
  hidden (+/ .* ) > 2 { y
)

NB. ---- GQA attention scores/softmax/output (shared by llama/qwen2/qwen3/granite) ----
NB. The group-major GQA matmul + direct-3D-softmax + output-projection block is
NB. identical across the standard-GQA archs (the per-arch Q/K/V projection,
NB. RoPE, and Q pre-scale happen BEFORE; only the scores/softmax/output differ
NB. by single-token vs batched).  Q/K/V stay (n_heads_kv, ctx, hd) — never
NB. expanded to n_heads.  Single-token: x = <Q; K; V; attn_o; n_heads; n_heads_kv;
NB. head_dim; n_groups> where Q = (n_heads,hd), K = (win,n_kv,hd), V = (win,n_kv,hd),
NB. attn_o = (emb,n_heads*hd).  Returns the (emb,) output row.
gqa_attn_single =: 3 : 0
  Q =. > 0 { y
  K =. > 1 { y
  V =. > 2 { y
  attn_o =. > 3 { y
  n_heads =. > 4 { y
  n_heads_kv =. > 5 { y
  head_dim =. > 6 { y
  n_groups =. > 7 { y
  scores2 =. ((n_heads_kv , n_groups , head_dim) $ , Q) (+/ .* "2) (1 2 0 |: K)   NB. (n_kv, n_groups, win)
  max_sf =. >./"1 scores2
  exp_sf =. ^ (scores2 - max_sf)
  softmax =. exp_sf % +/"1 exp_sf   NB. (n_kv, n_groups, win)
  attn_o linear_r ((n_heads * head_dim) $ , (softmax (+/ .* "2) (1 0 2 |: V)))
)

NB. Batched (prompt prefill): x = <Q; K; V; attn_o; mask_g2; n_heads; n_heads_kv;
NB. head_dim; n_groups; L> where Q = (L,n_heads,hd), K/V = (n_heads_kv,ctx,hd)
NB. (K includes the cache prefix), mask_g2 = (n_groups*L, ctx) tiled causal mask.
NB. Returns the (L, emb) output.
gqa_attn_batch =: 3 : 0
  Q =. > 0 { y
  K =. > 1 { y
  V =. > 2 { y
  attn_o =. > 3 { y
  mask_g2 =. > 4 { y
  n_heads =. > 5 { y
  n_heads_kv =. > 6 { y
  head_dim =. > 7 { y
  n_groups =. > 8 { y
  L =. > 9 { y
  Qp =. 1 0 2 |: Q
  Q_g2 =. (n_heads_kv , (n_groups * L) , head_dim) $ , Qp
  Kp2 =. 1 2 0 |: K
  scores2 =. Q_g2 (+/ .* "2) Kp2   NB. (n_kv, n_groups*L, ctx)
  scores2 =. scores2 -"2 (mask_g2 * 1e9)
  max_sf =. >./"1 scores2
  exp_sf =. ^ (scores2 - max_sf)
  softmax_f =. exp_sf % +/"1 exp_sf
  Vp =. 1 0 2 |: V
  attn2 =. softmax_f (+/ .* "2) Vp   NB. (n_kv, n_groups*L, hd)
  |: (attn_o (+/ .* ) |: ((L, n_heads * head_dim) $ , (1 0 2 |: ((n_heads, L, head_dim) $ , attn2))))
)

NB. ---- qwen2/qwen3 block_data accessors (shared — normalized layout) ----
NB. Both qwen2 and qwen3 build block_data with this COMMON 17-element layout:
NB. <attn_norm; attn_q; attn_k; attn_v; attn_o; q_bias; k_bias; v_bias;
NB. ffn_norm; ffn_gate; ffn_up; ffn_down; q_norm; k_norm; n_heads; head_dim;
NB. n_heads_kv>. qwen2 fills q_bias/k_bias/v_bias (absent q/k norm -> $0);
NB. qwen3 fills q_norm/k_norm (absent q/k/v bias -> $0). The forward verbs take
NB. a <has_bias; has_qk_norm> flag and skip the empty slots.
bd_attn_norm =: >@(0&{)
bd_attn_q    =: >@(1&{)
bd_attn_k    =: >@(2&{)
bd_attn_v    =: >@(3&{)
bd_attn_o    =: >@(4&{)
bd_q_bias    =: >@(5&{)
bd_k_bias    =: >@(6&{)
bd_v_bias    =: >@(7&{)
bd_ff_norm  =: >@(8&{)
bd_ff_gate  =: >@(9&{)
bd_ff_up    =: >@(10&{)
bd_ff_down  =: >@(11&{)
bd_q_norm    =: >@(12&{)
bd_k_norm    =: >@(13&{)
bd_n_heads   =: >@(14&{)
bd_head_dim  =: >@(15&{)
bd_n_heads_kv =: >@(16&{)

NB. ---- qwen2/qwen3 shared single-token forward verbs ----
NB. y carries the <has_bias; has_qk_norm> flags as the LAST 2 elements (qwen2:
NB. has_bias=1/has_qk_norm=0; qwen3: has_bias=0/has_qk_norm=1). The flags let
NB. ONE verb serve both arches (bias-add vs per-head Q/K RMSNorm).
qwen_attention =: 4 : 0
  'block_data pos mi rms_eps cos_tab sin_tab layer has_bias has_qk_norm' =. y
  n_heads =. bd_n_heads block_data
  head_dim =. bd_head_dim block_data
  n_heads_kv =. bd_n_heads_kv block_data
  n_groups =. n_heads % n_heads_kv
  hidden =. rms_norm ((< rms_eps) , (< (bd_attn_norm block_data)) , <x)
  qv =. (bd_attn_q block_data) linear_r hidden
  kv =. (bd_attn_k block_data) linear_r hidden
  vv =. (bd_attn_v block_data) linear_r hidden
  if. has_bias do.
    qv =. qv + bd_q_bias block_data
    kv =. kv + bd_k_bias block_data
    vv =. vv + bd_v_bias block_data
  end.
  Q =. (n_heads, head_dim) $ qv
  K =. (n_heads_kv, head_dim) $ kv
  V =. (n_heads_kv, head_dim) $ vv
  if. has_qk_norm do.
    Q =. rms_norm_rows ((< rms_eps) , (< bd_q_norm block_data) , <Q)
    K =. rms_norm_rows ((< rms_eps) , (< bd_k_norm block_data) , <K)
  end.
  cos_t =. pos { cos_tab
  sin_t =. pos { sin_tab
  rope_t =. (<cos_t) , (<sin_t)
  Q =. Q rope_apply2_neox_t rope_t
  K =. K rope_apply2_neox_t rope_t
  Q =. Q % head_dim ^ 0.5
  kv_write ((6 { y) , (1 { y) , (<K) , (<V))
  kv_result =. kv_read ((6 { y) , (1 { y))
  k_all =. > 0 { kv_result
  v_all =. > 1 { kv_result
  win =. pos + 1
  attn_out =. gqa_attn_single (Q ; k_all ; v_all ; (bd_attn_o block_data) ; n_heads ; n_heads_kv ; head_dim ; n_groups)
  (<attn_out)
)

qwen_block_forward =: 4 : 0
  hidden =. x
  'block_data pos mi rms_eps cos_tab sin_tab layer has_bias has_qk_norm' =. y
  attn_result =. hidden qwen_attention y
  attn_out =. > 0 { attn_result
  sa_out =. attn_out + hidden
  ffn_in =. rms_norm ((< rms_eps) , (< (bd_ff_norm block_data)) , <sa_out)
  gate =. (bd_ff_gate block_data) linear_r ffn_in
  up =. (bd_ff_up block_data) linear_r ffn_in
  ffn_raw =. (bd_ff_down block_data) linear_r (gate swiglu up)
  output =. ffn_raw + sa_out
  (<output)
)

qwen_run_blocks =: 4 : 0
  'llm pos has_bias has_qk_norm' =. y
  mi =. llm_mi llm
  head_dim =. mi_head_dim mi
  n_heads_kv =. mi_n_heads_kv mi
  block_count =. mi_block_count mi
  ctx_len =. mi_context_len mi
  rms_eps =. mi_rms_eps mi
  cos_tab =. mi_cos_tab mi
  sin_tab =. mi_sin_tab mi
  state =. x
  if. 0 = # kv_meta do.
    kv_create ((<block_count) , (<ctx_len) , (<n_heads_kv) , (<head_dim))
  end.
  b =. 0
  block_data_list =. llm_block_data llm
  bf_pre =. ((<pos) , (<mi) , (<rms_eps) , (<cos_tab) , (<sin_tab))
  while. b < block_count do.
    result =. state qwen_block_forward ((b { block_data_list) , bf_pre , (<b) , (<has_bias) , (<has_qk_norm))
    state =. > 0 { result
    b =. b + 1
  end.
  <state
)

NB. ---- qwen2/qwen3 shared BATCHED (prefill) verbs ----
NB. y carries <has_bias; has_qk_norm> as the LAST 2 elements (same flag
NB. convention as the single-token verbs). qwen2: has_bias=1/has_qk_norm=0;
NB. qwen3: has_bias=0/has_qk_norm=1.
qwen_attention_b =: 4 : 0
  block_data =. > 0 { y
  layer =. > 1 { y
  mi =. > 2 { y
  start_pos =. > 3 { y
  rope =. > 4 { y
  has_bias =. > 5 { y
  has_qk_norm =. > 6 { y
  L =. {. $ x
  n_heads =. bd_n_heads block_data
  head_dim =. bd_head_dim block_data
  n_heads_kv =. bd_n_heads_kv block_data
  n_groups =. n_heads % n_heads_kv
  cos_all =. > 0 { rope
  sin_all =. > 1 { rope
  half =. > 2 { rope
  cos_expq =. > 3 { rope
  sin_expq =. > 4 { rope
  cos_expk =. > 5 { rope
  sin_expk =. > 6 { rope
  mask_g2 =. > 7 { rope
  hidden =. rms_norm_rows ((< mi_rms_eps mi) , (< (bd_attn_norm block_data)) , <x)
  thin =. |: hidden
  qv =. |: ((bd_attn_q block_data) (+/ .* ) thin)
  kv =. |: ((bd_attn_k block_data) (+/ .* ) thin)
  vv =. |: ((bd_attn_v block_data) (+/ .* ) thin)
  if. has_bias do.
    qv =. qv + ((L , n_heads * head_dim) $ bd_q_bias block_data)
    kv =. kv + ((L , n_heads_kv * head_dim) $ bd_k_bias block_data)
    vv =. vv + ((L , n_heads_kv * head_dim) $ bd_v_bias block_data)
  end.
  Q =. (L, n_heads, head_dim) $ , qv
  K =. (L, n_heads_kv, head_dim) $ , kv
  V =. (L, n_heads_kv, head_dim) $ , vv
  if. has_qk_norm do.
    Qf =. ((L * n_heads) , head_dim) $ , Q
    Qf =. rms_norm_rows ((< mi_rms_eps mi) , (< bd_q_norm block_data) , <Qf)
    Q =. (L, n_heads, head_dim) $ , Qf
    Kf =. ((L * n_heads_kv) , head_dim) $ , K
    Kf =. rms_norm_rows ((< mi_rms_eps mi) , (< bd_k_norm block_data) , <Kf)
    K =. (L, n_heads_kv, head_dim) $ , Kf
  end.
  Q =. Q rope_apply2_neox_b ((> 3 { rope) ; (> 4 { rope))
  K =. K rope_apply2_neox_b ((> 5 { rope) ; (> 6 { rope))
  Q =. Q % head_dim ^ 0.5
  K_batch =. K
  V_batch =. V
  if. start_pos > 0 do.
    k_pre =. > 0 { (kv_read ((1 { y) , <(start_pos - 1)))
    v_pre =. > 1 { (kv_read ((1 { y) , <(start_pos - 1)))
    K =. k_pre , K
    V =. v_pre , V
  end.
  kv_write_rows ((<0) , (1 { y) , (3 { y) , <K_batch)
  kv_write_rows ((<1) , (<layer) , (<start_pos) , <V_batch)
  attn_out =. gqa_attn_batch (Q ; K ; V ; (bd_attn_o block_data) ; mask_g2 ; n_heads ; n_heads_kv ; head_dim ; n_groups ; L)
  (<attn_out)
)

qwen_block_forward_b =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  mi =. > 2 { y
  attn_result =. hidden qwen_attention_b y
  attn_out =. > 0 { attn_result
  sa_out =. attn_out + hidden
  ffn_in =. rms_norm_rows ((< mi_rms_eps mi) , (< (bd_ff_norm block_data)) , <sa_out)
  ft =. |: ffn_in
  ffn_raw =. |: ((bd_ff_down block_data) (+/ .* ) ((bd_ff_gate block_data) (+/ .* ) ft) swiglu ((bd_ff_up block_data) (+/ .* ) ft))
  output =. ffn_raw + sa_out
  (<output)
)

qwen_run_blocks_b =: 4 : 0
  'llm start_pos has_bias has_qk_norm' =. y
  mi =. llm_mi llm
  head_dim =. mi_head_dim mi
  n_heads =. mi_n_heads mi
  n_heads_kv =. mi_n_heads_kv mi
  block_count =. mi_block_count mi
  ctx_len =. mi_context_len mi
  state =. x
  if. 0 = # kv_meta do.
    kv_create ((<block_count) , (<ctx_len) , (<n_heads_kv) , (<head_dim))
  end.
  L =. {. $ x
  half =. <. head_dim % 2
  cos_all =. (start_pos + i. L) { mi_cos_tab mi
  sin_all =. (start_pos + i. L) { mi_sin_tab mi
  rope_exp =. (cos_all ; sin_all ; half ; n_heads ; n_heads_kv ; L) build_rope_expansions ''
  cos_expq =. > 0 { rope_exp
  sin_expq =. > 1 { rope_exp
  cos_expk =. > 2 { rope_exp
  sin_expk =. > 3 { rope_exp
  n_groups =. n_heads % n_heads_kv
  key_pos =. i. (start_pos + L)
  q_pos =. start_pos + i. L
  mask_2d =. q_pos </ key_pos
  mask_g2 =. ((n_groups * L) , start_pos + L) $ , (2 0 1 |: (mask_2d (*/) (n_groups $ 1)))
  rope =. (<cos_all) , (<sin_all) , (<half) , (<cos_expq) , (<sin_expq) , (<cos_expk) , (<sin_expk) , (<mask_g2)
  bfb_pre =. ((<mi) , (<start_pos) , <rope)
  b =. 0
  block_data_list =. llm_block_data llm
  while. b < block_count do.
    result =. state qwen_block_forward_b ((b { block_data_list) , (<b) , bfb_pre , (<has_bias) , (<has_qk_norm))
    state =. > 0 { result
    b =. b + 1
  end.
  <state
)

NB. ---- qwen2/qwen3 shared BATCHED-DECODE verbs (B sequences, one token each) ----
NB. y carries <has_bias; has_qk_norm> as the LAST 2 elements (same convention).
qwen_attention_bd =: 4 : 0
  block_data =. > 0 { y
  pos =. > 1 { y
  mi =. > 2 { y
  layer =. > 3 { y
  rope =. > 4 { y
  has_bias =. > 5 { y
  has_qk_norm =. > 6 { y
  B =. {. $ x
  n_heads =. bd_n_heads block_data
  head_dim =. bd_head_dim block_data
  n_heads_kv =. bd_n_heads_kv block_data
  n_groups =. n_heads % n_heads_kv
  half =. <. head_dim % 2
  hidden =. rms_norm_rows ((< mi_rms_eps mi) , (< (bd_attn_norm block_data)) , <x)
  thin =. |: hidden
  qv =. |: ((bd_attn_q block_data) (+/ .* ) thin)
  kv =. |: ((bd_attn_k block_data) (+/ .* ) thin)
  vv =. |: ((bd_attn_v block_data) (+/ .* ) thin)
  if. has_bias do.
    qv =. qv + ((B , n_heads * head_dim) $ bd_q_bias block_data)
    kv =. kv + ((B , n_heads_kv * head_dim) $ bd_k_bias block_data)
    vv =. vv + ((B , n_heads_kv * head_dim) $ bd_v_bias block_data)
  end.
  Q =. (B, n_heads, head_dim) $ , qv
  K =. (B, n_heads_kv, head_dim) $ , kv
  V =. (B, n_heads_kv, head_dim) $ , vv
  if. has_qk_norm do.
    Qf =. ((B * n_heads) , head_dim) $ , Q
    Qf =. rms_norm_rows ((< mi_rms_eps mi) , (< bd_q_norm block_data) , <Qf)
    Q =. (B, n_heads, head_dim) $ , Qf
    Kf =. ((B * n_heads_kv) , head_dim) $ , K
    Kf =. rms_norm_rows ((< mi_rms_eps mi) , (< bd_k_norm block_data) , <Kf)
    K =. (B, n_heads_kv, head_dim) $ , Kf
  end.
  Q =. Q rope_apply2_neox_b ((> 2 { rope) ; (> 3 { rope))
  K =. K rope_apply2_neox_b ((> 4 { rope) ; (> 5 { rope))
  Q =. Q % head_dim ^ 0.5
  if. (B > 1) *. (pos -: (B $ 0 { pos)) do.
    win =. (0 { pos) + 1
    eff_seq =. > 1 { kv_meta
    base_b =. ((layer * kv_batch_g) + i. B) * eff_seq
    idxw =. base_b + pos
    k_cache_g =: ((B , n_heads_kv * head_dim) $ , K) idxw} k_cache_g
    v_cache_g =: ((B , n_heads_kv * head_dim) $ , V) idxw} v_cache_g
    kv_pos_g =: kv_pos_g >. (0 { pos) + 1
    idxr =. base_b +/ i. win
    scores_b =. ((B , n_heads_kv , n_groups , head_dim) $ , Q) (+/ .* "2) ((0 2 3 1) |: ((B , win , n_heads_kv , head_dim) $ , (idxr { k_cache_g)))
    max_sf_b =. >./"1 scores_b
    exp_sf_b =. ^ (scores_b - max_sf_b)
    softmax_b =. exp_sf_b % +/"1 exp_sf_b
    attn_all =. (B , n_heads * head_dim) $ , (softmax_b (+/ .* "2) ((0 2 1 3) |: ((B , win , n_heads_kv , head_dim) $ , (idxr { v_cache_g))))
  else.
    attn_out =. ''
    b =. 0
    while. b < B do.
      q_b =. b { Q
      k_b =. b { K
      v_b =. b { V
      pos_b =. b { pos
      kv_write ((3 { y) , (<pos_b) , (<k_b) , (<v_b) , (<b))
      kv_result =. kv_read ((3 { y) , (<pos_b) , (<b))
      win =. pos_b + 1
      scores2 =. ((n_heads_kv , n_groups , head_dim) $ , q_b) (+/ .* "2) (1 2 0 |: (> 0 { kv_result))
      max_sf =. >./"1 scores2
      exp_sf =. ^ (scores2 - max_sf)
      softmax =. exp_sf % +/"1 exp_sf
      attn_out =. attn_out , <((n_heads * head_dim) $ , (softmax (+/ .* "2) (1 0 2 |: (> 1 { kv_result))))
      b =. b + 1
    end.
    attn_all =. > attn_out
  end.
  attn_result =. |: ((bd_attn_o block_data) (+/ .* ) |: attn_all)
  (<attn_result)
)

qwen_block_forward_bd =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  mi =. > 2 { y
  attn_result =. hidden qwen_attention_bd y
  attn_out =. > 0 { attn_result
  sa_out =. attn_out + hidden
  ffn_in =. rms_norm_rows ((< mi_rms_eps mi) , (< (bd_ff_norm block_data)) , <sa_out)
  ft =. |: ffn_in
  ffn_raw =. |: ((bd_ff_down block_data) (+/ .* ) ((bd_ff_gate block_data) (+/ .* ) ft) swiglu ((bd_ff_up block_data) (+/ .* ) ft))
  output =. ffn_raw + sa_out
  (<output)
)

qwen_run_blocks_bd =: 4 : 0
  'llm pos has_bias has_qk_norm' =. y
  mi =. llm_mi llm
  head_dim =. mi_head_dim mi
  n_heads =. mi_n_heads mi
  n_heads_kv =. mi_n_heads_kv mi
  block_count =. mi_block_count mi
  ctx_len =. mi_context_len mi
  state =. x
  if. 0 = # kv_meta do.
    kv_create ((<block_count) , (<ctx_len) , (<n_heads_kv) , (<head_dim))
  end.
  B =. {. $ state
  half =. <. head_dim % 2
  cos_all =. pos { mi_cos_tab mi
  sin_all =. pos { mi_sin_tab mi
  rope_exp =. (cos_all ; sin_all ; half ; n_heads ; n_heads_kv ; B) build_rope_expansions ''
  cos_expq =. > 0 { rope_exp
  sin_expq =. > 1 { rope_exp
  cos_expk =. > 2 { rope_exp
  sin_expk =. > 3 { rope_exp
  rope =. (<cos_all) , (<sin_all) , (<cos_expq) , (<sin_expq) , (<cos_expk) , (<sin_expk)
  b =. 0
  block_data_list =. llm_block_data llm
  bf_pre =. ((<pos) , (<mi))
  while. b < block_count do.
    result =. state qwen_block_forward_bd ((b { block_data_list) , bf_pre , (<b) , (<rope) , (<has_bias) , (<has_qk_norm))
    state =. > 0 { result
    b =. b + 1
  end.
  <state
)

NB. ---- qwen2/qwen3 shared BATCHED-PREFILL verbs ----
NB. y carries <has_bias; has_qk_norm> as the LAST 2 elements (same flag
NB. convention as the single/batched verbs). qwen2: has_bias=1/has_qk_norm=0;
NB. qwen3: has_bias=0/has_qk_norm=1. The prefill path always passes lens (the
NB. lens-masked per-sequence length, possibly empty) — at y index 4 (attention/
NB. block_forward) or 2 (run_blocks) — mirror the arch *_run_blocks_bp convention.
qwen_attention_bp =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  layer =. > 1 { y
  pos =. > 2 { y
  mi =. > 3 { y
  lens =. ''
  if. 4 < # y do. lens =. > 4 { y end.
  rope =. > 5 { y
  has_bias =. > 6 { y
  has_qk_norm =. > 7 { y
  B =. {. $ hidden
  c =. 1 { $ hidden
  emb_len =. 2 { $ hidden
  n_heads =. bd_n_heads block_data
  head_dim =. bd_head_dim block_data
  n_heads_kv =. bd_n_heads_kv block_data
  n_groups =. n_heads % n_heads_kv
  half =. <. head_dim % 2
  eff_seq =. > 1 { kv_meta

  hidden_flat =. rms_norm_rows ((< mi_rms_eps mi) , (< (bd_attn_norm block_data)) , <(((B*c) , emb_len) $ , hidden))

  thin =. |: hidden_flat
  qv =. |: ((bd_attn_q block_data) (+/ .* ) thin)
  kv =. |: ((bd_attn_k block_data) (+/ .* ) thin)
  vv =. |: ((bd_attn_v block_data) (+/ .* ) thin)
  if. has_bias do.
    qv =. qv + (((B*c) , n_heads * head_dim) $ bd_q_bias block_data)
    kv =. kv + (((B*c) , n_heads_kv * head_dim) $ bd_k_bias block_data)
    vv =. vv + (((B*c) , n_heads_kv * head_dim) $ bd_v_bias block_data)
  end.

  Q =. (B, c, n_heads, head_dim) $ , qv
  K =. (B, c, n_heads_kv, head_dim) $ , kv
  V =. (B, c, n_heads_kv, head_dim) $ , vv

  if. has_qk_norm do.
    Qf =. ((B*c*n_heads) , head_dim) $ , Q
    Qf =. rms_norm_rows ((< mi_rms_eps mi) , (< bd_q_norm block_data) , <Qf)
    Q =. (B, c, n_heads, head_dim) $ , Qf
    Kf =. ((B*c*n_heads_kv) , head_dim) $ , K
    Kf =. rms_norm_rows ((< mi_rms_eps mi) , (< bd_k_norm block_data) , <Kf)
    K =. (B, c, n_heads_kv, head_dim) $ , Kf
  end.

  Q =. Q rope_apply2_neox_b ((> 2 { rope) ; (> 3 { rope))
  K =. K rope_apply2_neox_b ((> 4 { rope) ; (> 5 { rope))

  Q =. Q % head_dim ^ 0.5

  attn_out =. ''
  b =. 0
  while. b < B do.
    pos_b =. b { pos
    base_b =. ((layer * kv_batch_g) + b) * eff_seq
    idxw =. base_b + pos_b + i. c
    k_cache_g =: ((c, n_heads_kv*head_dim) $ , (b { K)) idxw} k_cache_g
    v_cache_g =: ((c, n_heads_kv*head_dim) $ , (b { V)) idxw} v_cache_g
    kv_pos_g =: kv_pos_g >. pos_b + c
    win =. pos_b + c
    mask_2d =. (pos_b + i. c) </ i. win
    if. 0 < # lens do.
      mask_2d =. mask_2d +. ((i. win) >: b { lens)
    end.
    scores2 =. ((n_heads_kv, (n_groups*c), head_dim) $ , (1 0 2 |: (b { Q))) (+/ .* "2) (1 2 0 |: ((win, n_heads_kv, head_dim) $ , ((base_b + i. win) { k_cache_g)))
    scores2 =. scores2 -"2 (((n_groups * c), win) $ , (2 0 1 |: (mask_2d (*/) (n_groups $ 1))) * 1e9)
    max_sf =. >./"1 scores2
    exp_sf =. ^ (scores2 - max_sf)
    softmax_f =. exp_sf % +/"1 exp_sf
    attn_out =. attn_out , <((c, n_heads*head_dim) $ , (1 0 2 |: ((n_heads, c, head_dim) $ , (softmax_f (+/ .* "2) (1 0 2 |: ((win, n_heads_kv, head_dim) $ , ((base_b + i. win) { v_cache_g)))))))
    b =. b + 1
  end.
  attn_all =. > attn_out
  attn_result =. |: ((bd_attn_o block_data) (+/ .* ) |: (((B*c) , (n_heads*head_dim)) $ , attn_all))
  attn_result =. (B, c, emb_len) $ , attn_result
  (<attn_result)
)

qwen_block_forward_bp =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  mi =. > 3 { y
  B =. {. $ hidden
  c =. 1 { $ hidden
  emb_len =. 2 { $ hidden
  attn_result =. hidden qwen_attention_bp y
  attn_out =. > 0 { attn_result
  sa_out =. attn_out + hidden
  sa_flat =. ((B*c) , emb_len) $ , sa_out
  ffn_in =. rms_norm_rows ((< mi_rms_eps mi) , (< (bd_ff_norm block_data)) , <sa_flat)
  ft =. |: ffn_in
  ffn_raw =. |: ((bd_ff_down block_data) (+/ .* ) ((bd_ff_gate block_data) (+/ .* ) ft) swiglu ((bd_ff_up block_data) (+/ .* ) ft))
  output_flat =. ffn_raw + sa_flat
  output =. (B, c, emb_len) $ , output_flat
  (<output)
)

qwen_run_blocks_bp =: 4 : 0
  llm =. > 0 { y
  pos =. > 1 { y
  has_qk_norm =. > ((# y) - 1) { y
  has_bias =. > ((# y) - 2) { y
  lens =. ''
  if. 2 < (# y) - 2 do. lens =. > 2 { y end.
  mi =. llm_mi llm
  head_dim =. mi_head_dim mi
  n_heads =. mi_n_heads mi
  n_heads_kv =. mi_n_heads_kv mi
  block_count =. mi_block_count mi
  ctx_len =. mi_context_len mi
  state =. x
  if. 0 = # kv_meta do.
    kv_create ((<block_count) , (<ctx_len) , (<n_heads_kv) , (<head_dim))
  end.
  B =. {. $ state
  c =. 1 { $ state
  half =. <. head_dim % 2
  pos_bc_flat =. (B*c) $ , (pos +/ i. c)
  cos_all =. pos_bc_flat { mi_cos_tab mi
  sin_all =. pos_bc_flat { mi_sin_tab mi
  rope_exp =. (cos_all ; sin_all ; half ; n_heads ; n_heads_kv ; B*c) build_rope_expansions ''
  cos_expq =. (B, c, n_heads, half) $ , > 0 { rope_exp
  sin_expq =. (B, c, n_heads, half) $ , > 1 { rope_exp
  cos_expk =. (B, c, n_heads_kv, half) $ , > 2 { rope_exp
  sin_expk =. (B, c, n_heads_kv, half) $ , > 3 { rope_exp
  rope =. (<cos_all) , (<sin_all) , (<cos_expq) , (<sin_expq) , (<cos_expk) , (<sin_expk)
  bfb_pre =. ((<pos) , (<mi) , <lens)
  b =. 0
  block_data_list =. llm_block_data llm
  while. b < block_count do.
    result =. state qwen_block_forward_bp ((b { block_data_list) , (<b) , bfb_pre , (<rope) , (<has_bias) , (<has_qk_norm))
    state =. > 0 { result
    b =. b + 1
  end.
  <state
)

NB. ---- NEOX RoPE cos/sin table expansion (shared) ----
NB. x = <cos_all; sin_all; half; n_heads; n_heads_kv; lead> where cos_all/sin_all are
NB. (lead, half) tables and lead = L (run_blocks_b), B (run_blocks_bd), or B*c
NB. (run_blocks_bp — the caller reshapes the returned (lead,n_heads,half) to
NB. (B,c,n_heads,half)). Returns <cos_expq; sin_expq; cos_expk; sin_expk> each
NB. (lead, n_heads, half) — pre-expanded per-head, hoisted once per chunk and
NB. threaded through the layer loop (NEOX pairs (i, i+half)).
build_rope_expansions =: 4 : 0
  cos_all =. > 0 { x
  sin_all =. > 1 { x
  half =. > 2 { x
  n_heads =. > 3 { x
  n_heads_kv =. > 4 { x
  lead =. > 5 { x
  cos_expq =. (0 2 1) |: ((lead , half , n_heads) $ , (cos_all (*/) (n_heads $ 1)))
  sin_expq =. (0 2 1) |: ((lead , half , n_heads) $ , (sin_all (*/) (n_heads $ 1)))
  cos_expk =. (0 2 1) |: ((lead , half , n_heads_kv) $ , (cos_all (*/) (n_heads_kv $ 1)))
  sin_expk =. (0 2 1) |: ((lead , half , n_heads_kv) $ , (sin_all (*/) (n_heads_kv $ 1)))
  (cos_expq ; sin_expq ; cos_expk ; sin_expk)
)

NB. ---- Sample from logits ----
NB. y = <temp; k; p; min_p; logits>  (temp/k/p/min_p pre-boxed by the caller;
NB. sample_params_pack opens the boxed params list directly).
sample_from =: 3 : 0
  (0 1 2 3 { y) sampler_sample (> 4 { y)
)

NB. ---- Unified generation loop (all arches, fresh + resume) ----
NB. Replaces the four per-arch *_gen_loop copies (gem3/qw3/qw2/sm2) with ONE
NB. implementation. Per-arch differences (embedding scale, run_blocks /
NB. run_blocks_b verbs) are dispatched by llm_arch. The batched attention
NB. (attention_b) only attends within its batch, so a RESUME prefill processes
NB. the new segment tokens ONE AT A TIME through *_run_blocks (incremental,
NB. cache read at every position) — reusing the verified per-token path.
NB.
NB. x = llm ; y = <tokens; start_pos; max_steps; temp; k; p; min_p; stop_list>
NB.   tokens    = boxed list of token ids to (prefill + generate from)
NB.   start_pos = ''      -> FRESH: kv_create + batched prefill at position 0
NB.             = <number> -> RESUME: cache exists; prefill tokens incrementally
NB.                            at positions start_pos .. start_pos+#tokens-1
NB.   max_steps = generation cap; stop_list = stop-token ids (not appended).
NB.   Optional per-token callback: set the global verb gen_cb_g ('' = none) to a
NB.   monadic verb called on each GENERATED token (after sampling, before the
NB.   stop check). It may return a (possibly replaced) token id — returning a
NB.   stop-token id forces a stop — so it serves both as a streaming hook
NB.   (per-delta emit) and an interception point. Called for every token it
NB.   samples, including ones that would be dropped by stop_list. Verbs can't
NB.   be boxed into the y list (noun-verb syntax error), so it's a global.
NB. Returns the boxed token list: prompt tokens + generated (stops before stop).

NB. ---- Timing report ----
NB. x = <prefill_s; gen_s>   y = <prefill_toks; gen_toks>
NB. Echoes one line per phase: <time>s (<toks> tok, <tok/s> tok/s).
fmt2 =: 8j2 & ":

NB. y = <n; t> -> formatted tok/s (0 if t is 0).  Dyadic: x=n, y=t.
tokps =: 4 : 0
  if. y > 0 do. fmt2 x % y else. '0' end.
)

report_timing =: 4 : 0
  'pre_s gen_s' =. x
  'pre_toks gen_toks' =. y
  echo 'prefill: ', (fmt2 pre_s), 's (', (": pre_toks), ' tok, ', (pre_toks tokps pre_s), ' tok/s)'
  echo 'gen:     ', (fmt2 gen_s), 's (', (": gen_toks), ' tok, ', (gen_toks tokps gen_s), ' tok/s)'
)

NB. ---- Single forward-pass timing (infer verbs) ----
NB. y = <s; toks>  -> echoes a prefill line.
report_prefill =: 3 : 0
  's toks' =. y
  if. s > 0 do.
    echo 'prefill: ', (fmt2 s), 's (', (": toks), ' tok, ', (fmt2 toks % s), ' tok/s)'
  else.
    echo 'prefill: ', (fmt2 s), 's (', (": toks), ' tok)'
  end.
)
gen_loop_core =: 4 : 0
  llm =. x
  tokens =. > 0 { y
  start_pos =. > 1 { y
  max_steps =. > 2 { y
  temp =. > 3 { y
  k =. > 4 { y
  p =. > 5 { y
  min_p =. > 6 { y
  stop_list =. > 7 { y
  cb =. ''
  if. gen_cb_on_g do. cb =. gen_cb_g end.
  if. 0 < # sess_cur_g do. sess_kv_bind sess_cur_g end.

  arch =. llm_arch llm
  mi =. llm_mi llm
  emb_len =. mi_emb_len mi
  block_count =. mi_block_count mi
  ctx_len =. mi_context_len mi
  n_heads_kv =. mi_n_heads_kv mi
  head_dim =. mi_head_dim mi
  emb_w =. 'token_embd.weight' get_tensor_cached_d llm
  output_norm_w =. 'output_norm.weight' get_tensor_cached_d llm
  logit_div =. 1

  NB. Per-arch: embedding scale, logit scale, block-run verbs
  rec_reset =. ''
  select. arch
  case. 'gemma3' do.
    scale =. %: emb_len
    rb =. gem3_run_blocks
    rb_b =. gem3_run_blocks_b
  case. 'qwen3' do.
    scale =. 1
    rb =. qw3_run_blocks
    rb_b =. qw3_run_blocks_b
  case. 'qwen2' do.
    scale =. 1
    rb =. qw2_run_blocks
    rb_b =. qw2_run_blocks_b
  case. 'llama' do.
    scale =. 1
    rb =. llama_run_blocks
    rb_b =. llama_run_blocks_b
  case. 'qwen35' do.
    scale =. 1
    rb =. qw35_run_blocks
    rb_b =. qw35_run_blocks_b
    rec_reset =. rs_reset
  case. 'granite' do.
    scale =. granite_mi_embed_scale mi
    logit_div =. granite_mi_logit_scale mi
    rb =. granite_run_blocks
    rb_b =. granite_run_blocks_b
  case. 'ernie4_5' do.
    scale =. 1
    rb =. ernie_run_blocks
    rb_b =. ernie_run_blocks_b
  case. 'lfm2' do.
    scale =. 1
    rb =. lf2_run_blocks
    rb_b =. lf2_run_blocks_b
    rec_reset =. lf2_conv_reset
    NB. LFM2 has no output_norm.weight — the norm is token_embd_norm.weight
    output_norm_w =. 'token_embd_norm.weight' get_tensor_cached_d llm
  end.

  tok_list =. , > tokens
  L =. # tok_list
  NB. Effective context = model max, or the low-memory override. The cache is
  NB. pre-allocated to eff_seq and cannot grow past it, so truncate the prompt
  NB. (fresh) / new segment (resume) to fit. Fresh keeps the LAST eff_seq
  NB. tokens; resume keeps the last eff_seq-start_pos tokens of the segment.
  eff_seq =. ctx_len
  if. 0 < kv_max_seq_g do. eff_seq =. ctx_len <. kv_max_seq_g end.
  NB. Bound eff_seq so the KV cache fits J's array-size limit (2^31 elements)
  NB. — mirrors kv_create's bound (huge-ctx models, e.g. qwen35 ctx=262144).
  eff_seq =. eff_seq <. (<. (2^31) % (block_count * kv_batch_g * (n_heads_kv * head_dim)))
  if. '' -: start_pos do.
    if. L > eff_seq do. tok_list =. tok_list {~ (L - eff_seq) + i. eff_seq end.
  else.
    if. (start_pos + L) > eff_seq do.
      keep =. eff_seq - start_pos
      if. keep <: 0 do. keep =. 0 end.
      tok_list =. tok_list {~ (L - keep) + i. keep
    end.
  end.
  L =. # tok_list
  output =. <"0 tok_list
  NB. Box the llm noun once — reused by the prefill rb_b calls and decode rb.
  llm_box =. <llm

  if. '' -: start_pos do.
    NB. FRESH: create zeroed cache, then batched prefill the prompt in CHUNKS.
    NB. A one-shot full-batch prefill materializes a quadratic (n_heads, L, L)
    NB. scores matrix PER LAYER — OOM on deep contexts (e.g. 8k-token prompt:
    NB. llama-3.2-1b ~80 GB/layer). Chunking bounds the peak to
    NB. (n_heads, chunk, context) and is result-identical: each query still
    NB. attends to all keys <= its position (later positions are causally
    NB. masked anyway). rb_b is already cache-prefix aware, so successive
    NB. chunks write their K/V and the next chunk attends to them.
    kv_create ((<block_count) , (<ctx_len) , (<n_heads_kv) , (<head_dim))
    NB. Fresh generation: zero the arch recurrent-state caches (lfm2 conv,
    NB. qwen35 delta-net) so the prefill does NOT inherit a previous run's state.
    select. arch
    case. 'lfm2' do. rec_reset ((<lf2_n_conv_g) , (<2) , (<emb_len))
    case. 'qwen35' do. rec_reset ((<18) , (<3) , (<6144) , (<128) , <16)
    case. 'gemma3' do. '' end.
    chunk_sz =. prefill_chunk_sz
    pre_s =. 0
    i =. 0
    while. i < L do.
      c =. chunk_sz <. L - i
      seg =. (i + i. c) { tok_list
      emb_seg =. scale * |: (seg {"1 emb_w)
      if. gen_timing_g do.
        t =. 6!:2 'result_b =. emb_seg rb_b (llm_box , <i)'
        pre_s =. pre_s + t
      else.
        result_b =. emb_seg rb_b (llm_box , <i)
      end.
      h_b =. > 0 { result_b
      hidden =. > (c - 1) { h_b
      i =. i + c
    end.
    cur_pos =. L
  else.
    NB. RESUME: cache already holds positions 0..start_pos-1. Prefill the new
    NB. segment with ONE batched pass at start_pos — the batched attention
    NB. attends to the cache prefix (positions 0..start_pos-1) PLUS its own
    NB. batch, and writes the batch K/V at start_pos. (The old per-token
    NB. incremental loop is replaced: rb_b is now cache-prefix aware.)
    cur_pos =. start_pos
    emb_all =. |: (tok_list {"1 emb_w)
    if. scale ~: 1 do. emb_all =. emb_all * scale end.
    if. gen_timing_g do.
      t =. 6!:2 'result_b =. emb_all rb_b (llm_box , <start_pos)'
      pre_s =. pre_s + t
    else.
      result_b =. emb_all rb_b (llm_box , <start_pos)
    end.
    h_b =. > 0 { result_b
    hidden =. > (L - 1) { h_b
    cur_pos =. cur_pos + L
  end.

  NB. Generation loop (identical for fresh and resume). Step 0 predicts from
  NB. the last prefill hidden WITHOUT re-embedding (the old code double-
  NB. processed the last prompt token); later steps embed the previous token
  NB. and run blocks at cur_pos. All arches share output_head (rms_norm + lm_head).
  NB. Pre-box the generation-invariant args ONCE (output_head/sample_from prefixes,
  NB. llm box) so the per-token loop only boxes the varying piece.
  oh_pre =. ((< mi_rms_eps mi) , (<output_norm_w) , (<emb_w))
  sf_pre =. ((<temp) , (<k) , (<p) , (<min_p))
  gen_step =. 0
  gen_s =. 0
  while. gen_step < max_steps do.
    if. cur_pos >: eff_seq do. break. end.
    if. 0 = gen_step do.
      logits =. output_head (oh_pre , <hidden)
      if. logit_div ~: 1 do. logits =. logits % logit_div end.
    else.
      last_tok =. > {: output
      hidden =. |: (last_tok {"1 emb_w)
      if. scale ~: 1 do. hidden =. hidden * scale end.
      if. gen_timing_g do.
        t =. 6!:2 'result =. hidden rb (llm_box , <cur_pos)'
        gen_s =. gen_s + t
      else.
        result =. hidden rb (llm_box , <cur_pos)
      end.
      hidden =. > 0 { result
      logits =. output_head (oh_pre , <hidden)
      if. logit_div ~: 1 do. logits =. logits % logit_div end.
      cur_pos =. cur_pos + 1
    end.
        pred =. sample_from (sf_pre , <logits)
    if. gen_cb_on_g do. pred =. cb pred end.
    if. (stop_list i. pred) < # stop_list do. break. end.    output =. output , <pred
    gen_step =. gen_step + 1
  end.
  if. gen_timing_g do. (pre_s , gen_s) report_timing (L , gen_step) end.
  if. 0 < # sess_cur_g do.
    sess_cur_g =: (<kv_pos_g) (10) } sess_cur_g
    sess_cur_g =: (<kv_meta) (13) } sess_cur_g
  end.
  output
)

NB. ---- Batched generation loop: B independent sequences in parallel ----
NB. The weight matmuls are amortized across B (M=1 matvec floor is
NB. memory-bound on weight reads); each sequence keeps its own KV cache
NB. window + position (cache is B-sized via kv_batch_g; prefill sets
NB. kv_seq_g per sequence). Stop per sequence; returns boxed list of B
NB. token outputs (prompt tokens + generated, stops before stop tokens).
NB. x = llm ; y = <prompts_tok; max_steps; temp; k; p; min_p; stop_list>
NB.   prompts_tok = boxed list of B token lists (already chat-framed).
gen_loop_batch =: 4 : 0
  llm =. x
  prompts_tok =. > 0 { y
  max_steps =. > 1 { y
  temp =. > 2 { y
  k =. > 3 { y
  p =. > 4 { y
  min_p =. > 5 { y
  stop_list =. > 6 { y
  B =. # prompts_tok
  if. 0 < # sess_cur_g do. sess_kv_bind sess_cur_g end.

  arch =. llm_arch llm
  mi =. llm_mi llm
  emb_len =. mi_emb_len mi
  block_count =. mi_block_count mi
  ctx_len =. mi_context_len mi
  n_heads_kv =. mi_n_heads_kv mi
  head_dim =. mi_head_dim mi
  emb_w =. 'token_embd.weight' get_tensor_cached_d llm
  output_norm_w =. 'output_norm.weight' get_tensor_cached_d llm
  logit_div =. 1
  scale =. 1
  rb_b =. ''
  rb_bd =. ''
  rb_bp =. ''
  rb_bp_flag =. 0
  rec_reset =. ''
  select. arch
  case. 'gemma3' do.
    scale =. %: emb_len
    rb_b =. gem3_run_blocks_b
    rb_bd =. gem3_run_blocks_bd
    rb_bp =. gem3_run_blocks_bp
    rb_bp_flag =. 1
  case. 'qwen2' do.
    scale =. 1
    rb_b =. qw2_run_blocks_b
    rb_bd =. qw2_run_blocks_bd
    rb_bp =. qw2_run_blocks_bp
    rb_bp_flag =. 1
  case. 'qwen3' do.
    scale =. 1
    rb_b =. qw3_run_blocks_b
    rb_bd =. qw3_run_blocks_bd
    rb_bp =. qw3_run_blocks_bp
    rb_bp_flag =. 1
  case. 'llama' do.
    scale =. 1
    rb_b =. llama_run_blocks_b
    rb_bd =. llama_run_blocks_bd
    rb_bp =. llama_run_blocks_bp
    rb_bp_flag =. 1
  case. 'qwen35' do.
    scale =. 1
    rb_b =. qw35_run_blocks_b
    rb_bd =. qw35_run_blocks_bd
    rb_bp =. qw35_run_blocks_bp
    rb_bp_flag =. 1
    rec_reset =. rs_reset
  case. 'granite' do.
    scale =. granite_mi_embed_scale mi
    logit_div =. granite_mi_logit_scale mi
    rb_b =. granite_run_blocks_b
    rb_bd =. granite_run_blocks_bd
    rb_bp =. granite_run_blocks_bp
    rb_bp_flag =. 1
  case. 'ernie4_5' do.
    scale =. 1
    rb_b =. ernie_run_blocks_b
    rb_bd =. ernie_run_blocks_bd
    rb_bp =. ernie_run_blocks_bp
    rb_bp_flag =. 1
  case. 'lfm2' do.
    scale =. 1
    rb_b =. lf2_run_blocks_b
    rb_bd =. lf2_run_blocks_bd
    rb_bp =. lf2_run_blocks_bp
    rb_bp_flag =. 1
    rec_reset =. lf2_conv_reset
    output_norm_w =. 'token_embd_norm.weight' get_tensor_cached_d llm
  end.

  eff_seq =. ctx_len
  if. 0 < kv_max_seq_g do. eff_seq =. ctx_len <. kv_max_seq_g end.
  NB. Bound eff_seq so the KV cache fits J's array-size limit (2^31 elements)
  NB. — mirrors kv_create's bound (huge-ctx models, e.g. qwen35 ctx=262144).
  eff_seq =. eff_seq <. (<. (2^31) % (block_count * kv_batch_g * (n_heads_kv * head_dim)))

  NB. Prefill each sequence (chunked, per-sequence cache via kv_seq_g)
  kv_create ((<block_count) , (<ctx_len) , (<n_heads_kv) , (<head_dim))
  NB. Fresh generation: zero the arch recurrent-state caches (lfm2 conv,
  NB. qwen35 delta-net) so the per-sequence prefills do NOT inherit a previous
  NB. run's state.
  select. arch
  case. 'lfm2' do. rec_reset ((<lf2_n_conv_g) , (<2) , (<emb_len))
  case. 'qwen35' do. rec_reset ((<18) , (<3) , (<6144) , (<128) , <16)
  case. 'gemma3' do. '' end.
  pos =. B $ 0
  outputs =. ''
  hidden_all =. ''
  last_toks =. ''
  pre_s =. 0
  pre_toks =. 0
  NB. Box the llm noun once — reused by the prefill rb_b/rb_bp and decode rb_bd.
  llm_box =. <llm
  if. rb_bp_flag do.
    NB. ---- Batched prefill (B sequences in lockstep, padded, lens-masked) ----
    NB. Pad every sequence to the longest prompt length, then process ALL B
    NB. sequences' chunks together in one forward pass (weight-read amortized).
    NB. lens[b] = min(L_b, p+c) masks each sequence's padding keys; the padding
    NB. rows' outputs are discarded — only the hidden of each sequence's LAST
    NB. real token (captured in the chunk that covers position L_b-1) is kept.
    lens_b =. ''
    i =. 0
    while. i < B do.
      tok_list =. > i { prompts_tok
      L =. # tok_list
      if. L > eff_seq do. tok_list =. tok_list {~ (L - eff_seq) + i. eff_seq; L =. # tok_list end.
      prompts_tok =. (<tok_list) i} prompts_tok
      lens_b =. lens_b , L
      i =. i + 1
    end.
    max_L =. >./ lens_b
    pre_toks =. +/ lens_b
    NB. pad each seq to max_L with token 0 (padding, lens-masked)
    padded =. ''
    i =. 0
    while. i < B do.
      tok_list =. > i { prompts_tok
      padded =. padded , <(tok_list , (max_L - # tok_list) $ 0)
      i =. i + 1
    end.
    p =. 0
    hidden_all =. B $ <''   NB. one per seq; filled by index in seq order
    outputs =. ''
    last_toks =. B $ 0
    while. p < max_L do.
      c =. prefill_chunk_sz <. max_L - p
      NB. Build (B, c, emb) hidden: embed each seq's padded chunk
      emb_seg =. (B, c, emb_len) $ 0
      lens_cur =. B $ 0
      i =. 0
      while. i < B do.
        seg =. (p + i. c) { > i { padded
        es =. scale * |: (seg {"1 emb_w)   NB. (c, emb) — already the row shape, no re-reshape needed
        emb_seg =. es i} emb_seg
        lens_cur =. (p + (c <. ((i { lens_b) - p))) i} lens_cur
        i =. i + 1
      end.
      if. gen_timing_g do.
        t =. 6!:2 'result_b =. emb_seg rb_bp (llm_box , (<(B $ p)) , <lens_cur)'
        pre_s =. pre_s + t
      else.
        result_b =. emb_seg rb_bp (llm_box , (<(B $ p)) , <lens_cur)
      end.
      h_b =. > 0 { result_b   NB. (B, c, emb)
      i =. 0
      while. i < B do.
        L_i =. i { lens_b
        if. (L_i > p) *. (L_i <: p + c) do.
          NB. This chunk covers seq i's last real token (L_i-1 at row L_i-p-1)
          real_c =. L_i - p
          hid_i =. (real_c - 1) { i { h_b   NB. (emb,)
          hidden_all =. (<hid_i) i} hidden_all
          pos =. L_i i} pos
          last_toks =. ({: > i { prompts_tok) i} last_toks
        end.
        i =. i + 1
      end.
      p =. p + c
    end.
    i =. 0
    while. i < B do.
      outputs =. outputs , <(<"0 (> i { prompts_tok))
      i =. i + 1
    end.
    hidden =. > hidden_all   NB. (B, emb) — already the row shape, no re-reshape needed
    cur_pos =. pos
    done =. B $ 0
  else.
    NB. ---- Per-sequence prefill (fallback for arches without a bp verb) ----
    i =. 0
    while. i < B do.
      tok_list =. > i { prompts_tok
      L =. # tok_list
      if. L > eff_seq do. tok_list =. tok_list {~ (L - eff_seq) + i. eff_seq; L =. # tok_list end.
      pre_toks =. pre_toks + L
      kv_seq_g =: i
      j =. 0
      while. j < L do.
        c =. prefill_chunk_sz <. L - j
        seg =. (j + i. c) { tok_list
      emb_seg =. |: (seg {"1 emb_w)
      if. scale ~: 1 do. emb_seg =. emb_seg * scale end.
        if. gen_timing_g do.
          t =. 6!:2 'result_b =. emb_seg rb_b (llm_box , <j)'
          pre_s =. pre_s + t
        else.
          result_b =. emb_seg rb_b (llm_box , <j)
        end.
        h_b =. > 0 { result_b
        hidden =. > (c - 1) { h_b
        j =. j + c
      end.
      pos =. L i} pos
      outputs =. outputs , <(<"0 tok_list)
      hidden_all =. hidden_all , <hidden
      last_toks =. last_toks , {: tok_list
      i =. i + 1
    end.
    kv_seq_g =: 0
    hidden =. > hidden_all   NB. (B, emb) — already the row shape, no re-reshape needed
    cur_pos =. pos
    done =. B $ 0
  end.

  NB. Batched decode loop: embed B last tokens, one forward pass, sample B.
  NB. Pre-box the generation-invariant prefixes once (per-step loop only boxes
  NB. the varying hidden/logits slice): sample_from params + rms_norm_rows norm.
  sf_pre =. ((<temp) , (<k) , (<p) , (<min_p))
  rnn_pre =. ((< mi_rms_eps mi) , (<output_norm_w))
  gen_step =. 0
  gen_s =. 0
  while. gen_step < max_steps do.
    if. B = +/ done do. break. end.
    NB. Stop sequences whose position reached eff_seq (cache cannot grow past)
    b =. 0
    while. b < B do.
      if. (cur_pos >: eff_seq) *. -. b { done do.
        done =. 1 b} done
      end.
      b =. b + 1
    end.
    if. 0 = gen_step do.
      NB. Step 0 predicts from the prefill-last hidden WITHOUT re-embedding (the
      NB. prefill already ran blocks on the last prompt token) — mirrors
      NB. gen_loop_core. Re-embedding at pos L would duplicate the last prompt
      NB. token's K/V into the cache and shift the outputs by one step.
    else.
      hidden =. |: (last_toks {"1 emb_w)   NB. (B, emb)
      if. scale ~: 1 do. hidden =. hidden * scale end.
      if. gen_timing_g do.
        t =. 6!:2 'result =. hidden rb_bd (llm_box , <cur_pos)'
        gen_s =. gen_s + t
      else.
        result =. hidden rb_bd (llm_box , <cur_pos)
      end.
      hidden =. > 0 { result
    end.
    hidden_n =. rms_norm_rows (rnn_pre , <hidden)
    logits =. hidden_n (+/ .*) emb_w   NB. (B, vocab) — emb_w transposed (emb, vocab)
    if. logit_div ~: 1 do. logits =. logits % logit_div end.
    b =. 0
    while. b < B do.
      if. -. b { done do.
        pred =. sample_from (sf_pre , <(b { logits))
        if. (stop_list i. pred) < # stop_list do.
          done =. 1 b} done
        else.
          outputs =. (<((> b { outputs) , <pred)) b} outputs
          if. gen_step > 0 do.
            cur_pos_b =. b { cur_pos
            cur_pos =. (cur_pos_b + 1) b} cur_pos
          end.
          last_toks =. pred b} last_toks
        end.
      end.
      b =. b + 1
    end.
    gen_step =. gen_step + 1
  end.
  if. gen_timing_g do. (pre_s , gen_s) report_timing (pre_toks , gen_step) end.
  if. 0 < # sess_cur_g do.
    sess_cur_g =: (<kv_pos_g) (10) } sess_cur_g
    sess_cur_g =: (<kv_meta) (13) } sess_cur_g
  end.
  outputs
)

NB. ---- Infer arg parsing (canonical form) ----
NB. y = (text ; <params>)  where <params> is <temp;k;p;min_p> possibly double-boxed.
NB. Returns <text; temp; k; p; min_p>.
infer_args =: 3 : 0
  text =. > 0 { y
  params =. > 1 { y
  flat =. sample_params_pack params   NB. open <temp;k;p;min_p>, defaults filled
  temp =. 0 { flat
  k =. 1 { flat
  p =. 2 { flat
  min_p =. 3 { flat
  (<text) , (<temp) , (<k) , (<p) , (<min_p)
)

NB. ---- Generate arg parsing (canonical form) ----
NB. y = (text ; max_steps ; <params>)  where <params> is <temp;k;p;min_p> possibly double-boxed.
NB. Returns <text; max_steps; temp; k; p; min_p>.
gen_args =: 3 : 0
  text =. > 0 { y
  max_steps =. > 1 { y
  params =. > 2 { y
  flat =. sample_params_pack params   NB. open <temp;k;p;min_p>, defaults filled
  temp =. 0 { flat
  k =. 1 { flat
  p =. 2 { flat
  min_p =. 3 { flat
  (<text) , (<max_steps) , (<temp) , (<k) , (<p) , (<min_p)
)

NB. ---- Special-token helpers (shared, non-arch) ----
NB. Extract marker strings (e.g. '<start_of_turn>', '<|im_start|>') from a chat
NB. template string: scan for '<', take through the next '>'. Callers filter the
NB. result to real vocab entries.
template_markers =: 3 : 0
  t =. y
  res =. 0 $ <''
  i =. 0
  while. i < # t do.
    if. '<' = i { t do.
      j =. i + 1
      while. (j < # t) *. '>' ~: j { t do. j =. j + 1 end.
      if. j < # t do.
        res =. res , <(((j - i) + 1) {. i }. t)
        i =. j + 1
      else.
        i =. # t
      end.
    else.
      i =. i + 1
    end.
  end.
  ~. res
)

NB. Split text on special-token markers. x = specials (boxed list), y = text.
NB. Returns boxed list of segments; marker segments are the special strings.
split_specials =: 4 : 0
  specials =. x
  text =. , y   NB. ravel — handle scalar (single-char) input
  segs =. 0 $ <''
  while. 0 < # text do.
    best_pos =. # text
    best_s =. ''
    j =. 0
    while. j < # specials do.
      s =. > j { specials
      occ =. s E. text
      if. 1 e. occ do.
        p =. occ i. 1
        if. p < best_pos do.
          best_pos =. p
          best_s =. s
        end.
      end.
      j =. j + 1
    end.
    if. 0 = # best_s do.
      segs =. segs , <text
      text =. ''
    else.
      if. 0 < best_pos do. segs =. segs , <(best_pos {. text) end.
      segs =. segs , <best_s
      text =. (best_pos + (# best_s)) }. text
    end.
  end.
  segs
)

