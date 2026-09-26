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

NB. ---- llm noun layout (shared across architectures) ----
NB. llm = <path; ti; default_params; tokenizer; mi; kv_cache; tds; all_tensors;
NB.       block_data; kv_data; arch>
llm_path          =: >@(0&{)
llm_ti            =: >@(1&{)
llm_default_params =: >@(2&{)   NB. chat sampling defaults <temp;k;p;min_p>
llm_mi            =: >@(4&{)
llm_kv_cache      =: >@(5&{)   NB. llm[5]=<k;v>
llm_tds           =: >@(6&{)
llm_all_tensors   =: >@(7&{)
llm_block_data    =: >@(8&{)
llm_kv_data       =: >@(9&{)   NB. decoded-KV dict (query via kv_get)
llm_arch          =: >@(10&{)  NB. arch string ('gemma3'|'llama'|'qwen2')

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
NB. y = <rms_eps; output_norm_w; emb_w_final; hidden>
output_head =: 3 : 0
  eps =. > 0 { y
  onw =. > 1 { y
  efw =. > 2 { y
  hidden =. > 3 { y
  hidden =. rms_norm ((<eps) , (<onw) , <hidden)
  hidden (+/ .* ) efw
)

NB. ---- Sample from logits ----
NB. y = <temp; k; p; min_p; logits>
sample_from =: 3 : 0
  temp =. > 0 { y
  k =. > 1 { y
  p =. > 2 { y
  min_p =. > 3 { y
  logits =. > 4 { y
  flat =. temp , k , p , min_p
  params =. <"0 flat
  params sampler_sample logits
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
      t =. 6!:2 'result_b =. emb_seg rb_b ((<llm) , <i)'
      pre_s =. pre_s + t
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
    emb_all =. scale * |: (tok_list {"1 emb_w)
    pre_s =. 6!:2 'result_b =. emb_all rb_b ((<llm) , <start_pos)'
    h_b =. > 0 { result_b
    hidden =. > (L - 1) { h_b
    cur_pos =. cur_pos + L
  end.

  NB. Generation loop (identical for fresh and resume). Step 0 predicts from
  NB. the last prefill hidden WITHOUT re-embedding (the old code double-
  NB. processed the last prompt token); later steps embed the previous token
  NB. and run blocks at cur_pos. All arches share output_head (rms_norm + lm_head).
  gen_step =. 0
  gen_s =. 0
  while. gen_step < max_steps do.
    if. cur_pos >: eff_seq do. break. end.
    if. 0 = gen_step do.
      logits =. output_head ((< mi_rms_eps mi) , (<output_norm_w) , (<emb_w) , <hidden)
      logits =. logits % logit_div
    else.
      last_tok =. > {: output
      hidden =. scale * |: (last_tok {"1 emb_w)
      gen_s =. gen_s + 6!:2 'result =. hidden rb (<llm) , <cur_pos'
      hidden =. > 0 { result
      logits =. output_head ((< mi_rms_eps mi) , (<output_norm_w) , (<emb_w) , <hidden)
      logits =. logits % logit_div
      cur_pos =. cur_pos + 1
    end.
        pred =. sample_from ((<temp) , (<k) , (<p) , (<min_p) , <logits)
    if. gen_cb_on_g do. pred =. cb pred end.
    if. (stop_list i. pred) < # stop_list do. break. end.    output =. output , <pred
    gen_step =. gen_step + 1
  end.
  (pre_s , gen_s) report_timing (L , gen_step)
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
    rec_reset =. lf2_conv_reset
    output_norm_w =. 'token_embd_norm.weight' get_tensor_cached_d llm
  end.

  eff_seq =. ctx_len
  if. 0 < kv_max_seq_g do. eff_seq =. ctx_len <. kv_max_seq_g end.

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
        es =. scale * |: (seg {"1 emb_w)   NB. (c, emb)
        emb_seg =. ((c, emb_len) $ , es) i} emb_seg
        lens_cur =. (p + (c <. ((i { lens_b) - p))) i} lens_cur
        i =. i + 1
      end.
      t =. 6!:2 'result_b =. emb_seg rb_bp ((<llm) , (<(B $ p)) , <lens_cur)'
      pre_s =. pre_s + t
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
    hidden =. (B , emb_len) $ , > hidden_all
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
        emb_seg =. scale * |: (seg {"1 emb_w)
        t =. 6!:2 'result_b =. emb_seg rb_b ((<llm) , <j)'
        pre_s =. pre_s + t
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
    hidden =. (B , emb_len) $ , > hidden_all
    cur_pos =. pos
    done =. B $ 0
  end.

  NB. Batched decode loop: embed B last tokens, one forward pass, sample B.
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
      hidden =. scale * |: (last_toks {"1 emb_w)   NB. (B, emb)
      gen_s =. gen_s + 6!:2 'result =. hidden rb_bd (<llm) , <cur_pos'
      hidden =. > 0 { result
    end.
    hidden_n =. rms_norm_rows ((< mi_rms_eps mi) , (<output_norm_w) , <hidden)
    logits =. hidden_n (+/ .*) emb_w   NB. (B, vocab) — emb_w transposed (emb, vocab)
    logits =. logits % logit_div
    b =. 0
    while. b < B do.
      if. -. b { done do.
        pred =. sample_from ((<temp) , (<k) , (<p) , (<min_p) , <(b { logits))
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
  (pre_s , gen_s) report_timing (pre_toks , gen_step)
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
  if. 1 = # params do.
    flat =. > > params
  else.
    flat =. > params
  end.
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
  if. 1 = # params do.
    flat =. > > params
  else.
    flat =. > params
  end.
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

