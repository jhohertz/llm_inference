NB. ================================================================
NB. Qwen2.5-Coder (qwen2 arch) — standard decoder transformer, GQA, SwiGLU
NB. Separate QKV/O weights, no fused, no q/k norm, no post-attention/ffn norms.
NB. NEOX RoPE (pairs offset by head_dim/2), NOT interleaved. Q/K/V have biases.
NB. Tokenizer: GPT-2 byte-level BPE.
NB. Depends on: llm_core.ijs, kernels.ijs, gguf.ijs, kv_cache.ijs, sampler.ijs,
NB.           tokenizer_gpt2.ijs
NB. ================================================================
coclass 'inference'
require 'llm/inference/util/llm_core'
require 'llm/inference/tokenizers/tokenizer_gpt2'
require 'llm/inference/util/minja'
require 'llm/inference/util/chat_template'

NB. ---- Helper: move axes (dyadic |:) using variable axis list ----

NB. ---- Extract qwen2 model info from KV pairs ----
NB. mi = <block_count; context_len; emb_len; n_heads; n_heads_kv; head_dim; rope_freq; vocab_size; rms_eps; n_ff>
qw2_extract_hparams =: 3 : 0
  data =. y
  block_count =. 'qwen2.block_count' kv_uint data
  context_length =. 'qwen2.context_length' kv_uint data
  emb_len =. 'qwen2.embedding_length' kv_uint data
  n_heads =. 'qwen2.attention.head_count' kv_uint data
  n_heads_kv =. 'qwen2.attention.head_count_kv' kv_uint data
  rope_freq =. 'qwen2.rope.freq_base' kv_float data
  vocab_size =. 'qwen2.vocab_size' kv_uint data
  rms_eps =. 'qwen2.attention.layer_norm_rms_epsilon' kv_float data
  n_ff =. 'qwen2.feed_forward_length' kv_uint data
  key_len =. 'qwen2.attention.key_length' kv_uint data
  if. key_len <: 0 do. key_len =. emb_len % n_heads end.
  head_dim =. key_len
  (<"0) block_count , context_length , emb_len , n_heads , n_heads_kv , head_dim , rope_freq , vocab_size , rms_eps , n_ff
)

NB. ---- block_data accessors (shared bd_* in llm_core.ijs) ----
NB. ---- Build one block's data (rank-1; x=llm, y=block index) ----
qw2_build_block =: 4 : 0
  llm =. x
  b =. y
  mi =. llm_mi llm
  n_heads =. mi_n_heads mi
  n_heads_kv =. mi_n_heads_kv mi
  p =. 'blk.' , (": b) , '.'
  attn_norm =. (p , 'attn_norm.weight') get_tensor_cached_d llm
  attn_q =. (p , 'attn_q.weight') get_tensor_cached_d llm
  attn_k =. (p , 'attn_k.weight') get_tensor_cached_d llm
  attn_v =. (p , 'attn_v.weight') get_tensor_cached_d llm
  attn_o_w =. (p , 'attn_output.weight') get_tensor_cached_d llm
  q_bias =. (p , 'attn_q.bias') get_tensor_cached_d llm
  k_bias =. (p , 'attn_k.bias') get_tensor_cached_d llm
  v_bias =. (p , 'attn_v.bias') get_tensor_cached_d llm
  ffn_norm =. (p , 'ffn_norm.weight') get_tensor_cached_d llm
  ff_gate =. (p , 'ffn_gate.weight') get_tensor_cached_d llm
  ff_up =. (p , 'ffn_up.weight') get_tensor_cached_d llm
  ff_down =. (p , 'ffn_down.weight') get_tensor_cached_d llm
  head_dim =. (0 { $ attn_q) % n_heads
  (<attn_norm),(<attn_q),(<attn_k),(<attn_v),(<attn_o_w),(<q_bias),(<k_bias),(<v_bias),(<ffn_norm),(<ff_gate),(<ff_up),(<ff_down),(<0),(<0),(<n_heads),(<head_dim),(<n_heads_kv)
)

NB. ---- Pre-build all block data for Qwen2 ----
qw2_pre_build_block_data =: 3 : 0
  llm =. y
  block_count =. mi_block_count (llm_mi llm)
  (<llm) qw2_build_block each i. block_count
)

NB. ---- Expand KV axis (n_kv) -> n_heads (each KV head repeated n_groups times) ----

NB. ---- Single-token attention (qwen2: GQA, NEOX RoPE, Q/K/V biases) ----
NB. x = hidden (emb,) — the layer input; y = <block_data; pos; mi; layer>
qw2_attention =: 4 : 0
  x   qwen_attention ((y , <1) , <0)
)

NB. ---- Batched attention (prompt prefill, GQA, causal) ----
NB. x = hidden (L, emb); y = <block_data; mi; layer>
qw2_attention_b =: 4 : 0
  x   qwen_attention_b ((y , <1) , <0)
)

NB. ---- Single block forward (qwen2) ----
NB. x = hidden (emb,) — the layer input; y = <block_data; pos; mi; layer>
qw2_block_forward =: 4 : 0
  x   qwen_block_forward ((y , <1) , <0)
)

NB. ---- Batched block forward ----
NB. x = hidden (L, emb); y = <block_data; mi; layer; start_pos; rope>
qw2_block_forward_b =: 4 : 0
  x   qwen_block_forward_b ((y , <1) , <0)
)

NB. ---- Run all blocks (single token; cache lives in kv_cache_g global) ----
NB. x = hidden (emb,); y = <llm; pos>
qw2_run_blocks =: 4 : 0
  x   qwen_run_blocks ((y , <1) , <0)
)

NB. ---- Batched run all blocks (prompt prefill) ----
NB. x = hidden (L, emb); y = <llm; start_pos>  (positions start_pos..start_pos+L-1)
qw2_run_blocks_b =: 4 : 0
  x   qwen_run_blocks_b ((y , <1) , <0)
)

NB. ---- Batched-DECODE attention (B sequences, ONE token each at pos[b]) ----
NB. The weight projections (Q/K/V, O) are batched — (B, emb) x (emb, n) reads
NB. the weights ONCE for B sequences (the M=1 matvec amortization). The
NB. cache window scores/softmax/output stay PER-SEQUENCE (each has its own
NB. cache window). x = hidden (B, emb); y = <block_data; pos; mi; layer>
NB. pos = B-vector of positions.
qw2_attention_bd =: 4 : 0
  x   qwen_attention_bd ((y , <1) , <0)
)

NB. ---- Batched-decode block forward ----
NB. x = hidden (B, emb); y = <block_data; pos; mi; layer>  pos = B-vector
qw2_block_forward_bd =: 4 : 0
  x   qwen_block_forward_bd ((y , <1) , <0)
)

NB. ---- Run all blocks for B sequences (one token each at pos[b]) ----
NB. x = hidden (B, emb); y = <llm; pos>  pos = B-vector
qw2_run_blocks_bd =: 4 : 0
  x   qwen_run_blocks_bd ((y , <1) , <0)
)

NB. ---- Batched-PREFILL attention (qwen2): B chunks, one per seq at pos[b] ----
NB. Mirrors llama_attention_bp but for the qwen2 arch: Q/K/V biases, NEOX RoPE
NB. (split-first/second-half, no permute), Q scaled by 1/sqrt(head_dim), NO
NB. residual scale (sa_out = attn_out + input).  The Q/K/V projections + RoPE
NB. are batched across all B*c rows (weight-read amortized); scores/softmax/V
NB. run per-sequence (each chunk attends causally to its own window).
NB. x = hidden (B, c, emb); y = <block_data; pos; mi; layer>; optional 5th = lens.
qw2_attention_bp =: 4 : 0
  x   qwen_attention_bp ((y , <1) , <0)
)

NB. ---- Batched-prefill block forward (qwen2) ----
NB. x = hidden (B, c, emb); y = <block_data; pos; mi; layer>.  FFN matmuls batched
NB. across B*c rows.  Returns <(B, c, emb)>.  NO residual scale (qwen2).
qw2_block_forward_bp =: 4 : 0
  x   qwen_block_forward_bp ((y , <1) , <0)
)

NB. ---- Run all blocks for B sequences (one CHUNK each at pos[b]) ----
qw2_run_blocks_bp =: 4 : 0
  x   qwen_run_blocks_bp ((y , <1) , <0)
)

NB. ---- Load qwen2 GGUF into llm noun ----
qw2_load =: 3 : 0
  NB. y = <path; raw> — raw is the memory-mapped file (mapped by
  NB. load_gguf_to_llm; unmap'd after load — kvs_ctx is load-time only).
  data =. y
  path =. > 0 { data
  raw =. > 1 { data
  header =. parse_hdr_raw raw
  n_tensors =. > 2 { header
  kv_result =. parse_kv_pairs_raw raw
  kvs =. > 0 { kv_result
  kv_end =. > 3 { kv_result
  ti =. parse_tensor_infos (<raw) , (<kv_end) , (<n_tensors)
  ti_end_offset =. > ((n_tensors * 6) - 1) { ti
  tds =. 32 * <. (ti_end_offset + 31) % 32
  kvs_ctx =. (<kvs) , (<raw)
  kv_data =. build_kv_dict kvs_ctx
  mi =. qw2_extract_hparams kvs_ctx
  mi =. build_mi_dict mi
  rope_tables =. build_rope_tables ((< mi_context_len mi) , (< mi_head_dim mi) , (< mi_rope_freq mi))
  NB. ONE multi-put: the jdict single-put (1-key) path is broken in this J9.8
  NB. build (domain/length error); multi-put (2+ keys) and get/has work.
  ((> 0 { rope_tables) ; (> 1 { rope_tables)) put__mi 'cos_tab' ; 'sin_tab'
  NB. Real chat template from the GGUF ('' if absent → bespoke fallback).
  ct_tmpl =. 'tokenizer.chat_template' kv_string (0 1 { kv_result)
  tokenizer =. build_gpt2_tokenizer kv_result
  stop_tokens =. tokenizer_eos_g tokenizer
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
  n_heads_kv =. mi_n_heads_kv mi
  head_dim =. mi_head_dim mi
  ctx_len =. mi_context_len mi
  block_count =. mi_block_count mi
  p  =. <path
  t  =. <ti
  ze =. <0 0 0.95 0.0   NB. default_params (chat sampling defaults)
  tk =. <tokenizer
  mi_b =. <mi
  kc_b =. <(block_count ; ctx_len ; n_heads_kv ; head_dim)   NB. kv cache config (per-model, authoritative; mirror globals are the hot path)
  td =. <tds
  all_tensors =. emb_canonical all_tensors
  at =. <all_tensors
  llm =. p , t , ze , tk , mi_b , kc_b , td , at
  block_data =. qw2_pre_build_block_data llm
  llm =. llm , <block_data
  llm =. llm , <kv_data
  llm =. llm , <stop_tokens
  llm =. llm , <ct_tmpl
)



NB. ---- Chat-template support (Phase 1.1) ----
NB. y = messages: boxed list of message boxes; each = <role ; content>.
NB. Renders the REAL jinja chat_template pulled from the GGUF
NB. (tokenizer.chat_template) via the minja/chat_template port when present
NB. (chat_tmpl_render in chat.ijs); falls back to the bespoke qwen2 prompt
NB. when the GGUF has none. No BOS (llama.cpp qwen2 chat omits bos).

NB. Stop token: <|im_end|> (EOS). Computed once in qw2_load (dict member stop_tokens).
qw2_stop_tokens =: llm_stop_tokens
