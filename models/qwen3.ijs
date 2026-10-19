NB. ================================================================
NB. Qwen3-0.6B (qwen3 arch) — standard decoder transformer, GQA, SwiGLU
NB. Separate QKV/O weights, no fused. Q and K get a per-head RMSNorm
NB. (attn_q_norm/attn_k_norm, size = head_dim) BEFORE RoPE; NO Q/K/V biases.
NB. NEOX RoPE (pairs offset by head_dim/2), NOT interleaved.
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

NB. ---- Extract qwen3 model info from KV pairs ----
NB. mi = <block_count; context_len; emb_len; n_heads; n_heads_kv; head_dim; rope_freq; vocab_size; rms_eps; n_ff>
qw3_extract_hparams =: 3 : 0
  data =. y
  block_count =. 'qwen3.block_count' kv_uint data
  context_length =. 'qwen3.context_length' kv_uint data
  emb_len =. 'qwen3.embedding_length' kv_uint data
  n_heads =. 'qwen3.attention.head_count' kv_uint data
  n_heads_kv =. 'qwen3.attention.head_count_kv' kv_uint data
  rope_freq =. 'qwen3.rope.freq_base' kv_float data
  vocab_size =. 'qwen3.vocab_size' kv_uint data
  rms_eps =. 'qwen3.attention.layer_norm_rms_epsilon' kv_float data
  n_ff =. 'qwen3.feed_forward_length' kv_uint data
  key_len =. 'qwen3.attention.key_length' kv_uint data
  if. key_len <: 0 do. key_len =. emb_len % n_heads end.
  head_dim =. key_len
  (<"0) block_count , context_length , emb_len , n_heads , n_heads_kv , head_dim , rope_freq , vocab_size , rms_eps , n_ff
)

NB. ---- block_data accessors (shared bd_* in llm_core.ijs) ----
NB. ---- Build one block's data (rank-1; x=llm, y=block index) ----
qw3_build_block =: 4 : 0
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
  q_norm =. (p , 'attn_q_norm.weight') get_tensor_cached_d llm
  k_norm =. (p , 'attn_k_norm.weight') get_tensor_cached_d llm
  ffn_norm =. (p , 'ffn_norm.weight') get_tensor_cached_d llm
  ff_gate =. (p , 'ffn_gate.weight') get_tensor_cached_d llm
  ff_up =. (p , 'ffn_up.weight') get_tensor_cached_d llm
  ff_down =. (p , 'ffn_down.weight') get_tensor_cached_d llm
  head_dim =. (0 { $ attn_q) % n_heads
  (<attn_norm),(<attn_q),(<attn_k),(<attn_v),(<attn_o_w),(<0),(<0),(<0),(<ffn_norm),(<ff_gate),(<ff_up),(<ff_down),(<q_norm),(<k_norm),(<n_heads),(<head_dim),(<n_heads_kv)
)

NB. ---- Pre-build all block data for Qwen3 ----
qw3_pre_build_block_data =: 3 : 0
  llm =. y
  block_count =. mi_block_count (llm_mi llm)
  (<llm) qw3_build_block each i. block_count
)

NB. ---- Expand KV axis (n_kv) -> n_heads (each KV head repeated n_groups times) ----

NB. ---- Single-token attention (qwen3: GQA, NEOX RoPE, Q/K/V biases) ----
NB. x = hidden (emb,) — the layer input; y = <block_data; pos; mi; layer>
qw3_attention =: 4 : 0
  x   qwen_attention ((y , <0) , <1)
)

NB. ---- Batched attention (prompt prefill, GQA, causal) ----
NB. x = hidden (L, emb); y = <block_data; mi; layer>
qw3_attention_b =: 4 : 0
  x   qwen_attention_b ((y , <0) , <1)
)

NB. ---- Single block forward (qwen3) ----
NB. x = hidden (emb,) — the layer input; y = <block_data; pos; mi; layer>
qw3_block_forward =: 4 : 0
  x   qwen_block_forward ((y , <0) , <1)
)

NB. ---- Batched block forward ----
NB. x = hidden (L, emb); y = <block_data; mi; layer; start_pos; rope>
qw3_block_forward_b =: 4 : 0
  x   qwen_block_forward_b ((y , <0) , <1)
)

NB. ---- Run all blocks (single token; cache lives in kv_cache_g global) ----
NB. x = hidden (emb,); y = <llm; pos>
qw3_run_blocks =: 4 : 0
  x   qwen_run_blocks ((y , <0) , <1)
)

NB. ---- Batched run all blocks (prompt prefill) ----
NB. x = hidden (L, emb); y = <llm; start_pos>  (positions start_pos..start_pos+L-1)
qw3_run_blocks_b =: 4 : 0
  x   qwen_run_blocks_b ((y , <0) , <1)
)

NB. ---- Batched-DECODE attention (B sequences, ONE token each at pos[b]) ----
NB. Mirrors qw2_attention_bd; qwen3 adds per-head Q/K RMSNorm before RoPE and
NB. has NO Q/K/V biases. x = hidden (B, emb); y = <block_data; pos; mi; layer>
qw3_attention_bd =: 4 : 0
  x   qwen_attention_bd ((y , <0) , <1)
)

NB. ---- Batched-decode block forward ----
qw3_block_forward_bd =: 4 : 0
  x   qwen_block_forward_bd ((y , <0) , <1)
)

NB. ---- Run all blocks for B sequences (one token each at pos[b]) ----
qw3_run_blocks_bd =: 4 : 0
  x   qwen_run_blocks_bd ((y , <0) , <1)
)

NB. ---- Batched-PREFILL attention (qwen3): B chunks, one per seq at pos[b] ----
NB. Mirrors qw2_attention_bp but with qwen3 specifics: NO Q/K/V biases, per-head
NB. Q/K RMSNorm BEFORE RoPE (shared weight, size=head_dim), NEOX RoPE, Q scaled
NB. by 1/sqrt(head_dim), NO residual scale.  Q/K/V projections + RoPE batched
NB. across B*c rows; scores/softmax/V per-sequence.  Optional 5th arg = lens.
qw3_attention_bp =: 4 : 0
  x   qwen_attention_bp ((y , <0) , <1)
)

NB. ---- Batched-prefill block forward (qwen3) ----
qw3_block_forward_bp =: 4 : 0
  x   qwen_block_forward_bp ((y , <0) , <1)
)

NB. ---- Run all blocks for B sequences (one CHUNK each at pos[b]) ----
qw3_run_blocks_bp =: 4 : 0
  x   qwen_run_blocks_bp ((y , <0) , <1)
)

NB. ---- Load qwen3 GGUF into llm noun ----
qw3_load =: 3 : 0
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
  mi =. qw3_extract_hparams kvs_ctx
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
  block_data =. qw3_pre_build_block_data llm
  llm =. llm , <block_data
  llm =. llm , <kv_data
  llm =. llm , <stop_tokens
  llm =. llm , <ct_tmpl
)



NB. ---- Chat-template support (Phase 1.1) ----
NB. y = messages: boxed list of message boxes; each = <role ; content>.
NB. Renders the qwen3 chat template for the no-tools case: each message as
NB. '<|im_start|>role\ncontent<|im_end|>\n', then the generation prompt
NB. '<|im_start|>assistant\n'. The qwen3 template adds NO default system
NB. message (unlike qwen2). No BOS (gpt2 tokenize adds none).

NB. Stop token: <|im_end|> (EOS). Computed once in qw3_load (dict member stop_tokens).
qw3_stop_tokens =: llm_stop_tokens
