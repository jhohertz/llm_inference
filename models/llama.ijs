NB. ================================================================
NB. Llama arch — generic standard-decoder transformer (GQA, SwiGLU,
NB. interleaved RoPE, separate QKV/O weights, no fused, no q/k norm,
NB. no post-attention/ffn norms). Covers SmolLM2 AND Llama-3.2 (and any
NB. arch 'llama' model): all dims are read from the GGUF; the tokenizer
NB. dispatches on tokenizer.ggml.pre (smollm -> gpt2 byte-level BPE,
NB. llama-bpe -> llama3 regex pre + gpt2 BPE merges). Chat template
NB. dispatches the same way (SmolLM2 <|im_start|> vs Llama-3.2 llama3).
NB. Depends on: llm_core.ijs, kernels.ijs, gguf.ijs, kv_cache.ijs, sampler.ijs,
NB.           tokenizer_gpt2.ijs (which pulls in tokenizer_llama3.ijs)
NB. ================================================================
coclass 'inference'
require 'llm/inference/util/llm_core'
require 'llm/inference/tokenizers/tokenizer_gpt2'

NB. Chat-template dispatch marker. '' = SmolLM2 template; llama_load sets it
NB. to the tokenizer's pre ('llama-bpe' -> Llama-3.2 template). The default
NB. keeps llama_chat_prompt usable when a tokenizer is built standalone
NB. (test_chat builds minimal llms without llama_load).
llama_tokenizer_pre_g =: ''

NB. ---- Helper: move axes (dyadic |:) using variable axis list ----

NB. ---- KV helpers ----
llama_kv_uint =: 4 : 0
  key =. x
  data =. y
  key kv_uint data
)

llama_kv_float =: 4 : 0
  key =. x
  data =. y
  key kv_float data
)

NB. ---- Extract llama model info from KV pairs ----
NB. mi = <block_count; context_len; emb_len; n_heads; n_heads_kv; head_dim; rope_freq; vocab_size; rms_eps; n_ff>
llama_extract_hparams =: 3 : 0
  data =. y
  block_count =. 'llama.block_count' llama_kv_uint data
  context_length =. 'llama.context_length' llama_kv_uint data
  emb_len =. 'llama.embedding_length' llama_kv_uint data
  n_heads =. 'llama.attention.head_count' llama_kv_uint data
  n_heads_kv =. 'llama.attention.head_count_kv' llama_kv_uint data
  rope_freq =. 'llama.rope.freq_base' llama_kv_float data
  vocab_size =. 'llama.vocab_size' llama_kv_uint data
  rms_eps =. 'llama.attention.layer_norm_rms_epsilon' llama_kv_float data
  n_ff =. 'llama.feed_forward_length' llama_kv_uint data
  key_len =. 'llama.attention.key_length' llama_kv_uint data
  if. key_len <: 0 do. key_len =. emb_len % n_heads end.
  head_dim =. key_len
  (<"0) block_count , context_length , emb_len , n_heads , n_heads_kv , head_dim , rope_freq , vocab_size , rms_eps , n_ff
)

NB. ---- block_data accessors (llama: separate weights) ----
NB. block_data = <attn_norm; attn_q; attn_k; attn_v; attn_o; ffn_norm; ffn_gate; ffn_up; ffn_down; n_heads; head_dim; n_heads_kv>
llama_bd_attn_norm =: >@(0&{)
llama_bd_attn_q    =: >@(1&{)
llama_bd_attn_k    =: >@(2&{)
llama_bd_attn_v    =: >@(3&{)
llama_bd_attn_o    =: >@(4&{)
llama_bd_ff_norm   =: >@(5&{)
llama_bd_ff_gate   =: >@(6&{)
llama_bd_ff_up     =: >@(7&{)
llama_bd_ff_down   =: >@(8&{)
llama_bd_n_heads   =: >@(9&{)
llama_bd_head_dim  =: >@(10&{)
llama_bd_n_heads_kv=: >@(11&{)

NB. ---- Pre-build block data for llama arch ----
NB. ---- Build one block's data (rank-1; x=llm, y=block index) ----
llama_build_block =: 4 : 0
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
  ffn_norm =. (p , 'ffn_norm.weight') get_tensor_cached_d llm
  ff_gate =. (p , 'ffn_gate.weight') get_tensor_cached_d llm
  ff_up =. (p , 'ffn_up.weight') get_tensor_cached_d llm
  ff_down =. (p , 'ffn_down.weight') get_tensor_cached_d llm
  head_dim =. (0 { $ attn_q) % n_heads
  (<attn_norm),(<attn_q),(<attn_k),(<attn_v),(<attn_o_w),(<ffn_norm),(<ff_gate),(<ff_up),(<ff_down),(<n_heads),(<head_dim),(<n_heads_kv)
)

NB. ---- Pre-build all block data for SmolLM2 ----
llama_pre_build_block_data =: 3 : 0
  llm =. y
  block_count =. mi_block_count (llm_mi llm)
  (<llm) llama_build_block each i. block_count
)

NB. ---- Expand KV axis (n_kv) -> n_heads (each KV head repeated n_groups times) ----

NB. ---- Single-token attention (llama: GQA, interleaved RoPE, no SWA) ----
NB. x = hidden; y = <block_data; pos; mi; layer>
llama_attention =: 4 : 0
  hidden =. x
  'block_data pos mi layer' =. y
  n_heads =. llama_bd_n_heads block_data
  head_dim =. llama_bd_head_dim block_data
  n_heads_kv =. llama_bd_n_heads_kv block_data
  n_groups =. n_heads % n_heads_kv

  NB. Attention norm
  attn_norm_w =. llama_bd_attn_norm block_data
  hidden =. rms_norm ((< mi_rms_eps mi) , (< attn_norm_w) , <hidden)

  NB. Separate Q, K, V projections
  qv =. (llama_bd_attn_q block_data) linear_r hidden
  kv =. (llama_bd_attn_k block_data) linear_r hidden
  vv =. (llama_bd_attn_v block_data) linear_r hidden
  Q =. (n_heads, head_dim) $ qv
  K =. (n_heads_kv, head_dim) $ kv
  V =. (n_heads_kv, head_dim) $ vv

  NB. Interleaved RoPE (table-based)
  cos_t =. pos { mi_cos_tab mi
  sin_t =. pos { mi_sin_tab mi
  rope_t =. (<cos_t) , (<sin_t)
  Q =. Q rope_apply2_t rope_t
  K =. K rope_apply2_t rope_t

  NB. Llama scales Q by 1/sqrt(head_dim) before attention
  Q =. Q % head_dim ^ 0.5

  NB. Write K,V (n_kv, hd) to cache, read all up to pos
  kv_write ((<layer) , (<pos) , (<K) , (<V))
  kv_result =. kv_read ((<layer) , <pos)
  k_all =. > 0 { kv_result   NB. (win, n_kv, hd)
  v_all =. > 1 { kv_result
  win =. pos + 1

  NB. GQA without expanding KV: group the query heads (n_heads_kv groups of
  NB. n_groups) and matmul each group's Q against its shared K/V — k_all/v_all
  NB. stay (win, n_heads_kv, hd), never expanded to n_heads (4x llama).
  Q_g2 =. (n_heads_kv , n_groups , head_dim) $ , Q   NB. Q (n_heads, hd) -> (n_kv, n_g, hd)
  Kp2 =. 1 2 0 |: k_all   NB. (n_kv, hd, win) — one transpose (the 1 0 2 |: + |:"2 two-pass was ~4x slower)
  scores2 =. Q_g2 (+/ .* "2) Kp2   NB. (n_kv, n_groups, win): Q[g,r,d] vs K[j,g,d]
  scores =. (n_heads, win) $ , scores2   NB. [h,j], h = g*n_groups + r

  NB. Single-token decode: all cached j <= pos valid (causal), no mask.

  NB. Softmax over win per head (n_heads,) broadcasts as prefix of (n_heads, win)
  max_sf =. >./"1 scores
  exp_sf =. ^ (scores - max_sf)
  softmax =. exp_sf % +/"1 exp_sf   NB. (n_heads, win)

  NB. Output: attn[h] = sum_j softmax[h,j] * V[j,g(h)]
  softmax_g2 =. (n_heads_kv , n_groups , win) $ , softmax
  Vp =. 1 0 2 |: v_all   NB. (n_kv, win, hd)
  attn2 =. softmax_g2 (+/ .* "2) Vp   NB. (n_kv, n_groups, hd)
  attn_raw =. (n_heads, head_dim) $ , attn2   NB. [h,d]

  NB. Output projection
  attn_raw_flat =. (n_heads * head_dim) $ , attn_raw
  attn_out =. (llama_bd_attn_o block_data) linear_r attn_raw_flat
  (<attn_out)
)

NB. ---- Batched attention (prompt prefill, GQA, causal) ----
NB. x = hidden (L, emb); y = <block_data; mi; layer>
llama_attention_b =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  mi =. > 1 { y
  layer =. > 2 { y
  start_pos =. > 3 { y
  rope =. > 4 { y
  L =. {. $ hidden
  n_embd =. {: $ hidden
  n_heads =. llama_bd_n_heads block_data
  head_dim =. llama_bd_head_dim block_data
  n_heads_kv =. llama_bd_n_heads_kv block_data
  n_groups =. n_heads % n_heads_kv
  NB. rope = <cos_all; sin_all; idx; cos_expq; sin_expq; cos_expk; sin_expk; mask_g2>
  cos_all =. > 0 { rope
  sin_all =. > 1 { rope
  idx =. > 2 { rope
  cos_expq =. > 3 { rope
  sin_expq =. > 4 { rope
  cos_expk =. > 5 { rope
  sin_expk =. > 6 { rope
  mask_g2 =. > 7 { rope

  NB. Attention norm per row
  attn_norm_w =. llama_bd_attn_norm block_data
  hidden =. rms_norm_rows ((< mi_rms_eps mi) , (< attn_norm_w) , <hidden)

  NB. Separate Q,K,V batched projections
  qv =. |: ((llama_bd_attn_q block_data) (+/ .* ) |: hidden)   NB. (L, n_heads*hd)
  kv =. |: ((llama_bd_attn_k block_data) (+/ .* ) |: hidden)   NB. (L, n_kv*hd)
  vv =. |: ((llama_bd_attn_v block_data) (+/ .* ) |: hidden)   NB. (L, n_kv*hd)
  Q =. (L, n_heads, head_dim) $ , qv
  K =. (L, n_heads_kv, head_dim) $ , kv
  V =. (L, n_heads_kv, head_dim) $ , vv

  NB. Interleaved RoPE batched (table-based): pairs (i,i+1) per row, NORM style
  Qa =. idx {"1 Q          NB. (L, n_heads, half) even cols
  Qb =. (1 + idx) {"1 Q    NB. odd cols
  Qa_out =. (Qa * cos_expq) - (Qb * sin_expq)
  Qb_out =. (Qa * sin_expq) + (Qb * cos_expq)
  Q =. (L, n_heads, head_dim) $ , (1 2 3 0 |: (Qa_out ,: Qb_out))
  Ka =. idx {"1 K          NB. (L, n_heads_kv, half)
  Kb =. (1 + idx) {"1 K
  Ka_out =. (Ka * cos_expk) - (Kb * sin_expk)
  Kb_out =. (Ka * sin_expk) + (Kb * cos_expk)
  K =. (L, n_heads_kv, head_dim) $ , (1 2 3 0 |: (Ka_out ,: Kb_out))

  NB. Llama scales Q by 1/sqrt(head_dim)
  Q =. Q % head_dim ^ 0.5

  NB. RESUME: prepend the cache prefix (positions 0..start_pos-1, already
  NB. norm'd + RoPE'd) so this batch attends to the full history.
  K_batch =. K
  V_batch =. V
  if. start_pos > 0 do.
    k_pre =. > 0 { (kv_read ((<layer) , <(start_pos - 1)))
    v_pre =. > 1 { (kv_read ((<layer) , <(start_pos - 1)))
    K =. k_pre , K
    V =. v_pre , V
  end.

  NB. Bulk write the new batch's L K/V into cache at layer, starting at start_pos
  kv_write_rows ((<0) , (<layer) , (<start_pos) , <K_batch)
  kv_write_rows ((<1) , (<layer) , (<start_pos) , <V_batch)

  NB. GQA without expanding KV: group the query heads (n_heads_kv groups of
  NB. n_groups) and batched-matmul each group's Q against its shared K/V row —
  NB. K/V stay (n_heads_kv, ctx, hd), never expanded to n_heads (7x KV for
  NB. qwen2.5, 4x llama/granite, 2x qwen3). Q reshaped group-major so the
  NB. frames (n_heads_kv) align for +/ .*"2.
  Qp =. 1 0 2 |: Q        NB. (n_heads, L, hd)
  Q_g2 =. (n_heads_kv , (n_groups * L) , head_dim) $ , ((n_heads_kv , n_groups , L , head_dim) $ , Qp)
  Kp2 =. 1 2 0 |: K        NB. (n_heads_kv, hd, start_pos+L) — one transpose
  scores2 =. Q_g2 (+/ .* "2) Kp2   NB. (n_kv, n_groups*L, ctx): Q[t,h] vs K[j,g(h)]
  NB. causal mask: mask[h,t,j]=1 if j>t; query t at position start_pos+t, keys
  NB. 0..start_pos+L-1. Keep scores group-major: the tiled mask (row r*L+t
  NB. needs mask row t) is hoisted (item 7 of rope, built once per chunk) —
  NB. subtract with rank over the kv-head frame; no (n_heads, L, tot) 3D mask,
  NB. no per-layer mask build, no scores re-shape copy.
  scores2 =. scores2 -"2 (mask_g2 * 1e9)

  NB. Softmax over j per (g,r,t) row (order-independent)
  scores_f =. ((n_heads * L) , start_pos + L) $ , scores2
  max_sf =. >./"1 scores_f
  exp_sf =. ^ (scores_f - max_sf)
  softmax_f =. exp_sf % +/"1 exp_sf
  softmax_g2 =. (n_heads_kv , (n_groups * L) , start_pos + L) $ , softmax_f

  NB. Output: attn[h,t] = sum_j softmax[g(h),t,j] * V[g(h),j]
  Vp =. 1 0 2 |: V        NB. (n_heads_kv, start_pos+L, hd)
  attn2 =. softmax_g2 (+/ .* "2) Vp   NB. (n_kv, n_groups*L, hd)
  attn_raw =. (n_heads, L, head_dim) $ , attn2   NB. [h,t,d]

  NB. Output projection (batched)
  attn_o_w =. llama_bd_attn_o block_data
  attn_out =. |: (attn_o_w (+/ .* ) |: ((L, n_heads * head_dim) $ , (1 0 2 |: attn_raw)))   NB. (L, emb)

  (<attn_out)
)

NB. ---- Single block forward (llama) ----
NB. x = hidden; y = <block_data; pos; mi; layer>
llama_block_forward =: 4 : 0
  hidden =. x
  'block_data pos mi layer' =. y
  input =. hidden
  attn_result =. hidden llama_attention ((<block_data) , (<pos) , (<mi) , (<layer))
  attn_out =. > 0 { attn_result
  sa_out =. attn_out + input
  ffn_norm_w =. llama_bd_ff_norm block_data
  ffn_in =. rms_norm ((< mi_rms_eps mi) , (< ffn_norm_w) , <sa_out)
  gate =. (llama_bd_ff_gate block_data) linear_r ffn_in
  up =. (llama_bd_ff_up block_data) linear_r ffn_in
  ffn_raw =. (llama_bd_ff_down block_data) linear_r (gate swiglu up)
  output =. ffn_raw + sa_out
  (<output)
)

NB. ---- Batched block forward ----
NB. x = hidden (L, emb); y = <block_data; mi; layer; start_pos; rope>
llama_block_forward_b =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  mi =. > 1 { y
  layer =. > 2 { y
  start_pos =. > 3 { y
  rope =. > 4 { y
  input =. hidden
  attn_result =. hidden llama_attention_b ((<block_data) , (<mi) , (<layer) , (<start_pos) , <rope)
  attn_out =. > 0 { attn_result
  sa_out =. attn_out + input
  ffn_norm_w =. llama_bd_ff_norm block_data
  ffn_in =. rms_norm_rows ((< mi_rms_eps mi) , (< ffn_norm_w) , <sa_out)
  gate =. |: ((llama_bd_ff_gate block_data) (+/ .* ) |: ffn_in)   NB. (L, n_ff)
  up =. |: ((llama_bd_ff_up block_data) (+/ .* ) |: ffn_in)
  ffn_raw =. |: ((llama_bd_ff_down block_data) (+/ .* ) |: (gate swiglu up))
  output =. ffn_raw + sa_out
  (<output)
)

NB. ---- Run all blocks (single token; cache lives in kv_cache_g global) ----
NB. x = hidden; y = <llm; pos>
llama_run_blocks =: 4 : 0
  input =. x
  args =. y
  'llm pos' =. args
  mi =. llm_mi llm
  head_dim =. mi_head_dim mi
  n_heads_kv =. mi_n_heads_kv mi
  block_count =. mi_block_count mi
  ctx_len =. mi_context_len mi
  state =. input
  if. 0 = # kv_meta do.
    kv_create ((<block_count) , (<ctx_len) , (<n_heads_kv) , (<head_dim))
  end.
  b =. 0
  block_data_list =. llm_block_data llm
  while. b < block_count do.
    block_data =. > b { block_data_list
    result =. state llama_block_forward ((<block_data) , (<pos) , (<mi) , (<b))
    state =. > 0 { result
    b =. b + 1
  end.
  <state
)

NB. ---- Batched run all blocks (prompt prefill) ----
NB. x = hidden (L, emb); y = <llm; start_pos>  (positions start_pos..start_pos+L-1)
llama_run_blocks_b =: 4 : 0
  input =. x
  args =. y
  llm =. > 0 { args
  start_pos =. > 1 { args
  mi =. llm_mi llm
  head_dim =. mi_head_dim mi
  n_heads_kv =. mi_n_heads_kv mi
  block_count =. mi_block_count mi
  ctx_len =. mi_context_len mi
  state =. input
  if. 0 = # kv_meta do.
    kv_create ((<block_count) , (<ctx_len) , (<n_heads_kv) , (<head_dim))
  end.
  NB. RoPE tables are per-model and identical across layers (freq is model
  NB. level): compute the cos/sin tables + expansions ONCE per chunk and thread
  NB. through the layer loop, instead of recomputing them in every layer.
  L =. {. $ input
  half =. <. head_dim % 2
  idx =. 2 * i. half
  n_heads =. mi_n_heads mi
  cos_all =. (start_pos + i. L) { mi_cos_tab mi
  sin_all =. (start_pos + i. L) { mi_sin_tab mi
  cos_expq =. (0 2 1) |: ((L , half , n_heads) $ , (cos_all (*/) (n_heads $ 1)))
  sin_expq =. (0 2 1) |: ((L , half , n_heads) $ , (sin_all (*/) (n_heads $ 1)))
  cos_expk =. (0 2 1) |: ((L , half , n_heads_kv) $ , (cos_all (*/) (n_heads_kv $ 1)))
  sin_expk =. (0 2 1) |: ((L , half , n_heads_kv) $ , (sin_all (*/) (n_heads_kv $ 1)))
  NB. Causal mask is identical across layers in a chunk: build the group-major
  NB. tiled mask (n_groups*L, ctx) ONCE per chunk and thread it through the
  NB. layer loop (item 7 of the rope box) instead of per-layer.
  n_groups =. n_heads % n_heads_kv
  key_pos =. i. (start_pos + L)
  q_pos =. start_pos + i. L
  mask_2d =. q_pos </ key_pos
  NB. Fast r-major boolean tile via the (*/) broadcast (the cyclic boolean
  NB. reshape (n_groups,L,ctx)$mask_2d is ~100x slower); scaled at subtract.
  mask_g2 =. ((n_groups * L) , start_pos + L) $ , (2 0 1 |: (mask_2d (*/) (n_groups $ 1)))
  rope =. (<cos_all) , (<sin_all) , (<idx) , (<cos_expq) , (<sin_expq) , (<cos_expk) , (<sin_expk) , (<mask_g2)
  b =. 0
  block_data_list =. llm_block_data llm
  while. b < block_count do.
    block_data =. > b { block_data_list
    result =. state llama_block_forward_b ((<block_data) , (<mi) , (<b) , (<start_pos) , <rope)
    state =. > 0 { result
    b =. b + 1
  end.
  <state
)

NB. ---- Batched-DECODE attention (B sequences, ONE token each at pos[b]) ----
NB. Mirrors llama_attention_bd; llama = qwen2 minus QKV biases, INTERLEAVED RoPE.
NB. x = hidden (B, emb); y = <block_data; pos; mi; layer>
llama_attention_bd =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  pos =. > 1 { y
  mi =. > 2 { y
  layer =. > 3 { y
  B =. {. $ hidden
  n_heads =. llama_bd_n_heads block_data
  head_dim =. llama_bd_head_dim block_data
  n_heads_kv =. llama_bd_n_heads_kv block_data
  n_groups =. n_heads % n_heads_kv
  half =. <. head_dim % 2

  NB. Attention norm per row
  attn_norm_w =. llama_bd_attn_norm block_data
  hidden =. rms_norm_rows ((< mi_rms_eps mi) , (< attn_norm_w) , <hidden)

  NB. Batched Q,K,V projections (weight-read amortized across B)
  qv =. |: ((llama_bd_attn_q block_data) (+/ .* ) |: hidden)   NB. (B, n_heads*hd)
  kv =. |: ((llama_bd_attn_k block_data) (+/ .* ) |: hidden)   NB. (B, n_kv*hd)
  vv =. |: ((llama_bd_attn_v block_data) (+/ .* ) |: hidden)   NB. (B, n_kv*hd)

  Q =. (B, n_heads, head_dim) $ , qv
  K =. (B, n_heads_kv, head_dim) $ , kv
  V =. (B, n_heads_kv, head_dim) $ , vv

  NB. Interleaved RoPE batched at the B positions (NORM style, pairs (i,i+1))
  cos_all =. pos { mi_cos_tab mi
  sin_all =. pos { mi_sin_tab mi
  idx =. 2 * i. half
  cos_expq =. (0 2 1) |: ((B , half , n_heads) $ , (cos_all (*/) (n_heads $ 1)))
  sin_expq =. (0 2 1) |: ((B , half , n_heads) $ , (sin_all (*/) (n_heads $ 1)))
  cos_expk =. (0 2 1) |: ((B , half , n_heads_kv) $ , (cos_all (*/) (n_heads_kv $ 1)))
  sin_expk =. (0 2 1) |: ((B , half , n_heads_kv) $ , (sin_all (*/) (n_heads_kv $ 1)))
  Qa =. idx {"1 Q
  Qb =. (1 + idx) {"1 Q
  Qa_out =. (Qa * cos_expq) - (Qb * sin_expq)
  Qb_out =. (Qa * sin_expq) + (Qb * cos_expq)
  Q =. (B, n_heads, head_dim) $ , (1 2 3 0 |: (Qa_out ,: Qb_out))
  Ka =. idx {"1 K
  Kb =. (1 + idx) {"1 K
  Ka_out =. (Ka * cos_expk) - (Kb * sin_expk)
  Kb_out =. (Ka * sin_expk) + (Kb * cos_expk)
  K =. (B, n_heads_kv, head_dim) $ , (1 2 3 0 |: (Ka_out ,: Kb_out))

  NB. Llama scales Q by 1/sqrt(head_dim)
  Q =. Q % head_dim ^ 0.5

  NB. Per-sequence: write K/V at pos[b], read the window, scores/softmax/output
  NB. Vectorized path fires when all B sequences share one position (common
  NB. lockstep decode with equal-length prompts): ONE list-selector cache
  NB. write, ONE indexed gather of the B windows, then threaded batched
  NB. scores/softmax/V (J parallelizes the rank-4 matmuls over B). Falls back
  NB. to the per-seq loop when positions differ (variable window lengths).
  if. (B > 1) *. (pos -: (B $ 0 { pos)) do.
    win =. (0 { pos) + 1
    eff_seq =. > 1 { kv_meta
    base_b =. ((layer * kv_batch_g) + i. B) * eff_seq
    NB. Batch cache write: one list-selector amend for all B at pos[b]
    idxw =. base_b + pos
    k_cache_g =: ((B , n_heads_kv * head_dim) $ , K) idxw} k_cache_g
    v_cache_g =: ((B , n_heads_kv * head_dim) $ , V) idxw} v_cache_g
    kv_pos_g =: kv_pos_g >. (0 { pos) + 1
    NB. Gather all B windows in one indexed fetch (rows (base_b[b]+i.win))
    idxr =. base_b +/ i. win
    k_rows_b =. (B , win , n_heads_kv , head_dim) $ , (idxr { k_cache_g)
    v_rows_b =. (B , win , n_heads_kv , head_dim) $ , (idxr { v_cache_g)
    NB. Batched GQA scores, softmax, V (threaded over B; no causal mask —
    NB. all cached j <= pos valid in single-token decode)
    Kp_b =. (0 2 3 1) |: k_rows_b
    Q_g2_b =. (B , n_heads_kv , n_groups , head_dim) $ , Q
    scores_b =. Q_g2_b (+/ .* "2) Kp_b   NB. (B, n_kv, groups, win)
    scores_b2 =. (B , n_heads , win) $ , scores_b   NB. (B, n_heads, win)
    max_sf_b =. >./"1 scores_b2
    exp_sf_b =. ^ (scores_b2 - max_sf_b)
    softmax_b =. exp_sf_b % +/"1 exp_sf_b
    softmax_g2_b =. (B , n_heads_kv , n_groups , win) $ , softmax_b
    Vp_b =. (0 2 1 3) |: v_rows_b   NB. (B, n_kv, win, hd)
    attn2_b =. softmax_g2_b (+/ .* "2) Vp_b   NB. (B, n_kv, groups, hd)
    attn_all =. (B , n_heads * head_dim) $ , attn2_b   NB. (B, n_heads*hd) flat
  else.
    attn_out =. ''
    b =. 0
    while. b < B do.
      q_b =. (n_heads, head_dim) $ , (b { Q)
      k_b =. (n_heads_kv, head_dim) $ , (b { K)
      v_b =. (n_heads_kv, head_dim) $ , (b { V)
      pos_b =. b { pos
      kv_write ((<layer) , (<pos_b) , (<k_b) , (<v_b) , (<b))
      kv_result =. kv_read ((<layer) , (<pos_b) , (<b))
      k_all =. > 0 { kv_result
      v_all =. > 1 { kv_result
      win =. pos_b + 1
      Q_g2 =. (n_heads_kv , n_groups , head_dim) $ , q_b
      Kp2 =. 1 2 0 |: k_all
      scores2 =. Q_g2 (+/ .* "2) Kp2
      scores =. (n_heads, win) $ , scores2
      max_sf =. >./"1 scores
      exp_sf =. ^ (scores - max_sf)
      softmax =. exp_sf % +/"1 exp_sf
      softmax_g2 =. (n_heads_kv , n_groups , win) $ , softmax
      Vp =. 1 0 2 |: v_all
      attn2 =. softmax_g2 (+/ .* "2) Vp
      attn_raw =. (n_heads, head_dim) $ , attn2
      attn_raw_flat =. (n_heads * head_dim) $ , attn_raw
      attn_out =. attn_out , <attn_raw_flat
      b =. b + 1
    end.
    attn_all =. (B , n_heads * head_dim) $ , > attn_out
  end.
  attn_result =. |: ((llama_bd_attn_o block_data) (+/ .* ) |: attn_all)   NB. (B, emb)
  (<attn_result)
)

NB. ---- Batched-decode block forward (llama) ----
llama_block_forward_bd =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  pos =. > 1 { y
  mi =. > 2 { y
  layer =. > 3 { y
  input =. hidden
  attn_result =. hidden llama_attention_bd ((<block_data) , (<pos) , (<mi) , (<layer))
  attn_out =. > 0 { attn_result
  sa_out =. attn_out + input
  ffn_norm_w =. llama_bd_ff_norm block_data
  ffn_in =. rms_norm_rows ((< mi_rms_eps mi) , (< ffn_norm_w) , <sa_out)
  gate =. |: ((llama_bd_ff_gate block_data) (+/ .* ) |: ffn_in)   NB. (B, n_ff)
  up =. |: ((llama_bd_ff_up block_data) (+/ .* ) |: ffn_in)
  ffn_raw =. |: ((llama_bd_ff_down block_data) (+/ .* ) |: (gate swiglu up))
  output =. ffn_raw + sa_out
  (<output)
)

NB. ---- Run all blocks for B sequences (one token each at pos[b]) ----
llama_run_blocks_bd =: 4 : 0
  input =. x
  args =. y
  llm =. > 0 { args
  pos =. > 1 { args
  mi =. llm_mi llm
  head_dim =. mi_head_dim mi
  n_heads_kv =. mi_n_heads_kv mi
  block_count =. mi_block_count mi
  ctx_len =. mi_context_len mi
  state =. input
  if. 0 = # kv_meta do.
    kv_create ((<block_count) , (<ctx_len) , (<n_heads_kv) , (<head_dim))
  end.
  b =. 0
  block_data_list =. llm_block_data llm
  while. b < block_count do.
    block_data =. > b { block_data_list
    result =. state llama_block_forward_bd ((<block_data) , (<pos) , (<mi) , (<b))
    state =. > 0 { result
    b =. b + 1
  end.
  <state
)

llama_load =: 3 : 0
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
  mi =. llama_extract_hparams kvs_ctx
  rope_tables =. build_rope_tables ((< mi_context_len mi) , (< mi_head_dim mi) , (< mi_rope_freq mi))
  mi =. mi , rope_tables
  tokenizer =. build_gpt2_tokenizer kv_result
  NB. Chat-template dispatch marker: llama_chat_prompt takes messages only
  NB. (chat.ijs contract), so the llama module remembers which tokenizer
  NB. pre the current model uses ('llama-bpe' -> llama3 template, else SmolLM2).
  llama_tokenizer_pre_g =: tokenizer_pre_g tokenizer
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
  ze =. <$0
  tk =. <tokenizer
  mi_b =. <mi
  kc_b =. <''   NB. kv cache is the kv_cache_g global, not stored in the llm
  td =. <tds
  all_tensors =. emb_canonical all_tensors
  at =. <all_tensors
  llm =. p , t , ze , tk , mi_b , kc_b , td , at
  block_data =. llama_pre_build_block_data llm
  llm =. llm , <block_data
)

NB. ---- Generic tokenize/detokenize (llama arch) ----
NB. SmolLM2 (pre 'smollm'): gpt2 byte-level BPE, no bos. Llama-3.2
NB. (pre 'llama-bpe'): llama3 regex pre + gpt2 BPE merges, and llama.cpp
NB. adds BOS (add_bos=true), so the bos token is prepended here. The chat
NB. template renders no <|begin_of_text|> marker — this supplies it, so the
NB. token stream matches llama.cpp's _input_ids exactly (bos once).
llama_tokenize =: 3 : 0
  llm_data =. input_llm y
  text =. input_text y
  tokenizer =. llm_tokenizer llm_data
  tokens =. gpt2_tokenize (<llm_data) , <text
  if. 'llama-bpe' -: tokenizer_pre_g tokenizer do.
    bos =. tokenizer_bos_g tokenizer
    if. bos > 0 do. tokens =. (<bos) , tokens end.
  end.
  tokens
)
llama_detokenize =: 3 : 0
  gpt2_detokenize y
)

NB. ---- Single-token inference ----
NB. Usage: llm llama_infer (text ; <temp;k;p;min_p>)
NB.        Simple (default params): llm llama_infer_simple text
llama_infer =: 4 : 0
  llm =. x
  args =. infer_args y
  text =. > 0 { args
  temp =. > 1 { args
  k =. > 2 { args
  p =. > 3 { args
  min_p =. > 4 { args

  tokens =. llama_tokenize (<llm) , <text
  mi =. llm_mi llm
  emb_len =. mi_emb_len mi
  scale =. 1
  block_count =. mi_block_count mi
  ctx_len =. mi_context_len mi
  n_heads_kv =. mi_n_heads_kv mi
  head_dim =. mi_head_dim mi
  emb_w =. 'token_embd.weight' get_tensor_cached_d llm
  output_norm_w =. 'output_norm.weight' get_tensor_cached_d llm
  tok_list =. , > tokens
  n_tokens =. # tok_list
  kv_create ((<block_count) , (<ctx_len) , (<n_heads_kv) , (<head_dim))
  if. 1 = n_tokens do.
    tok =. 0 { tok_list
    hidden =. scale * |: (tok {"1 emb_w)
    pre_s =. 6!:2 'result =. hidden llama_run_blocks (<llm) , <0'
    hidden =. > 0 { result
  else.
    emb_all =. scale * |: (tok_list {"1 emb_w)
    pre_s =. 6!:2 'result_b =. emb_all llama_run_blocks_b ((<llm) , <0)'
    h_b =. > 0 { result_b
    hidden =. > (n_tokens - 1) { h_b
  end.
  logits =. output_head ((< mi_rms_eps mi) , (<output_norm_w) , (<emb_w) , <hidden)
  report_prefill (pre_s , n_tokens)
  pred_tok =. sample_from ((<temp) , (<k) , (<p) , (<min_p) , <logits)
  decoded =. llama_detokenize (<llm) , <pred_tok
  tokens ; pred_tok ; decoded ; logits
)

NB. ---- Multi-token generation ----
NB. Usage: llm llama_generate (text ; max_steps ; <temp;k;p;min_p>)
NB.        Simple (default params): llm llama_generate_simple (text ; max_steps)
NB. Generation is the UNIFIED gen_loop_core (llm_core.ijs) — per-arch
NB. differences (embedding scale, run_blocks/run_blocks_b) are dispatched by
NB. llm_arch. Fresh mode: kv_create + batched prefill; resume mode (chat
NB. sessions): incremental prefill of the new segment. Stop token not appended.
llama_generate =: 4 : 0
  llm =. x
  args =. gen_args y
  text =. > 0 { args
  max_steps =. > 1 { args
  temp =. > 2 { args
  k =. > 3 { args
  p =. > 4 { args
  min_p =. > 5 { args

  NB. Frame the prompt with the chat template (single user message) so the
  NB. instruct model emits its stop tokens; stop on the arch stop list.
  messages =. <('user') ; text
  prompt =. llama_chat_prompt messages
  tokens =. llama_tokenize (<llm) , <prompt
  stop =. llama_stop_tokens llm
  L =. # , > tokens
  output =. llm gen_loop_core (tokens ; '' ; max_steps ; temp ; k ; p ; min_p ; <stop)
  gen =. L }. output
  llama_detokenize (<llm) , <gen
)

NB. ---- Batched generation: B independent prompts in parallel ----
NB. Usage: llm llama_generate_batch (prompts ; max_steps ; <temp;k;p;min_p>)
llama_generate_batch =: 4 : 0
  llm =. x
  args =. gen_args y
  prompts =. > 0 { args
  max_steps =. > 1 { args
  temp =. > 2 { args
  k =. > 3 { args
  p =. > 4 { args
  min_p =. > 5 { args
  B =. # prompts
  prompts_tok =. ''
  prompts_len =. ''
  i =. 0
  while. i < B do.
    text =. > i { prompts
    messages =. <('user') ; text
    prompt =. llama_chat_prompt messages
    tokens =. llama_tokenize (<llm) , <prompt
    tok_list =. , > tokens
    prompts_tok =. prompts_tok , <tok_list
    prompts_len =. prompts_len , <(# tok_list)
    i =. i + 1
  end.
  stop =. llama_stop_tokens llm
  kv_batch_g =: B
  output =. llm gen_loop_batch (prompts_tok ; max_steps ; temp ; k ; p ; min_p ; <stop)
  answers =. ''
  i =. 0
  while. i < B do.
    L =. > i { prompts_len
    gen =. (L) }. (> i { output)
    answers =. answers , <(llama_detokenize (<llm) , <gen)
    i =. i + 1
  end.
  answers
)

NB. ---- Simple wrappers (default greedy/top-p params) ----
NB. llm llama_infer_simple text            | llm llama_generate_simple (text ; n)
llama_infer_simple =: llama_infer (] ; (<0 0 0.95 0.0)"_)
llama_generate_simple =: llama_generate (0&{ , 1&{ , (<0 0 0.95 0.0)"_)

NB. ---- Chat-template support (Phase 1.1) ----
NB. y = messages: boxed list of message boxes; each = <role ; content>.
NB. Dispatch by tokenizer pre (set at load): 'llama-bpe' (Llama-3.2) renders
NB. the llama3 template (always-emitted system block + current date); else the
NB. SmolLM2 <|im_start|> template.
llama_chat_prompt =: 3 : 0
  messages =. y
  if. 'llama-bpe' -: llama_tokenizer_pre_g do.
    llama32_chat_prompt messages
  else.
    smollm2_chat_prompt messages
  end.
)

NB. ---- SmolLM2 chat template ----
NB. Prepends the system message unless the first message is system; generation
NB. prompt '<|im_start|>assistant' appended. No BOS (llama.cpp smollm2 chat
NB. starts with <|im_start|>=1, no separate bos; gpt2 tokenize adds none).
smollm2_chat_prompt =: 3 : 0
  messages =. y
  res =. ''
  if. -. ('system' -: > 0 { > 0 { messages) do.
    res =. '<|im_start|>system' , LF , 'You are a helpful AI assistant named SmolLM, trained by Hugging Face' , '<|im_end|>' , LF
  end.
  for_i. i. # messages do.
    msg =. > i { messages
    role =. > 0 { msg
    content =. > 1 { msg
    res =. res , '<|im_start|>' , role , LF , content , '<|im_end|>' , LF
  end.
  res =. res , '<|im_start|>assistant' , LF
  res
)

NB. ---- Llama-3.2 (llama3) chat template ----
NB. The llama3 template ALWAYS emits a system block (Cutting Knowledge Date /
NB. Today Date) even with no system message, then <|eot_id|>, then the user
NB. messages, then the assistant generation prompt. Date is dynamic
NB. (llama-cpp-python injects strftime '%d %b %Y'); llama32_chat_date_g can pin
NB. it for stable test oracles. The <|begin_of_text|> marker is OMITTED —
NB. llama_tokenize prepends BOS for llama-bpe models, so the token stream
NB. matches llama.cpp (bos exactly once).
llama32_chat_date_g =: ''
llama32_today_date =: 3 : 0
  d =. 6!:0 'DD-MM-YYYY'
  day =. 0 1 { d
  mon =. 3 4 { d
  yr =. 6 7 8 9 { d
  months =. ;: 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'
  (day , ' ' , (> months {~ (0 ". mon) - 1)) , ' ' , yr
)
llama32_chat_prompt =: 3 : 0
  messages =. y
  date =. llama32_chat_date_g
  if. 0 = # date do. date =. llama32_today_date '' end.
  res =. '<|start_header_id|>system<|end_header_id|>' , LF , LF
  res =. res , 'Cutting Knowledge Date: December 2023' , LF
  res =. res , 'Today Date: ' , date , LF , LF
  res =. res , '<|eot_id|>'
  for_i. i. # messages do.
    msg =. > i { messages
    role =. > 0 { msg
    content =. strip_ws > 1 { msg
    res =. res , '<|start_header_id|>' , role , '<|end_header_id|>' , LF , LF , content , '<|eot_id|>'
  end.
  res =. res , '<|start_header_id|>assistant<|end_header_id|>' , LF , LF
  res
)

llama_default_params =: 0 0 0.95 0.0
NB. Stop token: EOS (SmolLM2 <|im_end|>=2; Llama-3.2 <|eot_id|>=128009).
llama_stop_tokens =: 3 : 0
  tk =. llm_tokenizer y
  tokenizer_eos_g tk
)
