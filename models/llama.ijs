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
require 'llm/inference/util/minja'
require 'llm/inference/util/chat_template'

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
  'block_data pos mi layer' =. y
  n_heads =. llama_bd_n_heads block_data
  head_dim =. llama_bd_head_dim block_data
  n_heads_kv =. llama_bd_n_heads_kv block_data
  n_groups =. n_heads % n_heads_kv

  NB. Attention norm
  attn_norm_w =. llama_bd_attn_norm block_data
  hidden =. rms_norm ((< mi_rms_eps mi) , (< attn_norm_w) , <x)

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
  Q =. Q * mi_attn_scale mi

  NB. Write K,V (n_kv, hd) to cache, read all up to pos
  NB. layer/pos are already boxed in y (indices 3/1) — pass them, no re-box.
  kv_write ((3 { y) , (1 { y) , (<K) , (<V))
  kv_result =. kv_read ((3 { y) , (1 { y))
  k_all =. > 0 { kv_result   NB. (win, n_kv, hd)
  v_all =. > 1 { kv_result
  win =. pos + 1

  NB. GQA without expanding KV: group the query heads (n_heads_kv groups of
  NB. n_groups) and matmul each group's Q against its shared K/V — k_all/v_all
  NB. stay (win, n_heads_kv, hd), never expanded to n_heads (4x llama).
  Q_g2 =. (n_heads_kv , n_groups , head_dim) $ , Q   NB. Q (n_heads, hd) -> (n_kv, n_g, hd)
  Kp2 =. 1 2 0 |: k_all   NB. (n_kv, hd, win) — one transpose (the 1 0 2 |: + |:"2 two-pass was ~4x slower)
  scores2 =. Q_g2 (+/ .* "2) Kp2   NB. (n_kv, n_groups, win): Q[g,r,d] vs K[j,g,d]
  NB. Single-token decode: all cached j <= pos valid (causal), no mask.

  NB. Softmax directly on the 3D scores2 (the old scores flatten +
  NB. softmax_g2 re-group were 2 redundant reshape copies per layer)
  max_sf =. >./"1 scores2
  exp_sf =. ^ (scores2 - max_sf)
  softmax =. exp_sf % +/"1 exp_sf   NB. (n_kv, n_groups, win)

  NB. Output: attn[h] = sum_j softmax[h,j] * V[j,g(h)]
  Vp =. 1 0 2 |: v_all   NB. (n_kv, win, hd)
  attn2 =. softmax (+/ .* "2) Vp   NB. (n_kv, n_groups, hd)
  attn_raw =. (n_heads, head_dim) $ , attn2   NB. [h,d]

  NB. Output projection
  attn_raw_flat =. (n_heads * head_dim) $ , attn_raw
  attn_out =. (llama_bd_attn_o block_data) linear_r attn_raw_flat
  (<attn_out)
)

NB. ---- Batched attention (prompt prefill, GQA, causal) ----
NB. x = hidden (L, emb); y = <block_data; mi; layer>
llama_attention_b =: 4 : 0
  block_data =. > 0 { y
  layer =. > 1 { y
  mi =. > 2 { y
  start_pos =. > 3 { y
  rope =. > 4 { y
  L =. {. $ x
  n_embd =. {: $ x
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
  hidden =. rms_norm_rows ((< mi_rms_eps mi) , (< attn_norm_w) , <x)

  NB. Separate Q,K,V batched projections (|: hidden hoisted once — 3 transposes
  NB. of the (L,emb) hidden were materializing 3 copies)
  thin =. |: hidden
  qv =. |: ((llama_bd_attn_q block_data) (+/ .* ) thin)   NB. (L, n_heads*hd)
  kv =. |: ((llama_bd_attn_k block_data) (+/ .* ) thin)   NB. (L, n_kv*hd)
  vv =. |: ((llama_bd_attn_v block_data) (+/ .* ) thin)   NB. (L, n_kv*hd)
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
  Q =. Q * mi_attn_scale mi

  NB. RESUME: prepend the cache prefix (positions 0..start_pos-1, already
  NB. norm'd + RoPE'd) so this batch attends to the full history.
  K_batch =. K
  V_batch =. V
  if. start_pos > 0 do.
    k_pre =. > 0 { (kv_read ((1 { y) , <(start_pos - 1)))
    v_pre =. > 1 { (kv_read ((1 { y) , <(start_pos - 1)))
    K =. k_pre , K
    V =. v_pre , V
  end.

  NB. Bulk write the new batch's L K/V into cache at layer, starting at start_pos
  kv_write_rows ((<0) , (1 { y) , (3 { y) , <K_batch)
  kv_write_rows ((<1) , (<layer) , (<start_pos) , <V_batch)

  NB. GQA without expanding KV: group the query heads (n_heads_kv groups of
  NB. n_groups) and batched-matmul each group's Q against its shared K/V row —
  NB. K/V stay (n_heads_kv, ctx, hd), never expanded to n_heads (7x KV for
  NB. qwen2.5, 4x llama/granite, 2x qwen3). Q reshaped group-major so the
  NB. frames (n_heads_kv) align for +/ .*"2.
  Qp =. 1 0 2 |: Q        NB. (n_heads, L, hd)
  Q_g2 =. (n_heads_kv , (n_groups * L) , head_dim) $ , Qp   NB. one ravel+reshape (the intermediate 4D reshape was redundant)
  Kp2 =. 1 2 0 |: K        NB. (n_heads_kv, hd, start_pos+L) — one transpose
  scores2 =. Q_g2 (+/ .* "2) Kp2   NB. (n_kv, n_groups*L, ctx): Q[t,h] vs K[j,g(h)]
  NB. causal mask: mask[h,t,j]=1 if j>t; query t at position start_pos+t, keys
  NB. 0..start_pos+L-1. Keep scores group-major: the tiled mask (row r*L+t
  NB. needs mask row t) is hoisted (item 7 of rope, built once per chunk) —
  NB. subtract with rank over the kv-head frame; no (n_heads, L, tot) 3D mask,
  NB. no per-layer mask build, no scores re-shape copy.
  scores2 =. scores2 -"2 (mask_g2 * 1e9)

  NB. Softmax over j per (g,r,t) row (order-independent)
  NB. Softmax directly on the 3D scores2 (the old scores_f flatten +
  NB. softmax_g2 re-group were 2 redundant reshape copies per layer)
  max_sf =. >./"1 scores2
  exp_sf =. ^ (scores2 - max_sf)
  softmax_f =. exp_sf % +/"1 exp_sf

  NB. Output: attn[h,t] = sum_j softmax[g(h),t,j] * V[g(h),j]
  Vp =. 1 0 2 |: V        NB. (n_heads_kv, start_pos+L, hd)
  attn2 =. softmax_f (+/ .* "2) Vp   NB. (n_kv, n_groups*L, hd)
  attn_raw =. (n_heads, L, head_dim) $ , attn2   NB. [h,t,d]

  NB. Output projection (batched)
  attn_o_w =. llama_bd_attn_o block_data
  attn_out =. |: (attn_o_w (+/ .* ) |: ((L, n_heads * head_dim) $ , (1 0 2 |: attn_raw)))   NB. (L, emb)

  (<attn_out)
)

NB. ---- Single block forward (llama) ----
NB. x = hidden (emb,) — the layer input; y = <block_data; pos; mi; layer>
llama_block_forward =: 4 : 0
  hidden =. x
  'block_data pos mi layer' =. y
  attn_result =. hidden llama_attention y
  attn_out =. > 0 { attn_result
  rs =. mi_resid_scale mi   NB. 1 for non-granite — skip the no-op copy
  if. rs ~: 1 do. attn_out =. attn_out * rs end.
  sa_out =. attn_out + hidden
  ffn_norm_w =. llama_bd_ff_norm block_data
  ffn_in =. rms_norm ((< mi_rms_eps mi) , (< ffn_norm_w) , <sa_out)
  gate =. (llama_bd_ff_gate block_data) linear_r ffn_in
  up =. (llama_bd_ff_up block_data) linear_r ffn_in
  ffn_raw =. (llama_bd_ff_down block_data) linear_r (gate swiglu up)
  if. rs ~: 1 do. ffn_raw =. ffn_raw * rs end.
  output =. ffn_raw + sa_out
  (<output)
)

NB. ---- Batched block forward ----
NB. x = hidden (L, emb); y = <block_data; mi; layer; start_pos; rope>
llama_block_forward_b =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  layer =. > 1 { y
  mi =. > 2 { y
  start_pos =. > 3 { y
  rope =. > 4 { y
  attn_result =. hidden llama_attention_b y
  attn_out =. > 0 { attn_result
  rs =. mi_resid_scale mi   NB. 1 for non-granite — skip the no-op copy
  if. rs ~: 1 do. attn_out =. attn_out * rs end.
  sa_out =. attn_out + hidden
  ffn_norm_w =. llama_bd_ff_norm block_data
  ffn_in =. rms_norm_rows ((< mi_rms_eps mi) , (< ffn_norm_w) , <sa_out)
  ft =. |: ffn_in   NB. |: ffn_in hoisted once — 2 transposes were materializing 2 copies
  gate =. |: ((llama_bd_ff_gate block_data) (+/ .* ) ft)   NB. (L, n_ff)
  up =. |: ((llama_bd_ff_up block_data) (+/ .* ) ft)
  ffn_raw =. |: ((llama_bd_ff_down block_data) (+/ .* ) |: (gate swiglu up))
  if. rs ~: 1 do. ffn_raw =. ffn_raw * rs end.
  output =. ffn_raw + sa_out
  (<output)
)

NB. ---- Run all blocks (single token; cache lives in kv_cache_g global) ----
NB. x = hidden (emb,); y = <llm; pos>
llama_run_blocks =: 4 : 0
  'llm pos' =. y
  mi =. llm_mi llm
  head_dim =. mi_head_dim mi
  n_heads_kv =. mi_n_heads_kv mi
  block_count =. mi_block_count mi
  ctx_len =. mi_context_len mi
  state =. x
  if. 0 = # kv_meta do.
    kv_create ((<block_count) , (<ctx_len) , (<n_heads_kv) , (<head_dim))
  end.
  b =. 0
  block_data_list =. llm_block_data llm
  NB. (<pos), (<mi) are layer-invariant — box once, reuse per layer.
  bf_pre =. ((<pos) , (<mi))
  while. b < block_count do.
    block_data =. > b { block_data_list
    result =. state llama_block_forward ((<block_data) , bf_pre , <b)
    state =. > 0 { result
    b =. b + 1
  end.
  <state
)

NB. ---- Batched run all blocks (prompt prefill) ----
NB. x = hidden (L, emb); y = <llm; start_pos>  (positions start_pos..start_pos+L-1)
llama_run_blocks_b =: 4 : 0
  llm =. > 0 { y
  start_pos =. > 1 { y
  mi =. llm_mi llm
  head_dim =. mi_head_dim mi
  n_heads_kv =. mi_n_heads_kv mi
  block_count =. mi_block_count mi
  ctx_len =. mi_context_len mi
  state =. x
  if. 0 = # kv_meta do.
    kv_create ((<block_count) , (<ctx_len) , (<n_heads_kv) , (<head_dim))
  end.
  NB. RoPE tables are per-model and identical across layers (freq is model
  NB. level): compute the cos/sin tables + expansions ONCE per chunk and thread
  NB. through the layer loop, instead of recomputing them in every layer.
  L =. {. $ x
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
  NB. (<mi), (<start_pos), <rope are layer-invariant — box once, reuse per layer.
  bfb_pre =. ((<mi) , (<start_pos) , <rope)
  b =. 0
  block_data_list =. llm_block_data llm
  while. b < block_count do.
    block_data =. > b { block_data_list
    result =. state llama_block_forward_b ((<block_data) , (<b) , bfb_pre)
    state =. > 0 { result
    b =. b + 1
  end.
  <state
)

NB. ---- Batched-DECODE attention (B sequences, ONE token each at pos[b]) ----
NB. Mirrors llama_attention_bd; llama = qwen2 minus QKV biases, INTERLEAVED RoPE.
NB. x = hidden (B, emb); y = <block_data; pos; mi; layer>
llama_attention_bd =: 4 : 0
  block_data =. > 0 { y
  pos =. > 1 { y
  mi =. > 2 { y
  layer =. > 3 { y
  rope =. > 4 { y
  B =. {. $ x
  n_heads =. llama_bd_n_heads block_data
  head_dim =. llama_bd_head_dim block_data
  n_heads_kv =. llama_bd_n_heads_kv block_data
  n_groups =. n_heads % n_heads_kv
  half =. <. head_dim % 2

  NB. Attention norm per row
  attn_norm_w =. llama_bd_attn_norm block_data
  hidden =. rms_norm_rows ((< mi_rms_eps mi) , (< attn_norm_w) , <x)

  NB. Batched Q,K,V projections (weight-read amortized across B; |: hidden
  NB. hoisted once — 3 transposes of the (B,emb) hidden were materializing 3 copies)
  thin =. |: hidden
  qv =. |: ((llama_bd_attn_q block_data) (+/ .* ) thin)   NB. (B, n_heads*hd)
  kv =. |: ((llama_bd_attn_k block_data) (+/ .* ) thin)   NB. (B, n_kv*hd)
  vv =. |: ((llama_bd_attn_v block_data) (+/ .* ) thin)   NB. (B, n_kv*hd)

  Q =. (B, n_heads, head_dim) $ , qv
  K =. (B, n_heads_kv, head_dim) $ , kv
  V =. (B, n_heads_kv, head_dim) $ , vv

  NB. Interleaved RoPE batched at the B positions (NORM style, pairs (i,i+1)).
  NB. cos/sin tables + expansions are layer-invariant (depend only on pos) —
  NB. hoisted once in run_blocks_bd, threaded through y as <cos_all; sin_all;
  NB. idx; cos_expq; sin_expq; cos_expk; sin_expk>.
  cos_all =. > 0 { rope
  sin_all =. > 1 { rope
  idx =. > 2 { rope
  cos_expq =. > 3 { rope
  sin_expq =. > 4 { rope
  cos_expk =. > 5 { rope
  sin_expk =. > 6 { rope
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
  Q =. Q * mi_attn_scale mi

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
    NB. Softmax directly on the 4D scores_b (the old scores_b2 flatten +
    NB. softmax_g2_b re-group were 2 redundant reshape copies per layer)
    max_sf_b =. >./"1 scores_b
    exp_sf_b =. ^ (scores_b - max_sf_b)
    softmax_b =. exp_sf_b % +/"1 exp_sf_b
    Vp_b =. (0 2 1 3) |: v_rows_b   NB. (B, n_kv, win, hd)
    attn2_b =. softmax_b (+/ .* "2) Vp_b   NB. (B, n_kv, groups, hd)
    attn_all =. (B , n_heads * head_dim) $ , attn2_b   NB. (B, n_heads*hd) flat
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
      k_all =. > 0 { kv_result
      v_all =. > 1 { kv_result
      win =. pos_b + 1
      Q_g2 =. (n_heads_kv , n_groups , head_dim) $ , q_b
      Kp2 =. 1 2 0 |: k_all
      scores2 =. Q_g2 (+/ .* "2) Kp2
      NB. Softmax directly on the 3D scores2 (the old scores flatten +
      NB. softmax_g2 re-group were 2 redundant reshape copies per layer)
      max_sf =. >./"1 scores2
      exp_sf =. ^ (scores2 - max_sf)
      softmax =. exp_sf % +/"1 exp_sf
      Vp =. 1 0 2 |: v_all
      attn2 =. softmax (+/ .* "2) Vp
      attn_raw =. (n_heads, head_dim) $ , attn2
      attn_raw_flat =. (n_heads * head_dim) $ , attn_raw
      attn_out =. attn_out , <attn_raw_flat
      b =. b + 1
    end.
    attn_all =. > attn_out
  end.
  attn_result =. |: ((llama_bd_attn_o block_data) (+/ .* ) |: attn_all)   NB. (B, emb)
  (<attn_result)
)

NB. ---- Batched-PREFILL attention (B sequences, one CHUNK each at pos[b]) ----
NB. Mirrors llama_attention_bd but for prefill: hidden = (B, c, emb) — B chunks
NB. of c tokens each at pos[b]..pos[b]+c-1.  The Q/K/V projections + RoPE are
NB. BATCHED across all B*c rows (weight-read amortized — the memory-bound win),
NB. then the scores/softmax/V run per-sequence (each chunk attends causally to
NB. its own window 0..pos[b]+c-1).  The causal mask excludes the chunk's future
NB. keys.  This is the projection-amortized prefill; a later vectorization can
NB. batch the attention too (lockstep equal-position chunks).
NB. x = hidden (B, c, emb); y = <block_data; pos; mi; layer>
llama_attention_bp =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  layer =. > 1 { y
  pos =. > 2 { y
  mi =. > 3 { y
  NB. Optional 5th arg: per-sequence REAL lengths (for padding masking).  If
  NB. provided, keys at positions >= lens[b] (padding) are excluded from
  NB. attention; if '' (empty), the mask is purely causal (equal-length).
  lens =. ''
  if. 4 < # y do. lens =. > 4 { y end.
  rope =. > 5 { y
  B =. {. $ hidden
  c =. 1 { $ hidden
  emb_len =. 2 { $ hidden
  n_heads =. llama_bd_n_heads block_data
  head_dim =. llama_bd_head_dim block_data
  n_heads_kv =. llama_bd_n_heads_kv block_data
  n_groups =. n_heads % n_heads_kv
  half =. <. head_dim % 2
  eff_seq =. > 1 { kv_meta

  NB. Attention norm per row (B*c)
  attn_norm_w =. llama_bd_attn_norm block_data
  hidden_flat =. ((B*c) , emb_len) $ , hidden
  hidden_flat =. rms_norm_rows ((< mi_rms_eps mi) , (< attn_norm_w) , <hidden_flat)

  NB. Batched Q,K,V projections (weight-read amortized across B*c rows; |: hidden_flat
  NB. hoisted once — 3 transposes were materializing 3 copies)
  thin =. |: hidden_flat
  qv =. |: ((llama_bd_attn_q block_data) (+/ .* ) thin)   NB. (B*c, n_heads*hd)
  kv =. |: ((llama_bd_attn_k block_data) (+/ .* ) thin)   NB. (B*c, n_kv*hd)
  vv =. |: ((llama_bd_attn_v block_data) (+/ .* ) thin)   NB. (B*c, n_kv*hd)

  Q =. (B, c, n_heads, head_dim) $ , qv
  K =. (B, c, n_heads_kv, head_dim) $ , kv
  V =. (B, c, n_heads_kv, head_dim) $ , vv

  NB. Interleaved RoPE at the (B, c) positions pos[b]+i.c. cos/sin tables +
  NB. expansions are layer-invariant (depend only on pos) — hoisted once in
  NB. run_blocks_bp, threaded through y as <cos_all; sin_all; idx; cos_expq;
  NB. sin_expq; cos_expk; sin_expk>.
  cos_all =. > 0 { rope
  sin_all =. > 1 { rope
  idx =. > 2 { rope
  cos_expq =. > 3 { rope
  sin_expq =. > 4 { rope
  cos_expk =. > 5 { rope
  sin_expk =. > 6 { rope
  Qa =. idx {"1 Q
  Qb =. (1 + idx) {"1 Q
  Qa_out =. (Qa * cos_expq) - (Qb * sin_expq)
  Qb_out =. (Qa * sin_expq) + (Qb * cos_expq)
  Q =. (B, c, n_heads, head_dim) $ , (1 2 3 4 0 |: (Qa_out ,: Qb_out))
  Ka =. idx {"1 K
  Kb =. (1 + idx) {"1 K
  Ka_out =. (Ka * cos_expk) - (Kb * sin_expk)
  Kb_out =. (Ka * sin_expk) + (Kb * cos_expk)
  K =. (B, c, n_heads_kv, head_dim) $ , (1 2 3 4 0 |: (Ka_out ,: Kb_out))

  NB. Llama scales Q by 1/sqrt(head_dim)
  Q =. Q * mi_attn_scale mi

  NB. Per-sequence attention (scores/softmax/V per seq; causal mask per chunk)
  attn_out =. ''
  b =. 0
  while. b < B do.
    q_b =. b { Q
    k_b =. b { K
    v_b =. b { V
    pos_b =. b { pos
    base_b =. ((layer * kv_batch_g) + b) * eff_seq
    idxw =. base_b + pos_b + i. c
    k_cache_g =: ((c, n_heads_kv*head_dim) $ , k_b) idxw} k_cache_g
    v_cache_g =: ((c, n_heads_kv*head_dim) $ , v_b) idxw} v_cache_g
    kv_pos_g =: kv_pos_g >. pos_b + c
    win =. pos_b + c
    k_all =. (win, n_heads_kv, head_dim) $ , ((base_b + i. win) { k_cache_g)
    v_all =. (win, n_heads_kv, head_dim) $ , ((base_b + i. win) { v_cache_g)
    NB. causal mask: row i (query at pos_b+i) invalid for keys j > pos_b+i
    mask_2d =. (pos_b + i. c) </ i. win   NB. (c, win)
    if. 0 < # lens do.
      NB. Exclude padding keys (positions >= lens[b]) so padded sequences do
      NB. not attend to their own padding.
      mask_2d =. mask_2d +. ((i. win) >: b { lens)
    end.
    mask_g2 =. ((n_groups * c), win) $ , (2 0 1 |: (mask_2d (*/) (n_groups $ 1)))
    Qp =. 1 0 2 |: q_b   NB. (n_heads, c, hd)
    Q_g2 =. (n_heads_kv, (n_groups*c), head_dim) $ , Qp   NB. one ravel+reshape (the intermediate 4D reshape was redundant)
    Kp2 =. 1 2 0 |: k_all   NB. (n_kv, hd, win)
    scores2 =. Q_g2 (+/ .* "2) Kp2   NB. (n_kv, groups*c, win)
    scores2 =. scores2 -"2 (mask_g2 * 1e9)
    NB. Softmax directly on the 3D scores2 (the old scores_f flatten +
    NB. softmax_g2 re-group were 2 redundant reshape copies per layer)
    max_sf =. >./"1 scores2
    exp_sf =. ^ (scores2 - max_sf)
    softmax_f =. exp_sf % +/"1 exp_sf
    Vp =. 1 0 2 |: v_all   NB. (n_kv, win, hd)
    attn2 =. softmax_f (+/ .* "2) Vp   NB. (n_kv, groups*c, hd)
    attn_raw =. (n_heads, c, head_dim) $ , attn2   NB. [h, row, d]
    attn_raw_flat =. (c, n_heads*head_dim) $ , (1 0 2 |: attn_raw)
    attn_out =. attn_out , <attn_raw_flat
    b =. b + 1
  end.
  attn_all =. > attn_out
  attn_result =. |: ((llama_bd_attn_o block_data) (+/ .* ) |: (((B*c) , (n_heads*head_dim)) $ , attn_all))
  attn_result =. (B, c, emb_len) $ , attn_result
  (<attn_result)
)

NB. ---- Batched-decode block forward (llama) ----
NB. x = hidden (B, emb); y = <block_data; pos; mi; layer>
llama_block_forward_bd =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  pos =. > 1 { y
  mi =. > 2 { y
  layer =. > 3 { y
  rope =. > 4 { y
  attn_result =. hidden llama_attention_bd y
  attn_out =. > 0 { attn_result
  rs =. mi_resid_scale mi   NB. 1 for non-granite — skip the no-op copy
  if. rs ~: 1 do. attn_out =. attn_out * rs end.
  sa_out =. attn_out + hidden
  ffn_norm_w =. llama_bd_ff_norm block_data
  ffn_in =. rms_norm_rows ((< mi_rms_eps mi) , (< ffn_norm_w) , <sa_out)
  ft =. |: ffn_in   NB. |: ffn_in hoisted once — 2 transposes were materializing 2 copies
  gate =. |: ((llama_bd_ff_gate block_data) (+/ .* ) ft)   NB. (B, n_ff)
  up =. |: ((llama_bd_ff_up block_data) (+/ .* ) ft)
  ffn_raw =. |: ((llama_bd_ff_down block_data) (+/ .* ) |: (gate swiglu up))
  if. rs ~: 1 do. ffn_raw =. ffn_raw * rs end.
  output =. ffn_raw + sa_out
  (<output)
)

NB. ---- Run all blocks for B sequences (one token each at pos[b]) ----
NB. x = hidden (B, emb); y = <llm; pos>
llama_run_blocks_bd =: 4 : 0
  llm =. > 0 { y
  pos =. > 1 { y
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
  NB. RoPE cos/sin tables + expansions are layer-invariant (depend only on
  NB. pos) — compute ONCE per step, thread through (mirror run_blocks_b).
  B =. {. $ state
  half =. <. head_dim % 2
  idx =. 2 * i. half
  cos_all =. pos { mi_cos_tab mi
  sin_all =. pos { mi_sin_tab mi
  cos_expq =. (0 2 1) |: ((B , half , n_heads) $ , (cos_all (*/) (n_heads $ 1)))
  sin_expq =. (0 2 1) |: ((B , half , n_heads) $ , (sin_all (*/) (n_heads $ 1)))
  cos_expk =. (0 2 1) |: ((B , half , n_heads_kv) $ , (cos_all (*/) (n_heads_kv $ 1)))
  sin_expk =. (0 2 1) |: ((B , half , n_heads_kv) $ , (sin_all (*/) (n_heads_kv $ 1)))
  rope =. (<cos_all) , (<sin_all) , (<idx) , (<cos_expq) , (<sin_expq) , (<cos_expk) , (<sin_expk)
  b =. 0
  block_data_list =. llm_block_data llm
  NB. (<pos), (<mi) are layer-invariant — box once, reuse per layer.
  bf_pre =. ((<pos) , (<mi))
  while. b < block_count do.
    block_data =. > b { block_data_list
    result =. state llama_block_forward_bd ((<block_data) , bf_pre , (<b) , <rope)
    state =. > 0 { result
    b =. b + 1
  end.
  <state
)

NB. ---- Batched-prefill block forward (llama) ----
NB. x = hidden (B, c, emb); y = <block_data; pos; mi; layer>.  The FFN weight
NB. matmuls are batched across B*c rows (amortized).  Returns <(B, c, emb)>.
llama_block_forward_bp =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  layer =. > 1 { y
  pos =. > 2 { y
  mi =. > 3 { y
  lens =. ''
  if. 4 < # y do. lens =. > 4 { y end.
  rope =. > 5 { y
  B =. {. $ hidden
  c =. 1 { $ hidden
  emb_len =. 2 { $ hidden
  attn_result =. hidden llama_attention_bp y
  attn_out =. > 0 { attn_result   NB. (B, c, emb)
  rs =. mi_resid_scale mi   NB. 1 for non-granite — skip the no-op copy
  if. rs ~: 1 do. attn_out =. attn_out * rs end.
  sa_out =. attn_out + hidden
  ffn_norm_w =. llama_bd_ff_norm block_data
  sa_flat =. ((B*c) , emb_len) $ , sa_out
  ffn_in =. rms_norm_rows ((< mi_rms_eps mi) , (< ffn_norm_w) , <sa_flat)
  ft =. |: ffn_in   NB. |: ffn_in hoisted once — 2 transposes were materializing 2 copies
  gate =. |: ((llama_bd_ff_gate block_data) (+/ .* ) ft)   NB. (B*c, n_ff)
  up =. |: ((llama_bd_ff_up block_data) (+/ .* ) ft)
  ffn_raw =. |: ((llama_bd_ff_down block_data) (+/ .* ) |: (gate swiglu up))
  if. rs ~: 1 do. ffn_raw =. ffn_raw * rs end.
  output_flat =. ffn_raw + sa_flat
  output =. (B, c, emb_len) $ , output_flat
  (<output)
)

NB. ---- Run all blocks for B sequences (one CHUNK each at pos[b]) ----
NB. x = hidden (B, c, emb); y = <llm; pos; lens?>
llama_run_blocks_bp =: 4 : 0
  llm =. > 0 { y
  pos =. > 1 { y
  lens =. ''
  if. 2 < # y do. lens =. > 2 { y end.
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
  NB. RoPE cos/sin tables + expansions are layer-invariant (depend only on
  NB. pos) — compute ONCE per chunk, thread through (mirror run_blocks_b).
  B =. {. $ state
  c =. 1 { $ state
  half =. <. head_dim % 2
  idx =. 2 * i. half
  pos_bc_flat =. (B*c) $ , (pos +/ i. c)
  cos_all =. pos_bc_flat { mi_cos_tab mi
  sin_all =. pos_bc_flat { mi_sin_tab mi
  cos_expq =. (B, c, n_heads, half) $ , ((0 2 1) |: (((B*c) , half , n_heads) $ , (cos_all (*/) (n_heads $ 1))))
  sin_expq =. (B, c, n_heads, half) $ , ((0 2 1) |: (((B*c) , half , n_heads) $ , (sin_all (*/) (n_heads $ 1))))
  cos_expk =. (B, c, n_heads_kv, half) $ , ((0 2 1) |: (((B*c) , half , n_heads_kv) $ , (cos_all (*/) (n_heads_kv $ 1))))
  sin_expk =. (B, c, n_heads_kv, half) $ , ((0 2 1) |: (((B*c) , half , n_heads_kv) $ , (sin_all (*/) (n_heads_kv $ 1))))
  rope =. (<cos_all) , (<sin_all) , (<idx) , (<cos_expq) , (<sin_expq) , (<cos_expk) , (<sin_expk)
  NB. (<pos), (<mi), <lens are layer-invariant — box once, reuse per layer.
  bfb_pre =. ((<pos) , (<mi) , <lens)
  b =. 0
  block_data_list =. llm_block_data llm
  while. b < block_count do.
    block_data =. > b { block_data_list
    result =. state llama_block_forward_bp ((<block_data) , (<b) , bfb_pre , <rope)
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
  kv_data =. build_kv_dict kvs_ctx
  mi =. llama_extract_hparams kvs_ctx
  mi =. build_mi_dict mi
  rope_tables =. build_rope_tables ((< mi_context_len mi) , (< mi_head_dim mi) , (< mi_rope_freq mi))
  NB. Default attention/residual scales: 1/sqrt(hd) and 1. Granite overrides
  NB. these with its GGUF scale KVs; the shared forward verbs read them.
  NB. ONE multi-put: the jdict single-put (1-key) path is broken in this J9.8
  NB. build (domain/length error); multi-put (2+ keys) and get/has work.
  head_dim =. mi_head_dim mi
  ((> 0 { rope_tables) ; (> 1 { rope_tables) ; (1 % head_dim ^ 0.5) ; 1) put__mi 'cos_tab' ; 'sin_tab' ; 'attn_scale' ; 'resid_scale'
  NB. Real chat template from the GGUF ('' if absent → bespoke fallback).
  ct_tmpl_g =: 'tokenizer.chat_template' kv_string (0 1 { kv_result)
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
  ze =. <0 0 0.95 0.0   NB. default_params (chat sampling defaults)
  tk =. <tokenizer
  mi_b =. <mi
  kc_b =. <''   NB. kv cache is the kv_cache_g global, not stored in the llm
  td =. <tds
  all_tensors =. emb_canonical all_tensors
  at =. <all_tensors
  llm =. p , t , ze , tk , mi_b , kc_b , td , at
  block_data =. llama_pre_build_block_data llm
  llm =. llm , <block_data
  llm =. llm , <kv_data
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
  llm_box =. <llm
  prompts_tok =. ''
  prompts_len =. ''
  i =. 0
  while. i < B do.
    text =. > i { prompts
    messages =. <('user') ; text
    prompt =. llama_chat_prompt messages
    tokens =. llama_tokenize (llm_box , <prompt)
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
    answers =. answers , <(llama_detokenize (llm_box , <gen))
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
NB. Renders the real GGUF jinja chat_template (tokenizer.chat_template) via
NB. chat_tmpl_render for both SmolLM2 and Llama-3.2 (the llama arch).


NB. Stop token: EOS (SmolLM2 <|im_end|>=2; Llama-3.2 <|eot_id|>=128009).
llama_stop_tokens =: 3 : 0
  tk =. llm_tokenizer y
  tokenizer_eos_g tk
)
