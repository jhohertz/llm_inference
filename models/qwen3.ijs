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

NB. ---- KV helpers ----
qw3_kv_uint =: 4 : 0
  key =. x
  data =. y
  key kv_uint data
)

qw3_kv_float =: 4 : 0
  key =. x
  data =. y
  key kv_float data
)

NB. ---- Extract qwen3 model info from KV pairs ----
NB. mi = <block_count; context_len; emb_len; n_heads; n_heads_kv; head_dim; rope_freq; vocab_size; rms_eps; n_ff>
qw3_extract_hparams =: 3 : 0
  data =. y
  block_count =. 'qwen3.block_count' qw3_kv_uint data
  context_length =. 'qwen3.context_length' qw3_kv_uint data
  emb_len =. 'qwen3.embedding_length' qw3_kv_uint data
  n_heads =. 'qwen3.attention.head_count' qw3_kv_uint data
  n_heads_kv =. 'qwen3.attention.head_count_kv' qw3_kv_uint data
  rope_freq =. 'qwen3.rope.freq_base' qw3_kv_float data
  vocab_size =. 'qwen3.vocab_size' qw3_kv_uint data
  rms_eps =. 'qwen3.attention.layer_norm_rms_epsilon' qw3_kv_float data
  n_ff =. 'qwen3.feed_forward_length' qw3_kv_uint data
  key_len =. 'qwen3.attention.key_length' qw3_kv_uint data
  if. key_len <: 0 do. key_len =. emb_len % n_heads end.
  head_dim =. key_len
  (<"0) block_count , context_length , emb_len , n_heads , n_heads_kv , head_dim , rope_freq , vocab_size , rms_eps , n_ff
)

NB. ---- block_data accessors (qwen3: separate weights + q/k norms, no biases) ----
NB. block_data = <attn_norm; attn_q; attn_k; attn_v; attn_o; ffn_norm; ffn_gate; ffn_up; ffn_down; q_norm; k_norm; n_heads; head_dim; rope_freq; n_heads_kv; n_ff; block_idx>
qw3_bd_attn_norm =: >@(0&{)
qw3_bd_attn_q    =: >@(1&{)
qw3_bd_attn_k    =: >@(2&{)
qw3_bd_attn_v    =: >@(3&{)
qw3_bd_attn_o    =: >@(4&{)
qw3_bd_ff_norm   =: >@(5&{)
qw3_bd_ff_gate   =: >@(6&{)
qw3_bd_ff_up     =: >@(7&{)
qw3_bd_ff_down   =: >@(8&{)
qw3_bd_q_norm    =: >@(9&{)
qw3_bd_k_norm    =: >@(10&{)
qw3_bd_n_heads   =: >@(11&{)
qw3_bd_head_dim  =: >@(12&{)

qw3_bd_n_heads_kv=: >@(13&{)
NB. ---- Pre-build block data for qwen3 arch ----
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
  (<attn_norm),(<attn_q),(<attn_k),(<attn_v),(<attn_o_w),(<ffn_norm),(<ff_gate),(<ff_up),(<ff_down),(<q_norm),(<k_norm),(<n_heads),(<head_dim),(<n_heads_kv)
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
  'block_data pos mi layer' =. y
  n_heads =. qw3_bd_n_heads block_data
  head_dim =. qw3_bd_head_dim block_data
  n_heads_kv =. qw3_bd_n_heads_kv block_data
  n_groups =. n_heads % n_heads_kv

  NB. Attention norm
  hidden =. rms_norm ((< mi_rms_eps mi) , (< (qw3_bd_attn_norm block_data)) , <x)

  NB. Separate Q, K, V projections
  qv =. (qw3_bd_attn_q block_data) linear_r hidden
  kv =. (qw3_bd_attn_k block_data) linear_r hidden
  vv =. (qw3_bd_attn_v block_data) linear_r hidden

  Q =. (n_heads, head_dim) $ qv
  K =. (n_heads_kv, head_dim) $ kv
  V =. (n_heads_kv, head_dim) $ vv

  NB. Qwen3: per-head RMSNorm on Q and K BEFORE RoPE (shared weight, size=head_dim)
  Q =. rms_norm_rows ((< mi_rms_eps mi) , (< qw3_bd_q_norm block_data) , <Q)
  K =. rms_norm_rows ((< mi_rms_eps mi) , (< qw3_bd_k_norm block_data) , <K)

  NB. NEOX RoPE (pairs offset by head_dim/2), table-based
  cos_t =. pos { mi_cos_tab mi
  sin_t =. pos { mi_sin_tab mi
  rope_t =. (<cos_t) , (<sin_t)
  Q =. Q rope_apply2_neox_t rope_t
  K =. K rope_apply2_neox_t rope_t

  NB. Qwen3 scales Q by 1/sqrt(head_dim) before attention
  Q =. Q % head_dim ^ 0.5

  NB. Write K,V (n_kv, hd) to cache, read all up to pos
  kv_write ((3 { y) , (1 { y) , (<K) , (<V))
  kv_result =. kv_read ((3 { y) , (1 { y))
  k_all =. > 0 { kv_result   NB. (win, n_kv, hd)
  v_all =. > 1 { kv_result
  win =. pos + 1

  NB. GQA without expanding KV: group the query heads (n_heads_kv groups of
  NB. n_groups) and matmul each group's Q against its shared K/V — k_all/v_all
  NB. stay (win, n_heads_kv, hd), never expanded to n_heads (2x qwen3).
  scores2 =. ((n_heads_kv , n_groups , head_dim) $ , Q) (+/ .* "2) (1 2 0 |: k_all)   NB. (n_kv, n_groups, win): Q[g,r,d] vs K[j,g,d]
  NB. Single-token decode: all cached j <= pos valid (causal), no mask.

  NB. Softmax directly on the 3D scores2 (the old scores flatten +
  NB. softmax_g2 re-group were 2 redundant reshape copies per layer)
  max_sf =. >./"1 scores2
  exp_sf =. ^ (scores2 - max_sf)
  softmax =. exp_sf % +/"1 exp_sf   NB. (n_kv, n_groups, win)

  NB. Output: attn[h] = sum_j softmax[h,j] * V[j,g(h)]; output projection
  attn_out =. (qw3_bd_attn_o block_data) linear_r ((n_heads * head_dim) $ , (softmax (+/ .* "2) (1 0 2 |: v_all)))
  (<attn_out)
)

NB. ---- Batched attention (prompt prefill, GQA, causal) ----
NB. x = hidden (L, emb); y = <block_data; mi; layer>
qw3_attention_b =: 4 : 0
  block_data =. > 0 { y
  layer =. > 1 { y
  mi =. > 2 { y
  start_pos =. > 3 { y
  rope =. > 4 { y
  L =. {. $ x
  n_embd =. {: $ x
  n_heads =. qw3_bd_n_heads block_data
  head_dim =. qw3_bd_head_dim block_data
  n_heads_kv =. qw3_bd_n_heads_kv block_data
  n_groups =. n_heads % n_heads_kv
  NB. rope = <cos_all; sin_all; half; cos_expq; sin_expq; cos_expk; sin_expk; mask_g2>
  cos_all =. > 0 { rope
  sin_all =. > 1 { rope
  half =. > 2 { rope
  cos_expq =. > 3 { rope
  sin_expq =. > 4 { rope
  cos_expk =. > 5 { rope
  sin_expk =. > 6 { rope
  mask_g2 =. > 7 { rope

  NB. Attention norm per row
  hidden =. rms_norm_rows ((< mi_rms_eps mi) , (< (qw3_bd_attn_norm block_data)) , <x)

  NB. Separate Q,K,V batched projections (|: hidden hoisted once — 3 transposes
  NB. of the (L,emb) hidden were materializing 3 copies)
  thin =. |: hidden
  qv =. |: ((qw3_bd_attn_q block_data) (+/ .* ) thin)   NB. (L, n_heads*hd)
  kv =. |: ((qw3_bd_attn_k block_data) (+/ .* ) thin)   NB. (L, n_kv*hd)
  vv =. |: ((qw3_bd_attn_v block_data) (+/ .* ) thin)   NB. (L, n_kv*hd)

  Q =. (L, n_heads, head_dim) $ , qv
  K =. (L, n_heads_kv, head_dim) $ , kv
  V =. (L, n_heads_kv, head_dim) $ , vv

  NB. Qwen3: per-head RMSNorm on Q and K BEFORE RoPE (shared weight, size=head_dim)
  Qf =. ((L * n_heads) , head_dim) $ , Q
  Qf =. rms_norm_rows ((< mi_rms_eps mi) , (< qw3_bd_q_norm block_data) , <Qf)
  Q =. (L, n_heads, head_dim) $ , Qf
  Kf =. ((L * n_heads_kv) , head_dim) $ , K
  Kf =. rms_norm_rows ((< mi_rms_eps mi) , (< qw3_bd_k_norm block_data) , <Kf)
  K =. (L, n_heads_kv, head_dim) $ , Kf

  NB. NEOX RoPE batched (table-based): pairs (i, i+half) per row
  Qa =. half {. "1 Q        NB. (L, n_heads, half) first half
  Qb =. half }. "1 Q        NB. (L, n_heads, half) second half
  Qa_out =. (Qa * cos_expq) - (Qb * sin_expq)
  Qb_out =. (Qa * sin_expq) + (Qb * cos_expq)
  Q =. (L, n_heads, head_dim) $ , (Qa_out ,"1 Qb_out)
  Ka =. half {. "1 K        NB. (L, n_heads_kv, half)
  Kb =. half }. "1 K
  Ka_out =. (Ka * cos_expk) - (Kb * sin_expk)
  Kb_out =. (Ka * sin_expk) + (Kb * cos_expk)
  K =. (L, n_heads_kv, head_dim) $ , (Ka_out ,"1 Kb_out)

  NB. Qwen3 scales Q by 1/sqrt(head_dim)
  Q =. Q % head_dim ^ 0.5

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

  NB. Softmax directly on the 3D scores2 (the old scores_f flatten +
  NB. softmax_g2 re-group were 2 redundant reshape copies per layer)
  max_sf =. >./"1 scores2
  exp_sf =. ^ (scores2 - max_sf)
  softmax_f =. exp_sf % +/"1 exp_sf

  NB. Output: attn[h,t] = sum_j softmax[g(h),t,j] * V[g(h),j]
  Vp =. 1 0 2 |: V        NB. (n_heads_kv, start_pos+L, hd)
  attn2 =. softmax_f (+/ .* "2) Vp   NB. (n_kv, n_groups*L, hd)
  NB. Output projection (batched)
  attn_out =. |: ((qw3_bd_attn_o block_data) (+/ .* ) |: ((L, n_heads * head_dim) $ , (1 0 2 |: ((n_heads, L, head_dim) $ , attn2))))   NB. (L, emb)

  (<attn_out)
)

NB. ---- Single block forward (qwen3) ----
NB. x = hidden (emb,) — the layer input; y = <block_data; pos; mi; layer>
qw3_block_forward =: 4 : 0
  hidden =. x
  'block_data pos mi layer' =. y
  attn_result =. hidden qw3_attention y
  attn_out =. > 0 { attn_result
  sa_out =. attn_out + hidden
  ffn_in =. rms_norm ((< mi_rms_eps mi) , (< (qw3_bd_ff_norm block_data)) , <sa_out)
  gate =. (qw3_bd_ff_gate block_data) linear_r ffn_in
  up =. (qw3_bd_ff_up block_data) linear_r ffn_in
  ffn_raw =. (qw3_bd_ff_down block_data) linear_r (gate swiglu up)
  output =. ffn_raw + sa_out
  (<output)
)

NB. ---- Batched block forward ----
NB. x = hidden (L, emb); y = <block_data; mi; layer; start_pos; rope>
qw3_block_forward_b =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  layer =. > 1 { y
  mi =. > 2 { y
  start_pos =. > 3 { y
  rope =. > 4 { y
  attn_result =. hidden qw3_attention_b y
  attn_out =. > 0 { attn_result
  sa_out =. attn_out + hidden
  ffn_in =. rms_norm_rows ((< mi_rms_eps mi) , (< (qw3_bd_ff_norm block_data)) , <sa_out)
  ft =. |: ffn_in   NB. |: ffn_in hoisted once — 2 transposes were materializing 2 copies
  gate =. |: ((qw3_bd_ff_gate block_data) (+/ .* ) ft)   NB. (L, n_ff)
  up =. |: ((qw3_bd_ff_up block_data) (+/ .* ) ft)
  ffn_raw =. |: ((qw3_bd_ff_down block_data) (+/ .* ) |: (gate swiglu up))
  output =. ffn_raw + sa_out
  (<output)
)

NB. ---- Run all blocks (single token; cache lives in kv_cache_g global) ----
NB. x = hidden (emb,); y = <llm; pos>
qw3_run_blocks =: 4 : 0
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
    result =. state qw3_block_forward ((<block_data) , bf_pre , <b)
    state =. > 0 { result
    b =. b + 1
  end.
  <state
)

NB. ---- Batched run all blocks (prompt prefill) ----
NB. x = hidden (L, emb); y = <llm; start_pos>  (positions start_pos..start_pos+L-1)
qw3_run_blocks_b =: 4 : 0
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
  rope =. (<cos_all) , (<sin_all) , (<half) , (<cos_expq) , (<sin_expq) , (<cos_expk) , (<sin_expk) , (<mask_g2)
  NB. (<mi), (<start_pos), <rope are layer-invariant — box once, reuse per layer.
  bfb_pre =. ((<mi) , (<start_pos) , <rope)
  b =. 0
  block_data_list =. llm_block_data llm
  while. b < block_count do.
    block_data =. > b { block_data_list
    result =. state qw3_block_forward_b ((<block_data) , (<b) , bfb_pre)
    state =. > 0 { result
    b =. b + 1
  end.
  <state
)

NB. ---- Batched-DECODE attention (B sequences, ONE token each at pos[b]) ----
NB. Mirrors qw2_attention_bd; qwen3 adds per-head Q/K RMSNorm before RoPE and
NB. has NO Q/K/V biases. x = hidden (B, emb); y = <block_data; pos; mi; layer>
qw3_attention_bd =: 4 : 0
  block_data =. > 0 { y
  pos =. > 1 { y
  mi =. > 2 { y
  layer =. > 3 { y
  rope =. > 4 { y
  B =. {. $ x
  n_heads =. qw3_bd_n_heads block_data
  head_dim =. qw3_bd_head_dim block_data
  n_heads_kv =. qw3_bd_n_heads_kv block_data
  n_groups =. n_heads % n_heads_kv
  half =. <. head_dim % 2

  NB. Attention norm per row
  hidden =. rms_norm_rows ((< mi_rms_eps mi) , (< (qw3_bd_attn_norm block_data)) , <x)

  NB. Batched Q,K,V projections (weight-read amortized across B; |: hidden
  NB. hoisted once — 3 transposes of the (B,emb) hidden were materializing 3 copies)
  thin =. |: hidden
  qv =. |: ((qw3_bd_attn_q block_data) (+/ .* ) thin)   NB. (B, n_heads*hd)
  kv =. |: ((qw3_bd_attn_k block_data) (+/ .* ) thin)   NB. (B, n_kv*hd)
  vv =. |: ((qw3_bd_attn_v block_data) (+/ .* ) thin)   NB. (B, n_kv*hd)

  Q =. (B, n_heads, head_dim) $ , qv
  K =. (B, n_heads_kv, head_dim) $ , kv
  V =. (B, n_heads_kv, head_dim) $ , vv

  NB. Qwen3: per-head RMSNorm on Q and K BEFORE RoPE (shared weight, size=head_dim)
  Qf =. ((B * n_heads) , head_dim) $ , Q
  Qf =. rms_norm_rows ((< mi_rms_eps mi) , (< qw3_bd_q_norm block_data) , <Qf)
  Q =. (B, n_heads, head_dim) $ , Qf
  Kf =. ((B * n_heads_kv) , head_dim) $ , K
  Kf =. rms_norm_rows ((< mi_rms_eps mi) , (< qw3_bd_k_norm block_data) , <Kf)
  K =. (B, n_heads_kv, head_dim) $ , Kf

  NB. NEOX RoPE batched at the B positions. cos/sin tables + expansions are
  NB. layer-invariant (depend only on pos) — hoisted once in run_blocks_bd,
  NB. threaded through y as <cos_all; sin_all; cos_expq; sin_expq; cos_expk;
  NB. sin_expk>.
  cos_all =. > 0 { rope
  sin_all =. > 1 { rope
  cos_expq =. > 2 { rope
  sin_expq =. > 3 { rope
  cos_expk =. > 4 { rope
  sin_expk =. > 5 { rope
  Qa =. half {. "1 Q
  Qb =. half }. "1 Q
  Qa_out =. (Qa * cos_expq) - (Qb * sin_expq)
  Qb_out =. (Qa * sin_expq) + (Qb * cos_expq)
  Q =. (B, n_heads, head_dim) $ , (Qa_out ,"1 Qb_out)
  Ka =. half {. "1 K
  Kb =. half }. "1 K
  Ka_out =. (Ka * cos_expk) - (Kb * sin_expk)
  Kb_out =. (Ka * sin_expk) + (Kb * cos_expk)
  K =. (B, n_heads_kv, head_dim) $ , (Ka_out ,"1 Kb_out)

  NB. Qwen3 scales Q by 1/sqrt(head_dim)
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
    NB. Batched GQA scores, softmax, V (threaded over B; no causal mask —
    NB. all cached j <= pos valid in single-token decode)
    scores_b =. ((B , n_heads_kv , n_groups , head_dim) $ , Q) (+/ .* "2) ((0 2 3 1) |: ((B , win , n_heads_kv , head_dim) $ , (idxr { k_cache_g)))   NB. (B, n_kv, groups, win)
    NB. Softmax directly on the 4D scores_b (the old scores_b2 flatten +
    NB. softmax_g2_b re-group were 2 redundant reshape copies per layer)
    max_sf_b =. >./"1 scores_b
    exp_sf_b =. ^ (scores_b - max_sf_b)
    softmax_b =. exp_sf_b % +/"1 exp_sf_b
    attn_all =. (B , n_heads * head_dim) $ , (softmax_b (+/ .* "2) ((0 2 1 3) |: ((B , win , n_heads_kv , head_dim) $ , (idxr { v_cache_g))))   NB. (B, n_heads*hd) flat
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
      NB. Softmax directly on the 3D scores2 (the old scores flatten +
      NB. softmax_g2 re-group were 2 redundant reshape copies per layer)
      max_sf =. >./"1 scores2
      exp_sf =. ^ (scores2 - max_sf)
      softmax =. exp_sf % +/"1 exp_sf
      attn_out =. attn_out , <((n_heads * head_dim) $ , (softmax (+/ .* "2) (1 0 2 |: (> 1 { kv_result))))
      b =. b + 1
    end.
    attn_all =. > attn_out
  end.
  attn_result =. |: ((qw3_bd_attn_o block_data) (+/ .* ) |: attn_all)   NB. (B, emb)
  (<attn_result)
)

NB. ---- Batched-decode block forward ----
qw3_block_forward_bd =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  pos =. > 1 { y
  mi =. > 2 { y
  layer =. > 3 { y
  rope =. > 4 { y
  attn_result =. hidden qw3_attention_bd y
  attn_out =. > 0 { attn_result
  sa_out =. attn_out + hidden
  ffn_in =. rms_norm_rows ((< mi_rms_eps mi) , (< (qw3_bd_ff_norm block_data)) , <sa_out)
  ft =. |: ffn_in   NB. |: ffn_in hoisted once — 2 transposes were materializing 2 copies
  gate =. |: ((qw3_bd_ff_gate block_data) (+/ .* ) ft)   NB. (B, n_ff)
  up =. |: ((qw3_bd_ff_up block_data) (+/ .* ) ft)
  ffn_raw =. |: ((qw3_bd_ff_down block_data) (+/ .* ) |: (gate swiglu up))
  output =. ffn_raw + sa_out
  (<output)
)

NB. ---- Run all blocks for B sequences (one token each at pos[b]) ----
qw3_run_blocks_bd =: 4 : 0
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
  cos_all =. pos { mi_cos_tab mi
  sin_all =. pos { mi_sin_tab mi
  cos_expq =. (0 2 1) |: ((B , half , n_heads) $ , (cos_all (*/) (n_heads $ 1)))
  sin_expq =. (0 2 1) |: ((B , half , n_heads) $ , (sin_all (*/) (n_heads $ 1)))
  cos_expk =. (0 2 1) |: ((B , half , n_heads_kv) $ , (cos_all (*/) (n_heads_kv $ 1)))
  sin_expk =. (0 2 1) |: ((B , half , n_heads_kv) $ , (sin_all (*/) (n_heads_kv $ 1)))
  rope =. (<cos_all) , (<sin_all) , (<cos_expq) , (<sin_expq) , (<cos_expk) , (<sin_expk)
  b =. 0
  block_data_list =. llm_block_data llm
  NB. (<pos), (<mi) are layer-invariant — box once, reuse per layer.
  bf_pre =. ((<pos) , (<mi))
  while. b < block_count do.
    block_data =. > b { block_data_list
    result =. state qw3_block_forward_bd ((<block_data) , bf_pre , (<b) , <rope)
    state =. > 0 { result
    b =. b + 1
  end.
  <state
)

NB. ---- Batched-PREFILL attention (qwen3): B chunks, one per seq at pos[b] ----
NB. Mirrors qw2_attention_bp but with qwen3 specifics: NO Q/K/V biases, per-head
NB. Q/K RMSNorm BEFORE RoPE (shared weight, size=head_dim), NEOX RoPE, Q scaled
NB. by 1/sqrt(head_dim), NO residual scale.  Q/K/V projections + RoPE batched
NB. across B*c rows; scores/softmax/V per-sequence.  Optional 5th arg = lens.
qw3_attention_bp =: 4 : 0
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
  n_heads =. qw3_bd_n_heads block_data
  head_dim =. qw3_bd_head_dim block_data
  n_heads_kv =. qw3_bd_n_heads_kv block_data
  n_groups =. n_heads % n_heads_kv
  half =. <. head_dim % 2
  eff_seq =. > 1 { kv_meta

  hidden_flat =. rms_norm_rows ((< mi_rms_eps mi) , (< (qw3_bd_attn_norm block_data)) , <(((B*c) , emb_len) $ , hidden))

  NB. Batched Q,K,V projections (|: hidden_flat hoisted once — 3 transposes
  NB. were materializing 3 copies)
  thin =. |: hidden_flat
  qv =. |: ((qw3_bd_attn_q block_data) (+/ .* ) thin)
  kv =. |: ((qw3_bd_attn_k block_data) (+/ .* ) thin)
  vv =. |: ((qw3_bd_attn_v block_data) (+/ .* ) thin)
  Q =. (B, c, n_heads, head_dim) $ , qv
  K =. (B, c, n_heads_kv, head_dim) $ , kv
  V =. (B, c, n_heads_kv, head_dim) $ , vv

  NB. Per-head RMSNorm on Q/K BEFORE RoPE (B*c*n_heads rows)
  Qf =. ((B*c*n_heads) , head_dim) $ , Q
  Qf =. rms_norm_rows ((< mi_rms_eps mi) , (< qw3_bd_q_norm block_data) , <Qf)
  Q =. (B, c, n_heads, head_dim) $ , Qf
  Kf =. ((B*c*n_heads_kv) , head_dim) $ , K
  Kf =. rms_norm_rows ((< mi_rms_eps mi) , (< qw3_bd_k_norm block_data) , <Kf)
  K =. (B, c, n_heads_kv, head_dim) $ , Kf

  NB. NEOX RoPE at the (B, c) positions pos[b]+i.c. cos/sin tables + expansions
  NB. are layer-invariant (depend only on pos) — hoisted once in run_blocks_bp,
  NB. threaded through y as <cos_all; sin_all; cos_expq; sin_expq; cos_expk;
  NB. sin_expk>.
  cos_all =. > 0 { rope
  sin_all =. > 1 { rope
  cos_expq =. > 2 { rope
  sin_expq =. > 3 { rope
  cos_expk =. > 4 { rope
  sin_expk =. > 5 { rope
  Qa =. half {. "1 Q
  Qb =. half }. "1 Q
  Qa_out =. (Qa * cos_expq) - (Qb * sin_expq)
  Qb_out =. (Qa * sin_expq) + (Qb * cos_expq)
  Q =. (B, c, n_heads, head_dim) $ , (Qa_out ,"1 Qb_out)
  Ka =. half {. "1 K
  Kb =. half }. "1 K
  Ka_out =. (Ka * cos_expk) - (Kb * sin_expk)
  Kb_out =. (Ka * sin_expk) + (Kb * cos_expk)
  K =. (B, c, n_heads_kv, head_dim) $ , (Ka_out ,"1 Kb_out)

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
    NB. Softmax directly on the 3D scores2 (the old scores_f flatten +
    NB. softmax_g2 re-group were 2 redundant reshape copies per layer)
    max_sf =. >./"1 scores2
    exp_sf =. ^ (scores2 - max_sf)
    softmax_f =. exp_sf % +/"1 exp_sf
    attn_out =. attn_out , <((c, n_heads*head_dim) $ , (1 0 2 |: ((n_heads, c, head_dim) $ , (softmax_f (+/ .* "2) (1 0 2 |: ((win, n_heads_kv, head_dim) $ , ((base_b + i. win) { v_cache_g)))))))
    b =. b + 1
  end.
  attn_all =. > attn_out
  attn_result =. |: ((qw3_bd_attn_o block_data) (+/ .* ) |: (((B*c) , (n_heads*head_dim)) $ , attn_all))
  attn_result =. (B, c, emb_len) $ , attn_result
  (<attn_result)
)

NB. ---- Batched-prefill block forward (qwen3) ----
qw3_block_forward_bp =: 4 : 0
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
  attn_result =. hidden qw3_attention_bp y
  attn_out =. > 0 { attn_result   NB. (B, c, emb)
  sa_out =. attn_out + hidden
  sa_flat =. ((B*c) , emb_len) $ , sa_out
  ffn_in =. rms_norm_rows ((< mi_rms_eps mi) , (< (qw3_bd_ff_norm block_data)) , <sa_flat)
  ft =. |: ffn_in   NB. |: ffn_in hoisted once — 2 transposes were materializing 2 copies
  gate =. |: ((qw3_bd_ff_gate block_data) (+/ .* ) ft)
  up =. |: ((qw3_bd_ff_up block_data) (+/ .* ) ft)
  ffn_raw =. |: ((qw3_bd_ff_down block_data) (+/ .* ) |: (gate swiglu up))
  output_flat =. ffn_raw + sa_flat
  output =. (B, c, emb_len) $ , output_flat
  (<output)
)

NB. ---- Run all blocks for B sequences (one CHUNK each at pos[b]) ----
qw3_run_blocks_bp =: 4 : 0
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
  pos_bc_flat =. (B*c) $ , (pos +/ i. c)
  cos_all =. pos_bc_flat { mi_cos_tab mi
  sin_all =. pos_bc_flat { mi_sin_tab mi
  cos_expq =. (B, c, n_heads, half) $ , ((0 2 1) |: (((B*c) , half , n_heads) $ , (cos_all (*/) (n_heads $ 1))))
  sin_expq =. (B, c, n_heads, half) $ , ((0 2 1) |: (((B*c) , half , n_heads) $ , (sin_all (*/) (n_heads $ 1))))
  cos_expk =. (B, c, n_heads_kv, half) $ , ((0 2 1) |: (((B*c) , half , n_heads_kv) $ , (cos_all (*/) (n_heads_kv $ 1))))
  sin_expk =. (B, c, n_heads_kv, half) $ , ((0 2 1) |: (((B*c) , half , n_heads_kv) $ , (sin_all (*/) (n_heads_kv $ 1))))
  rope =. (<cos_all) , (<sin_all) , (<cos_expq) , (<sin_expq) , (<cos_expk) , (<sin_expk)
  NB. (<pos), (<mi), <lens are layer-invariant — box once, reuse per layer.
  bfb_pre =. ((<pos) , (<mi) , <lens)
  b =. 0
  block_data_list =. llm_block_data llm
  while. b < block_count do.
    block_data =. > b { block_data_list
    result =. state qw3_block_forward_bp ((<block_data) , (<b) , bfb_pre , <rope)
    state =. > 0 { result
    b =. b + 1
  end.
  <state
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
  ct_tmpl_g =: 'tokenizer.chat_template' kv_string (0 1 { kv_result)
  tokenizer =. build_gpt2_tokenizer kv_result
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
  block_data =. qw3_pre_build_block_data llm
  llm =. llm , <block_data
  llm =. llm , <kv_data
)

NB. ---- Single-token inference ----
NB. Usage: llm qw3_infer (text ; <temp;k;p;min_p>)
NB.        Simple (default params): llm qw3_infer_simple text
qw3_infer =: 4 : 0
  llm =. x
  args =. infer_args y
  text =. > 0 { args
  temp =. > 1 { args
  k =. > 2 { args
  p =. > 3 { args
  min_p =. > 4 { args

  tokens =. gpt2_tokenize (<llm) , <text
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
    pre_s =. 6!:2 'result =. hidden qw3_run_blocks (<llm) , <0'
    hidden =. > 0 { result
  else.
    emb_all =. scale * |: (tok_list {"1 emb_w)
    pre_s =. 6!:2 'result_b =. emb_all qw3_run_blocks_b ((<llm) , <0)'
    h_b =. > 0 { result_b
    hidden =. > (n_tokens - 1) { h_b
  end.
  logits =. output_head ((< mi_rms_eps mi) , (<output_norm_w) , (<emb_w) , <hidden)
  report_prefill (pre_s , n_tokens)
  pred_tok =. sample_from ((<temp) , (<k) , (<p) , (<min_p) , <logits)
  decoded =. gpt2_detokenize (<llm) , <pred_tok
  tokens ; pred_tok ; decoded ; logits
)

NB. ---- Multi-token generation ----
NB. Usage: llm qw3_generate (text ; max_steps ; <temp;k;p;min_p>)
NB.        Simple (default params): llm qw3_generate_simple (text ; max_steps)
NB. Generation is the UNIFIED gen_loop_core (llm_core.ijs) — per-arch
NB. differences (embedding scale, run_blocks/run_blocks_b) are dispatched by
NB. llm_arch. Fresh mode: kv_create + batched prefill; resume mode (chat
NB. sessions): incremental prefill of the new segment. Stop token not appended.
qw3_generate =: 4 : 0
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
  prompt =. qw3_chat_prompt messages
  tokens =. gpt2_tokenize (<llm) , <prompt
  stop =. qw3_stop_tokens llm
  L =. # , > tokens
  output =. llm gen_loop_core (tokens ; '' ; max_steps ; temp ; k ; p ; min_p ; <stop)
  gen =. L }. output
  gpt2_detokenize (<llm) , <gen
)

NB. ---- Batched generation: B independent prompts in parallel ----
NB. Usage: llm qw3_generate_batch (prompts ; max_steps ; <temp;k;p;min_p>)
qw3_generate_batch =: 4 : 0
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
    prompt =. qw3_chat_prompt messages
    tokens =. gpt2_tokenize (llm_box , <prompt)
    tok_list =. , > tokens
    prompts_tok =. prompts_tok , <tok_list
    prompts_len =. prompts_len , <(# tok_list)
    i =. i + 1
  end.
  stop =. qw3_stop_tokens llm
  kv_batch_g =: B
  output =. llm gen_loop_batch (prompts_tok ; max_steps ; temp ; k ; p ; min_p ; <stop)
  answers =. ''
  i =. 0
  while. i < B do.
    L =. > i { prompts_len
    gen =. (L) }. (> i { output)
    answers =. answers , <(gpt2_detokenize (llm_box , <gen))
    i =. i + 1
  end.
  answers
)

NB. ---- Simple wrappers (default greedy/top-p params) ----
NB. llm qw3_infer_simple text            | llm qw3_generate_simple (text ; n)
qw3_infer_simple =: qw3_infer (] ; (<0 0 0.95 0.0)"_)
qw3_generate_simple =: qw3_generate (0&{ , 1&{ , (<0 0 0.95 0.0)"_)

NB. ---- Chat-template support (Phase 1.1) ----
NB. y = messages: boxed list of message boxes; each = <role ; content>.
NB. Renders the qwen3 chat template for the no-tools case: each message as
NB. '<|im_start|>role\ncontent<|im_end|>\n', then the generation prompt
NB. '<|im_start|>assistant\n'. The qwen3 template adds NO default system
NB. message (unlike qwen2). No BOS (gpt2 tokenize adds none).

NB. Stop token: <|im_end|> (EOS).
qw3_stop_tokens =: 3 : 0
  tk =. llm_tokenizer y
  tokenizer_eos_g tk
)
