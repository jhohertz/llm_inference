NB. ================================================================
NB. LFM2-350M (lfm2 arch) — hybrid: 6 attention layers (per-head Q/K
NB. RMSNorm qwen3-style, NEOX RoPE) + 10 shortconv layers (conv1d with a
NB. b*x gate). Dense SwiGLU FFN every layer. TIED embeddings (no
NB. output.weight — lm_head = token_embd.weight). New pre-tokenizer
NB. 'lfm2' (llama.cpp maps it to the llama3 regex pre, add_bos=true).
NB.
NB. Shortconv block (llama.cpp src/models/lfm2.cpp, ggml_ssm_conv):
NB.   normed -> in_proj (3*emb) -> split b,c,x (each emb)
NB.   bx = b*x; input = conv_state(2 rows) , bx
NB.   conv_out[t,c] = sum_tap kernel[c,tap] * input[t+tap,c]  (l_cache=3 taps)
NB.   y = c * conv_out; out = out_proj(y); residual + hidden
NB.   conv_state update: last 2 rows of input
NB. Depends on: llm_core.ijs, kernels.ijs, gguf.ijs, kv_cache.ijs,
NB.           sampler.ijs, tokenizer_gpt2.ijs
NB. ================================================================
coclass 'inference'
require 'llm/inference/util/llm_core'
require 'llm/inference/kernels/jfloat'
require 'llm/inference/util/kv_cache'
require 'llm/inference/util/sampler'
require 'llm/inference/tokenizers/tokenizer_gpt2'

NB. ---- KV helpers ----
lf2_kv_uint =: 4 : 0
  key =. x
  data =. y
  key kv_uint data
)

lf2_kv_float =: 4 : 0
  key =. x
  data =. y
  key kv_float data
)

NB. ---- Extract LFM2 model info from KV pairs ----
NB. mi = <block_count; context_len; emb_len; n_heads; n_heads_kv; head_dim;
NB.     rope_freq; vocab_size; rms_eps; n_ff>
NB. head_count_kv is an ARRAY KV (0 for shortconv layers, 8 for attention) —
NB. kv_uint returns _1; take the max for n_heads_kv.
lf2_extract_hparams =: 3 : 0
  data =. y
  block_count =. 'lfm2.block_count' lf2_kv_uint data
  context_length =. 'lfm2.context_length' lf2_kv_uint data
  emb_len =. 'lfm2.embedding_length' lf2_kv_uint data
  n_heads =. 'lfm2.attention.head_count' lf2_kv_uint data
  hc_kv =. 'lfm2.attention.head_count_kv' kv_array data
  n_heads_kv =. >./ hc_kv
  rope_freq =. 'lfm2.rope.freq_base' lf2_kv_float data
  vocab_size =. 'lfm2.vocab_size' lf2_kv_uint data
  rms_eps =. 'lfm2.attention.layer_norm_rms_epsilon' lf2_kv_float data
  n_ff =. 'lfm2.feed_forward_length' lf2_kv_uint data
  key_len =. 'lfm2.attention.key_length' lf2_kv_uint data
  if. key_len <: 0 do. key_len =. emb_len % n_heads end.
  head_dim =. key_len
  (<"0) block_count , context_length , emb_len , n_heads , n_heads_kv , head_dim , rope_freq , vocab_size , rms_eps , n_ff
)

NB. ---- block_data accessors ----
NB. ATTENTION layers (15 items): <attn_norm; attn_q; attn_k; attn_v; attn_o;
NB.   ffn_norm; ffn_gate; ffn_up; ffn_down; q_norm; k_norm; n_heads; head_dim;
NB.   n_heads_kv; is_conv=0>
NB. SHORTCONV layers (9 items): <attn_norm; ffn_norm; ffn_gate; ffn_up;
NB.   ffn_down; conv_w; in_proj; out_proj; is_conv=1>
NB. is_conv is always the LAST item (lf2_bd_is_conv = >@({:)).
lf2_bd_attn_norm =: >@(0&{)
lf2_bd_attn_q    =: >@(1&{)
lf2_bd_attn_k    =: >@(2&{)
lf2_bd_attn_v    =: >@(3&{)
lf2_bd_attn_o    =: >@(4&{)
lf2_bd_ffn_norm  =: >@(5&{)
lf2_bd_ffn_gate  =: >@(6&{)
lf2_bd_ffn_up    =: >@(7&{)
lf2_bd_ffn_down  =: >@(8&{)
lf2_bd_q_norm    =: >@(9&{)
lf2_bd_k_norm    =: >@(10&{)
lf2_bd_n_heads   =: >@(11&{)
lf2_bd_head_dim  =: >@(12&{)

lf2_bd_n_heads_kv=: >@(13&{)
lf2_bd_is_conv   =: >@({:)   NB. last item (0 = attention, 1 = shortconv)
lf2_cv_ffn_norm  =: >@(1&{)
lf2_cv_ffn_gate  =: >@(2&{)
lf2_cv_ffn_up    =: >@(3&{)
lf2_cv_ffn_down  =: >@(4&{)
lf2_bd_conv      =: >@(5&{)
lf2_bd_in_proj   =: >@(6&{)
lf2_bd_out_proj  =: >@(7&{)

NB. ---- Build one layer's block_data ----
NB. x = llm, y = layer index. Shortconv layers carry shortconv.conv.weight.
lf2_build_block_data =: 4 : 0
  llm =. x
  b =. y
  p =. 'blk.' , (": b) , '.'
  attn_norm =. (p , 'attn_norm.weight') get_tensor_cached_d llm
  ffn_norm =. (p , 'ffn_norm.weight') get_tensor_cached_d llm
  ffn_gate =. (p , 'ffn_gate.weight') get_tensor_cached_d llm
  ffn_up =. (p , 'ffn_up.weight') get_tensor_cached_d llm
  ffn_down =. (p , 'ffn_down.weight') get_tensor_cached_d llm
  is_conv =. 0
  if. 0 < # ((p , 'shortconv.conv.weight') get_tensor_cached_d llm) do. is_conv =. 1 end.
  if. is_conv do.
    conv_w =. (p , 'shortconv.conv.weight') get_tensor_cached_d llm   NB. (emb, l_cache)
    in_proj =. (p , 'shortconv.in_proj.weight') get_tensor_cached_d llm   NB. (3*emb, emb)
    out_proj =. (p , 'shortconv.out_proj.weight') get_tensor_cached_d llm   NB. (emb, emb)
    (<attn_norm) , (<ffn_norm) , (<ffn_gate) , (<ffn_up) , (<ffn_down) , (<conv_w) , (<in_proj) , (<out_proj) , <is_conv
  else.
    q_w =. (p , 'attn_q.weight') get_tensor_cached_d llm
    k_w =. (p , 'attn_k.weight') get_tensor_cached_d llm
    v_w =. (p , 'attn_v.weight') get_tensor_cached_d llm
    o_w =. (p , 'attn_output.weight') get_tensor_cached_d llm
    q_norm =. (p , 'attn_q_norm.weight') get_tensor_cached_d llm
    k_norm =. (p , 'attn_k_norm.weight') get_tensor_cached_d llm
    mi =. llm_mi llm
    n_heads =. mi_n_heads mi
    head_dim =. mi_head_dim mi
    n_heads_kv =. mi_n_heads_kv mi
      (<attn_norm) , (<q_w) , (<k_w) , (<v_w) , (<o_w) , (<ffn_norm) , (<ffn_gate) , (<ffn_up) , (<ffn_down) , (<q_norm) , (<k_norm) , (<n_heads) , (<head_dim) , (<n_heads_kv) , <is_conv
  end.
)

lf2_pre_build_block_data =: 3 : 0
  llm =. y
  (<llm) lf2_build_block_data each i. mi_block_count llm_mi llm
)

NB. ---- Expand KV heads to n_heads (GQA) ----

NB. ---- Shortconv recurrent state cache ----
NB. lf2_conv_meta = <n_conv; d_conv; emb>; lf2_conv_cache_g = ONE FLAT array
NB. (n_conv * kv_batch_g * d_conv * emb) conv state, positions-leading
NB. (layer ord; seq), amended IN-PLACE via list-selector (mirrors qwen35
NB. rs_conv_g — the old boxed-per-layer amend unboxed a refcount-2 global and
NB. COPIED the whole batch state per write; the flat amend is ~78x faster).
NB. BATCH dimension = kv_batch_g so batched decode keeps per-sequence states.
NB. d_conv = l_cache - 1 = 2. Monadic, no threading — one global, amended in place.
lf2_conv_meta =: ''
lf2_conv_cache_g =: ''
lf2_conv_layers =: 0 $ 0
lf2_conv_batch_g =: 1

lf2_conv_slice =: 3 : '(> 1 { y) * (> 2 { y)'

lf2_conv_create =: 3 : 0
  n_conv =. > 0 { y
  d_conv =. > 1 { y
  emb =. > 2 { y
  NB. Reuse if same dims + same batch (mirror kv_create) — a repeat prefill
  NB. per sequence must NOT reset earlier sequences' states.
  if. -. '' -: lf2_conv_meta do.
    if. (n_conv = > 0 { lf2_conv_meta) *. (d_conv = > 1 { lf2_conv_meta) *. (emb = > 2 { lf2_conv_meta) *. (lf2_conv_batch_g = kv_batch_g) do.
      '' return.
    end.
  end.
  lf2_conv_meta =: (<n_conv) , (<d_conv) , (<emb)
  lf2_conv_cache_g =: (n_conv * kv_batch_g * d_conv * emb) $ 0.0
  lf2_conv_batch_g =: kv_batch_g
  ''
)

NB. Force zero the conv cache (fresh generation start). The guarded create in
NB. lf2_run_blocks_b keeps per-sequence prefill from resetting sibling states,
NB. but a new generation must NOT inherit the previous one's recurrent state.
lf2_conv_reset =: 3 : 0
  n_conv =. > 0 { y
  d_conv =. > 1 { y
  emb =. > 2 { y
  lf2_conv_meta =: (<n_conv) , (<d_conv) , (<emb)
  lf2_conv_cache_g =: (n_conv * kv_batch_g * d_conv * emb) $ 0.0
  lf2_conv_batch_g =: kv_batch_g
  ''
)

lf2_conv_read =: 3 : 0
  layer =. y
  ord =. lf2_conv_layers i. layer
  cs =. lf2_conv_slice lf2_conv_meta
  cb =. ((ord * lf2_conv_batch_g) + kv_seq_g) * cs
  ((> 1 { lf2_conv_meta) , (> 2 { lf2_conv_meta)) $ , ((cb + i. cs) { lf2_conv_cache_g)
)

lf2_conv_write =: 3 : 0
  layer =. > 0 { y
  new_conv =. > 1 { y
  ord =. lf2_conv_layers i. layer
  cs =. lf2_conv_slice lf2_conv_meta
  cb =. ((ord * lf2_conv_batch_g) + kv_seq_g) * cs
  lf2_conv_cache_g =: (, new_conv) (cb + i. cs)} lf2_conv_cache_g
  ''
)

NB. Batched-decode variants: explicit sequence index (decode keeps kv_seq_g=0)
lf2_conv_read_b =: 3 : 0
  layer =. > 0 { y
  b =. > 1 { y
  ord =. lf2_conv_layers i. layer
  cs =. lf2_conv_slice lf2_conv_meta
  cb =. ((ord * lf2_conv_batch_g) + b) * cs
  ((> 1 { lf2_conv_meta) , (> 2 { lf2_conv_meta)) $ , ((cb + i. cs) { lf2_conv_cache_g)
)

lf2_conv_write_b =: 3 : 0
  layer =. > 0 { y
  new_conv =. > 1 { y
  b =. > 2 { y
  ord =. lf2_conv_layers i. layer
  cs =. lf2_conv_slice lf2_conv_meta
  cb =. ((ord * lf2_conv_batch_g) + b) * cs
  lf2_conv_cache_g =: (, new_conv) (cb + i. cs)} lf2_conv_cache_g
  ''
)

NB. ---- Attention layer forward (batched) ----
NB. x = hidden (L, emb); y = <block_data; mi; layer; start_pos>
NB. LFM2 attention layers are qwen3-style: per-head Q/K RMSNorm before RoPE,
NB. NEOX RoPE, scale 1/sqrt(head_dim), GQA, no QKV biases.
lf2_attention_b =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  mi =. > 1 { y
  layer =. > 2 { y
  start_pos =. > 3 { y
  L =. {. $ hidden
  n_embd =. {: $ hidden
  n_heads =. lf2_bd_n_heads block_data
  head_dim =. lf2_bd_head_dim block_data
  n_heads_kv =. lf2_bd_n_heads_kv block_data
  n_groups =. n_heads % n_heads_kv
  half =. <. head_dim % 2

  NB. Attention norm per row
  attn_norm_w =. lf2_bd_attn_norm block_data
  hidden =. rms_norm_rows ((< mi_rms_eps mi) , (< attn_norm_w) , <hidden)

  NB. Separate Q,K,V batched projections (|: hidden hoisted once — 3 transposes
  NB. of the (L,emb) hidden were materializing 3 copies)
  thin =. |: hidden
  qv =. |: ((lf2_bd_attn_q block_data) (+/ .* ) thin)   NB. (L, n_heads*hd)
  kv =. |: ((lf2_bd_attn_k block_data) (+/ .* ) thin)   NB. (L, n_kv*hd)
  vv =. |: ((lf2_bd_attn_v block_data) (+/ .* ) thin)   NB. (L, n_kv*hd)

  NB. NOTE: `(L, n_heads, hd) $ , qv` — the ravel is REQUIRED: J's `$` on a
  NB. rank-2/3 right operand is rank-dependent (`(2 3) $ (4,3)` → (2,3,3)),
  NB. so flatten-then-reshape is the safe form, not a removable copy.
  Q =. (L, n_heads, head_dim) $ , qv
  K =. (L, n_heads_kv, head_dim) $ , kv
  V =. (L, n_heads_kv, head_dim) $ , vv

  NB. Per-head RMSNorm on Q and K BEFORE RoPE (shared weight, size=head_dim)
  Qf =. ((L * n_heads) , head_dim) $ , Q
  Qf =. rms_norm_rows ((< mi_rms_eps mi) , (< lf2_bd_q_norm block_data) , <Qf)
  Q =. (L, n_heads, head_dim) $ , Qf
  Kf =. ((L * n_heads_kv) , head_dim) $ , K
  Kf =. rms_norm_rows ((< mi_rms_eps mi) , (< lf2_bd_k_norm block_data) , <Kf)
  K =. (L, n_heads_kv, head_dim) $ , Kf

  NB. NEOX RoPE batched (table-based): pairs (i, i+half) per row
  cos_all =. (start_pos + i. L) { mi_cos_tab mi    NB. (L, half)
  sin_all =. (start_pos + i. L) { mi_sin_tab mi
  Qa =. half {. "1 Q        NB. (L, n_heads, half) first half
  Qb =. half }. "1 Q        NB. (L, n_heads, half) second half
  cos_expq =. (0 2 1) |: ((L , half , n_heads) $ , (cos_all (*/) (n_heads $ 1)))
  sin_expq =. (0 2 1) |: ((L , half , n_heads) $ , (sin_all (*/) (n_heads $ 1)))
  Qa_out =. (Qa * cos_expq) - (Qb * sin_expq)
  Qb_out =. (Qa * sin_expq) + (Qb * cos_expq)
  Q =. (L, n_heads, head_dim) $ , (Qa_out ,"1 Qb_out)
  Ka =. half {. "1 K        NB. (L, n_heads_kv, half)
  Kb =. half }. "1 K
  cos_expk =. (0 2 1) |: ((L , half , n_heads_kv) $ , (cos_all (*/) (n_heads_kv $ 1)))
  sin_expk =. (0 2 1) |: ((L , half , n_heads_kv) $ , (sin_all (*/) (n_heads_kv $ 1)))
  Ka_out =. (Ka * cos_expk) - (Kb * sin_expk)
  Kb_out =. (Ka * sin_expk) + (Kb * cos_expk)
  K =. (L, n_heads_kv, head_dim) $ , (Ka_out ,"1 Kb_out)

  NB. Scale Q by 1/sqrt(head_dim)
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
  NB. 0..start_pos+L-1. Keep scores group-major: tile the 2D mask r-major
  NB. (row r*L+t needs mask row t) and subtract with rank over the kv-head
  NB. frame — no (n_heads, L, tot) 3D mask and no scores re-shape copy.
  key_pos =. i. (start_pos + L)
  q_pos =. start_pos + i. L
  mask_2d =. q_pos </ key_pos
  NB. Fast r-major boolean tile via the (*/) broadcast (the cyclic boolean
  NB. reshape (n_groups,L,ctx)$mask_2d is ~100x slower); scaled at subtract.
  mask_g2 =. ((n_groups * L) , start_pos + L) $ , (2 0 1 |: (mask_2d (*/) (n_groups $ 1)))
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
  attn_o_w =. lf2_bd_attn_o block_data
  attn_out =. |: (attn_o_w (+/ .* ) |: ((L, n_heads * head_dim) $ , (1 0 2 |: attn_raw)))   NB. (L, emb)

  (<attn_out)
)

NB. ---- Attention block forward (batched) ----
NB. x = hidden (L, emb); y = <block_data; mi; layer; start_pos>
lf2_block_forward_b =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  mi =. > 1 { y
  layer =. > 2 { y
  start_pos =. > 3 { y
  input =. hidden
  attn_result =. hidden lf2_attention_b ((<block_data) , (<mi) , (<layer) , (<start_pos))
  attn_out =. > 0 { attn_result
  sa_out =. attn_out + input
  ffn_norm_w =. lf2_bd_ffn_norm block_data
  ffn_in =. rms_norm_rows ((< mi_rms_eps mi) , (< ffn_norm_w) , <sa_out)
  gate =. |: ((lf2_bd_ffn_gate block_data) (+/ .* ) |: ffn_in)   NB. (L, n_ff)
  up =. |: ((lf2_bd_ffn_up block_data) (+/ .* ) |: ffn_in)
  ffn_raw =. |: ((lf2_bd_ffn_down block_data) (+/ .* ) |: (gate swiglu up))
  output =. ffn_raw + sa_out
  (<output)
)

NB. ---- Shortconv block forward (batched) ----
NB. x = hidden (L, emb); y = <block_data; mi; layer>
lf2_conv_forward_b =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  mi =. > 1 { y
  layer =. > 2 { y
  L =. {. $ hidden
  emb =. {: $ hidden

  NB. Operator norm
  attn_norm_w =. lf2_bd_attn_norm block_data
  normed =. rms_norm_rows ((< mi_rms_eps mi) , (< attn_norm_w) , <hidden)

  NB. in_proj -> split b,c,x chunks (each emb)
  bcx =. |: ((lf2_bd_in_proj block_data) (+/ .* ) |: normed)   NB. (L, 3*emb)
  b =. (emb {. "1 bcx)   NB. (L, emb)
  c =. (emb {. "1 (emb }."1 bcx))   NB. (L, emb)
  x =. (2 * emb) }."1 bcx   NB. (L, emb)
  bx =. b * x   NB. (L, emb)

  NB. Conv1d: input = [conv_state(2); bx], kernel (emb, l_cache=3),
  NB. conv_out[t,c] = sum_tap kernel[c,tap] * input[t+tap,c]
  conv_state =. lf2_conv_read layer   NB. (2, emb)
  input =. conv_state , bx   NB. (2+L, emb)
  conv_w =. lf2_bd_conv block_data   NB. (emb, 3)
  conv_out =. (L , emb) $ 0
  conv_out =. conv_out + ((L , emb) $ (0 {"1 conv_w)) * ((0 + i. L) { input)
  conv_out =. conv_out + ((L , emb) $ (1 {"1 conv_w)) * ((1 + i. L) { input)
  conv_out =. conv_out + ((L , emb) $ (2 {"1 conv_w)) * ((2 + i. L) { input)

  NB. Gate: y = c * conv_out, then out_proj
  y =. c * conv_out   NB. (L, emb)
  out =. |: ((lf2_bd_out_proj block_data) (+/ .* ) |: y)   NB. (L, emb)

  NB. Residual (prev_cur)
  sa_out =. out + hidden

  NB. Update conv state: last 2 rows of input
  new_conv =. (L + i. 2) { input
  lf2_conv_write ((<layer) , <new_conv)

  NB. Post-attention norm + SwiGLU FFN (no inner residual) — conv-layout accessors
  ffn_norm_w =. lf2_cv_ffn_norm block_data
  ffn_in =. rms_norm_rows ((< mi_rms_eps mi) , (< ffn_norm_w) , <sa_out)
  tfin =. |: ffn_in
  gate_f =. |: ((lf2_cv_ffn_gate block_data) (+/ .* ) tfin)
  up_f =. |: ((lf2_cv_ffn_up block_data) (+/ .* ) tfin)
  ffn_raw =. |: ((lf2_cv_ffn_down block_data) (+/ .* ) |: (gate_f swiglu up_f))
  output =. ffn_raw + sa_out
  <output
)

NB. ---- Batched run all blocks ----
NB. x = hidden (L, emb); y = <llm; start_pos> (start_pos=0 -> fresh: conv state
NB. zeroed; start_pos>0 -> resume: attention cache prefix + conv state persist).
lf2_run_blocks_b =: 4 : 0
  input =. x
  args =. y
  llm =. > 0 { args
  start_pos =. > 1 { args
  mi =. llm_mi llm
  head_dim =. mi_head_dim mi
  n_heads_kv =. mi_n_heads_kv mi
  block_count =. mi_block_count mi
  ctx_len =. mi_context_len mi
  emb_len =. mi_emb_len mi
  state =. input
  if. 0 = # kv_meta do.
    kv_create ((<block_count) , (<ctx_len) , (<n_heads_kv) , (<head_dim))
  end.
  NB. Shortconv state: fresh prefill zeroes it (start_pos=0); resume keeps it.
  if. 0 = start_pos do.
    lf2_conv_create ((<lf2_n_conv_g) , (<2) , (<emb_len))
  else.
    if. 0 = # lf2_conv_meta do.
      lf2_conv_create ((<lf2_n_conv_g) , (<2) , (<emb_len))
    end.
  end.
  b =. 0
  block_data_list =. llm_block_data llm
  while. b < block_count do.
    block_data =. > b { block_data_list
    if. lf2_bd_is_conv block_data do.
      result =. state lf2_conv_forward_b ((<block_data) , (<mi) , (<b))
    else.
      result =. state lf2_block_forward_b ((<block_data) , (<mi) , (<b) , (<start_pos))
    end.
    state =. > 0 { result
    b =. b + 1
  end.
  <state
)

NB. ---- Batched-DECODE attention (B sequences, ONE token each at pos[b]) ----
NB. LFM2 attention = qwen3-style (per-head Q/K norm, NEOX RoPE, GQA, no biases).
NB. x = hidden (B, emb); y = <block_data; pos; mi; layer>
lf2_attention_bd =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  pos =. > 1 { y
  mi =. > 2 { y
  layer =. > 3 { y
  B =. {. $ hidden
  n_heads =. lf2_bd_n_heads block_data
  head_dim =. lf2_bd_head_dim block_data
  n_heads_kv =. lf2_bd_n_heads_kv block_data
  n_groups =. n_heads % n_heads_kv
  half =. <. head_dim % 2

  NB. Attention norm per row
  attn_norm_w =. lf2_bd_attn_norm block_data
  hidden =. rms_norm_rows ((< mi_rms_eps mi) , (< attn_norm_w) , <hidden)

  NB. Batched Q,K,V projections (weight-read amortized across B; |: hidden hoisted)
  thin =. |: hidden
  qv =. |: ((lf2_bd_attn_q block_data) (+/ .* ) thin)   NB. (B, n_heads*hd)
  kv =. |: ((lf2_bd_attn_k block_data) (+/ .* ) thin)   NB. (B, n_kv*hd)
  vv =. |: ((lf2_bd_attn_v block_data) (+/ .* ) thin)   NB. (B, n_kv*hd)

  Q =. (B, n_heads, head_dim) $ , qv
  K =. (B, n_heads_kv, head_dim) $ , kv
  V =. (B, n_heads_kv, head_dim) $ , vv

  NB. Per-head RMSNorm on Q and K BEFORE RoPE (shared weight, size=head_dim)
  Qf =. ((B * n_heads) , head_dim) $ , Q
  Qf =. rms_norm_rows ((< mi_rms_eps mi) , (< lf2_bd_q_norm block_data) , <Qf)
  Q =. (B, n_heads, head_dim) $ , Qf
  Kf =. ((B * n_heads_kv) , head_dim) $ , K
  Kf =. rms_norm_rows ((< mi_rms_eps mi) , (< lf2_bd_k_norm block_data) , <Kf)
  K =. (B, n_heads_kv, head_dim) $ , Kf

  NB. NEOX RoPE batched at the B positions (table-based)
  cos_all =. pos { mi_cos_tab mi
  sin_all =. pos { mi_sin_tab mi
  cos_expq =. (0 2 1) |: ((B , half , n_heads) $ , (cos_all (*/) (n_heads $ 1)))
  sin_expq =. (0 2 1) |: ((B , half , n_heads) $ , (sin_all (*/) (n_heads $ 1)))
  cos_expk =. (0 2 1) |: ((B , half , n_heads_kv) $ , (cos_all (*/) (n_heads_kv $ 1)))
  sin_expk =. (0 2 1) |: ((B , half , n_heads_kv) $ , (sin_all (*/) (n_heads_kv $ 1)))
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

  NB. Scale Q by 1/sqrt(head_dim)
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
  attn_result =. |: ((lf2_bd_attn_o block_data) (+/ .* ) |: attn_all)   NB. (B, emb)
  (<attn_result)
)

NB. ---- Batched-DECODE attention block forward ----
lf2_block_forward_bd =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  pos =. > 1 { y
  mi =. > 2 { y
  layer =. > 3 { y
  input =. hidden
  attn_result =. hidden lf2_attention_bd ((<block_data) , (<pos) , (<mi) , (<layer))
  attn_out =. > 0 { attn_result
  sa_out =. attn_out + input
  ffn_norm_w =. lf2_bd_ffn_norm block_data
  ffn_in =. rms_norm_rows ((< mi_rms_eps mi) , (< ffn_norm_w) , <sa_out)
  gate =. |: ((lf2_bd_ffn_gate block_data) (+/ .* ) |: ffn_in)   NB. (B, n_ff)
  up =. |: ((lf2_bd_ffn_up block_data) (+/ .* ) |: ffn_in)
  ffn_raw =. |: ((lf2_bd_ffn_down block_data) (+/ .* ) |: (gate swiglu up))
  output =. ffn_raw + sa_out
  (<output)
)

NB. ---- Batched-DECODE shortconv block forward (B sequences, 1 token each) ----
lf2_conv_forward_bd =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  pos =. > 1 { y
  mi =. > 2 { y
  layer =. > 3 { y
  B =. {. $ hidden
  emb =. {: $ hidden

  NB. Operator norm
  attn_norm_w =. lf2_bd_attn_norm block_data
  normed =. rms_norm_rows ((< mi_rms_eps mi) , (< attn_norm_w) , <hidden)

  NB. in_proj -> split b,c,x chunks (each emb)
  bcx =. |: ((lf2_bd_in_proj block_data) (+/ .* ) |: normed)   NB. (B, 3*emb)
  b =. (emb {. "1 bcx)   NB. (B, emb)
  c =. (emb {. "1 (emb }."1 bcx))   NB. (B, emb)
  x =. (2 * emb) }."1 bcx   NB. (B, emb)
  bx =. b * x   NB. (B, emb)

  NB. Conv1d per sequence: input = [conv_state(2); bx] (3, emb), one token out,
  NB. state update = last 2 rows (mirror lf2_conv_forward_b with L=1).
  conv_w =. lf2_bd_conv block_data   NB. (emb, 3)
  outs =. ''
  bi =. 0
  while. bi < B do.
    conv_state =. lf2_conv_read_b ((<layer) , <bi)   NB. (2, emb)
    input =. conv_state , (bi { bx)   NB. (3, emb)
    conv_out =. (1 , emb) $ 0
    conv_out =. conv_out + ((1 , emb) $ (0 {"1 conv_w)) * ((0 + i. 1) { input)
    conv_out =. conv_out + ((1 , emb) $ (1 {"1 conv_w)) * ((1 + i. 1) { input)
    conv_out =. conv_out + ((1 , emb) $ (2 {"1 conv_w)) * ((2 + i. 1) { input)
    y =. ((bi + i. 1) { c) * conv_out   NB. (1, emb)
    new_conv =. (1 + i. 2) { input
    lf2_conv_write_b ((<layer) , (<new_conv) , <bi)
    outs =. outs , <y
    bi =. bi + 1
  end.

  NB. Gate: out_proj over all B, then residual
  y_all =. (B , emb) $ , > outs
  out =. |: ((lf2_bd_out_proj block_data) (+/ .* ) |: y_all)   NB. (B, emb)
  sa_out =. out + hidden

  NB. Post-attention norm + SwiGLU FFN (no inner residual) — conv-layout accessors
  ffn_norm_w =. lf2_cv_ffn_norm block_data
  ffn_in =. rms_norm_rows ((< mi_rms_eps mi) , (< ffn_norm_w) , <sa_out)
  tfin =. |: ffn_in
  gate_f =. |: ((lf2_cv_ffn_gate block_data) (+/ .* ) tfin)
  up_f =. |: ((lf2_cv_ffn_up block_data) (+/ .* ) tfin)
  ffn_raw =. |: ((lf2_cv_ffn_down block_data) (+/ .* ) |: (gate_f swiglu up_f))
  output =. ffn_raw + sa_out
  <output
)

NB. ---- Batched-DECODE run all blocks (B sequences, one token each at pos[b]) ----
lf2_run_blocks_bd =: 4 : 0
  input =. x
  args =. y
  llm =. > 0 { args
  pos =. > 1 { args
  mi =. llm_mi llm
  head_dim =. mi_head_dim mi
  n_heads_kv =. mi_n_heads_kv mi
  block_count =. mi_block_count mi
  ctx_len =. mi_context_len mi
  emb_len =. mi_emb_len mi
  state =. input
  if. 0 = # kv_meta do.
    kv_create ((<block_count) , (<ctx_len) , (<n_heads_kv) , (<head_dim))
  end.
  NB. Conv cache created during prefill (lf2_run_blocks_b, start_pos=0); for
  NB. decode it must exist with the batch dimension. If absent (direct decode),
  NB. allocate it.
  if. 0 = # lf2_conv_meta do.
    lf2_conv_create ((<lf2_n_conv_g) , (<2) , (<emb_len))
  end.
  b =. 0
  block_data_list =. llm_block_data llm
  while. b < block_count do.
    block_data =. > b { block_data_list
    if. lf2_bd_is_conv block_data do.
      result =. state lf2_conv_forward_bd ((<block_data) , (<pos) , (<mi) , (<b))
    else.
      result =. state lf2_block_forward_bd ((<block_data) , (<pos) , (<mi) , (<b))
    end.
    state =. > 0 { result
    b =. b + 1
  end.
  <state
)

NB. ---- Single-token run all blocks (1D hidden) ----
NB. x = hidden (emb,); y = <llm; pos>. Wraps to (1, emb) and runs the batched path.
NB. Returns <row> (boxed) — gen_loop_core's resume step does > 0 { result.
lf2_run_blocks =: 4 : 0
  input =. x
  args =. y
  llm =. > 0 { args
  pos =. > 1 { args
  input2 =. (1 , $ input) $ input
  result_b =. input2 lf2_run_blocks_b ((<llm) , <pos)
  state =. > 0 { result_b
  < > 0 { state
)

NB. ---- Load LFM2 GGUF into llm noun ----
lf2_load =: 3 : 0
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
  mi =. lf2_extract_hparams kvs_ctx
  rope_tables =. build_rope_tables ((< mi_context_len mi) , (< mi_head_dim mi) , (< mi_rope_freq mi))
  mi =. mi , rope_tables
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
  NB. Shortconv layer indices: head_count_kv array value 0 -> conv layer.
  hc_kv =. 'lfm2.attention.head_count_kv' kv_array kvs_ctx
  lf2_conv_layers =: I. 0 = hc_kv
  lf2_n_conv_g =: # lf2_conv_layers
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
  block_data =. lf2_pre_build_block_data llm
  llm =. llm , <block_data
)

NB. ---- Tokenize/detokenize (gpt2 BPE + llama3 pre, BOS prepended) ----
NB. pre 'lfm2' maps to the llama3 regex pre (llama.cpp PRE_TYPE_LLAMA3,
NB. add_bos=true) — gpt2_tokenize dispatches 'lfm2' to llama3_pre_tokenize.
NB. BOS token 1 (<|startoftext|>) is prepended here; the chat template's
NB. {{bos_token}} is rendered as '' (lf2_chat_prompt) so the stream matches
NB. llama.cpp (bos once).
lf2_tokenize =: 3 : 0
  llm_data =. input_llm y
  text =. input_text y
  tokenizer =. llm_tokenizer llm_data
  tokens =. gpt2_tokenize (<llm_data) , <text
  bos =. tokenizer_bos_g tokenizer
  if. bos > 0 do. tokens =. (<bos) , tokens end.
  tokens
)
lf2_detokenize =: 3 : 0
  gpt2_detokenize y
)

NB. ---- Single-token inference ----
NB. Usage: llm lf2_infer (text ; <temp;k;p;min_p>)
NB.        Simple (default params): llm lf2_infer_simple text
lf2_infer =: 4 : 0
  llm =. x
  args =. infer_args y
  text =. > 0 { args
  temp =. > 1 { args
  k =. > 2 { args
  p =. > 3 { args
  min_p =. > 4 { args

  tokens =. lf2_tokenize (<llm) , <text
  mi =. llm_mi llm
  emb_len =. mi_emb_len mi
  scale =. 1
  block_count =. mi_block_count mi
  ctx_len =. mi_context_len mi
  n_heads_kv =. mi_n_heads_kv mi
  head_dim =. mi_head_dim mi
  emb_w =. 'token_embd.weight' get_tensor_cached_d llm
  output_norm_w =. 'token_embd_norm.weight' get_tensor_cached_d llm
  tok_list =. , > tokens
  n_tokens =. # tok_list
  kv_create ((<block_count) , (<ctx_len) , (<n_heads_kv) , (<head_dim))
  NB. Fresh single-token infer: zero the conv recurrent state (lf2_run_blocks_b
  NB. create is guarded to protect batched per-sequence prefill).
  lf2_conv_reset ((<lf2_n_conv_g) , (<2) , (<emb_len))
  if. 1 = n_tokens do.
    tok =. 0 { tok_list
    hidden =. scale * |: (tok {"1 emb_w)
    pre_s =. 6!:2 'result =. hidden lf2_run_blocks (<llm) , <0'
    hidden =. > 0 { result
  else.
    emb_all =. scale * |: (tok_list {"1 emb_w)
    pre_s =. 6!:2 'result_b =. emb_all lf2_run_blocks_b ((<llm) , <0)'
    h_b =. > 0 { result_b
    hidden =. > (n_tokens - 1) { h_b
  end.
  logits =. output_head ((< mi_rms_eps mi) , (<output_norm_w) , (<emb_w) , <hidden)
  report_prefill (pre_s , n_tokens)
  pred_tok =. sample_from ((<temp) , (<k) , (<p) , (<min_p) , <logits)
  decoded =. lf2_detokenize (<llm) , <pred_tok
  tokens ; pred_tok ; decoded ; logits
)

NB. ---- Multi-token generation ----
NB. Usage: llm lf2_generate (text ; max_steps ; <temp;k;p;min_p>)
NB.        Simple (default params): llm lf2_generate_simple (text ; max_steps)
NB. Generation is the UNIFIED gen_loop_core (llm_core.ijs) — per-arch
NB. differences (embedding scale, logit scale, run_blocks/run_blocks_b) are
NB. dispatched by llm_arch. Fresh mode: kv_create + batched prefill; resume
NB. mode (chat sessions): incremental prefill of the new segment. Stop token
NB. not appended.
lf2_generate =: 4 : 0
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
  prompt =. lf2_chat_prompt messages
  tokens =. lf2_tokenize (<llm) , <prompt
  stop =. lf2_stop_tokens llm
  L =. # , > tokens
  output =. llm gen_loop_core (tokens ; '' ; max_steps ; temp ; k ; p ; min_p ; <stop)
  gen =. L }. output
  lf2_detokenize (<llm) , <gen
)

NB. ---- Batched generation: B independent prompts in parallel ----
NB. Usage: llm lf2_generate_batch (prompts ; max_steps ; <temp;k;p;min_p>)
lf2_generate_batch =: 4 : 0
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
    prompt =. lf2_chat_prompt messages
    tokens =. lf2_tokenize (<llm) , <prompt
    tok_list =. , > tokens
    prompts_tok =. prompts_tok , <tok_list
    prompts_len =. prompts_len , <(# tok_list)
    i =. i + 1
  end.
  stop =. lf2_stop_tokens llm
  kv_batch_g =: B
  output =. llm gen_loop_batch (prompts_tok ; max_steps ; temp ; k ; p ; min_p ; <stop)
  answers =. ''
  i =. 0
  while. i < B do.
    L =. > i { prompts_len
    gen =. (L) }. (> i { output)
    answers =. answers , <(lf2_detokenize (<llm) , <gen)
    i =. i + 1
  end.
  answers
)

NB. ---- Simple wrappers (default greedy/top-p params) ----
NB. llm lf2_infer_simple text            | llm lf2_generate_simple (text ; n)
lf2_infer_simple =: lf2_infer (] ; (<0 0 0.95 0.0)"_)
lf2_generate_simple =: lf2_generate (0&{ , 1&{ , (<0 0 0.95 0.0)"_)

NB. ---- LFM2 chat template ----
NB. qwen-style: {{bos_token}} + <|im_start|>role\n content <|im_end|>\n per
NB. message + generation prompt <|im_start|>assistant\n. No default system
NB. block. {{bos_token}} is rendered as '' — lf2_tokenize prepends BOS so the
NB. token stream matches llama.cpp (bos once).
lf2_chat_prompt =: 3 : 0
  messages =. y
  res =. ''
  for_i. i. # messages do.
    msg =. > i { messages
    role =. > 0 { msg
    content =. > 1 { msg
    res =. res , '<|im_start|>' , role , LF , content , '<|im_end|>' , LF
  end.
  res =. res , '<|im_start|>assistant' , LF
  res
)

lf2_default_params =: 0 0 0.95 0.0
NB. Stop tokens: EOS = <|im_end|> (7).
lf2_stop_tokens =: 3 : 0
  tk =. llm_tokenizer y
  tokenizer_eos_g tk
)
