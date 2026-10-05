NB. ================================================================
NB. Qwen3.5-0.8B (qwen35 arch) — hybrid decoder: 6 full-attention layers
NB. (il+1 divisible by 4: 3,7,11,15,19,23) + 18 gated-delta-net (SSM)
NB. layers. Attention: Q+GATE fused projection (wq = 2*head_dim per head),
NB. per-head Q/K RMSNorm, sigmoid-gated output, NEOX-style MRoPE (text:
NB. p_t=p_h=p_w=pos, p_e=0 -> plain NEOX over the first n_rot=64 dims).
NB. SSM: conv1d (kernel 4) + gated delta net recurrence (state S_v x S_v
NB. per v-head), L2-normed q/k, norm_gated output, SwiGLU FFN.
NB. MTP/nextn head (blk.24.nextn.*) and mmproj are NOT in scope: blk.24
NB. tensors are skipped at load.
NB. Tokenizer: GPT-2 byte-level BPE (pre=qwen35; regex adds \p{M} combining
NB. marks to the letter class — identical to qwen2 for ASCII).
NB. Depends on: llm_core.ijs, kernels.ijs, gguf.ijs, kv_cache.ijs,
NB.           sampler.ijs, tokenizer_gpt2.ijs
NB. ================================================================
coclass 'inference'
require 'llm/inference/util/llm_core'
require 'llm/inference/tokenizers/tokenizer_gpt2'
require 'llm/inference/util/minja'
require 'llm/inference/util/chat_template'

NB. ---- KV helpers ----
qw35_kv_uint =: 4 : 0
  key =. x
  data =. y
  key kv_uint data
)

qw35_kv_float =: 4 : 0
  key =. x
  data =. y
  key kv_float data
)

NB. ---- Extract qwen35 model info from KV pairs ----
NB. Returns the shared 12-item mi prefix (vocab patched by qw35_load from the
NB. token_embd dims; qwen35.rope.dimension_sections is not needed for text).
NB. mi = <block_count; context_len; emb_len; n_heads; n_heads_kv; head_dim;
NB.      rope_freq; vocab_size; rms_eps; n_ff; cos_tab; sin_tab;
NB.      key_len; n_rot; ssm_d_inner; ssm_d_state; ssm_dt_rank; ssm_n_group>
NB.      (18 items after load adds rope tables)
qw35_extract_hparams =: 3 : 0
  data =. y
  block_count =. 'qwen35.block_count' qw35_kv_uint data
  nextn =. 'qwen35.nextn_predict_layers' qw35_kv_uint data
  if. _1 ~: nextn do.
    NB. MTP flavour: block_count includes the nextn head (blk.24) — the trunk
    NB. is block_count - nextn. Non-MTP has no nextn KV (kv_uint -> _1).
    block_count =. block_count - nextn
  end.
  context_length =. 'qwen35.context_length' qw35_kv_uint data
  emb_len =. 'qwen35.embedding_length' qw35_kv_uint data
  n_heads =. 'qwen35.attention.head_count' qw35_kv_uint data
  n_heads_kv =. 'qwen35.attention.head_count_kv' qw35_kv_uint data
  rope_freq =. 'qwen35.rope.freq_base' qw35_kv_float data
  rms_eps =. 'qwen35.attention.layer_norm_rms_epsilon' qw35_kv_float data
  n_ff =. 'qwen35.feed_forward_length' qw35_kv_uint data
  key_len =. 'qwen35.attention.key_length' qw35_kv_uint data
  head_dim =. key_len
  (<"0) block_count , context_length , emb_len , n_heads , n_heads_kv , head_dim , rope_freq , 0 , rms_eps , n_ff
)

NB. ---- qwen35-specific mi accessors (shared mi_* cover indices 0..11) ----
qw35_mi_key_len     =: 3 : 0
  d =. y
  > get__d (<'key_len')
)
qw35_mi_n_rot       =: 3 : 0
  d =. y
  > get__d (<'n_rot')
)
qw35_mi_ssm_d_inner =: 3 : 0
  d =. y
  > get__d (<'ssm_d_inner')
)
qw35_mi_ssm_d_state =: 3 : 0
  d =. y
  > get__d (<'ssm_d_state')
)
qw35_mi_ssm_dt_rank =: 3 : 0
  d =. y
  > get__d (<'ssm_dt_rank')
)
qw35_mi_ssm_n_group =: 3 : 0
  d =. y
  > get__d (<'ssm_n_group')
)

NB. ---- block_data accessors ----
NB. Shared (all layers): <attn_norm; post_norm; ffn_gate; ffn_up; ffn_down;
NB.   is_ssm>  (indices 0..5)
NB. Attention layers: + <attn_q; attn_k; attn_v; attn_o; q_norm; k_norm;
NB.   n_heads; head_dim; n_heads_kv>  (indices 8..16)
NB. SSM layers: + <wqkv; wqkv_gate; ssm_beta; ssm_alpha; ssm_dt; ssm_a;
NB.   ssm_conv1d; ssm_norm; ssm_out; head_k_dim; num_k_heads; head_v_dim;
NB.   num_v_heads; d_inner>  (indices 8..21)
qw35_bd_attn_norm    =: >@(0&{)
qw35_bd_post_norm    =: >@(1&{)
qw35_bd_ff_gate      =: >@(2&{)
qw35_bd_ff_up        =: >@(3&{)
qw35_bd_ff_down      =: >@(4&{)


qw35_bd_is_ssm       =: >@(5&{)
qw35_bd_a_q          =: >@(6&{)
qw35_bd_a_k          =: >@(7&{)
qw35_bd_a_v          =: >@(8&{)
qw35_bd_a_o          =: >@(9&{)
qw35_bd_a_q_norm     =: >@(10&{)
qw35_bd_a_k_norm     =: >@(11&{)
qw35_bd_a_n_heads    =: >@(12&{)
qw35_bd_a_head_dim   =: >@(13&{)
qw35_bd_a_n_heads_kv =: >@(14&{)
qw35_bd_s_wqkv       =: >@(6&{)
qw35_bd_s_gate       =: >@(7&{)
qw35_bd_s_beta       =: >@(8&{)
qw35_bd_s_alpha      =: >@(9&{)
qw35_bd_s_dt         =: >@(10&{)
qw35_bd_s_a          =: >@(11&{)
qw35_bd_s_conv1d     =: >@(12&{)
qw35_bd_s_norm       =: >@(13&{)
qw35_bd_s_out        =: >@(14&{)
qw35_bd_s_head_k_dim =: >@(15&{)
qw35_bd_s_n_k_heads  =: >@(16&{)
qw35_bd_s_head_v_dim =: >@(17&{)
qw35_bd_s_n_v_heads  =: >@(18&{)
qw35_bd_s_d_inner    =: >@(19&{)
NB. ---- Build one block's data (rank-1; x=llm, y=block index) ----
qw35_build_block =: 4 : 0
  llm =. x
  b =. y
  mi =. llm_mi llm
  n_heads =. mi_n_heads mi
  n_heads_kv =. mi_n_heads_kv mi
  key_len =. qw35_mi_key_len mi
  p =. 'blk.' , (": b) , '.'
  attn_norm =. (p , 'attn_norm.weight') get_tensor_cached_d llm
  post_norm =. (p , 'post_attention_norm.weight') get_tensor_cached_d llm
  ff_gate =. (p , 'ffn_gate.weight') get_tensor_cached_d llm
  ff_up =. (p , 'ffn_up.weight') get_tensor_cached_d llm
  ff_down =. (p , 'ffn_down.weight') get_tensor_cached_d llm
  is_ssm =. 1
  NB. Determine layer type: SSM layers carry ssm_a; attention layers carry attn_q.
  if. 0 = # ((p , 'ssm_a') get_tensor_cached_d llm) do. is_ssm =. 0 end.
  shared =. (<attn_norm) , (<post_norm) , (<ff_gate) , (<ff_up) , (<ff_down) , <is_ssm
  if. is_ssm do.
    wqkv =. (p , 'attn_qkv.weight') get_tensor_cached_d llm
    zgate =. (p , 'attn_gate.weight') get_tensor_cached_d llm
    beta =. (p , 'ssm_beta.weight') get_tensor_cached_d llm
    alpha =. (p , 'ssm_alpha.weight') get_tensor_cached_d llm
    dt =. (p , 'ssm_dt.bias') get_tensor_cached_d llm
    a =. (p , 'ssm_a') get_tensor_cached_d llm
    conv1d =. (p , 'ssm_conv1d.weight') get_tensor_cached_d llm
    ssm_norm =. (p , 'ssm_norm.weight') get_tensor_cached_d llm
    ssm_out =. (p , 'ssm_out.weight') get_tensor_cached_d llm
    head_k_dim =. qw35_mi_ssm_d_state mi
    num_k_heads =. qw35_mi_ssm_n_group mi
    head_v_dim =. <. (qw35_mi_ssm_d_inner mi) % (qw35_mi_ssm_dt_rank mi)
    num_v_heads =. qw35_mi_ssm_dt_rank mi
    d_inner =. qw35_mi_ssm_d_inner mi
    shared , (<wqkv) , (<zgate) , (<beta) , (<alpha) , (<dt) , (<a) , (<conv1d) , (<ssm_norm) , (<ssm_out) , (<head_k_dim) , (<num_k_heads) , (<head_v_dim) , (<num_v_heads) , (<d_inner)
  else.
    attn_q =. (p , 'attn_q.weight') get_tensor_cached_d llm
    attn_k =. (p , 'attn_k.weight') get_tensor_cached_d llm
    attn_v =. (p , 'attn_v.weight') get_tensor_cached_d llm
    attn_o =. (p , 'attn_output.weight') get_tensor_cached_d llm
    q_norm =. (p , 'attn_q_norm.weight') get_tensor_cached_d llm
    k_norm =. (p , 'attn_k_norm.weight') get_tensor_cached_d llm
    head_dim =. key_len
    shared , (<attn_q) , (<attn_k) , (<attn_v) , (<attn_o) , (<q_norm) , (<k_norm) , (<n_heads) , (<head_dim) , (<n_heads_kv)
  end.
)

NB. ---- Pre-build all block data ----
qw35_pre_build_block_data =: 3 : 0
  llm =. y
  block_count =. mi_block_count (llm_mi llm)
  (<llm) qw35_build_block each i. block_count
)

NB. ---- Sigmoid ----
sigmoid =: 1 % 1 + ^@-

NB. ---- Softplus (stable; ggml: x > 20 -> x, else log(1+exp(x))) ----
NB. Tacit: (x>20)*x + (not x>20)*log(1+exp x)
softplus =: (((20 < ]) * ]) + ((-.@(20 < ])) * ^.@(1 + ^)))

NB. ---- L2 norm per row (ggml_l2_norm: scale = 1 / max(sqrt(sum sq), eps)) ----
NB. y = <eps; matrix>  -> matrix scaled. Tacit: mat * (1 % sqrt(row sumsq) >. eps)
l2norm_rows =: ((>@(1&{)) * (1 % ((%: @ (+/"1 @ (*: @ >@(1&{)))) >. >@(0&{))))

NB. ---- Partial NEOX RoPE batched (qwen35 MRoPE for text) ----
NB. ---- Recurrent state cache (SSM layers: conv state + delta-net state) ----
NB. rs_meta = <n_ssm_layers; conv_rows; conv_channels; S_v; n_v_heads>
NB. rs cache is TWO FLAT arrays (positions-leading, the kv_cache pattern —
NB. a boxed per-layer cache forces whole-batch COPY amends on write, ~40x
NB. slower; the flat in-place amend fires ~78x faster):
NB.   rs_conv_g = (n_ssm * kv_batch_g * conv_rows * conv_channels) conv state
NB.   rs_s_g    = (n_ssm * kv_batch_g * n_v_heads * S_v * S_v) s state
NB.   layer ord = qw35_ssm_layers i. layer; seq base = (ord*kv_batch_g + seq)
NB.   * slice (conv: conv_rows*conv_channels; s: n_v_heads*S_v*S_v)
NB.   BATCH dimension = kv_batch_g so batched decode keeps per-sequence states.
rs_meta =: ''
rs_conv_g =: ''
rs_s_g =: ''
rs_batch_g =: 1
NB. Block indices of the 18 SSM (gated delta net) layers: il+1 % 4 != 0.
qw35_ssm_layers =: 0 1 2 4 5 6 8 9 10 12 13 14 16 17 18 20 21 22

rs_create =: 3 : 0
  n_ssm =. > 0 { y
  conv_rows =. > 1 { y
  conv_channels =. > 2 { y
  s_v =. > 3 { y
  n_v_heads =. > 4 { y
  NB. Reuse if same dims + same batch (mirror kv_create) — a repeat prefill per
  NB. sequence must NOT reset earlier sequences' states.
  if. -. '' -: rs_meta do.
    if. (n_ssm = > 0 { rs_meta) *. (conv_rows = > 1 { rs_meta) *. (conv_channels = > 2 { rs_meta) *. (s_v = > 3 { rs_meta) *. (n_v_heads = > 4 { rs_meta) *. (rs_batch_g = kv_batch_g) do.
      '' return.
    end.
  end.
  rs_meta =: (<n_ssm) , (<conv_rows) , (<conv_channels) , (<s_v) , (<n_v_heads)
  rs_conv_g =: (n_ssm * kv_batch_g * conv_rows * conv_channels) $ 0.0
  rs_s_g =: (n_ssm * kv_batch_g * n_v_heads * s_v * s_v) $ 0.0
  rs_batch_g =: kv_batch_g
  ''
)

NB. Force zero the recurrent-state cache (fresh generation start). The guarded
NB. create in qw35_run_blocks_b keeps per-sequence prefill from resetting
NB. sibling states, but a new generation must NOT inherit the previous one's
NB. conv + delta-net state.
rs_reset =: 3 : 0
  n_ssm =. > 0 { y
  conv_rows =. > 1 { y
  conv_channels =. > 2 { y
  s_v =. > 3 { y
  n_v_heads =. > 4 { y
  rs_meta =: (<n_ssm) , (<conv_rows) , (<conv_channels) , (<s_v) , (<n_v_heads)
  rs_conv_g =: (n_ssm * kv_batch_g * conv_rows * conv_channels) $ 0.0
  rs_s_g =: (n_ssm * kv_batch_g * n_v_heads * s_v * s_v) $ 0.0
  rs_batch_g =: kv_batch_g
  ''
)

NB. ---- Slice sizes per (layer, seq): conv_rows*conv_channels; n_v_heads*S_v*S_v ----
rs_conv_slice =: 3 : '(> 1 { y) * (> 2 { y)'
rs_s_slice =: 3 : '(> 4 { y) * (> 3 { y) * (> 3 { y)'

rs_read =: 3 : 0
  layer =. y
  ord =. qw35_ssm_layers i. layer
  seq =. kv_seq_g
  cs =. rs_conv_slice rs_meta
  ss =. rs_s_slice rs_meta
  cb =. ((ord * rs_batch_g) + seq) * cs
  sb =. ((ord * rs_batch_g) + seq) * ss
  (<((> 1 { rs_meta) , (> 2 { rs_meta)) $ , ((cb + i. cs) { rs_conv_g)) , <((> 4 { rs_meta) , (> 3 { rs_meta) , (> 3 { rs_meta)) $ , ((sb + i. ss) { rs_s_g)
)

rs_write =: 3 : 0
  layer =. > 0 { y
  conv =. > 1 { y
  s =. > 2 { y
  ord =. qw35_ssm_layers i. layer
  seq =. kv_seq_g
  cs =. rs_conv_slice rs_meta
  ss =. rs_s_slice rs_meta
  cb =. ((ord * rs_batch_g) + seq) * cs
  sb =. ((ord * rs_batch_g) + seq) * ss
  rs_conv_g =: (, conv) (cb + i. cs)} rs_conv_g
  rs_s_g =: (, s) (sb + i. ss)} rs_s_g
  ''
)

NB. Batched-decode variants: explicit sequence index (decode keeps kv_seq_g=0)
rs_read_b =: 3 : 0
  layer =. > 0 { y
  b =. > 1 { y
  ord =. qw35_ssm_layers i. layer
  cs =. rs_conv_slice rs_meta
  ss =. rs_s_slice rs_meta
  cb =. ((ord * rs_batch_g) + b) * cs
  sb =. ((ord * rs_batch_g) + b) * ss
  (<((> 1 { rs_meta) , (> 2 { rs_meta)) $ , ((cb + i. cs) { rs_conv_g)) , <((> 4 { rs_meta) , (> 3 { rs_meta) , (> 3 { rs_meta)) $ , ((sb + i. ss) { rs_s_g)
)

rs_write_b =: 3 : 0
  layer =. > 0 { y
  conv =. > 1 { y
  s =. > 2 { y
  b =. > 3 { y
  ord =. qw35_ssm_layers i. layer
  cs =. rs_conv_slice rs_meta
  ss =. rs_s_slice rs_meta
  cb =. ((ord * rs_batch_g) + b) * cs
  sb =. ((ord * rs_batch_g) + b) * ss
  rs_conv_g =: (, conv) (cb + i. cs)} rs_conv_g
  rs_s_g =: (, s) (sb + i. ss)} rs_s_g
  ''
)

NB. Batched read/write: read (or write) all B sequences' conv+s recurrent state
NB. in ONE indexed pass, mirroring rs_read_b/rs_write_b's per-seq index math.
NB. y = <layer; B> (read) / <layer; conv_states(B,rows,ch); s_states(B,nv,Sv,Sv); B> (write).
rs_read_b_all =: 3 : 0
  layer =. > 0 { y
  B =. > 1 { y
  ord =. qw35_ssm_layers i. layer
  cs =. rs_conv_slice rs_meta
  ss =. rs_s_slice rs_meta
  cb0 =. (ord * rs_batch_g) * cs
  sb0 =. (ord * rs_batch_g) * ss
  idxc =. cb0 + ((i. B) * cs) +/ (i. cs)
  idxs =. sb0 + ((i. B) * ss) +/ (i. ss)
  cshape =. B , (> 1 { rs_meta) , (> 2 { rs_meta)
  sshape =. B , (> 4 { rs_meta) , (> 3 { rs_meta) , (> 3 { rs_meta)
  (<cshape $ , (idxc { rs_conv_g)) , <sshape $ , (idxs { rs_s_g)
)
rs_write_b_all =: 3 : 0
  layer =. > 0 { y
  conv =. > 1 { y
  s =. > 2 { y
  B =. > 3 { y
  ord =. qw35_ssm_layers i. layer
  cs =. rs_conv_slice rs_meta
  ss =. rs_s_slice rs_meta
  cb0 =. (ord * rs_batch_g) * cs
  sb0 =. (ord * rs_batch_g) * ss
  idxc =. cb0 + ((i. B) * cs) +/ (i. cs)
  idxs =. sb0 + ((i. B) * ss) +/ (i. ss)
  rs_conv_g =: (, conv) (, idxc)} rs_conv_g
  rs_s_g =: (, s) (, idxs)} rs_s_g
  ''
)


NB. ---- GQA expand ----

NB. ---- Batched attention layer (cache-prefix aware) ----
NB. x = hidden (L, emb); y = <block_data; mi; layer; start_pos>
qw35_attention_b =: 4 : 0
  block_data =. > 0 { y
  layer =. > 1 { y
  mi =. > 2 { y
  start_pos =. > 3 { y
  rope =. > 4 { y
  NB. rope = <cos_all; sin_all; cos_expq; sin_expq; cos_expk; sin_expk; mask_g2>
  cos_all =. > 0 { rope
  sin_all =. > 1 { rope
  cos_expq =. > 2 { rope
  sin_expq =. > 3 { rope
  cos_expk =. > 4 { rope
  sin_expk =. > 5 { rope
  mask_g2 =. > 6 { rope
  L =. {. $ x
  n_heads =. qw35_bd_a_n_heads block_data
  head_dim =. qw35_bd_a_head_dim block_data
  n_heads_kv =. qw35_bd_a_n_heads_kv block_data
  n_groups =. n_heads % n_heads_kv
  n_rot =. qw35_mi_n_rot mi
  half =. <. n_rot % 2

  NB. Attention norm
  normed =. rms_norm_rows ((< mi_rms_eps mi) , (< qw35_bd_attn_norm block_data) , <x)
  nt =. |: normed   NB. |: normed hoisted once — 3 (attn) / 4 (ssm) transposes were materializing copies

  NB. Fused Q+GATE projection (wq = 2*head_dim per head), then K/V
  qg =. |: ((qw35_bd_a_q block_data) (+/ .*) nt)   NB. (L, 2*head_dim*n_heads)
  qg3 =. (L , n_heads , 2 * head_dim) $ , qg
  Q =. head_dim {."1 qg3    NB. (L, n_heads, head_dim)
  gate =. head_dim }."1 qg3  NB. (L, n_heads, head_dim)
  kv =. |: ((qw35_bd_a_k block_data) (+/ .*) nt)   NB. (L, n_kv*head_dim)
  vv =. |: ((qw35_bd_a_v block_data) (+/ .*) nt)
  K =. (L , n_heads_kv , head_dim) $ , kv
  V =. (L , n_heads_kv , head_dim) $ , vv

  NB. Per-head RMSNorm on Q and K BEFORE RoPE
  Qf =. ((L * n_heads) , head_dim) $ , Q
  Qf =. rms_norm_rows ((< mi_rms_eps mi) , (< qw35_bd_a_q_norm block_data) , <Qf)
  Q =. (L , n_heads , head_dim) $ , Qf
  Kf =. ((L * n_heads_kv) , head_dim) $ , K
  Kf =. rms_norm_rows ((< mi_rms_eps mi) , (< qw35_bd_a_k_norm block_data) , <Kf)
  K =. (L , n_heads_kv , head_dim) $ , Kf

  NB. Partial NEOX RoPE (n_rot dims, pairs (i, i+half)) — tables + expansions
  NB. hoisted once per chunk (mirror llama_run_blocks_b).
  Qa =. half {."1 Q
  Qb =. half {."1 (half }."1 Q)
  Qa_out =. (Qa * cos_expq) - (Qb * sin_expq)
  Qb_out =. (Qa * sin_expq) + (Qb * cos_expq)
  Qtail =. (2 * half) }."1 Q
  Q =. (L , n_heads , head_dim) $ , ((Qa_out ,"1 Qb_out) ,"1 Qtail)
  Ka =. half {."1 K
  Kb =. half {."1 (half }."1 K)
  Ka_out =. (Ka * cos_expk) - (Kb * sin_expk)
  Kb_out =. (Ka * sin_expk) + (Kb * cos_expk)
  Ktail =. (2 * half) }."1 K
  K =. (L , n_heads_kv , head_dim) $ , ((Ka_out ,"1 Kb_out) ,"1 Ktail)

  NB. Scale Q by 1/sqrt(head_dim)
  Q =. Q % head_dim ^ 0.5

  NB. RESUME: prepend the cache prefix (positions 0..start_pos-1)
  K_batch =. K
  V_batch =. V
  if. start_pos > 0 do.
    k_pre =. > 0 { (kv_read ((<layer) , <(start_pos - 1)))
    v_pre =. > 1 { (kv_read ((<layer) , <(start_pos - 1)))
    K =. k_pre , K
    V =. v_pre , V
  end.
  kv_write_rows ((<0) , (<layer) , (<start_pos) , <K_batch)
  kv_write_rows ((<1) , (<layer) , (<start_pos) , <V_batch)

  NB. GQA without expanding KV over the combined keys: group the query heads
  NB. (n_heads_kv groups of n_groups) and batched-matmul each group's Q against
  NB. its shared K/V — K/V stay (n_heads_kv, ctx, hd), never expanded to n_heads.
  Qp =. 1 0 2 |: Q        NB. (n_heads, L, hd)
  Q_g2 =. (n_heads_kv , (n_groups * L) , head_dim) $ , Qp   NB. one ravel+reshape (the intermediate 4D reshape was redundant)
  Kp2 =. 1 2 0 |: K        NB. (n_heads_kv, hd, start_pos+L) — one transpose
  scores2 =. Q_g2 (+/ .* "2) Kp2   NB. (n_kv, n_groups*L, ctx): Q[t,h] vs K[j,g(h)]
  NB. causal mask: query t at start_pos+t, keys 0..start_pos+L-1. Keep scores
  NB. group-major: tile the 2D mask r-major (row r*L+t needs mask row t) and
  NB. Causal mask: hoisted once per chunk (mask_g2).
  scores2 =. scores2 -"2 (mask_g2 * 1e9)

  NB. Softmax directly on the 3D scores2 (the old scores_f flatten +
  NB. softmax_g2 re-group were 2 redundant reshape copies per layer)
  max_sf =. >./"1 scores2
  exp_sf =. ^ (scores2 - max_sf)
  softmax_f =. exp_sf % +/"1 exp_sf

  NB. Output: attn[h,t] = sum_j softmax[g(h),t,j] * V[g(h),j]
  Vp =. 1 0 2 |: V        NB. (n_heads_kv, start_pos+L, hd)
  attn2 =. softmax_f (+/ .* "2) Vp   NB. (n_kv, n_groups*L, hd)
  NB. Gated output: multiply by sigmoid(gate) per head, then wo
  attn_flat =. (L , n_heads * head_dim) $ , (((L, n_heads, head_dim) $ , (1 0 2 |: ((n_heads, L, head_dim) $ , attn2))) * (sigmoid gate))
  out =. |: ((qw35_bd_a_o block_data) (+/ .*) |: attn_flat)   NB. (L, emb)
  <out
)

NB. ---- Sequential gated-delta-net recurrence over a batch ----
NB. y = <s_state; q; k; v; beta; gate; scale>
NB. s_state (n_v_heads, S_v, S_v) TRANSPOSED layout; q/k/v (L, n_v_heads, S_v);
NB. beta/gate (L, n_v_heads). Returns <final_s_state; o> where o (L, n_v_heads, S_v).
qw35_ssm_recur =: 3 : 0
  s =. > 0 { y
  q =. > 1 { y
  k =. > 2 { y
  v =. > 3 { y
  beta =. > 4 { y
  gate =. > 5 { y
  scale =. > 6 { y
  L =. {. $ q
  n_v_heads =. {. $ s
  s_v =. 2 { $ s
  out =. (L , n_v_heads , s_v) $ 0
  NB. Precompute all decay factors once (one vectorized ^ instead of L scalar ^)
  decay_all =. ^ gate   NB. (L, n_v_heads)
  t =. 0
  while. t < L do.
    decay =. t { decay_all
    s =. s * decay
    k_t =. t { k
    v_t =. t { v
    q_t =. t { q
    sk =. k_t (+/ .* "1 2) s
    delta =. (v_t - sk) * (t { beta)
    NB. Outer-product state update, reformulated as a batched matmat
    NB. (n_v_heads, S_v, 1) x (n_v_heads, 1, S_v) -> (n_v_heads, S_v, S_v):
    NB. J threads the rank-4 matmul over the leading v-head axis (~5x vs the
    NB. rank-1 outer (*/)"1 1). Bit-exact (same values, verified -: match).
    k3 =. (n_v_heads, s_v, 1) $ , k_t
    d3 =. (n_v_heads, 1, s_v) $ , delta
    s =. s + (k3 (+/ .* "2) d3)
    o_t =. (q_t (+/ .* "1 2) s) * scale
    out =. o_t t} out
    t =. t + 1
  end.
  (<s) , <out
)

NB. Batched-decode recurrence: one vectorized delta-net step across B sequences
NB. in a single pass (J threads the rank-4 matmuls over the (B, n_v_heads) cell
NB. axes). Bit-exact vs the per-seq qw35_ssm_recur loop (verified -: on real
NB. layer data). y = <s_states(B,nv,S_v,S_v); q(B,nv,hd); k; v; beta(B,nv);
NB. gate(B,nv); scale>.
qw35_ssm_recur_b =: 3 : 0
  s =. > 0 { y
  q =. > 1 { y
  k =. > 2 { y
  v =. > 3 { y
  beta =. > 4 { y
  gate =. > 5 { y
  scale =. > 6 { y
  B =. {. $ s
  n_v_heads =. 1 { $ s
  s_v =. 3 { $ s
  NB. decay ^gate broadcast to (B,nv,S_v,S_v): (*/ ) outer over the two S_v axes
  s =. s * ((B , n_v_heads , s_v , s_v) $ , ((^ gate) (*/) (s_v $ 1) (*/) (s_v $ 1)))
  sk =. k (+/ .* "1 2) s
  delta =. (v - sk) * beta
  k3 =. (B , n_v_heads , s_v , 1) $ , k
  d3 =. (B , n_v_heads , 1 , s_v) $ , delta
  s =. s + (k3 (+/ .* "2) d3)
  o =. (q (+/ .* "1 2) s) * scale
  (<s) , <o
)

NB. ---- Batched SSM (gated delta net) layer forward ----
NB. x = hidden (L, emb); y = <block_data; mi; layer>
qw35_ssm_forward_b =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  mi =. > 1 { y
  layer =. > 2 { y
  L =. {. $ hidden
  head_v_dim =. qw35_bd_s_head_v_dim block_data
  num_v_heads =. qw35_bd_s_n_v_heads block_data
  d_inner =. qw35_bd_s_d_inner block_data

  NB. Attention norm
  normed =. rms_norm_rows ((< mi_rms_eps mi) , (< qw35_bd_attn_norm block_data) , <hidden)
  nt =. |: normed   NB. |: normed hoisted once — 3 (attn) / 4 (ssm) transposes were materializing copies

  key_dim =. (qw35_bd_s_head_k_dim block_data) * (qw35_bd_s_n_k_heads block_data)
  value_dim =. head_v_dim * num_v_heads
  conv_dim =. (key_dim * 2) + value_dim

  NB. Projections (batched)
  qkv_mixed =. |: ((qw35_bd_s_wqkv block_data) (+/ .*) nt)   NB. (L, conv_dim)
  z =. |: ((qw35_bd_s_gate block_data) (+/ .*) nt)   NB. (L, value_dim)
  beta_p =. |: ((qw35_bd_s_beta block_data) (+/ .*) nt)   NB. (L, num_v_heads)
  beta =. sigmoid beta_p
  alpha_p =. |: ((qw35_bd_s_alpha block_data) (+/ .*) nt)   NB. (L, num_v_heads)
  alpha =. alpha_p + ((L , num_v_heads) $ qw35_bd_s_dt block_data)
  alpha =. softplus alpha
  gate =. alpha * ((L , num_v_heads) $ qw35_bd_s_a block_data)   NB. decay base

  NB. Conv1d: input = [conv_state; qkv_mixed], kernel 4, out[t] = sum_k input[t+k]*W[k]
  rs_state =. rs_read layer
  conv_state =. > 0 { rs_state
  input =. conv_state , qkv_mixed   NB. (3+L, conv_dim)
  conv_w =. qw35_bd_s_conv1d block_data   NB. (conv_dim, 4)
  conv_out =. ((i. L) { input) * ((L , conv_dim) $ (0 {"1 conv_w))
  conv_out =. conv_out + ((1 + i. L) { input) * ((L , conv_dim) $ (1 {"1 conv_w))
  conv_out =. conv_out + ((2 + i. L) { input) * ((L , conv_dim) $ (2 {"1 conv_w))
  conv_out =. conv_out + ((3 + i. L) { input) * ((L , conv_dim) $ (3 {"1 conv_w))
  conv_out =. silu conv_out   NB. (L, conv_dim)

  NB. Split Q/K/V channels: q (first key_dim), k (next), v (last)
  q_conv =. key_dim {."1 conv_out
  k_conv =. key_dim {."1 (key_dim }."1 conv_out)
  v_conv =. (2 * key_dim) }."1 conv_out

  NB. Reshape to (L, num_k_heads, head_k_dim) etc; L2-norm q and k
  q3 =. (L , num_v_heads , head_v_dim) $ , q_conv
  k3 =. (L , num_v_heads , head_v_dim) $ , k_conv
  v3 =. (L , num_v_heads , head_v_dim) $ , v_conv
  qf =. ((L * num_v_heads) , head_v_dim) $ , q3
  qf =. l2norm_rows ((< mi_rms_eps mi) , <qf)
  q =. (L , num_v_heads , head_v_dim) $ , qf
  kf =. ((L * num_v_heads) , head_v_dim) $ , k3
  kf =. l2norm_rows ((< mi_rms_eps mi) , <kf)
  k =. (L , num_v_heads , head_v_dim) $ , kf

  NB. Recurrence
  s_state =. > 1 { rs_state
  scale =. 1 % head_v_dim ^ 0.5
  rec =. qw35_ssm_recur ((<s_state) , (<q) , (<k) , (<v3) , (<beta) , (<gate) , <scale)
  s_new =. > 0 { rec
  o =. > 1 { rec   NB. (L, num_v_heads, head_v_dim)

  NB. Norm-gated output: rms_norm(o, ssm_norm) * silu(z)
  o_flat =. ((L * num_v_heads) , head_v_dim) $ , o
  o_n =. rms_norm_rows ((< mi_rms_eps mi) , (< qw35_bd_s_norm block_data) , <o_flat)
  z3 =. (L , num_v_heads , head_v_dim) $ , z
  z_flat =. ((L * num_v_heads) , head_v_dim) $ , z3
  gated =. o_n * silu z_flat
  final =. (L , d_inner) $ , gated
  out =. |: ((qw35_bd_s_out block_data) (+/ .*) |: final)   NB. (L, emb)

  NB. Residual
  out =. out + hidden

  NB. Update conv state (last 3 rows of input) and s state
  new_conv =. (L + i. 3) { input
  rs_write ((<layer) , (<new_conv) , <s_new)

  NB. Post-attention norm + SwiGLU FFN (no inner residual)
  post =. rms_norm_rows ((< mi_rms_eps mi) , (< qw35_bd_post_norm block_data) , <out)
  pt =. |: post   NB. |: post hoisted once — 2 transposes were materializing 2 copies
  gate_f =. |: ((qw35_bd_ff_gate block_data) (+/ .*) pt)
  up_f =. |: ((qw35_bd_ff_up block_data) (+/ .*) pt)
  ffn_raw =. |: ((qw35_bd_ff_down block_data) (+/ .*) |: (gate_f swiglu up_f))
  output =. ffn_raw + out
  <output
)

NB. ---- Batched attention block forward ----
NB. x = hidden (L, emb); y = <block_data; mi; layer; start_pos>
qw35_block_forward_a_b =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  layer =. > 1 { y
  mi =. > 2 { y
  start_pos =. > 3 { y
  rope =. > 4 { y
  attn_result =. hidden qw35_attention_b y
  attn_out =. > 0 { attn_result
  sa_out =. attn_out + hidden
  post =. rms_norm_rows ((< mi_rms_eps mi) , (< qw35_bd_post_norm block_data) , <sa_out)
  pt =. |: post   NB. |: post hoisted once — 2 transposes were materializing 2 copies
  gate_f =. |: ((qw35_bd_ff_gate block_data) (+/ .*) pt)
  up_f =. |: ((qw35_bd_ff_up block_data) (+/ .*) pt)
  ffn_raw =. |: ((qw35_bd_ff_down block_data) (+/ .*) |: (gate_f swiglu up_f))
  output =. ffn_raw + sa_out
  <output
)

NB. ---- Batched SSM block forward ----
NB. x = hidden (L, emb); y = <block_data; mi; layer> — identical to
NB. qw35_ssm_forward_b (same y shape); alias instead of a wrapper.
qw35_block_forward_s_b =: qw35_ssm_forward_b

NB. ---- Batched run all blocks ----
NB. x = hidden (L, emb); y = <llm; start_pos> (start_pos=0 -> fresh: rs state zeroed;
NB. start_pos>0 -> resume: attention cache prefix + rs state persist).
qw35_run_blocks_b =: 4 : 0
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
  NB. SSM recurrent state: fresh prefill zeroes it (start_pos=0); resume keeps it.
  if. 0 = start_pos do.
    rs_create ((<18) , (<3) , (<6144) , (<128) , <16)
  else.
    if. 0 = # rs_meta do.
      rs_create ((<18) , (<3) , (<6144) , (<128) , <16)
    end.
  end.
  NB. RoPE tables are per-model and identical across layers (freq is model
  NB. level): compute the cos/sin tables + expansions ONCE per chunk and thread
  NB. through the layer loop (mirror llama_run_blocks_b), instead of recomputing
  NB. them in every layer.  The causal mask is layer-invariant too.
  L =. {. $ x
  n_heads =. mi_n_heads mi
  n_rot =. qw35_mi_n_rot mi
  half =. <. n_rot % 2
  cos_all =. (start_pos + i. L) { mi_cos_tab mi
  sin_all =. (start_pos + i. L) { mi_sin_tab mi
  cos_expq =. (0 2 1) |: ((L , half , n_heads) $ , (cos_all (*/) (n_heads $ 1)))
  sin_expq =. (0 2 1) |: ((L , half , n_heads) $ , (sin_all (*/) (n_heads $ 1)))
  cos_expk =. (0 2 1) |: ((L , half , n_heads_kv) $ , (cos_all (*/) (n_heads_kv $ 1)))
  sin_expk =. (0 2 1) |: ((L , half , n_heads_kv) $ , (sin_all (*/) (n_heads_kv $ 1)))
  n_groups =. n_heads % n_heads_kv
  key_pos =. i. (start_pos + L)
  q_pos =. start_pos + i. L
  mask_2d =. q_pos </ key_pos
  mask_g2 =. ((n_groups * L) , start_pos + L) $ , (2 0 1 |: (mask_2d (*/) (n_groups $ 1)))
  rope =. (<cos_all) , (<sin_all) , (<cos_expq) , (<sin_expq) , (<cos_expk) , (<sin_expk) , (<mask_g2)
  b =. 0
  block_data_list =. llm_block_data llm
  NB. <mi, <start_pos, <rope are layer-invariant — box once, reuse per layer.
  bf_pre =. ((<mi) , (<start_pos) , <rope)
  while. b < block_count do.
    block_data =. > b { block_data_list
    if. qw35_bd_is_ssm block_data do.
      result =. state qw35_block_forward_s_b ((<block_data) , (<mi) , <b)
    else.
      result =. state qw35_block_forward_a_b ((<block_data) , (<b) , bf_pre)
    end.
    state =. > 0 { result
    b =. b + 1
  end.
  <state
)

NB. ---- Batched-DECODE attention layer (B sequences, ONE token each at pos[b]) ----
NB. qwen35 attention = qwen3-style + fused Q+GATE, PARTIAL NEOX RoPE (n_rot),
NB. gated output (sigmoid(gate)). x = hidden (B, emb); y = <block_data; pos; mi; layer>
qw35_attention_bd =: 4 : 0
  block_data =. > 0 { y
  pos =. > 1 { y
  mi =. > 2 { y
  layer =. > 3 { y
  rope =. > 4 { y
  B =. {. $ x
  n_heads =. qw35_bd_a_n_heads block_data
  head_dim =. qw35_bd_a_head_dim block_data
  n_heads_kv =. qw35_bd_a_n_heads_kv block_data
  n_groups =. n_heads % n_heads_kv
  n_rot =. qw35_mi_n_rot mi
  half =. <. n_rot % 2

  NB. Attention norm per row
  normed =. rms_norm_rows ((< mi_rms_eps mi) , (< qw35_bd_attn_norm block_data) , <x)
  nt =. |: normed   NB. |: normed hoisted once — 3 (attn) / 4 (ssm) transposes were materializing copies

  NB. Fused Q+GATE projection (wq = 2*head_dim per head), then K/V
  qg =. |: ((qw35_bd_a_q block_data) (+/ .*) nt)   NB. (B, 2*head_dim*n_heads)
  qg3 =. (B , n_heads , 2 * head_dim) $ , qg
  Q =. head_dim {."1 qg3    NB. (B, n_heads, head_dim)
  gate =. head_dim }."1 qg3  NB. (B, n_heads, head_dim)
  kv =. |: ((qw35_bd_a_k block_data) (+/ .*) nt)   NB. (B, n_kv*head_dim)
  vv =. |: ((qw35_bd_a_v block_data) (+/ .*) nt)
  K =. (B , n_heads_kv , head_dim) $ , kv
  V =. (B , n_heads_kv , head_dim) $ , vv

  NB. Per-head RMSNorm on Q and K BEFORE RoPE
  Qf =. ((B * n_heads) , head_dim) $ , Q
  Qf =. rms_norm_rows ((< mi_rms_eps mi) , (< qw35_bd_a_q_norm block_data) , <Qf)
  Q =. (B , n_heads , head_dim) $ , Qf
  Kf =. ((B * n_heads_kv) , head_dim) $ , K
  Kf =. rms_norm_rows ((< mi_rms_eps mi) , (< qw35_bd_a_k_norm block_data) , <Kf)
  K =. (B , n_heads_kv , head_dim) $ , Kf

  NB. Partial NEOX RoPE batched at the B positions (n_rot dims, pairs (i, i+half)).
  NB. cos/sin tables + expansions are layer-invariant (depend only on pos) —
  NB. hoisted once in run_blocks_bd, threaded through y as <cos_all; sin_all;
  NB. cos_expq; sin_expq; cos_expk; sin_expk>.
  cos_all =. > 0 { rope    NB. (B, half)
  sin_all =. > 1 { rope
  cos_expq =. > 2 { rope
  sin_expq =. > 3 { rope
  Qa =. half {."1 Q
  Qb =. half {."1 (half }."1 Q)
  Qa_out =. (Qa * cos_expq) - (Qb * sin_expq)
  Qb_out =. (Qa * sin_expq) + (Qb * cos_expq)
  Qtail =. (2 * half) }."1 Q
  Q =. (B , n_heads , head_dim) $ , ((Qa_out ,"1 Qb_out) ,"1 Qtail)
  cos_expk =. > 4 { rope
  sin_expk =. > 5 { rope
  Ka =. half {."1 K
  Kb =. half {."1 (half }."1 K)
  Ka_out =. (Ka * cos_expk) - (Kb * sin_expk)
  Kb_out =. (Ka * sin_expk) + (Kb * cos_expk)
  Ktail =. (2 * half) }."1 K
  K =. (B , n_heads_kv , head_dim) $ , ((Ka_out ,"1 Kb_out) ,"1 Ktail)

  NB. Scale Q by 1/sqrt(head_dim)
  Q =. Q % head_dim ^ 0.5

  NB. Per-sequence: write/read, scores, softmax, gated output
  NB. Vectorized path fires when all B sequences share one position (common
  NB. lockstep decode with equal-length prompts): ONE list-selector cache
  NB. write, ONE indexed gather of the B windows, then threaded batched
  NB. scores/softmax/V + sigmoid-gate (J parallelizes over B). Falls back to
  NB. the per-seq loop when positions differ (variable window lengths).
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
    NB. Batched GQA scores, softmax, V, sigmoid-gate (threaded over B)
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
    attn_b3 =. (B , n_heads , head_dim) $ , attn2_b
    attn_gated_b =. attn_b3 * (sigmoid gate)   NB. gate (B, n_heads, hd)
    attn_all =. (B , n_heads * head_dim) $ , attn_gated_b   NB. (B, n_heads*hd) flat
  else.
    attn_out =. ''
    b =. 0
    while. b < B do.
      q_b =. b { Q
      k_b =. b { K
      v_b =. b { V
      gate_b =. b { gate
      pos_b =. b { pos
      kv_write ((<layer) , (<pos_b) , (<k_b) , (<v_b) , (<b))
      kv_result =. kv_read ((<layer) , (<pos_b) , (<b))
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
      attn_out =. attn_out , <((n_heads * head_dim) $ , (((n_heads, head_dim) $ , attn2) * (sigmoid gate_b)))
      b =. b + 1
    end.
    attn_all =. > attn_out
  end.
  out =. |: ((qw35_bd_a_o block_data) (+/ .*) |: attn_all)   NB. (B, emb)
  <out
)

NB. ---- Batched-DECODE attention block forward ----
qw35_block_forward_a_bd =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  pos =. > 1 { y
  mi =. > 2 { y
  layer =. > 3 { y
  rope =. > 4 { y
  attn_result =. hidden qw35_attention_bd y
  attn_out =. > 0 { attn_result
  sa_out =. attn_out + hidden
  post =. rms_norm_rows ((< mi_rms_eps mi) , (< qw35_bd_post_norm block_data) , <sa_out)
  pt =. |: post   NB. |: post hoisted once — 2 transposes were materializing 2 copies
  gate_f =. |: ((qw35_bd_ff_gate block_data) (+/ .*) pt)
  up_f =. |: ((qw35_bd_ff_up block_data) (+/ .*) pt)
  ffn_raw =. |: ((qw35_bd_ff_down block_data) (+/ .*) |: (gate_f swiglu up_f))
  output =. ffn_raw + sa_out
  <output
)

NB. ---- Batched-DECODE SSM (gated delta net) layer forward ----
NB. x = hidden (B, emb); y = <block_data; pos; mi; layer>
qw35_ssm_forward_bd =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  pos =. > 1 { y
  mi =. > 2 { y
  layer =. > 3 { y
  B =. {. $ hidden
  head_v_dim =. qw35_bd_s_head_v_dim block_data
  num_v_heads =. qw35_bd_s_n_v_heads block_data
  d_inner =. qw35_bd_s_d_inner block_data

  NB. Attention norm per row
  normed =. rms_norm_rows ((< mi_rms_eps mi) , (< qw35_bd_attn_norm block_data) , <hidden)
  nt =. |: normed   NB. |: normed hoisted once — 3 (attn) / 4 (ssm) transposes were materializing copies

  key_dim =. (qw35_bd_s_head_k_dim block_data) * (qw35_bd_s_n_k_heads block_data)
  value_dim =. head_v_dim * num_v_heads
  conv_dim =. (key_dim * 2) + value_dim

  NB. Projections (batched)
  qkv_mixed =. |: ((qw35_bd_s_wqkv block_data) (+/ .*) nt)   NB. (B, conv_dim)
  z =. |: ((qw35_bd_s_gate block_data) (+/ .*) nt)   NB. (B, value_dim)
  beta_p =. |: ((qw35_bd_s_beta block_data) (+/ .*) nt)   NB. (B, num_v_heads)
  beta =. sigmoid beta_p
  alpha_p =. |: ((qw35_bd_s_alpha block_data) (+/ .*) nt)   NB. (B, num_v_heads)
  alpha =. alpha_p + ((B , num_v_heads) $ qw35_bd_s_dt block_data)
  alpha =. softplus alpha
  gate =. alpha * ((B , num_v_heads) $ qw35_bd_s_a block_data)   NB. decay base

  NB. Conv1d + delta-net recurrence per sequence (1 token each)
  conv_w =. qw35_bd_s_conv1d block_data   NB. (conv_dim, 4)
  scale =. 1 % head_v_dim ^ 0.5
  NB. Batch the norm-gated silu(z) once over all B (one vectorized op per layer
  NB. instead of B per-seq silu calls — cuts the per-call activation overhead)
  z3 =. (B , num_v_heads , head_v_dim) $ , z
  z_silu =. silu z3
  NB. Batched conv + recurrence + gated norm across all B sequences in ONE
  NB. vectorized pass (bit-exact vs the per-seq loop it replaces): J threads the
  NB. rank-4 matmuls over the (B, n_v_heads) cell axes, so the delta-net
  NB. recurrence runs once for the whole batch instead of B times.
  'conv_states_b s_states_b' =. rs_read_b_all (layer ; B)
  input_b =. (B , 4 , conv_dim) $ , (conv_states_b ,"2 (B , 1 , conv_dim) $ , qkv_mixed)
  c0 =. |: (0 {"2 |: input_b)
  c1 =. |: (1 {"2 |: input_b)
  c2 =. |: (2 {"2 |: input_b)
  c3 =. |: (3 {"2 |: input_b)
  conv_out_b =. (B , conv_dim) $ , ((c0 * ((B , conv_dim) $ , 0 {"1 conv_w)) + (c1 * ((B , conv_dim) $ , 1 {"1 conv_w)) + (c2 * ((B , conv_dim) $ , 2 {"1 conv_w)) + (c3 * ((B , conv_dim) $ , 3 {"1 conv_w)))
  conv_out_b =. silu conv_out_b
  q3_b =. (B , num_v_heads , head_v_dim) $ , (key_dim {."1 conv_out_b)
  k3_b =. (B , num_v_heads , head_v_dim) $ , (key_dim {."1 (key_dim }."1 conv_out_b))
  v3_b =. (B , num_v_heads , head_v_dim) $ , ((2 * key_dim) }."1 conv_out_b)
  qf_b =. l2norm_rows ((< mi_rms_eps mi) , <((B * num_v_heads) , head_v_dim) $ , q3_b)
  q_b =. (B , num_v_heads , head_v_dim) $ , qf_b
  kf_b =. l2norm_rows ((< mi_rms_eps mi) , <((B * num_v_heads) , head_v_dim) $ , k3_b)
  k_b =. (B , num_v_heads , head_v_dim) $ , kf_b
  rec_b =. qw35_ssm_recur_b ((<s_states_b) , (<q_b) , (<k_b) , (<v3_b) , (<beta) , (<gate) , <scale)
  s_new_b =. > 0 { rec_b
  o_b =. > 1 { rec_b
  o_n_b =. rms_norm_rows ((< mi_rms_eps mi) , (< qw35_bd_s_norm block_data) , <((B * num_v_heads) , head_v_dim) $ , o_b)
  gated_b =. ((B , num_v_heads , head_v_dim) $ , o_n_b) * z_silu
  final_all =. (B , d_inner) $ , gated_b
  NB. Update conv state (last 3 rows of input_b) + s state for all B at once
  new_conv_b =. 1 0 2 |: ((1 + i. 3) { 1 0 2 |: input_b)
  rs_write_b_all ((<layer) , (<new_conv_b) , (<s_new_b) , <B)
  NB. out_proj + residual + FFN (batched)
  out =. |: ((qw35_bd_s_out block_data) (+/ .*) |: final_all)   NB. (B, emb)
  out =. out + hidden
  post =. rms_norm_rows ((< mi_rms_eps mi) , (< qw35_bd_post_norm block_data) , <out)
  pt =. |: post   NB. |: post hoisted once — 2 transposes were materializing 2 copies
  gate_f =. |: ((qw35_bd_ff_gate block_data) (+/ .*) pt)
  up_f =. |: ((qw35_bd_ff_up block_data) (+/ .*) pt)
  ffn_raw =. |: ((qw35_bd_ff_down block_data) (+/ .*) |: (gate_f swiglu up_f))
  output =. ffn_raw + out
  <output
)

NB. ---- Batched-DECODE run all blocks (B sequences, one token each at pos[b]) ----
qw35_block_forward_s_bd =: qw35_ssm_forward_bd

qw35_run_blocks_bd =: 4 : 0
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
  NB. SSM recurrent state created during prefill (qw35_run_blocks_b start_pos=0);
  NB. for decode it must exist with the batch dimension. If absent, allocate it.
  if. 0 = # rs_meta do.
    rs_create ((<18) , (<3) , (<6144) , (<128) , <16)
  end.
  NB. RoPE cos/sin tables + expansions are layer-invariant (depend only on
  NB. pos) — compute ONCE per step, thread through (attention layers only;
  NB. SSM layers don't use RoPE).
  n_rot =. qw35_mi_n_rot mi
  half =. <. n_rot % 2
  B =. {. $ state
  cos_all =. pos { mi_cos_tab mi    NB. (B, half)
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
    if. qw35_bd_is_ssm block_data do.
      result =. state qw35_block_forward_s_bd ((<block_data) , bf_pre , <b)
    else.
      result =. state qw35_block_forward_a_bd ((<block_data) , bf_pre , (<b) , <rope)
    end.
    state =. > 0 { result
    b =. b + 1
  end.
  <state
)

NB. ---- Single-token run all blocks (1D hidden) ----
NB. x = hidden (emb,); y = <llm; pos>. Wraps to (1, emb) and runs the batched path.
qw35_run_blocks =: 4 : 0
  llm =. > 0 { y
  pos =. > 1 { y
  input2 =. (1 , $ x) $ x
  result_b =. input2 qw35_run_blocks_b ((<llm) , <pos)
  state =. > 0 { result_b
  < > 0 { state
)

NB. ---- Load qwen35 GGUF into llm noun ----
qw35_load =: 3 : 0
  NB. y = <path; raw> — raw is the memory-mapped file.
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
  mi =. qw35_extract_hparams kvs_ctx
  mi =. build_mi_dict mi
  NB. vocab from token_embd dims (qwen35 has no vocab_size KV)
  te_dims =. > ((0 * 6) + 1) { ti
  vocab =. {: te_dims
  NB. qwen35-specific config fields: key_len, n_rot, ssm_*
  key_len =. 'qwen35.attention.key_length' qw35_kv_uint kvs_ctx
  n_rot =. 'qwen35.rope.dimension_count' qw35_kv_uint kvs_ctx
  ssm_d_inner =. 'qwen35.ssm.inner_size' qw35_kv_uint kvs_ctx
  ssm_d_state =. 'qwen35.ssm.state_size' qw35_kv_uint kvs_ctx
  ssm_dt_rank =. 'qwen35.ssm.time_step_rank' qw35_kv_uint kvs_ctx
  ssm_n_group =. 'qwen35.ssm.group_count' qw35_kv_uint kvs_ctx
  rope_tables =. build_rope_tables ((< mi_context_len mi) , (<n_rot) , (< mi_rope_freq mi))
  NB. ONE multi-put: the jdict single-put (1-key) path is broken in this J9.8
  NB. build (domain/length error); multi-put (2+ keys) and get/has work.
  NB. vocab_size overwrites the build_mi_dict placeholder (qwen35 has no KV).
  (vocab ; (> 0 { rope_tables) ; (> 1 { rope_tables) ; key_len ; n_rot ; ssm_d_inner ; ssm_d_state ; ssm_dt_rank ; ssm_n_group) put__mi 'vocab_size' ; 'cos_tab' ; 'sin_tab' ; 'key_len' ; 'n_rot' ; 'ssm_d_inner' ; 'ssm_d_state' ; 'ssm_dt_rank' ; 'ssm_n_group'
  NB. Real chat template from the GGUF ('' if absent → bespoke fallback).
  ct_tmpl_g =: 'tokenizer.chat_template' kv_string (0 1 { kv_result)
  tokenizer =. build_gpt2_tokenizer kv_result
  all_tensors =. ''
  tensor_idx =. 0
  while. tensor_idx < n_tensors do.
    tname =. > (tensor_idx * 6) { ti
    NB. MTP/nextn block (blk.24) is NOT in scope — skip those tensors.
    if. -. 'blk.24.' -: 6 {. tname do.
      tdata =. (<path) , (<ti) , (<tds) , (<tname) , (<raw)
      td =. load_tdata tdata
      if. 0 < # td do.
        ti_row =. tname get_tensor_info ti
        dims =. ti_dims ti_row
        etype =. ti_etype ti_row
        all_tensors =. all_tensors , (<tname) , (<td) , (<dims) , (<etype)
      end.
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
  block_data =. qw35_pre_build_block_data llm
  llm =. llm , <block_data
  llm =. llm , <kv_data
)

NB. ---- Infer (raw single forward, no chat template) ----
qw35_infer =: 4 : 0
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
  NB. Fresh single-token infer: zero the SSM recurrent state (qw35_run_blocks_b
  NB. create is guarded to protect batched per-sequence prefill).
  rs_reset ((<18) , (<3) , (<6144) , (<128) , <16)
  if. 1 = n_tokens do.
    tok =. 0 { tok_list
    hidden =. scale * |: (tok {"1 emb_w)
    pre_s =. 6!:2 'result =. hidden qw35_run_blocks (<llm) , <0'
    hidden =. > 0 { result
  else.
    emb_all =. scale * |: (tok_list {"1 emb_w)
    pre_s =. 6!:2 'result_b =. emb_all qw35_run_blocks_b ((<llm) , <0)'
    h_b =. > 0 { result_b
    hidden =. > (n_tokens - 1) { h_b
  end.
  logits =. output_head ((< mi_rms_eps mi) , (<output_norm_w) , (<emb_w) , <hidden)
  report_prefill (pre_s , n_tokens)
  pred_tok =. sample_from ((<temp) , (<k) , (<p) , (<min_p) , <logits)
  decoded =. gpt2_detokenize (<llm) , <pred_tok
  tokens ; pred_tok ; decoded ; logits
)

NB. ---- Chat template prompt (qwen3.5: thinking/response generation prompt) ----
NB. The real template (GGUF tokenizer.chat_template) renders the generation
NB. prompt with angle-bracket tags: thinking LF LF response LF LF (default) or
NB. thinking LF (enable_thinking=true). Rendered via chat_tmpl_render.



NB. ---- Generate (chat-template single turn) ----
qw35_generate =: 4 : 0
  llm =. x
  args =. gen_args y
  text =. > 0 { args
  max_steps =. > 1 { args
  temp =. > 2 { args
  k =. > 3 { args
  p =. > 4 { args
  min_p =. > 5 { args

  messages =. <('user') ; text
  prompt =. qw35_chat_prompt messages
  tokens =. gpt2_tokenize (<llm) , <prompt
  stop =. qw35_stop_tokens llm
  L =. # , > tokens
  output =. llm gen_loop_core (tokens ; '' ; max_steps ; temp ; k ; p ; min_p ; <stop)
  gen =. L }. output
  gpt2_detokenize (<llm) , <gen
)

NB. ---- Batched generation: B independent prompts in parallel ----
NB. Usage: llm qw35_generate_batch (prompts ; max_steps ; <temp;k;p;min_p>)
qw35_generate_batch =: 4 : 0
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
    prompt =. qw35_chat_prompt messages
    tokens =. gpt2_tokenize (llm_box , <prompt)
    tok_list =. , > tokens
    prompts_tok =. prompts_tok , <tok_list
    prompts_len =. prompts_len , <(# tok_list)
    i =. i + 1
  end.
  stop =. qw35_stop_tokens llm
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

NB. ---- Batched-PREFILL attention (qwen35): B chunks, one per seq at pos[b] ----
NB. Mirrors qw35_attention_b but for prefill: hidden = (B, c, emb).  Q+GATE (fused),
NB. K/V projections + per-head Q/K norm + PARTIAL NEOX RoPE + Q pre-scale are
NB. BATCHED across all B*c rows; the GQA scores/softmax/V + sigmoid-gate run
NB. per-sequence.  Optional 5th arg = lens (padding masking).
NB. x = hidden (B, c, emb); y = <block_data; pos; mi; layer> (+ lens).
qw35_attention_bp =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  pos =. > 1 { y
  mi =. > 2 { y
  layer =. > 3 { y
  lens =. ''
  if. 4 < # y do. lens =. > 4 { y end.
  rope =. > 5 { y
  B =. {. $ hidden
  c =. 1 { $ hidden
  emb_len =. 2 { $ hidden
  n_heads =. qw35_bd_a_n_heads block_data
  head_dim =. qw35_bd_a_head_dim block_data
  n_heads_kv =. qw35_bd_a_n_heads_kv block_data
  n_groups =. n_heads % n_heads_kv
  n_rot =. qw35_mi_n_rot mi
  half =. <. n_rot % 2
  eff_seq =. > 1 { kv_meta

  NB. Attention norm per row (batched)
  hidden_flat =. ((B*c) , emb_len) $ , hidden
  normed =. rms_norm_rows ((< mi_rms_eps mi) , (< qw35_bd_attn_norm block_data) , <hidden_flat)
  nt =. |: normed   NB. |: normed hoisted once — 3 (attn) / 4 (ssm) transposes were materializing copies

  NB. Fused Q+GATE projection (wq = 2*head_dim per head), then K/V
  qg =. |: ((qw35_bd_a_q block_data) (+/ .*) nt)   NB. (B*c, 2*head_dim*n_heads)
  qg3 =. (B , c , n_heads , 2 * head_dim) $ , qg
  Q =. head_dim {."1 qg3    NB. (B, c, n_heads, head_dim)
  gate =. head_dim }."1 qg3  NB. (B, c, n_heads, head_dim)
  kv =. |: ((qw35_bd_a_k block_data) (+/ .*) nt)   NB. (B*c, n_kv*head_dim)
  vv =. |: ((qw35_bd_a_v block_data) (+/ .*) nt)
  K =. (B , c , n_heads_kv , head_dim) $ , kv
  V =. (B , c , n_heads_kv , head_dim) $ , vv

  NB. Per-head RMSNorm on Q and K BEFORE RoPE
  Qf =. ((B*c*n_heads) , head_dim) $ , Q
  Qf =. rms_norm_rows ((< mi_rms_eps mi) , (< qw35_bd_a_q_norm block_data) , <Qf)
  Q =. (B , c , n_heads , head_dim) $ , Qf
  Kf =. ((B*c*n_heads_kv) , head_dim) $ , K
  Kf =. rms_norm_rows ((< mi_rms_eps mi) , (< qw35_bd_a_k_norm block_data) , <Kf)
  K =. (B , c , n_heads_kv , head_dim) $ , Kf

  NB. Partial NEOX RoPE at the (B, c) positions pos[b]+i.c (n_rot dims, pairs
  NB. (i, i+half)). cos/sin tables + expansions are layer-invariant (depend only
  NB. on pos) — hoisted once in run_blocks_bp, threaded through y as <cos_all;
  NB. sin_all; cos_expq; sin_expq; cos_expk; sin_expk>.
  cos_all =. > 0 { rope   NB. (B*c, half)
  sin_all =. > 1 { rope
  cos_expq =. > 2 { rope
  sin_expq =. > 3 { rope
  Qa =. half {."1 Q
  Qb =. half {."1 (half }."1 Q)
  Qa_out =. (Qa * cos_expq) - (Qb * sin_expq)
  Qb_out =. (Qa * sin_expq) + (Qb * cos_expq)
  Qtail =. (2 * half) }."1 Q
  Q =. (B , c , n_heads , head_dim) $ , ((Qa_out ,"1 Qb_out) ,"1 Qtail)
  cos_expk =. > 4 { rope
  sin_expk =. > 5 { rope
  Ka =. half {."1 K
  Kb =. half {."1 (half }."1 K)
  Ka_out =. (Ka * cos_expk) - (Kb * sin_expk)
  Kb_out =. (Ka * sin_expk) + (Kb * cos_expk)
  Ktail =. (2 * half) }."1 K
  K =. (B , c , n_heads_kv , head_dim) $ , ((Ka_out ,"1 Kb_out) ,"1 Ktail)

  NB. Scale Q by 1/sqrt(head_dim)
  Q =. Q % head_dim ^ 0.5

  NB. Per-sequence GQA + gated output (scores/softmax/V per seq; causal + lens mask)
  attn_out =. ''
  b =. 0
  while. b < B do.
    q_b =. b { Q
    k_b =. b { K
    v_b =. b { V
    gate_b =. b { gate
    pos_b =. b { pos
    base_b =. ((layer * kv_batch_g) + b) * eff_seq
    idxw =. base_b + pos_b + i. c
    k_cache_g =: ((c, n_heads_kv*head_dim) $ , k_b) idxw} k_cache_g
    v_cache_g =: ((c, n_heads_kv*head_dim) $ , v_b) idxw} v_cache_g
    kv_pos_g =: kv_pos_g >. pos_b + c
    win =. pos_b + c
    k_all =. (win, n_heads_kv, head_dim) $ , ((base_b + i. win) { k_cache_g)
    v_all =. (win, n_heads_kv, head_dim) $ , ((base_b + i. win) { v_cache_g)
    mask_2d =. (pos_b + i. c) </ i. win
    if. 0 < # lens do.
      mask_2d =. mask_2d +. ((i. win) >: b { lens)
    end.
    mask_g2 =. ((n_groups * c), win) $ , (2 0 1 |: (mask_2d (*/) (n_groups $ 1)))
    Qp =. 1 0 2 |: q_b
    Q_g2 =. (n_heads_kv, (n_groups*c), head_dim) $ , Qp   NB. one ravel+reshape (the intermediate 4D reshape was redundant)
    Kp2 =. 1 2 0 |: k_all
    scores2 =. Q_g2 (+/ .* "2) Kp2
    scores2 =. scores2 -"2 (mask_g2 * 1e9)
    NB. Softmax directly on the 3D scores2 (the old scores_f flatten +
    NB. softmax_g2 re-group were 2 redundant reshape copies per layer)
    max_sf =. >./"1 scores2
    exp_sf =. ^ (scores2 - max_sf)
    softmax_f =. exp_sf % +/"1 exp_sf
    Vp =. 1 0 2 |: v_all
    attn2 =. softmax_f (+/ .* "2) Vp
    attn_out =. attn_out , <((c, n_heads*head_dim) $ , (((c, n_heads, head_dim) $ , (1 0 2 |: ((n_heads, c, head_dim) $ , attn2))) * (sigmoid gate_b)))
    b =. b + 1
  end.
  attn_all =. > attn_out
  out =. |: ((qw35_bd_a_o block_data) (+/ .*) |: (((B*c) , (n_heads*head_dim)) $ , attn_all))
  out =. (B, c, emb_len) $ , out
  (<out)
)

NB. ---- Batched-PREFILL SSM (gated delta-net) layer (B chunks, c tokens each) ----
NB. Batched projections (qkv/gate/beta/alpha) across B*c rows; the conv1d +
NB. delta-net recurrence run per sequence (conv + s state per seq via rs_read_b/
NB. rs_write_b); the norm-gated output + out_proj + residual + FFN are batched.
NB. x = hidden (B, c, emb); y = <block_data; pos; mi; layer> (+ lens).
qw35_ssm_forward_bp =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  pos =. > 1 { y
  mi =. > 2 { y
  layer =. > 3 { y
  lens =. ''
  if. 4 < # y do. lens =. > 4 { y end.
  B =. {. $ hidden
  c =. 1 { $ hidden
  emb_len =. 2 { $ hidden
  head_v_dim =. qw35_bd_s_head_v_dim block_data
  num_v_heads =. qw35_bd_s_n_v_heads block_data
  d_inner =. qw35_bd_s_d_inner block_data

  NB. Attention norm per row (batched)
  hidden_flat =. ((B*c) , emb_len) $ , hidden
  normed =. rms_norm_rows ((< mi_rms_eps mi) , (< qw35_bd_attn_norm block_data) , <hidden_flat)
  nt =. |: normed   NB. |: normed hoisted once — 3 (attn) / 4 (ssm) transposes were materializing copies

  key_dim =. (qw35_bd_s_head_k_dim block_data) * (qw35_bd_s_n_k_heads block_data)
  value_dim =. head_v_dim * num_v_heads
  conv_dim =. (key_dim * 2) + value_dim

  NB. Projections (batched across B*c rows)
  qkv_mixed =. |: ((qw35_bd_s_wqkv block_data) (+/ .*) nt)   NB. (B*c, conv_dim)
  z =. |: ((qw35_bd_s_gate block_data) (+/ .*) nt)   NB. (B*c, value_dim)
  beta_p =. |: ((qw35_bd_s_beta block_data) (+/ .*) nt)   NB. (B*c, num_v_heads)
  beta =. sigmoid beta_p
  alpha_p =. |: ((qw35_bd_s_alpha block_data) (+/ .*) nt)   NB. (B*c, num_v_heads)
  alpha =. alpha_p + (((B*c) , num_v_heads) $ qw35_bd_s_dt block_data)
  alpha =. softplus alpha
  gate =. alpha * (((B*c) , num_v_heads) $ qw35_bd_s_a block_data)   NB. decay base

  NB. Conv1d + recurrence per sequence (c tokens each; conv + s state per seq)
  conv_w =. qw35_bd_s_conv1d block_data   NB. (conv_dim, 4)
  scale =. 1 % head_v_dim ^ 0.5
  finals =. ''
  b =. 0
  while. b < B do.
    qkv_b =. (c, conv_dim) $ , ((b * c) + i. c) { qkv_mixed
    z_b =. (c, value_dim) $ , ((b * c) + i. c) { z
    beta_b =. (c, num_v_heads) $ , ((b * c) + i. c) { beta
    gate_b =. (c, num_v_heads) $ , ((b * c) + i. c) { gate
    rs_state =. rs_read_b ((<layer) , <b)
    conv_state =. > 0 { rs_state   NB. (3, conv_dim)
    input =. conv_state , qkv_b   NB. (3+c, conv_dim)
    conv_out =. ((c , conv_dim) $ (0 {"1 conv_w)) * ((i. c) { input)
    conv_out =. conv_out + ((c , conv_dim) $ (1 {"1 conv_w)) * ((1 + i. c) { input)
    conv_out =. conv_out + ((c , conv_dim) $ (2 {"1 conv_w)) * ((2 + i. c) { input)
    conv_out =. conv_out + ((c , conv_dim) $ (3 {"1 conv_w)) * ((3 + i. c) { input)
    conv_out =. silu conv_out   NB. (c, conv_dim)

    NB. Split Q/K/V channels
    q_conv =. key_dim {."1 conv_out
    k_conv =. key_dim {."1 (key_dim }."1 conv_out)
    v_conv =. (2 * key_dim) }."1 conv_out

    NB. Reshape + L2-norm q and k
    q3 =. (c , num_v_heads , head_v_dim) $ , q_conv
    k3 =. (c , num_v_heads , head_v_dim) $ , k_conv
    v3 =. (c , num_v_heads , head_v_dim) $ , v_conv
    qf =. ((c * num_v_heads) , head_v_dim) $ , q3
    qf =. l2norm_rows ((< mi_rms_eps mi) , <qf)
    q =. (c , num_v_heads , head_v_dim) $ , qf
    kf =. ((c * num_v_heads) , head_v_dim) $ , k3
    kf =. l2norm_rows ((< mi_rms_eps mi) , <kf)
    k =. (c , num_v_heads , head_v_dim) $ , kf

    NB. Recurrence (c tokens)
    s_state =. > 1 { rs_state
    rec =. qw35_ssm_recur ((<s_state) , (<q) , (<k) , (<v3) , (<beta_b) , (<gate_b) , <scale)
    s_new =. > 0 { rec
    o =. > 1 { rec   NB. (c, num_v_heads, head_v_dim)

    NB. Norm-gated output: rms_norm(o, ssm_norm) * silu(z)
    o_flat =. ((c * num_v_heads) , head_v_dim) $ , o
    o_n =. rms_norm_rows ((< mi_rms_eps mi) , (< qw35_bd_s_norm block_data) , <o_flat)
    z3 =. (c , num_v_heads , head_v_dim) $ , z_b
    z_flat =. ((c * num_v_heads) , head_v_dim) $ , z3
    gated =. o_n * silu z_flat
    final =. (c , d_inner) $ , gated
    finals =. finals , <final

    NB. Update conv state (last 3 REAL rows) + s state.  input = [conv_state(3);
    NB. qkv(c)], so the last 3 real qkv rows sit at input rows (real_c, real_c+1,
    NB. real_c+2) — padding must NOT contaminate the sliding window.
    if. 0 < # lens do.
      real_c =. c <. ((b { lens) - (0 { pos))
    else.
      real_c =. c
    end.
    if. real_c > 0 do.
      new_conv =. (real_c + i. 3) { input
      rs_write_b ((<layer) , (<new_conv) , (<s_new) , <b)
    end.
    b =. b + 1
  end.

  NB. out_proj + residual + FFN (batched)
  final_all =. (B , c , d_inner) $ , > finals
  final_flat =. ((B*c) , d_inner) $ , final_all
  out =. |: ((qw35_bd_s_out block_data) (+/ .*) |: final_flat)   NB. (B*c, emb)
  out =. out + hidden_flat
  post =. rms_norm_rows ((< mi_rms_eps mi) , (< qw35_bd_post_norm block_data) , <out)
  pt =. |: post   NB. |: post hoisted once — 2 transposes were materializing 2 copies
  gate_f =. |: ((qw35_bd_ff_gate block_data) (+/ .*) pt)
  up_f =. |: ((qw35_bd_ff_up block_data) (+/ .*) pt)
  ffn_raw =. |: ((qw35_bd_ff_down block_data) (+/ .*) |: (gate_f swiglu up_f))
  output_flat =. ffn_raw + out
  output =. (B , c , emb_len) $ , output_flat
  (<output)
)

NB. ---- Batched-PREFILL block forwards (qwen35) ----
qw35_block_forward_a_bp =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  pos =. > 1 { y
  mi =. > 2 { y
  layer =. > 3 { y
  lens =. ''
  if. 4 < # y do. lens =. > 4 { y end.
  rope =. > 5 { y
  B =. {. $ hidden
  c =. 1 { $ hidden
  emb_len =. 2 { $ hidden
  input =. hidden
  attn_result =. hidden qw35_attention_bp y
  attn_out =. > 0 { attn_result
  sa_out =. attn_out + input
  post =. rms_norm_rows ((< mi_rms_eps mi) , (< qw35_bd_post_norm block_data) , <(((B*c) , emb_len) $ , sa_out))
  pt =. |: post   NB. |: post hoisted once — 2 transposes were materializing 2 copies
  gate_f =. |: ((qw35_bd_ff_gate block_data) (+/ .*) pt)
  up_f =. |: ((qw35_bd_ff_up block_data) (+/ .*) pt)
  ffn_raw =. |: ((qw35_bd_ff_down block_data) (+/ .*) |: (gate_f swiglu up_f))
  output_flat =. ffn_raw + (((B*c) , emb_len) $ , sa_out)
  output =. (B , c , emb_len) $ , output_flat
  (<output)
)

qw35_block_forward_s_bp =: qw35_ssm_forward_bp

NB. ---- Run all blocks for B sequences (one CHUNK each at pos[b]) ----
qw35_run_blocks_bp =: 4 : 0
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
  NB. SSM recurrent state: batched prefill needs it (rs_batch_g = kv_batch_g = B).
  if. 0 = # rs_meta do.
    rs_create ((<18) , (<3) , (<6144) , (<128) , <16)
  end.
  NB. RoPE cos/sin tables + expansions are layer-invariant (depend only on
  NB. pos) — compute ONCE per chunk, thread through (attention layers only;
  NB. SSM layers don't use RoPE).
  n_rot =. qw35_mi_n_rot mi
  half =. <. n_rot % 2
  B =. {. $ state
  c =. 1 { $ state
  pos_bc_flat =. (B*c) $ , (pos +/ i. c)
  cos_all =. pos_bc_flat { mi_cos_tab mi   NB. (B*c, half)
  sin_all =. pos_bc_flat { mi_sin_tab mi
  cos_expq =. (B , c , n_heads , half) $ , ((0 2 1) |: (((B*c) , half , n_heads) $ , (cos_all (*/) (n_heads $ 1))))
  sin_expq =. (B , c , n_heads , half) $ , ((0 2 1) |: (((B*c) , half , n_heads) $ , (sin_all (*/) (n_heads $ 1))))
  cos_expk =. (B , c , n_heads_kv , half) $ , ((0 2 1) |: (((B*c) , half , n_heads_kv) $ , (cos_all (*/) (n_heads_kv $ 1))))
  sin_expk =. (B , c , n_heads_kv , half) $ , ((0 2 1) |: (((B*c) , half , n_heads_kv) $ , (sin_all (*/) (n_heads_kv $ 1))))
  rope =. (<cos_all) , (<sin_all) , (<cos_expq) , (<sin_expq) , (<cos_expk) , (<sin_expk)
  b =. 0
  block_data_list =. llm_block_data llm
  while. b < block_count do.
    block_data =. > b { block_data_list
    if. qw35_bd_is_ssm block_data do.
      result =. state qw35_block_forward_s_bp ((<block_data) , (<pos) , (<mi) , (<b) , (<lens))
    else.
      result =. state qw35_block_forward_a_bp ((<block_data) , (<pos) , (<mi) , (<b) , (<lens) , <rope)
    end.
    state =. > 0 { result
    b =. b + 1
  end.
  <state
)

NB. ---- Stop tokens ----
qw35_stop_tokens =: 3 : 0
  tk =. llm_tokenizer y
  tokenizer_eos_g tk
)

NB. ---- Simple wrappers ----
qw35_infer_simple =: qw35_infer (] ; (<0 0 0.95 0.0)"_)
qw35_generate_simple =: qw35_generate (0&{ , 1&{ , (<0 0 0.95 0.0)"_)
