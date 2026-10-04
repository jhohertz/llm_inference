NB. ================================================================
NB. Gemma 3 270M Architecture — standard decoder transformer
NB. Depends on: llm_core.ijs (kernels/gguf/kv_cache/sampler), tokenizer_llama3.ijs
NB. Tensor names from gemma-3-270m-it-F16.gguf
NB. ================================================================
coclass 'inference'
require 'llm/inference/util/llm_core'
require 'llm/inference/tokenizers/tokenizer_llama3'
require 'llm/inference/util/minja'
require 'llm/inference/util/chat_template'

NB. ---- Helper: move axes (dyadic |:) using variable axis list ----

NB. ---- Gemma3-specific mi accessor ----
NB. mi layout matches llm_core (indices 0-11: block_count..sin_tab); swa appended at 12.
mi_swa         =: 3 : 0
  d =. y
  > get__d (<'swa')
)

NB. ---- block_data accessors (gemma3-specific) ----
NB. block_data = <attn_q; attn_o; q_norm; k_norm; attn_pn; ff_norm; ff_gate; ff_up; ff_down; ff_pn; attn_norm; n_heads; head_dim; rope_freq; n_heads_kv; n_ff; fused_ff_gu; fused_qkv; cos_tab; sin_tab; swa_l>
NB. cos_tab/sin_tab are PER-LAYER RoPE tables: gemma3 uses rope_freq_base=1e6 for the
NB. dense layers (5,11,17) and rope_freq_base=10000 for the SWA (sliding-attention)
NB. layers — the SWA pattern is `il % 6 < 5` (SWA), `il % 6 = 5` (dense), matching
NB. llama.cpp set_swa_pattern(6, dense_first=0) and HF layer_types.
NB. swa_l = the layer's sliding window (512 for SWA layers, 0 for dense layers).
gem3_bd_attn_q      =: >@(0&{)
gem3_bd_attn_o      =: >@(1&{)
gem3_bd_q_norm      =: >@(2&{)
gem3_bd_k_norm      =: >@(3&{)
gem3_bd_attn_pn     =: >@(4&{)
gem3_bd_ff_norm     =: >@(5&{)
gem3_bd_ff_gate     =: >@(6&{)
gem3_bd_ff_up       =: >@(7&{)
gem3_bd_ff_down     =: >@(8&{)
gem3_bd_ff_pn       =: >@(9&{)
gem3_bd_attn_norm   =: >@(10&{)
gem3_bd_n_heads     =: >@(11&{)
gem3_bd_head_dim    =: >@(12&{)
gem3_bd_rope_freq   =: >@(13&{)
gem3_bd_n_heads_kv  =: >@(14&{)
gem3_bd_n_ff        =: >@(15&{)
gem3_bd_fused_ff_gu =: >@(16&{)
gem3_bd_fused_qkv   =: >@(17&{)
gem3_bd_cos_tab     =: >@(18&{)
gem3_bd_sin_tab     =: >@(19&{)
gem3_bd_swa_l       =: >@(20&{)
NB. KV helpers: key kv_uint/kv_float data
gem3_kv_uint =: 4 : 0
  key =. x
  data =. y
  key kv_uint data
)

gem3_kv_float =: 4 : 0
  key =. x
  data =. y
  key kv_float data
)

NB. ---- Extract Gemma3 model info from KV pairs ----
gem3_extract_hparams =: 3 : 0
  data =. y
  block_count =. 'gemma3.block_count' gem3_kv_uint data
  context_length =. 'gemma3.context_length' gem3_kv_uint data
  emb_len =. 'gemma3.embedding_length' gem3_kv_uint data
  n_heads =. 'gemma3.attention.head_count' gem3_kv_uint data
  n_heads_kv =. 'gemma3.attention.head_count_kv' gem3_kv_uint data
  rope_freq =. 'gemma3.rope.freq_base' gem3_kv_float data
  vocab_size =. 'llama.vocab_size' gem3_kv_uint data
  rms_eps =. 'gemma3.attention.layer_norm_rms_epsilon' gem3_kv_float data
  swa =. 'gemma3.attention.sliding_window' gem3_kv_uint data
  n_ff =. 'gemma3.feed_forward_length' gem3_kv_uint data
   key_len =. 'gemma3.attention.key_length' gem3_kv_uint data
   head_dim =. key_len % n_heads_kv
  (<"0) block_count , context_length , emb_len , n_heads , n_heads_kv , head_dim , rope_freq , vocab_size , rms_eps , n_ff
)

NB. ---- Build one block's data (rank-1; x=llm, y=block index) ----
gem3_build_block =: 4 : 0
  llm =. x
  b =. y
  mi =. llm_mi llm
  n_embd =. mi_emb_len mi
  n_heads =. mi_n_heads mi
  rope_freq =. mi_rope_freq mi
  n_heads_kv =. mi_n_heads_kv mi
  n_ff =. mi_n_ff mi
  p =. 'blk.' , (": b) , '.'
  attn_norm =. (p , 'attn_norm.weight') get_tensor_cached_d llm
  attn_q =. (p , 'attn_q.weight') get_tensor_cached_d llm
  attn_k =. (p , 'attn_k.weight') get_tensor_cached_d llm
  attn_v =. (p , 'attn_v.weight') get_tensor_cached_d llm
  attn_o_w =. (p , 'attn_output.weight') get_tensor_cached_d llm
  attn_o =. attn_o_w
  q_norm =. (p , 'attn_q_norm.weight') get_tensor_cached_d llm
  k_norm =. (p , 'attn_k_norm.weight') get_tensor_cached_d llm
  attn_pn =. (p , 'post_attention_norm.weight') get_tensor_cached_d llm
  ff_norm =. (p , 'ffn_norm.weight') get_tensor_cached_d llm
  ff_gate =. (p , 'ffn_gate.weight') get_tensor_cached_d llm
  ff_up =. (p , 'ffn_up.weight') get_tensor_cached_d llm
  ff_down =. (p , 'ffn_down.weight') get_tensor_cached_d llm
  ff_pn =. (p , 'post_ffw_norm.weight') get_tensor_cached_d llm
  head_dim =. (0 { ( $ attn_q)) % n_heads
  NB. Weights are [out, in]; fuse by concatenating along output axis (rows)
  fused_ff_gu =. ff_gate , ff_up          NB. (2*n_ff, n_embd)
  fused_qkv =. attn_q , attn_k , attn_v   NB. ((n_heads+n_kv*2)*head_dim, n_embd)

  NB. Per-layer RoPE tables + sliding window (gemma3 SWA pattern: il%6<5 => SWA)
  NB. SWA layers use rope_freq_base=10000; dense layers (il%6=5: 5,11,17) use 1e6.
  NB. Dense layers have NO sliding window (swa_l=0).
  is_swa_l =. (6 | b) < 5
  swa_l =. (mi_swa mi) * is_swa_l
  freq_l =. 10000 + ((mi_rope_freq mi) - 10000) * -. is_swa_l
  tabs =. build_rope_tables ((< mi_context_len mi) , (< head_dim) , (< freq_l))
  cos_tab =. > 0 { tabs
  sin_tab =. > 1 { tabs

  (<attn_q),(<attn_o),(<q_norm),(<k_norm),(<attn_pn),(<ff_norm),(<ff_gate),(<ff_up),(<ff_down),(<ff_pn),(<attn_norm),(<n_heads),(<head_dim),(<rope_freq),(<n_heads_kv),(<n_ff),(<fused_ff_gu),(<fused_qkv),(<cos_tab),(<sin_tab),(<swa_l)
)

NB. ---- Pre-build all block data for Gemma3 ----
gem3_pre_build_block_data =: 3 : 0
  llm =. y
  block_count =. mi_block_count (llm_mi llm)
  (<llm) gem3_build_block each i. block_count
)

NB. ---- Standard self-attention for Gemma3 ----
NB. Computes attention for the CURRENT token only against all cached K/V
NB. x = hidden (emb,) — the layer input; y = <block_data; pos; swa; mi; layer>
gem3_attention =: 4 : 0
  'block_data pos swa mi layer' =. y
  
  n_heads =. gem3_bd_n_heads block_data
  head_dim =. gem3_bd_head_dim block_data
  rope_freq =. gem3_bd_rope_freq block_data
  n_heads_kv =. gem3_bd_n_heads_kv block_data
  
  NB. Attention norm
  attn_norm_w =. gem3_bd_attn_norm block_data
  hidden =. rms_norm ((< mi_rms_eps mi) , (< attn_norm_w) , <x)
  
   NB. Q, K, V projections — fused single matmul
   fused_qkv_w =. gem3_bd_fused_qkv block_data
   qkv =. fused_qkv_w linear_r hidden
   q_len =. n_heads * head_dim
   kv_len =. n_heads_kv * head_dim
   Q =. q_len {. qkv
   K =. kv_len {. q_len }. qkv
   V =. kv_len {. (q_len + kv_len) }. qkv
   Q =. (n_heads, head_dim) $ Q
   K =. (n_heads_kv, head_dim) $ K
   V =. (n_heads_kv, head_dim) $ V
  
  NB. Q/K norm
  q_norm_w =. gem3_bd_q_norm block_data
  k_norm_w =. gem3_bd_k_norm block_data
  Q =. rms_norm_rows ((< mi_rms_eps mi) , (< q_norm_w) , <Q)
  K =. rms_norm_rows ((< mi_rms_eps mi) , (< k_norm_w) , <K)
  
    NB. RoPE — apply per head (Gemma3 uses NEOX layout: pairs offset by head_dim/2)
    NB. cos/sin precomputed once per position in PER-LAYER tables (dense vs SWA freq)
    cos_t =. pos { gem3_bd_cos_tab block_data
    sin_t =. pos { gem3_bd_sin_tab block_data
    rope_t =. (<cos_t) , (<sin_t)
    Q =. (n_heads, head_dim) $ ,Q
    Q =. Q rope_apply2_neox_t rope_t
    K =. (n_heads_kv, head_dim) $ ,K
    K =. K rope_apply2_neox_t rope_t
   NB. llama.cpp: pre-scale Q by 1/sqrt(head_dim) for Gemma (after RoPE)
   Q =. Q % head_dim ^ 0.5
  
   NB. Write K, V to KV cache at current position
   kv_write ((4 { y) , (1 { y) , (<K) , (<V))
   
   NB. Read all K, V from cache up to current position
   kv_result =. kv_read ((4 { y) , (1 { y))
   k_all =. > 0 { kv_result
   v_all =. > 1 { kv_result
  
  NB. k_all: (win, n_heads_kv, head_dim)
  NB. v_all: (win, n_heads_kv, head_dim)
   win =. pos + 1
   
    NB. Q: (n_heads, head_dim) — current token only
    NB. k_all: (win, n_heads_kv, head_dim)
    NB. v_all: (win, n_heads_kv, head_dim)
      
        n_groups =. n_heads % n_heads_kv
      
        NB. Compute Q·K^T: (n_heads, head_dim) · (head_dim, n_heads_kv, win) → (n_heads, n_heads_kv, win)
         NB. Move last axis to front: (win,nhkv,hd) -> (hd,nhkv,win)
          k_trans =. 2 0 1 |: k_all
          scores =. Q (+/ .* ) k_trans
       
        NB. Causal + sliding window mask: 1=masked, 0=valid
        NB. swa_l is per-layer: SWA layers use 512, dense layers (5,11,17) use 0 (no window)
        swa_l =. gem3_bd_swa_l block_data
        mask_1d =. (i. win) < (win - swa_l)
        mask_1d =. (swa_l > 0) *. mask_1d
       NB. Scores shape is (n_heads, win, n_heads_kv) due to axis permute 2 0 1
       NB. Build mask to match: (n_heads, win, n_heads_kv)
        mask_3d =. (n_heads, win, n_heads_kv) $ mask_1d
       NB. Apply mask to scores before softmax
       scores =. scores - (mask_3d * 1e9)
      
       NB. Reshape to (n_heads, n_heads_kv * win) for per-row softmax
       scores_f =. (n_heads, n_heads_kv * win) $ ,scores
      
        NB. Softmax per row (each row = one query-head's distribution)
        max_sf =. >./"1 scores_f
        NB. Broadcast max_sf (n_heads,) along rows (J: (n_heads,) is a prefix of (n_heads, win))
        exp_sf =. ^ (scores_f - max_sf)
        softmax_f =. exp_sf % +/"1 exp_sf
      
      NB. Reshape softmax back to (n_heads, n_heads_kv, win)
      softmax =. (n_heads, n_heads_kv, win) $ ,softmax_f
      
      NB. Permute v_all from (win, nk, hd) to (nk, win, hd), then flatten
      v_flat =. (n_heads_kv * win, head_dim) $ , 1 0 2 |: v_all
      
      NB. Flatten softmax to 2D for matrix multiply
      softmax_flat =. (n_heads, n_heads_kv * win) $ , softmax
        attn_raw =. softmax_flat (+/ .* ) v_flat   NB. (n_heads, head_dim)
      
         NB. Output projection: attn_o is [in, out] = [n_heads*head_dim, emb_len]
         NB. Flatten attn_raw from (n_heads, head_dim) to (n_heads*head_dim,) for matmul
         attn_o_w =. gem3_bd_attn_o block_data
         attn_raw_flat =. (n_heads * head_dim) $ , attn_raw
         attn_out =. attn_o_w (+/ .* ) attn_raw_flat
     
     NB. Post-attention norm
    attn_pn_w =. gem3_bd_attn_pn block_data
   attn_out =. rms_norm ((< mi_rms_eps mi) , (< attn_pn_w) , <attn_out)
   
    (<attn_out)
)

NB. ---- Batched self-attention (prompt prefill) ----
NB. hidden = (L, emb) matrix for L prompt tokens; computes causal attention for ALL positions
NB. in one batched pass (blocks outer, tokens inner). KV cache filled in bulk.
NB. y = <block_data; swa; mi; layer; start_pos>   (positions start_pos .. start_pos+L-1)
NB. start_pos=0 -> fresh (no prefix); start_pos>0 -> RESUME: attends to the cache prefix
NB. (positions 0..start_pos-1) PLUS this batch, and writes the batch K/V at start_pos.
gem3_attention_b =: 4 : 0
  block_data =. > 0 { y
  layer =. > 1 { y
  swa =. > 2 { y
  mi =. > 3 { y
  start_pos =. > 4 { y
  L =. {. $ x
  n_embd =. {: $ x
  
  n_heads =. gem3_bd_n_heads block_data
  head_dim =. gem3_bd_head_dim block_data
  n_heads_kv =. gem3_bd_n_heads_kv block_data
  half =. <. head_dim % 2
  
  NB. Attention norm per row
  attn_norm_w =. gem3_bd_attn_norm block_data
  hidden =. rms_norm_rows ((< mi_rms_eps mi) , (< attn_norm_w) , <x)
  
  NB. Q, K, V projections — batched single matmul
  fused_qkv_w =. gem3_bd_fused_qkv block_data
  qkv =. |: (fused_qkv_w (+/ .* ) |: hidden)   NB. (L, qkv_len)
  q_len =. n_heads * head_dim
  kv_len =. n_heads_kv * head_dim
  Q =. (q_len {. "1 qkv)                        NB. (L, q_len)
  K =. (kv_len {. "1 (q_len }. "1 qkv))        NB. (L, kv_len)
  V =. (kv_len {. "1 ((q_len + kv_len) }. "1 qkv)) NB. (L, kv_len)
  Q =. (L, n_heads, head_dim) $ ,Q
  K =. (L, n_heads_kv, head_dim) $ ,K
  V =. (L, n_heads_kv, head_dim) $ ,V
  
  NB. Q/K norm per head (flatten heads, norm rows, reshape)
  q_norm_w =. gem3_bd_q_norm block_data
  k_norm_w =. gem3_bd_k_norm block_data
  Qf =. ((L * n_heads) , head_dim) $ ,Q
  Qf =. rms_norm_rows ((< mi_rms_eps mi) , (< q_norm_w) , <Qf)
  Q =. (L, n_heads, head_dim) $ ,Qf
  Kf =. ((L * n_heads_kv) , head_dim) $ ,K
  Kf =. rms_norm_rows ((< mi_rms_eps mi) , (< k_norm_w) , <Kf)
  K =. (L, n_heads_kv, head_dim) $ ,Kf
  
  NB. RoPE — batched, table-based (NEOX). cos/sin per position, PER-LAYER tables.
  cos_all =. (start_pos + i. L) { gem3_bd_cos_tab block_data    NB. (L, half)
  sin_all =. (start_pos + i. L) { gem3_bd_sin_tab block_data
  Qa =. half {. "1 Q                     NB. (L, n_heads, half)
  Qb =. half }. "1 Q
  cos_exp =. (0 2 1) |: ((L , half , n_heads) $ , (cos_all (*/) (n_heads $ 1)))
  sin_exp =. (0 2 1) |: ((L , half , n_heads) $ , (sin_all (*/) (n_heads $ 1)))
  Qa_out =. (Qa * cos_exp) - (Qb * sin_exp)
  Qb_out =. (Qa * sin_exp) + (Qb * cos_exp)
  Q =. (L, n_heads, head_dim) $ ,(Qa_out ,"1 Qb_out)
  Ka =. half {. "1 K                     NB. (L, n_heads_kv, half)
  Kb =. half }. "1 K
  cos_expk =. (0 2 1) |: ((L , half , n_heads_kv) $ , (cos_all (*/) (n_heads_kv $ 1)))
  sin_expk =. (0 2 1) |: ((L , half , n_heads_kv) $ , (sin_all (*/) (n_heads_kv $ 1)))
  Ka_out =. (Ka * cos_expk) - (Kb * sin_expk)
  Kb_out =. (Ka * sin_expk) + (Kb * cos_expk)
  K =. (L, n_heads_kv, head_dim) $ ,(Ka_out ,"1 Kb_out)
  
  NB. Pre-scale Q by 1/sqrt(head_dim) (Gemma)
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

  NB. Bulk-write the new batch's L K/V rows into cache at layer, starting at start_pos
  NB. (one amend per row box). K/V above include the prefix — write the saved batch.
  kv_write_rows ((<0) , (1 { y) , (4 { y) , <K_batch)
  kv_write_rows ((<1) , (<layer) , (<start_pos) , <V_batch)

  NB. Batched causal+sliding-window scores: Qf·Kf^T over all (t,j) in
  NB. [prefix + batch] keys.
  Qf =. ((L * n_heads) , head_dim) $ ,Q
  Kf =. (((start_pos + L) * n_heads_kv) , head_dim) $ ,K
  scores_all =. Qf (+/ .* ) |: Kf            NB. (L*nh, (start_pos+L)*nk)

  NB. mask[t,j]=1 if j>t (future) or j < t-swa_l+1 (out of window); swa_l=0 -> causal only.
  NB. Query t sits at position start_pos+t; keys are positions 0..start_pos+L-1.
  key_pos =. i. (start_pos + L)
  q_pos =. start_pos + i. L
  mask_future =. q_pos </ key_pos           NB. (start_pos+t) < j -> 1
  swa_l =. gem3_bd_swa_l block_data
  mask_swa =. ((q_pos - swa_l) + 1) >/ key_pos  NB. mask j < (start_pos+t)-swa_l+1 (window swa_l)
  if. swa_l <: 0 do. mask_swa =. (L, start_pos + L) $ 0 end.
  mask_2d =. mask_future +. mask_swa
  NB. Fused mask: expand mask_2d (L,L) straight to (L*nh*nk, L) — row t*nh*nk + h*nk + k,
  NB. col j — skipping the (L,L,nh,nk) 4D expansion + (0 2 1 3) permute + (L,nh,L,nk) reshape.
  NB. Outer product gives (L,L,nh*nk) in [t,j,g] order; permute (0 2 1) -> (L,nh*nk,L) [t,g,j]
  NB. so the ravel/reshape rows are (t,g) with g = h*nk+k, matching scores_f's row order.
  mask_3d =. (0 2 1) |: (mask_2d (*/) ((n_heads * n_heads_kv) $ 1))
  mask_f =. ((L * n_heads * n_heads_kv) , start_pos + L) $ ,mask_3d
  scores_f =. ((L * n_heads * n_heads_kv) , start_pos + L) $ ,scores_all
  scores_f =. scores_f - (mask_f * 1e9)

  NB. Per-row softmax over j axis
  max_sf =. >./"1 scores_f
  exp_sf =. ^ (scores_f - max_sf)
  softmax_f =. exp_sf % +/"1 exp_sf
  softmax =. (L, n_heads, n_heads_kv, start_pos + L) $ ,softmax_f

  NB. attn_raw[t,h] = sum_{k,j} softmax[t,h,k,j] * V[j,k]
  softmax_flat =. ((L * n_heads) , (n_heads_kv * (start_pos + L))) $ , softmax
  v_flat =. ((n_heads_kv * (start_pos + L)) , head_dim) $ , (1 0 2 |: V)   NB. (L,nk,hd)->(nk,L,hd)->flat
  attn_raw_flat =. softmax_flat (+/ .* ) v_flat          NB. (L*nh, hd)
  attn_raw =. (L, n_heads, head_dim) $ ,attn_raw_flat
  
  NB. Output projection attn_o: [emb, n_heads*head_dim]; batched
  attn_o_w =. gem3_bd_attn_o block_data
  attn_out =. |: (attn_o_w (+/ .* ) |: ((L, n_heads * head_dim) $ ,attn_raw))   NB. (L, emb)
  
  NB. Post-attention norm per row
  attn_pn_w =. gem3_bd_attn_pn block_data
  attn_out =. rms_norm_rows ((< mi_rms_eps mi) , (< attn_pn_w) , <attn_out)
  
  (<attn_out)
)

NB. ---- Batched Gemma3 block forward (prefill) ----
NB. hidden = (L, emb); y = <block_data; swa; mi; layer; start_pos>
gem3_block_forward_b =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  layer =. > 1 { y
  swa =. > 2 { y
  mi =. > 3 { y
  start_pos =. > 4 { y
  
  attn_result =. hidden gem3_attention_b y
  attn_out =. > 0 { attn_result
  
  sa_out =. attn_out + hidden
  
  ff_norm_w =. gem3_bd_ff_norm block_data
  ffn_in =. rms_norm_rows ((< mi_rms_eps mi) , (< ff_norm_w) , <sa_out)
  
  fused_ff_gu_w =. gem3_bd_fused_ff_gu block_data
  gate_up =. |: (fused_ff_gu_w (+/ .* ) |: ffn_in)   NB. (L, 2*n_ff)
  n_ff =. gem3_bd_n_ff block_data
  gate_out =. (n_ff {. "1 gate_up)
  up_out =. (n_ff }. "1 gate_up)
  
  ff_down_w =. gem3_bd_ff_down block_data
  ffn_raw =. |: (ff_down_w (+/ .* ) |: (gate_out geglu up_out))   NB. GEGLU(GELU(gate)*up) -> (L, emb)
  
  ff_pn_w =. gem3_bd_ff_pn block_data
  ffn_out =. rms_norm_rows ((< mi_rms_eps mi) , (< ff_pn_w) , <ffn_raw)
  
  output =. ffn_out + sa_out
  
  (<output)
)

NB. ---- Run all Gemma3 blocks in a batched prefill pass ----
NB. input = (L, emb); args = <llm; start_pos>  (positions start_pos..start_pos+L-1;
NB. cache lives in kv_cache_g). start_pos=0 -> fresh; >0 -> resume (attends to cache prefix).
gem3_run_blocks_b =: 4 : 0
  llm =. > 0 { y
  start_pos =. > 1 { y
  mi =. llm_mi llm
  swa =. mi_swa mi
  head_dim =. mi_head_dim mi
  n_heads_kv =. mi_n_heads_kv mi
  block_count =. mi_block_count mi
  ctx_len =. mi_context_len mi
  L =. {. $ x

  state =. x
  if. 0 = # kv_meta do.
    kv_create ((<block_count) , (<ctx_len) , (<n_heads_kv) , (<head_dim))
  end.

   b =. 0
   block_data_list =. llm_block_data llm
   NB. (<swa), (<mi), (<start_pos) are layer-invariant — box once, reuse per layer.
   bfb_pre =. (<swa) , (<mi) , (<start_pos)
   while. b < block_count do.
     block_data =. > b { block_data_list
    result =. state gem3_block_forward_b (<block_data) , (<b) , bfb_pre
    state =. > 0 { result
     b =. b + 1
   end.

  <state
)

NB. ---- Batched-PREFILL attention (gemma3): B chunks, one per seq at pos[b] ----
NB. Mirrors gem3_attention_b but for prefill: hidden = (B, c, emb) — B chunks of
NB. c tokens each at pos[b]..pos[b]+c-1.  Fused QKV + per-head Q/K norm + NEOX
NB. RoPE + Q pre-scale are BATCHED across all B*c rows (weight-read amortized);
NB. the SWA/causal-masked scores/softmax/V run per-sequence.  Optional 6th arg
NB. = lens (padding masking).  x = hidden (B, c, emb); y = <block_data; swa;
NB. pos; mi; layer>.
gem3_attention_bp =: 4 : 0
  hidden =. x   NB. (B, c, emb)
  block_data =. > 0 { y
  layer =. > 1 { y
  swa =. > 2 { y
  pos =. > 3 { y
  mi =. > 4 { y
  lens =. ''
  if. 5 < # y do. lens =. > 5 { y end.
  rope =. > 6 { y
  B =. {. $ hidden
  c =. 1 { $ hidden
  emb_len =. 2 { $ hidden
  n_heads =. gem3_bd_n_heads block_data
  head_dim =. gem3_bd_head_dim block_data
  n_heads_kv =. gem3_bd_n_heads_kv block_data
  half =. <. head_dim % 2
  eff_seq =. > 1 { kv_meta

  NB. Attention norm per row (B*c)
  attn_norm_w =. gem3_bd_attn_norm block_data
  hidden_flat =. ((B*c) , emb_len) $ , hidden
  hidden_flat =. rms_norm_rows ((< mi_rms_eps mi) , (< attn_norm_w) , <hidden_flat)

  NB. Fused QKV projection batched across B*c rows
  fused_qkv_w =. gem3_bd_fused_qkv block_data
  qkv =. |: (fused_qkv_w (+/ .* ) |: hidden_flat)   NB. (B*c, qkv_len)
  q_len =. n_heads * head_dim
  kv_len =. n_heads_kv * head_dim
  Q =. (q_len {. "1 qkv)   NB. (B*c, q_len)
  K =. (kv_len {. "1 (q_len }. "1 qkv))
  V =. (kv_len {. "1 ((q_len + kv_len) }. "1 qkv))
  Q =. (B, c, n_heads, head_dim) $ , Q
  K =. (B, c, n_heads_kv, head_dim) $ , K
  V =. (B, c, n_heads_kv, head_dim) $ , V

  NB. Q/K norm per head (B*c*n_heads rows)
  q_norm_w =. gem3_bd_q_norm block_data
  k_norm_w =. gem3_bd_k_norm block_data
  Qf =. ((B*c*n_heads) , head_dim) $ , Q
  Qf =. rms_norm_rows ((< mi_rms_eps mi) , (< q_norm_w) , <Qf)
  Q =. (B, c, n_heads, head_dim) $ , Qf
  Kf =. ((B*c*n_heads_kv) , head_dim) $ , K
  Kf =. rms_norm_rows ((< mi_rms_eps mi) , (< k_norm_w) , <Kf)
  K =. (B, c, n_heads_kv, head_dim) $ , Kf

  NB. NEOX RoPE at the (B, c) positions pos[b]+i.c. cos/sin tables + expansions
  NB. are layer-invariant (identical across layers; depend only on pos) —
  NB. hoisted once in run_blocks_bp, threaded through y as <cos_all; sin_all;
  NB. cos_exp; sin_exp; cos_expk; sin_expk>.
  cos_all =. > 0 { rope   NB. (B*c, half)
  sin_all =. > 1 { rope
  Qa =. half {. "1 Q   NB. (B, c, n_heads, half)
  Qb =. half }. "1 Q
  cos_exp =. > 2 { rope
  sin_exp =. > 3 { rope
  Qa_out =. (Qa * cos_exp) - (Qb * sin_exp)
  Qb_out =. (Qa * sin_exp) + (Qb * cos_exp)
  Q =. (B, c, n_heads, head_dim) $ , (Qa_out ,"1 Qb_out)
  Ka =. half {. "1 K   NB. (B, c, n_heads_kv, half)
  Kb =. half }. "1 K
  cos_expk =. > 4 { rope
  sin_expk =. > 5 { rope
  Ka_out =. (Ka * cos_expk) - (Kb * sin_expk)
  Kb_out =. (Ka * sin_expk) + (Kb * cos_expk)
  K =. (B, c, n_heads_kv, head_dim) $ , (Ka_out ,"1 Kb_out)

  NB. Pre-scale Q by 1/sqrt(head_dim) (Gemma)
  Q =. Q % head_dim ^ 0.5

  NB. Per-sequence attention: write K/V to cache, read back, SWA+causal+padding
  NB. masked scores, softmax with n_heads_kv expansion, output projection.
  attn_raw_all =. ''
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

    NB. scores: Qf·Kf^T over the window (c*nh, win*nk)
    Qf =. ((c * n_heads) , head_dim) $ , q_b
    Kf =. ((win * n_heads_kv) , head_dim) $ , k_all
    scores_all =. Qf (+/ .* ) |: Kf   NB. (c*nh, win*nk)

    NB. mask: causal + SWA (window swa_l) + lens (padding)
    key_pos =. i. win
    q_pos =. pos_b + i. c
    mask_future =. q_pos </ key_pos
    swa_l =. gem3_bd_swa_l block_data
    mask_swa =. ((q_pos - swa_l) + 1) >/ key_pos
    if. swa_l <: 0 do. mask_swa =. (c, win) $ 0 end.
    mask_2d =. mask_future +. mask_swa
    if. 0 < # lens do.
      mask_2d =. mask_2d +. ((i. win) >: b { lens)
    end.
    NB. Fused mask (c,win) -> (c*nh*nk, win) [t,g,j] row order g = h*nk+k
    mask_3d =. (0 2 1) |: (mask_2d (*/) ((n_heads * n_heads_kv) $ 1))
    mask_f =. ((c * n_heads * n_heads_kv) , win) $ , mask_3d
    scores_f =. ((c * n_heads * n_heads_kv) , win) $ , scores_all
    scores_f =. scores_f - (mask_f * 1e9)

    NB. per-row softmax over j
    max_sf =. >./"1 scores_f
    exp_sf =. ^ (scores_f - max_sf)
    softmax_f =. exp_sf % +/"1 exp_sf
    softmax =. (c, n_heads, n_heads_kv, win) $ , softmax_f

    NB. attn_raw[t,h] = sum_{k,j} softmax[t,h,k,j] * V[j,k]
    softmax_flat =. ((c * n_heads) , (n_heads_kv * win)) $ , softmax
    v_flat =. ((n_heads_kv * win) , head_dim) $ , (1 0 2 |: v_all)
    attn_raw_flat =. softmax_flat (+/ .* ) v_flat   NB. (c*nh, hd)
    attn_raw =. (c, n_heads, head_dim) $ , attn_raw_flat
    attn_raw_all =. attn_raw_all , <attn_raw
    b =. b + 1
  end.

  NB. Batched output projection (B*c rows) + post-attention norm
  attn_all =. (B, c, n_heads*head_dim) $ , > attn_raw_all
  attn_o_w =. gem3_bd_attn_o block_data
  attn_out =. |: (attn_o_w (+/ .* ) |: (((B*c) , (n_heads*head_dim)) $ , attn_all))
  attn_out =. ((B*c) , emb_len) $ , attn_out
  attn_pn_w =. gem3_bd_attn_pn block_data
  attn_out =. rms_norm_rows ((< mi_rms_eps mi) , (< attn_pn_w) , <attn_out)
  attn_out =. (B, c, emb_len) $ , attn_out
  (<attn_out)
)

NB. ---- Batched-prefill block forward (gemma3) ----
NB. x = hidden (B, c, emb); y = <block_data; swa; pos; mi; layer>.  Returns <(B, c, emb)>.
gem3_block_forward_bp =: 4 : 0
  hidden =. x   NB. (B, c, emb)
  block_data =. > 0 { y
  layer =. > 1 { y
  swa =. > 2 { y
  pos =. > 3 { y
  mi =. > 4 { y
  lens =. ''
  if. 5 < # y do. lens =. > 5 { y end.
  rope =. > 6 { y
  B =. {. $ hidden
  c =. 1 { $ hidden
  emb_len =. 2 { $ hidden
  attn_result =. hidden gem3_attention_bp y
  attn_out =. > 0 { attn_result
  sa_out =. attn_out + hidden
  ff_norm_w =. gem3_bd_ff_norm block_data
  sa_flat =. ((B*c) , emb_len) $ , sa_out
  ffn_in =. rms_norm_rows ((< mi_rms_eps mi) , (< ff_norm_w) , <sa_flat)
  fused_ff_gu_w =. gem3_bd_fused_ff_gu block_data
  gate_up =. |: (fused_ff_gu_w (+/ .* ) |: ffn_in)   NB. (B*c, 2*n_ff)
  n_ff =. gem3_bd_n_ff block_data
  gate_out =. (n_ff {. "1 gate_up)
  up_out =. (n_ff }. "1 gate_up)
  ff_down_w =. gem3_bd_ff_down block_data
  ffn_raw =. |: (ff_down_w (+/ .* ) |: (gate_out geglu up_out))   NB. (B*c, emb)
  ff_pn_w =. gem3_bd_ff_pn block_data
  ffn_out =. rms_norm_rows ((< mi_rms_eps mi) , (< ff_pn_w) , <ffn_raw)
  output_flat =. ffn_out + sa_flat
  output =. (B, c, emb_len) $ , output_flat
  (<output)
)

NB. ---- Run all blocks for B sequences (one CHUNK each at pos[b]) ----
gem3_run_blocks_bp =: 4 : 0
  llm =. > 0 { y
  pos =. > 1 { y
  lens =. ''
  if. 2 < # y do. lens =. > 2 { y end.
  mi =. llm_mi llm
  swa =. mi_swa mi
  head_dim =. mi_head_dim mi
  n_heads =. mi_n_heads mi
  n_heads_kv =. mi_n_heads_kv mi
  block_count =. mi_block_count mi
  ctx_len =. mi_context_len mi
  state =. x
  if. 0 = # kv_meta do.
    kv_create ((<block_count) , (<ctx_len) , (<n_heads_kv) , (<head_dim))
  end.
  NB. RoPE cos/sin tables + expansions are layer-invariant (identical across
  NB. layers; depend only on pos) — compute ONCE per chunk, thread through.
  B =. {. $ state
  c =. 1 { $ state
  half =. <. head_dim % 2
  pos_bc_flat =. (B*c) $ , (pos +/ i. c)
  cos_all =. pos_bc_flat { mi_cos_tab mi   NB. (B*c, half)
  sin_all =. pos_bc_flat { mi_sin_tab mi
  cos_exp =. (B, c, n_heads, half) $ , ((0 2 1) |: (((B*c) , half , n_heads) $ , (cos_all (*/) (n_heads $ 1))))
  sin_exp =. (B, c, n_heads, half) $ , ((0 2 1) |: (((B*c) , half , n_heads) $ , (sin_all (*/) (n_heads $ 1))))
  cos_expk =. (B, c, n_heads_kv, half) $ , ((0 2 1) |: (((B*c) , half , n_heads_kv) $ , (cos_all (*/) (n_heads_kv $ 1))))
  sin_expk =. (B, c, n_heads_kv, half) $ , ((0 2 1) |: (((B*c) , half , n_heads_kv) $ , (sin_all (*/) (n_heads_kv $ 1))))
  rope =. (<cos_all) , (<sin_all) , (<cos_exp) , (<sin_exp) , (<cos_expk) , (<sin_expk)
   b =. 0
   block_data_list =. llm_block_data llm
   NB. (<swa), (<pos), (<mi), <lens are layer-invariant — box once, reuse per layer.
   bfb_pre =. (<swa) , (<pos) , (<mi) , <lens
   while. b < block_count do.
     block_data =. > b { block_data_list
     result =. state gem3_block_forward_bp (<block_data) , (<b) , bfb_pre , <rope
     state =. > 0 { result
     b =. b + 1
   end.
  <state
)

NB. ---- Batched-DECODE attention (B sequences, ONE token each at pos[b]) ----
NB. Mirrors gem3_attention (single-token): fused QKV, per-head Q/K norm, NEOX
NB. RoPE via PER-LAYER tables, SWA mask, post-attn norm. x = hidden (B, emb);
NB. y = <block_data; pos; swa; mi; layer>
gem3_attention_bd =: 4 : 0
  block_data =. > 0 { y
  pos =. > 1 { y
  swa =. > 2 { y
  mi =. > 3 { y
  layer =. > 4 { y
  rope =. > 5 { y
  B =. {. $ x
  n_heads =. gem3_bd_n_heads block_data
  head_dim =. gem3_bd_head_dim block_data
  n_heads_kv =. gem3_bd_n_heads_kv block_data
  half =. <. head_dim % 2
  q_len =. n_heads * head_dim
  kv_len =. n_heads_kv * head_dim

  NB. Attention norm per row
  attn_norm_w =. gem3_bd_attn_norm block_data
  hidden =. rms_norm_rows ((< mi_rms_eps mi) , (< attn_norm_w) , <x)

  NB. Fused QKV projection (batched single matmul)
  fused_qkv_w =. gem3_bd_fused_qkv block_data
  qkv =. |: (fused_qkv_w (+/ .* ) |: hidden)   NB. (B, q_len + 2*kv_len)
  Q =. (q_len {. "1 qkv)
  K =. (kv_len {. "1 (q_len }. "1 qkv))
  V =. (kv_len {. "1 ((q_len + kv_len) }. "1 qkv))
  Q =. (B, n_heads, head_dim) $ ,Q
  K =. (B, n_heads_kv, head_dim) $ ,K
  V =. (B, n_heads_kv, head_dim) $ ,V

  NB. Q/K norm per head (flatten heads, norm rows, reshape)
  q_norm_w =. gem3_bd_q_norm block_data
  k_norm_w =. gem3_bd_k_norm block_data
  Qf =. ((B * n_heads) , head_dim) $ ,Q
  Qf =. rms_norm_rows ((< mi_rms_eps mi) , (< q_norm_w) , <Qf)
  Q =. (B, n_heads, head_dim) $ ,Qf
  Kf =. ((B * n_heads_kv) , head_dim) $ ,K
  Kf =. rms_norm_rows ((< mi_rms_eps mi) , (< k_norm_w) , <Kf)
  K =. (B, n_heads_kv, head_dim) $ ,Kf

  NB. RoPE — batched, table-based (NEOX). cos/sin tables + expansions are
  NB. layer-invariant (identical across layers; depend only on pos) — hoisted
  NB. once in run_blocks_bd, threaded through y as <cos_all; sin_all; cos_exp;
  NB. sin_exp; cos_expk; sin_expk>.
  cos_all =. > 0 { rope    NB. (B, half)
  sin_all =. > 1 { rope
  cos_exp =. > 2 { rope
  sin_exp =. > 3 { rope
  Qa =. half {. "1 Q
  Qb =. half }. "1 Q
  Qa_out =. (Qa * cos_exp) - (Qb * sin_exp)
  Qb_out =. (Qa * sin_exp) + (Qb * cos_exp)
  Q =. (B, n_heads, head_dim) $ ,(Qa_out ,"1 Qb_out)
  cos_expk =. > 4 { rope
  sin_expk =. > 5 { rope
  Ka =. half {. "1 K
  Kb =. half }. "1 K
  Ka_out =. (Ka * cos_expk) - (Kb * sin_expk)
  Kb_out =. (Ka * sin_expk) + (Kb * cos_expk)
  K =. (B, n_heads_kv, head_dim) $ ,(Ka_out ,"1 Kb_out)

  NB. Pre-scale Q by 1/sqrt(head_dim) (Gemma)
  Q =. Q % head_dim ^ 0.5

  NB. Batched lockstep decode (all B sequences share one position): ONE
  NB. list-selector cache write, ONE indexed gather of the B windows, then
  NB. threaded batched scores/softmax/V (J parallelizes the rank-4 matmuls over
  NB. B). gemma3 is MQA (n_groups = n_heads/n_heads_kv) with a per-layer SWA
  NB. causal mask that is B-independent in lockstep (win is shared). Falls back
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
    NB. Gather all B windows in one indexed fetch
    idxr =. base_b +/ i. win
    k_rows_b =. (B , win , n_heads_kv , head_dim) $ , (idxr { k_cache_g)
    v_rows_b =. (B , win , n_heads_kv , head_dim) $ , (idxr { v_cache_g)
    NB. Batched MQA scores (each query head attends the shared n_kv heads)
    n_groups =. n_heads % n_heads_kv
    Q_g2_b =. (B , n_heads_kv , n_groups , head_dim) $ , Q
    Kp_b =. (0 2 3 1) |: k_rows_b   NB. (B, n_kv, hd, win)
    scores_b =. Q_g2_b (+/ .* "2) Kp_b   NB. (B, n_kv, groups, win)
    scores_b2 =. (B , n_heads , win) $ , scores_b
    NB. SWA + causal mask (B-independent in lockstep: win is shared)
    swa_l =. gem3_bd_swa_l block_data
    mask_1d =. (i. win) < (win - swa_l)
    mask_1d =. (swa_l > 0) *. mask_1d
    mask_3d_b =. (B , n_heads , win) $ , (((B * n_heads) $ 1) (*/) mask_1d)
    scores_b2 =. scores_b2 - (mask_3d_b * 1e9)
    max_sf_b =. >./"1 scores_b2
    exp_sf_b =. ^ (scores_b2 - max_sf_b)
    softmax_b =. exp_sf_b % +/"1 exp_sf_b
    softmax_g2_b =. (B , n_heads_kv , n_groups , win) $ , softmax_b
    Vp_b =. (0 2 1 3) |: v_rows_b   NB. (B, n_kv, win, hd)
    attn2_b =. softmax_g2_b (+/ .* "2) Vp_b   NB. (B, n_kv, groups, hd)
    attn_all =. (B , n_heads * head_dim) $ , attn2_b
  else.
    NB. Per-sequence: SWA-masked causal scores/softmax/output (gem3_attention shape)
    attn_out =. ''
    b =. 0
    while. b < B do.
      q_b =. (n_heads, head_dim) $ ,(b { Q)
      k_b =. (n_heads_kv, head_dim) $ ,(b { K)
      v_b =. (n_heads_kv, head_dim) $ ,(b { V)
      pos_b =. b { pos
      kv_write ((4 { y) , (<pos_b) , (<k_b) , (<v_b) , (<b))
      kv_result =. kv_read ((4 { y) , (<pos_b) , (<b))
      k_all =. > 0 { kv_result   NB. (win, n_kv, hd)
      v_all =. > 1 { kv_result
      win =. pos_b + 1
      NB. Causal + sliding window mask (swa_l per-layer; dense layers swa_l=0)
      swa_l =. gem3_bd_swa_l block_data
      mask_1d =. (i. win) < (win - swa_l)
      mask_1d =. (swa_l > 0) *. mask_1d
      mask_3d =. (n_heads, win, n_heads_kv) $ mask_1d
      k_trans =. 2 0 1 |: k_all   NB. (hd, win, n_kv)
      scores =. q_b (+/ .* ) k_trans   NB. (n_heads, win, n_kv)
      scores =. scores - (mask_3d * 1e9)
      scores_f =. (n_heads, n_heads_kv * win) $ ,scores
      max_sf =. >./"1 scores_f
      exp_sf =. ^ (scores_f - max_sf)
      softmax_f =. exp_sf % +/"1 exp_sf
      softmax =. (n_heads, n_heads_kv, win) $ ,softmax_f
      v_flat =. (n_heads_kv * win, head_dim) $ ,(1 0 2 |: v_all)
      softmax_flat =. (n_heads, n_heads_kv * win) $ ,softmax
      attn_raw =. softmax_flat (+/ .* ) v_flat   NB. (n_heads, hd)
      attn_raw_flat =. (n_heads * head_dim) $ ,attn_raw
      attn_out =. attn_out , <attn_raw_flat
      b =. b + 1
    end.
    attn_all =. > attn_out
  end.

  NB. Output projection + post-attention norm
  attn_o_w =. gem3_bd_attn_o block_data
  attn_out =. |: (attn_o_w (+/ .* ) |: attn_all)   NB. (B, emb)
  attn_pn_w =. gem3_bd_attn_pn block_data
  attn_out =. rms_norm_rows ((< mi_rms_eps mi) , (< attn_pn_w) , <attn_out)
  (<attn_out)
)

NB. ---- Batched-DECODE Gemma3 block forward ----
NB. x = hidden (B, emb); y = <block_data; pos; swa; mi; layer>
gem3_block_forward_bd =: 4 : 0
  hidden =. x
  block_data =. > 0 { y
  pos =. > 1 { y
  swa =. > 2 { y
  mi =. > 3 { y
  layer =. > 4 { y
  rope =. > 5 { y
  attn_result =. hidden gem3_attention_bd y
  attn_out =. > 0 { attn_result
  sa_out =. attn_out + hidden
  ff_norm_w =. gem3_bd_ff_norm block_data
  ffn_in =. rms_norm_rows ((< mi_rms_eps mi) , (< ff_norm_w) , <sa_out)
  fused_ff_gu_w =. gem3_bd_fused_ff_gu block_data
  gate_up =. |: (fused_ff_gu_w (+/ .* ) |: ffn_in)   NB. (B, 2*n_ff)
  n_ff =. gem3_bd_n_ff block_data
  gate_out =. (n_ff {. "1 gate_up)
  up_out =. (n_ff }. "1 gate_up)
  ff_down_w =. gem3_bd_ff_down block_data
  ffn_raw =. |: (ff_down_w (+/ .* ) |: (gate_out geglu up_out))   NB. GEGLU
  ff_pn_w =. gem3_bd_ff_pn block_data
  ffn_out =. rms_norm_rows ((< mi_rms_eps mi) , (< ff_pn_w) , <ffn_raw)
  output =. ffn_out + sa_out
  (<output)
)

NB. ---- Run all Gemma3 blocks for B sequences (one token each at pos[b]) ----
gem3_run_blocks_bd =: 4 : 0
  llm =. > 0 { y
  pos =. > 1 { y
  mi =. llm_mi llm
  swa =. mi_swa mi
  head_dim =. mi_head_dim mi
  n_heads =. mi_n_heads mi
  n_heads_kv =. mi_n_heads_kv mi
  block_count =. mi_block_count mi
  ctx_len =. mi_context_len mi
  state =. x
  if. 0 = # kv_meta do.
    kv_create ((<block_count) , (<ctx_len) , (<n_heads_kv) , (<head_dim))
  end.
  NB. RoPE cos/sin tables + expansions are layer-invariant (identical across
  NB. layers; depend only on pos) — compute ONCE per step, thread through.
  B =. {. $ state
  half =. <. head_dim % 2
  cos_all =. pos { mi_cos_tab mi    NB. (B, half)
  sin_all =. pos { mi_sin_tab mi
  cos_exp =. (0 2 1) |: ((B , half , n_heads) $ , (cos_all (*/) (n_heads $ 1)))
  sin_exp =. (0 2 1) |: ((B , half , n_heads) $ , (sin_all (*/) (n_heads $ 1)))
  cos_expk =. (0 2 1) |: ((B , half , n_heads_kv) $ , (cos_all (*/) (n_heads_kv $ 1)))
  sin_expk =. (0 2 1) |: ((B , half , n_heads_kv) $ , (sin_all (*/) (n_heads_kv $ 1)))
  rope =. (<cos_all) , (<sin_all) , (<cos_exp) , (<sin_exp) , (<cos_expk) , (<sin_expk)
  b =. 0
  block_data_list =. llm_block_data llm
  NB. (<pos), (<swa), (<mi) are layer-invariant — box once, reuse per layer.
  bf_pre =. (<pos) , (<swa) , (<mi)
  while. b < block_count do.
    block_data =. > b { block_data_list
    result =. state gem3_block_forward_bd ((<block_data) , bf_pre , (<b) , <rope)
    state =. > 0 { result
    b =. b + 1
  end.
  <state
)

NB. ---- Single Gemma3 block forward ----
NB. x = hidden (emb,) — the layer input; y = <block_data; pos; swa; mi; layer>
gem3_block_forward =: 4 : 0
  hidden =. x
  'block_data pos swa mi layer' =. y
  
  attn_result =. hidden gem3_attention y
  attn_out =. > 0 { attn_result
  
  sa_out =. attn_out + hidden
  
  ff_norm_w =. gem3_bd_ff_norm block_data
  ffn_in =. rms_norm ((< mi_rms_eps mi) , (< ff_norm_w) , <sa_out)
  
   fused_ff_gu_w =. gem3_bd_fused_ff_gu block_data
   gate_up =. fused_ff_gu_w linear_r ffn_in
   n_ff =. gem3_bd_n_ff block_data
   gate_out =. n_ff {. gate_up
   up_out =. n_ff }. gate_up
  
     ff_down_w =. gem3_bd_ff_down block_data
     ffn_raw =. ff_down_w linear_r (gate_out geglu up_out)  NB. GEGLU: GELU(gate) * up
  
    ff_pn_w =. gem3_bd_ff_pn block_data
    ffn_out =. rms_norm ((< mi_rms_eps mi) , (< ff_pn_w) , <ffn_raw)
    
    output =. ffn_out + sa_out
  
  (<output)
)

NB. ---- Run all Gemma3 blocks (single token; cache lives in kv_cache_g) ----
NB. x = hidden (emb,); y = <llm; pos>
gem3_run_blocks =: 4 : 0
  'llm pos' =. y
  mi =. llm_mi llm
  swa =. mi_swa mi
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
   NB. (<pos), (<swa), (<mi) are layer-invariant — box once, reuse per layer.
   bf_pre =. (<pos) , (<swa) , (<mi)
   while. b < block_count do.
     block_data =. > b { block_data_list
    result =. state gem3_block_forward (<block_data) , bf_pre , (<b)
    state =. > 0 { result
     b =. b + 1
   end.

   <state
)

NB. ---- Load Gemma3 GGUF into llm noun ----
gem3_load =: 3 : 0
  NB. y = <path; raw> — raw is the memory-mapped file (mapped by
  NB. load_gguf_to_llm; unmap'd after load — kvs_ctx is load-time only).
  data =. y
  path =. > 0 { data
  raw =. > 1 { data
  
  header =. parse_hdr_raw raw
  n_tensors =. > 2 { header
  n_kv =. > 3 { header
  
  kv_result =. parse_kv_pairs_raw raw
  kvs =. > 0 { kv_result
  kv_end =. > 3 { kv_result
  
  ti =. parse_tensor_infos (<raw) , (<kv_end) , (<n_tensors)
  
  ti_end_offset =. > ((n_tensors * 6) - 1) { ti
  tds =. 32 * <. (ti_end_offset + 31) % 32
  
  kvs_ctx =. (<kvs) , (<raw)
  kv_data =. build_kv_dict kvs_ctx
  mi =. gem3_extract_hparams kvs_ctx
  mi =. build_mi_dict mi

  NB. Precompute RoPE cos/sin tables for all positions (removes per-token trig)
  rope_tables =. build_rope_tables ((< mi_context_len mi) , (< mi_head_dim mi) , (< mi_rope_freq mi))
  NB. Gemma3-specific sliding window
  NB. ONE multi-put: the jdict single-put (1-key) path is broken in this J9.8
  NB. build (domain/length error); multi-put (2+ keys) and get/has work.
  swa =. 'gemma3.attention.sliding_window' gem3_kv_uint kvs_ctx
  ((> 0 { rope_tables) ; (> 1 { rope_tables) ; swa) put__mi 'cos_tab' ; 'sin_tab' ; 'swa'
  
  NB. Real chat template from the GGUF ('' if absent → bespoke fallback).
  ct_tmpl_g =: 'tokenizer.chat_template' kv_string (0 1 { kv_result)
  
  tokenizer =. build_llama3_tokenizer kv_result
  
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
  
   swa =. mi_swa mi
   n_heads_kv =. mi_n_heads_kv mi
   head_dim =. mi_head_dim mi
   ctx_len =. mi_context_len mi
   block_count =. mi_block_count mi
  
  NB. Build llm as boxed vector: each element individually boxed
  NB. Must box EVERY element to prevent razing during , concatenation
  p  =. <path
  t  =. <ti
  ze =. <1.0 64 0.95 0.001   NB. default_params (chat sampling defaults)
  tk =. <tokenizer
  mi_b =. <mi
  kc_b =. <''   NB. kv cache is the kv_cache_g global, not stored in the llm
  td =. <tds
  all_tensors =. emb_canonical all_tensors
  at =. <all_tensors
  llm =. p , t , ze , tk , mi_b , kc_b , td , at
  block_data =. gem3_pre_build_block_data llm
  llm =. llm , <block_data
  llm =. llm , <kv_data
)

NB. ---- Single-token inference for Gemma3 ----
NB. Usage: llm infer (text ; <temp;k;p;min_p>)
NB.        Simple (default params): llm infer_simple text
NB. ----------------------------------------------------------------
gem3_infer =: 4 : 0
  llm =. x
  args =. infer_args y
  text =. > 0 { args
  temp =. > 1 { args
  k =. > 2 { args
  p =. > 3 { args
  min_p =. > 4 { args

  tokens =. llama3_tokenize (<llm) , <text
  
  mi =. llm_mi llm
  emb_len =. mi_emb_len mi
  n_embd =. emb_len
  block_count =. mi_block_count mi
  ctx_len =. mi_context_len mi
  n_heads_kv =. mi_n_heads_kv mi
  head_dim =. mi_head_dim mi
  
    emb_w =. 'token_embd.weight' get_tensor_cached_d llm
     scale =. %: n_embd
     vsz =. {: $ emb_w
      NB. token_embd is stored transposed (emb, vocab) — column tok is embedding
      
     tok_list =. , > tokens
    
    output_norm_w =. 'output_norm.weight' get_tensor_cached_d llm
    block_data_list =. llm_block_data llm
    swa =. mi_swa mi
    mi_for_blocks =. mi
    
    NB. Embed all input tokens (llama.cpp: no sqrt(n_embd) scale, no output_norm before blocks)
    n_tokens =. # tok_list
    kv_create ((<block_count) , (<ctx_len) , (<n_heads_kv) , (<head_dim))
    if. 1 = n_tokens do.
      tok =. 0 { tok_list
      hidden =. scale * |: (tok {"1 emb_w)
      pre_s =. 6!:2 'result =. hidden gem3_run_blocks (<llm) , <0'
      hidden =. > 0 { result
    else.
      NB. Batched prompt prefill: run all blocks once over (n_tokens x emb)
      NB. gemma3 tokenizer is SentencePiece with a 262144 real-token vocab — NO byte
      NB. tokens, so token ids are used directly (no 65536 clamp; that clamp is only
      NB. valid for llama3-style BPE / gpt2 byte-level tokenizers).
      emb_all =. scale * |: (tok_list {"1 emb_w)
      pre_s =. 6!:2 'result_b =. emb_all gem3_run_blocks_b ((<llm) , <0)'
      h_b =. > 0 { result_b
      hidden =. > (n_tokens-1) { h_b   NB. last prompt token's hidden predicts next
    end.
  
   NB. Final norm + lm_head
   hidden =. rms_norm ((< mi_rms_eps mi) , (<output_norm_w) , <hidden)
  emb_w_final =. 'token_embd.weight' get_tensor_cached_d llm
  logits =. hidden (+/ .* ) emb_w_final
  
  NB. Sample from logits using sampler module
  report_prefill (pre_s , n_tokens)
  flat =. temp , k , p , min_p
  params =. <"0 flat
  pred_tok =. params sampler_sample logits
  
  decoded =. llama3_detokenize (<llm) , <pred_tok
  
  tokens ; pred_tok ; decoded ; logits
)

NB. ---- Multi-token generation for Gemma3 ----
NB. Usage: llm generate (text ; max_steps ; <temp;k;p;min_p>)
NB.        Simple (default params): llm generate_simple (text ; max_steps)
NB. Generation is the UNIFIED gen_loop_core (llm_core.ijs) — per-arch
NB. differences (embedding scale, run_blocks/run_blocks_b) are dispatched by
NB. llm_arch. Fresh mode: kv_create + batched prefill; resume mode (chat
NB. sessions): incremental prefill of the new segment. Stop token not appended.
gem3_generate =: 4 : 0
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
  prompt =. gem3_chat_prompt messages
  tokens =. llama3_tokenize (<llm) , <prompt
  stop =. gem3_stop_tokens llm
  L =. # , > tokens
  output =. llm gen_loop_core (tokens ; '' ; max_steps ; temp ; k ; p ; min_p ; <stop)
  gen =. L }. output
  llama3_detokenize (<llm) , <gen
)

NB. ---- Batched generation: B independent prompts in parallel ----
NB. Usage: llm gem3_generate_batch (prompts ; max_steps ; <temp;k;p;min_p>)
gem3_generate_batch =: 4 : 0
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
    prompt =. gem3_chat_prompt messages
    tokens =. llama3_tokenize (llm_box , <prompt)
    tok_list =. , > tokens
    prompts_tok =. prompts_tok , <tok_list
    prompts_len =. prompts_len , <(# tok_list)
    i =. i + 1
  end.
  stop =. gem3_stop_tokens llm
  kv_batch_g =: B
  output =. llm gen_loop_batch (prompts_tok ; max_steps ; temp ; k ; p ; min_p ; <stop)
  answers =. ''
  i =. 0
  while. i < B do.
    L =. > i { prompts_len
    gen =. (L) }. (> i { output)
    answers =. answers , <(llama3_detokenize (llm_box , <gen))
    i =. i + 1
  end.
  answers
)

NB. ---- Simple wrappers (default greedy/top-p params) ----
NB. llm gem3_infer_simple text            | llm gem3_generate_simple (text ; n)
gem3_infer_simple =: gem3_infer (] ; (<0 0 0.95 0.0)"_)
gem3_generate_simple =: gem3_generate (0&{ , 1&{ , (<0 0 0.95 0.0)"_)

NB. ---- Chat-template support (Phase 1.1) ----
NB. y = messages: boxed list of message boxes; each message = <role ; content>
NB. (2-item boxed list, built with (role) ; content). Renders the gemma3 chat
NB. template: assistant role -> 'model', content trimmed (template `| trim`),
NB. generation prompt '<start_of_turn>model' appended. BOS is added by the
NB. llama3 tokenizer (llama.cpp also prepends bos for gemma chat).

NB. Stop tokens: <end_of_turn> (EOS) and <eos> (token 1) — per gemma params file.
gem3_stop_tokens =: 3 : 0
  tk =. llm_tokenizer y
  (tokenizer_eos tk) , 1
)
