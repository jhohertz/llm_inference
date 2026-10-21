NB. ================================================================
NB. Shared Transformer Kernels — used by all architecture models
NB. ================================================================

coclass 'inference'

NB. ---- Raw matvec/matmat (no boxing) — for hot fused projections ----

NB. ---- Raw matvec/matmat (no boxing) — for hot fused projections ----
NB. x = weight, y = raw input array (no bias). Skips the box/unbox of `linear`.
linear_r =: +/ .*

NB. ---- Numeric closeness: 1 iff every atom of x matches y within atol+rtol*|y| ----
NB. x allclose y -> 1/0. atol = absolute tolerance, rtol = relative tolerance.
NB. Use for near-equality checks (batched-vs-single, path-vs-path); exact
NB. bit-level verification keeps the tighter absolute-sum patterns.
ATOL =: 1e_9
RTOL =: 1e_6
allclose =: 4 : 0
  *./ , (| x - y) <: ATOL + RTOL * | y
)

NB. ---- NumPy-style broadcasting via verb rank (Żołek 2026) ----
NB. u Broadcastly  applies verb u as a NumPy universal function: broadcast
NB. dimensions (length-1 axes) are expanded and the element-wise verb is
NB. applied with the broadcast dims removed, exactly matching NumPy.
NB. E.g.  (2 3 $ 10 20 30 40 50 60) + Broadcastly (3 $ 1 2 3)  ->  the 3-vector
NB. broadcast across the 2 rows.  J's plain `+` (rank 0) REJECTS (2 3)+(3,) —
NB. the frames don't conform — this is the case NumPy broadcasts but J rank
NB. does not.  Adverb from reference/broadcastly.ijs (paper
NB. "Expressing NumPy Broadcasting via Verb Rank in J", arXiv:2609.16064).
Broadcastly =: {{
  'l r' =. }. u b. 0
  cs =. (( - @ <. # ) {. ] ) $
  fr =. - @ [ }. $ @ ]
  csx =. l cs x [ csy =. r cs y
  fxy =. ( l fr x ) ,: !. 1 & |. r fr y
  assert. ( +. / @: ( = & 1 ) *. / @: +. = / ) fxy
  mxy =. ( = / +. " 1 ~: & 1 ) fxy
  ranks =. ( +. & ( 2 & ( > / \ )) / @ ( 1 & ,. ) # |: @ ( - ~ + / \ " 1 )) mxy
  ranks =. ( l , r ) -. ~ ranks ( ] , + " 1 ) csx , & # csy
  repr =. u ` '' < F.. ( '"' , & < ( , & < '0' & ; ) ~ ) ranks
  'fx fy' =. mxy < @ |. @ # " 1 fxy
  ( x ( $ , ) ~ fx , csx ) repr `: 6 y ( $ , ) ~ fy , csy
}}

NB. ---- GELU Activation (tanh approximation) ----
NB. For architectures that use GELU (e.g. Gemma, Qwen)
NB. Formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
gelu =: 0.5&* * (1 + 7 o. ((%: 2 % o. 1)&* @ (] + 0.044715&* @ (] * *:))))

NB. ---- SiLU / Swish Activation ----
NB. For LFM2 architecture: silu(x) = x / (1 + exp(-x))
silu =: ] % 1 + ^@-

NB. ---- GEGLU Activation ----
NB. For Gemma 3 architecture: GEGLU(gate, up) = GELU(gate) * up
NB. Dyadic: gate geglu up -> gelu(gate) * up. Tacit fork (gelu@[) * ]; f.
NB. flattens gelu in (Ch 41: removes the gelu name lookup per FFN call).
geglu =: ((gelu@[) * ]) f.

NB. ---- SwiGLU Activation ----
NB. SwiGLU(gate, up) = Swish(gate) * up
NB. Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
NB. Dyadic: gate swiglu up -> silu(gate) * up   (llama.cpp LLM_FFN_SILU/PAR: swiglu_split)
swiglu =: ((silu@[) * ]) f.

NB. ---- RMSNorm (vector) ----
NB. rms_norm eps <weight, input>
NB. Normalizes the vector by RMS, then scales by weight
NB. Tacit: `>@(N&{)` accessors (Ch 36); weight * input % sqrt((sum sq % #) + eps)
rms_norm =: ((>@(1&{)) * ((>@(2&{)) % (%: @ (((+/ @ (*: @ >@(2&{))) % (# @ >@(2&{))) + >@(0&{)))))

NB. ---- RMSNorm per row of 2D array ----
NB. rms_norm_rows eps <weight, matrix>
NB. Applies RMSNorm independently to each row. Broadcast the weight vector
NB. across the matrix rows by `$`-replication (J's `$` is special-coded and
NB. near-free). NOTE: Broadcastly (NumPy-style broadcasting) was tried here but
NB. changed the result (off by sqrt(row_len)) — the broadcast-multiply via
NB. Broadcastly does not reproduce rms_norm_rows exactly, so keep `$`.
rms_norm_rows =: ((($ @ >@(2&{)) $ >@(1&{)) * ((>@(2&{)) % (%: @ (((+/"1 @ (*: @ >@(2&{))) % ({: @ $ @ >@(2&{))) + >@(0&{)))))

NB. ---- RoPE for 1D vector ----









NB. ---- NEOX RoPE for 2D array — table-based (no per-call trig) ----
NB. x = (n_rows, dim) qk matrix, y = <cos_t; sin_t> where cos_t/sin_t are (half,) vectors for the current position
NB. cos_t/sin_t precomputed once per position (build_rope_tables); removes all sin/cos from hot loop.
NB. half = dim % 2; pairs are (i, i+half).
rope_apply2_neox_t =: 4 : 0
  qk =. x
  cos_t =. > 0 { y
  sin_t =. > 1 { y
  half =. # cos_t
  a =. half {. "1 qk       NB. first half of each row
  b =. half }. "1 qk       NB. second half of each row
  cos_exp =. ($ a) $ cos_t
  sin_exp =. ($ a) $ sin_t
  a_out =. (a * cos_exp) - (b * sin_exp)
  b_out =. (a * sin_exp) + (b * cos_exp)
  a_out ,"1 b_out          NB. per-row: x[i]=a_out[i], x[i+half]=b_out[i]
)

NB. ---- INTERLEAVED RoPE for 2D array — table-based (Llama style) ----
NB. x = (n_rows, dim) qk matrix, y = <cos_t; sin_t> (half,) for the current position.
NB. Pairs are consecutive (i, i+1), unlike NEOX (i, i+half). half = dim % 2.
NB. Same cos/sin tables as NEOX (theta_i = pos * freq^(-2i/dim)).
rope_apply2_t =: 4 : 0
  qk =. x
  cos_t =. > 0 { y
  sin_t =. > 1 { y
  dim =. {: $ qk
  pair_count =. dim % 2
  indices =. 2 * i. pair_count
  a =. indices {"1 qk          NB. even-indexed cols per row
  b =. (1 + indices) {"1 qk    NB. odd-indexed cols per row
  cos_exp =. ($ a) $ cos_t
  sin_exp =. ($ a) $ sin_t
  a_out =. (a * cos_exp) - (b * sin_exp)
  b_out =. (a * sin_exp) + (b * cos_exp)
  NB. Interleave back: pair (a_out[i], b_out[i]) per row, flatten to (n_rows, dim)
  n_rows =. # a_out
  pc =. 1 { $ a_out
  stacked =. a_out ,: b_out
  (n_rows , pc * 2) $ , (1 2 0 |: stacked)
)

NB. ---- NEOX RoPE for BATCHED arrays — table-based, rank-generic ----
NB. x = (lead..., dim) qk array (any rank >= 2, dim on the LAST axis; e.g.
NB. (L,n_heads,dim), (B,n_heads,dim), (B,c,n_heads,dim)). y = <cos_exp; sin_exp>
NB. where cos_exp/sin_exp are PRE-EXPANDED to (lead..., half) — the leading shape
NB. of x minus the dim axis, with half = dim % 2 (hoisted once per chunk by the
NB. caller and threaded through). Pairs are (i, i+half) (NEOX). The `"1` rotate
NB. + `,"1` concat are rank-generic (no permute needed for NEOX).
NB. CALLER CONVENTION: y MUST be a BOXED pair — pass `(cos) ; (sin)` (box), NOT
NB. `(cos) , (sin)` (append). `,` builds a (2*lead, n_heads, half) ARRAY and
NB. `> 0 { y` grabs the 0th ROW (n_heads, half) — only correct when lead=1; for
NB. lead>1 the cos/sin shapes stop conforming (2 16 64 vs 16 64).
rope_apply2_neox_b =: 4 : 0
  qk =. x
  cos_exp =. > 0 { y
  sin_exp =. > 1 { y
  half =. {: $ cos_exp
  a =. half {. "1 qk       NB. first half of each row
  b =. half }. "1 qk        NB. second half of each row
  a_out =. (a * cos_exp) - (b * sin_exp)
  b_out =. (a * sin_exp) + (b * cos_exp)
  a_out ,"1 b_out           NB. per-row: x[i]=a_out[i], x[i+half]=b_out[i]
)

NB. ---- INTERLEAVED RoPE for BATCHED arrays — table-based, rank-generic ----
NB. x = (lead..., dim) qk array (any rank >= 2, dim on the LAST axis). y =
NB. <cos_exp; sin_exp; idx> where cos_exp/sin_exp are PRE-EXPANDED to (lead..., half)
NB. and idx = 2 * i. half. Pairs are consecutive (i, i+1), unlike NEOX. The
NB. interleave stacks via `,:` then permutes the new axis to last (rank-generic:
NB. perm = (1 + i. r-1) , 0 for the stacked rank r) before ravel+reshape.
rope_apply2_t_b =: 4 : 0
  qk =. x
  cos_exp =. > 0 { y
  sin_exp =. > 1 { y
  idx =. > 2 { y
  a =. idx {"1 qk           NB. even-indexed cols per row
  b =. (1 + idx) {"1 qk     NB. odd-indexed cols per row
  a_out =. (a * cos_exp) - (b * sin_exp)
  b_out =. (a * sin_exp) + (b * cos_exp)
  lead =. }: $ a_out
  half =. {: $ a_out
  r =. 1 + # $ a_out
  perm =. (1 + i. r - 1) , 0
  (lead , 2 * half) $ , (perm |: (a_out ,: b_out))
)

NB. ---- Build RoPE cos/sin tables for all positions ----
NB. y = <context_len; head_dim; rope_freq>
NB. Returns <cos_tab; sin_tab>, each (context_len, half) where half = head_dim % 2
NB. theta(i, p) = p * freq^(-2i/head_dim), i in 0..half-1
build_rope_tables =: 3 : 0
  ctx_len =. > 0 { y
  head_dim =. > 1 { y
  freq =. > 2 { y
  half =. <. head_dim % 2
  pos_idx =. i. ctx_len
  theta =. pos_idx (*/) freq ^ (0 - 2 * i. half) % head_dim   NB. (ctx_len, half)
  (<(2 o. theta)) , (<(1 o. theta))
)
NB. Used by Gemma models for logit scaling
NB. Dyadic: scale softcap x — scale on left, value on right
softcap =: 4 : 0
  scale =. x
  ez2 =. ^ (2 * y % scale)
  (ez2 - 1) % (ez2 + 1) * scale
)


