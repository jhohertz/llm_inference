NB. ================================================================
NB. Shared Transformer Kernels — used by all architecture models
NB. ================================================================

NB. ---- Matrix Multiplication (idiomatic J) ----
coclass 'inference'
matmul =: +/ .*

NB. ---- Linear Layer Forward ----
NB. <weight> linear <input, bias>
linear =: 4 : 0
  weight =. x
  input =. > 0 { y
  bias =. > 1 { y
  ws =. $ weight
  ins =. $ input
  if. 0 = #ws do. echo 'LINEAR: EMPTY WEIGHT, wtype:', ": 3!:0 weight; echo 'y was:', ": 3!:0 y; return. $0 end.
  if. 0 = #ins do. echo 'LINEAR: EMPTY INPUT, itype:', ": 3!:0 input; echo 'y was:', ": 3!:0 y; return. $0 end.
  result =. weight (+/ .* ) input
  if. #bias > 0 do. result =. result + bias end.
  result
)

NB. ---- Raw matvec/matmat (no boxing) — for hot fused projections ----
NB. x = weight, y = raw input array (no bias). Skips the box/unbox of `linear`.
linear_r =: +/ .*

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
NB. Applies RMSNorm independently to each row
NB. Tacit: broadcast weight to matrix shape ((($ @ >@(2&{)) $ >@(1&{))), then
NB. matrix % sqrt((row sumsq % row_len) + eps). Hot path (every norm).
rms_norm_rows =: ((($ @ >@(2&{)) $ >@(1&{)) * ((>@(2&{)) % (%: @ (((+/"1 @ (*: @ >@(2&{))) % ({: @ $ @ >@(2&{))) + >@(0&{)))))

NB. ---- RoPE for 1D vector ----
NB. x rope_apply y: x = qk vector (1D), y = <dim, pos, freq>
NB. freq = rope_freq_base from KV pairs
NB. Formula: freq_i = (pos * freq_base) ^ (-2i / dim)
rope_apply =: 4 : 0
  qk =. x
  dim =. > 0 { y
  pos =. > 1 { y
  freq =. > 2 { y
  pair_count =. dim % 2
  indices =. 2 * i. pair_count
  theta =. pos * freq ^ (0 - indices) % dim
  cos_t =. 2 o. theta
  sin_t =. 1 o. theta
  NB. Extract even/odd indexed elements directly (no reshape)
  a =. indices { qk
  b =. (1 + indices) { qk
  a_out =. (a * cos_t) - (b * sin_t)
  b_out =. (a * sin_t) + (b * cos_t)
  NB. Interleave back: pair each (a_out[i], b_out[i]) and flatten
  , a_out ,. b_out
)

   NB. ---- RoPE for 2D array — vectorized row-wise ----
   NB. x = (n_rows, dim) qk matrix, y = <dim, pos, freq>
   NB. Applies rope_apply to each row in a vectorized operation
   rope_apply2 =: 4 : 0
     qk =. x
     dim =. > 0 { y
     pos =. > 1 { y
     freq =. > 2 { y
     pair_count =. dim % 2
     indices =. 2 * i. pair_count
     theta =. pos * freq ^ (0 - indices) % dim
     cos_t =. 2 o. theta
     sin_t =. 1 o. theta
     NB. Extract a (even-indexed) and b (odd-indexed) for all rows
     a =. indices {"1 qk     NB. (n_rows, pair_count)
     b =. (1 + indices) {"1 qk  NB. (n_rows, pair_count)
     NB. Broadcast cos/sin to match a/b shape
     cos_exp =. ($ a) $ cos_t
     sin_exp =. ($ a) $ sin_t
     a_out =. (a * cos_exp) - (b * sin_exp)
     b_out =. (a * sin_exp) + (b * cos_exp)
      NB. Interleave: pair (a[i],b[i]) and reshape to [n_rows, dim]
      n_rows =. # a_out
      sh =. $ a_out
      pair_count =. 1 { sh
      stacked =. a_out ,: b_out
      (n_rows, pair_count * 2) $ , (1 2 0 |: stacked)
    )
NB. ---- RoPE for 2D array (legacy alias) ----
rope_apply2_v1 =: rope_apply"1

NB. ---- NEOX-style RoPE (GPT-NeoX / Gemma3 / Qwen / Falcon) ----
NB. Pairs element i with element i + dim/2 (offset by half), NOT consecutive pairs.
NB. Same frequencies as rope_apply: theta_i = pos * freq_base^(-2i/dim), i in 0..dim/2-1.
NB. Monadic on a 1D qk vector: x rope_apply_neox <dim; pos; freq>
rope_apply_neox =: 4 : 0
  qk =. x
  dim =. > 0 { y
  pos =. > 1 { y
  freq =. > 2 { y
  half =. dim % 2
  theta =. pos * freq ^ (0 - 2 * i. half) % dim
  cos_t =. 2 o. theta
  sin_t =. 1 o. theta
  a =. half {. qk          NB. elements 0..half-1
  b =. half }. qk          NB. elements half..dim-1
  a_out =. (a * cos_t) - (b * sin_t)
  b_out =. (a * sin_t) + (b * cos_t)
  a_out , b_out            NB. x[i]=a_out[i], x[i+half]=b_out[i]
)

NB. ---- NEOX RoPE for 2D array (vectorized row-wise) ----
NB. x = (n_rows, dim) qk matrix, y = <dim, pos, freq>
rope_apply2_neox =: 4 : 0
  qk =. x
  dim =. > 0 { y
  pos =. > 1 { y
  freq =. > 2 { y
  half =. dim % 2
  theta =. pos * freq ^ (0 - 2 * i. half) % dim
  cos_t =. 2 o. theta
  sin_t =. 1 o. theta
  a =. half {. "1 qk       NB. first half of each row
  b =. half }. "1 qk       NB. second half of each row
  cos_exp =. ($ a) $ cos_t
  sin_exp =. ($ a) $ sin_t
  a_out =. (a * cos_exp) - (b * sin_exp)
  b_out =. (a * sin_exp) + (b * cos_exp)
  a_out ,"1 b_out          NB. per-row: x[i]=a_out[i], x[i+half]=b_out[i]
)

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
  z =. y % scale
  ez2 =. ^ (2 * z)
  (ez2 - 1) % (ez2 + 1) * scale
)


