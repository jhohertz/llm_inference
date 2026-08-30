NB. ================================================================
NB. Kernel Test Suite — unit tests for all transformer kernels
NB. Uses generated random arrays
NB. ================================================================
coclass 'inference'
load './kernels/jfloat.ijs'
load './tests/j/test_harness.ijs'
load './tests/j/pm_fixture.ijs'

NB. ---- rand_float ----
rand_float =: 3 : '_1 + 2 * (y ?@$ 10000) % 10000'

NB. ---- RMS helper: correct formula ----
NB. RMS(x) = sqrt(mean(x^2)) = sqrt((+/ *: x) % # x)
RMS =: 3 : '%: ((+/ *: x) % # x)'

NB. ---- Main test runner (all if. must be inside explicit def) ----
test_kernels =: 3 : 0
  init_counters ''
  suite_start =. 6!:1 ''
  echo '========================================'
  echo 'Kernel Test Suite (small arrays)'
  echo '========================================'

  NB. matmul
  echo ''
  echo '--- matmul ---'
  a =. 3 4 $ rand_float 12
  b =. 4 5 $ rand_float 20
  c =. a matmul b
  if. 3 5 -: $ c do. pass 'matmul shape 3x4@4x5=3x5'
  else. fail 'matmul shape' end.
  if. (1.0e_6 > +/ , | c - (a +/ .* b)) do. pass 'matmul matches +/.*'
  else. fail 'matmul value' end.
  if. (1.0e10 > >./ | c) do. pass 'matmul values OK'
  else. fail 'matmul values' end.
  I3 =. (i.3) =/ i.3
  IA =. I3 matmul a
  if. (0.1e_10 > +/ , | IA - a) do. pass 'matmul identity'
  else. fail 'matmul identity' end.

  NB. gelu
  echo ''
  echo '--- gelu ---'
  g_in =. 3 4 $ rand_float 12
  g_out =. gelu g_in
  if. 3 4 -: $ g_out do. pass 'gelu shape'
  else. fail 'gelu shape' end.
  if. (1.0e10 > >./ | g_out) do. pass 'gelu values OK'
  else. fail 'gelu values' end.
  if. 0.5 > | gelu 0 do. pass 'gelu(0)=0'
  else. fail 'gelu(0)' end.

  NB. silu
  echo ''
  echo '--- silu ---'
  s_in =. 3 4 $ rand_float 12
  s_out =. silu s_in
  if. 3 4 -: $ s_out do. pass 'silu shape'
  else. fail 'silu shape' end.
  if. (1.0e10 > >./ | s_out) do. pass 'silu values OK'
  else. fail 'silu values' end.
  if. 0.5 > | silu 0 do. pass 'silu(0)=0'
  else. fail 'silu(0)' end.

  NB. rms_norm
  echo ''
  echo '--- rms_norm ---'
  eps =. 1.0e_5
  w =. 8 $ 1
  x =. 8 $ 1 2 3 4 5 6 7 8
  rn_out =. rms_norm ((<eps) , <w) , <x
  if. 8 -: # rn_out do. pass 'rms_norm shape'
  else. fail 'rms_norm shape' end.
  rn_rms =. %: ((+/ *: rn_out) % # rn_out)
  w_rms =. %: ((+/ *: w) % # w)
  if. 0.001 > | rn_rms - w_rms do. pass 'rms_norm RMS~weight RMS'
  else. fail 'rms_norm RMS' end.

  NB. rms_norm_rows
  echo ''
  echo '--- rms_norm_rows ---'
  rnr_w =. 6 $ 1
  rnr_m =. 4 6 $ 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  rnr_out =. rms_norm_rows ((<1.0e_5) , <rnr_w) , <rnr_m
  if. 4 6 -: $ rnr_out do. pass 'rms_norm_rows shape'
  else. fail 'rms_norm_rows shape' end.

  NB. rope_apply
  echo ''
  echo '--- rope_apply ---'
  rope_qk =. 64 $ 1
  rope_out0 =. rope_qk rope_apply ((<64) , <0) , <1000000
  if. 64 -: # > rope_out0 do. pass 'rope shape'
  else. fail 'rope shape' end.
  if. 0.1e_10 > +/ | , > rope_out0 - 64$1 do. pass 'rope pos=0 identity'
  else. fail 'rope pos=0' end.
  rope_out1 =. rope_qk rope_apply ((<64) , <1) , <1000000
  if. -. (0.1e_10 > +/ | , > rope_out1 - 64$1) do. pass 'rope pos>0 changes'
  else. fail 'rope pos>0' end.

  NB. rope_apply2
  echo ''
  echo '--- rope_apply2 ---'
  rope2_qk =. 4 64 $ rand_float 256
  rope2_out =. rope2_qk rope_apply2 ((<64) , <5) , <1000000
  if. 4 64 -: $ rope2_out do. pass 'rope_apply2 shape'
  else. fail 'rope_apply2 shape' end.

  NB. rope_apply_neox (Gemma3/Qwen/Falcon style: pairs offset by dim/2)
  echo ''
  echo '--- rope_apply_neox ---'
  neox_qk =. 8 $ i. 8
  neox_out =. neox_qk rope_apply_neox ((<8) , <2) , <1000000
  if. 8 -: # > neox_out do. pass 'rope_apply_neox shape'
  else. fail 'rope_apply_neox shape' end.
  NB. Reference computed from ggml NEOX algorithm (dim=8, pos=2, base=1e6)
  neox_exp =. _3.637189707302727 0.6819836769119373 1.9879960080013317 2.9995572751278714 _1.6645873461885696 5.05320673082209 6.003987997337333 7.000189722659483
  if. 0.1e_4 > +/ | , > neox_out - neox_exp do. pass 'rope_apply_neox matches ggml'
  else. fail 'rope_apply_neox matches ggml' end.
  if. 0.1e_10 > +/ | , > (neox_qk rope_apply_neox ((<8) , <0) , <1000000) - 8 $ i. 8 do. pass 'rope_apply_neox pos=0 identity'
  else. fail 'rope_apply_neox pos=0' end.

  NB. rope_apply2_neox
  echo ''
  echo '--- rope_apply2_neox ---'
  neox2_qk =. 3 8 $ i. 24
  neox2_out =. neox2_qk rope_apply2_neox ((<8) , <2) , <1000000
  if. 3 8 -: $ neox2_out do. pass 'rope_apply2_neox shape'
  else. fail 'rope_apply2_neox shape' end.
  neox2_exp =. 3 8 $ _3.637189707302727 0.6819836769119373 1.9879960080013317 2.9995572751278714 _1.6645873461885696 5.05320673082209 6.003987997337333 7.000189722659483 _14.240743814285318 8.160361826068856 9.971980018673328 10.999051294702582 2.2806173760397446 13.542839246909717 14.019971986676001 15.000695671084772 _24.844297921267913 15.638739975225773 17.955964029345328 18.998545314277294 6.225822098268059 22.032471762997346 22.035955976014673 23.001201619510063
  if. 0.1e_4 > +/ | , > neox2_out - neox2_exp do. pass 'rope_apply2_neox matches ggml'
  else. fail 'rope_apply2_neox matches ggml' end.

  NB. softcap
  echo ''
  echo '--- softcap ---'
  sc_in =. 7 $ _5 _2 _1 0 1 2 5
  sc_out =. 3.0 softcap sc_in
  if. 7 -: # sc_out do. pass 'softcap shape'
  else. fail 'softcap shape' end.
  if. (3.0 > >./ | sc_out) do. pass 'softcap clips to 3.0'
  else. fail 'softcap clip' end.
  if. 0.5 > | {. sc_out do. pass 'softcap(0)=0'
  else. fail 'softcap(0)' end.

  NB. linear — weight (out,in), input vector (in,) → result (out,)
  echo ''
  echo '--- linear ---'
  w =. 6 4 $ rand_float 24
  x =. 4 $ rand_float 4
  b =. 6 $ rand_float 6
  lb =. w linear (<x) , <b
  if. 6 = # lb do. pass 'linear shape'
  else. fail 'linear shape' end.

  show_summary 1
)
pm_start 1e8

test_kernels 0

pm_report ''
