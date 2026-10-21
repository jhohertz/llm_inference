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


  NB. Broadcastly — NumPy-style broadcasting via verb rank (arXiv:2609.16064)
  echo ''
  echo '--- Broadcastly ---'
  assert_test ((2 3) -: $ (2 3 $ 10 20 30 40 50 60) + Broadcastly (3 $ 1 2 3)) ; 'broadcast (2 3)+(3,) -> 2 3'
  assert_test (((2 3 $ 11 22 33 41 52 63) -: (2 3 $ 10 20 30 40 50 60) + Broadcastly (3 $ 1 2 3)) ; 'broadcast (2 3)+(3,) values match manual')
  assert_test ((4 4) -: $ (1 4 $ 10 11 12 13) + Broadcastly (4 1 $ 1 2 3 4)) ; 'broadcast (1 4)+(4 1) -> 4 4'
  assert_test ((2 2 3) -: $ (0) + Broadcastly (2 2 3 $ 0)) ; 'scalar + (2 2 3) -> 2 2 3'
  assert_test ((1 2 3 4 1 5 6) -: $ (1 4 1 1 6 $ i. 24) + Broadcastly (1 2 3 1 1 5 6 $ i. 30)) ; 'documented 1 4 1 1 6 + 1 2 3 1 1 5 6'
  bad =. 'incompatible rejected'
  try.
    _ =. (3 1 4 $ 0) + Broadcastly (1 4 2 $ 0)
    fail 'broadcast rejects incompatible (3 1 4)+(1 4 2)'
  catch.
    pass 'broadcast rejects incompatible (3 1 4)+(1 4 2)'
  end.

  show_summary 1
)
pm_start 1e8

test_kernels 0

pm_report ''
