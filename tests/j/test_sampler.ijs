NB. test_sampler.ijs — Test suite for util/sampler.ijs
NB. Run: load 'test_sampler'; test_sampler_all

coclass 'inference'
load './util/sampler.ijs'
load './tests/j/pm_fixture.ijs'

tc =: 0
pc =: 0
fc =: 0

NB. ----------------------------------------------------------------
NB. Test harness
NB. ----------------------------------------------------------------
test_pass =: 4 : 0
  name =. x
  result =. y
  tc =: tc + 1
  if. result do.
    pc =: pc + 1
    echo name, ': PASS'
  else.
    fc =: fc + 1
    echo name, ': FAIL'
  end.
  result
)

test_eq =: 3 : 0
  NB. y = <expected; actual>
  expected =. 0 { y
  actual =. 1 { y
  expected -: actual
)

NB. ----------------------------------------------------------------
NB. Test: Temperature scaling
NB. ----------------------------------------------------------------
test_temp_basic =: 3 : 0
  logits =. 1 2 3
  scaled =. 2 sampler_temp logits
  expected =. 0.5 1 1.5
  test_eq (<expected), (<scaled)
)

test_temp_zero =: 3 : 0
  logits =. 1 2 3
  scaled =. 0 sampler_temp logits
  test_eq (<logits), (<scaled)
)

test_temp_high =: 3 : 0
  logits =. _10 0 10
  scaled =. 100 sampler_temp logits
  max_diff =. >./ scaled - <./ scaled
  max_diff <= 0.2
)

test_temp_low =: 3 : 0
  logits =. _10 0 10
  scaled =. 0.1 sampler_temp logits
  diffs =. 2 }. scaled
  abs_diffs =. | diffs
  >./ abs_diffs >= 100
)

NB. ----------------------------------------------------------------
NB. Test: Softmax
NB. ----------------------------------------------------------------
test_softmax_basic =: 3 : 0
  logits =. 1 2 3
  probs =. sampler_softmax logits
  prob_sum =. +/ probs
  ok =. prob_sum >: 0.999
  test_eq (<1.0), (<ok)
)

test_softmax_large =: 3 : 0
  logits =. 1000 1001 1002
  probs =. sampler_softmax logits
  prob_sum =. +/ probs
  ok =. prob_sum >: 0.999
  test_eq (<1.0), (<ok)
)

test_softmax_uniform =: 3 : 0
  logits =. 5 5 5 5
  probs =. sampler_softmax logits
  expected =. 0.25 0.25 0.25 0.25
  test_eq (<expected), (<probs)
)

test_softmax_dominant =: 3 : 0
  logits =. _100 0 100
  probs =. sampler_softmax logits
  dominant =. 2 { probs
  dominant > 0.99
)

NB. ----------------------------------------------------------------
NB. Test: Top-k
NB. ----------------------------------------------------------------
test_topk_indices_basic =: 3 : 0
  logits =. 3 1 4 1 5 9 2 6
  idx =. 3 sampler_topk_indices logits
  vals =. idx { logits
  sorted_vals =. \:~ vals
  test_eq (<3), (<# vals)
  test_eq (<9 6 5), (<sorted_vals)
)

test_topk_indices_more_than_vocab =: 3 : 0
  logits =. 1 2 3
  idx =. 100 sampler_topk_indices logits
  test_eq (<3), (<# idx)
  vals =. idx { logits
  \:~ vals -: 3 2 1
)

test_topk_indices_zero =: 3 : 0
  logits =. 1 2 3
  idx =. 0 sampler_topk_indices logits
  test_eq (<0), (<# idx)
)

test_topk_filter =: 3 : 0
  logits =. 1 2 3 4 5
  masked =. 3 sampler_topk logits
  NB. excluded set to -1e30 (NOT 0) so softmax's 2^(shifted) underflows to ~0
  masked -: _1e30 _1e30 3 4 5
)

NB. ----------------------------------------------------------------
NB. Test: Top-p
NB. ----------------------------------------------------------------
test_topp_mask_basic =: 3 : 0
  probs =. 0.01 0.02 0.05 0.32 0.60
  mask =. 0.9 sampler_topp_mask probs
  expected =. 0 0 0 1 1
  test_eq (<expected), (<mask)
)

test_topp_mask_all =: 3 : 0
  probs =. 0.1 0.1 0.1 0.1 0.6
  mask =. 0.99 sampler_topp_mask probs
  test_eq (<5), (<# mask)
  ok =. 0 -: +/ 1 -: mask
  ok
)

test_topp_mask_strict =: 3 : 0
  probs =. 0.2 0.2 0.2 0.2 0.2
  mask =. 0.3 sampler_topp_mask probs
  expected =. 1 1 0 0 0  NB. cumsum crosses 0.3 at index 1 (0.4)
  test_eq (<expected), (<mask)
)

test_topp_filter =: 3 : 0
  probs =. 0.01 0.02 0.05 0.32 0.60
  filtered =. 0.9 sampler_topp probs
  filtered -: 0 0 0 0.32 0.60
)

NB. ----------------------------------------------------------------
NB. Test: Min-p
NB. ----------------------------------------------------------------
test_minp_mask_basic =: 3 : 0
  probs =. 0.01 0.1 0.2 0.4 0.3
  mask =. 0.1 sampler_minp_mask probs
  expected =. 0 1 1 1 1
  test_eq (<expected), (<mask)
)

test_minp_disabled =: 3 : 0
  probs =. 0.1 0.2 0.3
  mask =. 0 sampler_minp_mask probs
  expected =. 1 1 1
  test_eq (<expected), (<mask)
)

test_minp_strict =: 3 : 0
  probs =. 0.01 0.1 0.5 0.3 0.05
  mask =. 0.3 sampler_minp_mask probs
  expected =. 0 0 1 1 0
  test_eq (<expected), (<mask)
)

test_minp_filter =: 3 : 0
  probs =. 0.01 0.1 0.2 0.4 0.3
  filtered =. 0.1 sampler_minp probs
  expected =. 0 0.1 0.2 0.4 0.3
  test_eq (<expected), (<filtered)
)

NB. ----------------------------------------------------------------
NB. Test: Weighted sample
NB. ----------------------------------------------------------------
test_weighted_sample_shape =: 3 : 0
  probs =. 0.2 0.3 0.5
  sample =. sampler_weighted_sample probs
  0 <= sample *. sample < 3
)

test_weighted_sample_rejects_invalid =: 3 : 0
  probs =. 0.25 0.25 0.25 0.25
  n =. 100
  samples =. > sampler_weighted_sample each n $ probs
  min_sample =. <./ samples
  max_sample =. >./ samples
  (min_sample >: 0) *. (max_sample < 4)
)

test_weighted_sample_bias =: 3 : 0
  probs =. 0.1 0.2 0.3 0.4
  NB. Test that sampler can produce different values (not all same)
  s1 =. sampler_weighted_sample probs
  s2 =. sampler_weighted_sample probs
  s3 =. sampler_weighted_sample probs
  s4 =. sampler_weighted_sample probs
  NB. At least one sample should differ from the first
  (s1 -: s2) +. (s1 -: s3) +. (s1 -: s4) < 3
)

NB. ----------------------------------------------------------------
NB. Test: Full sampler
NB. ----------------------------------------------------------------
test_sampler_greedy =: 3 : 0
  logits =. 1 5 3 2
  params =. <0
  idx =. params sampler_sample logits
  test_eq (<1), (<idx)
)

test_sampler_temperature =: 3 : 0
  logits =. 0 0 0
  params =. <2
  sample =. params sampler_sample logits
  0 <= sample *. sample < 3
)

test_sampler_topk =: 3 : 0
  logits =. 0 100 0 0 0
  params =. <1 3
  sample =. params sampler_sample logits
  0 <= sample *. sample < 3
)

test_sampler_topp =: 3 : 0
  logits =. _100 _100 0 100 100
  params =. <1 0 0.95
  sample =. params sampler_sample logits
  (sample >: 3) *. (sample < 5)
)

test_sampler_renormalization =: 3 : 0
  logits =. 0 0 0 0 0
  params =. <1 0 0.3 0.5
  sample =. params sampler_sample logits
  0 <= sample *. sample < 5
)

NB. ----------------------------------------------------------------
NB. Test: Sampler edge cases
NB. ----------------------------------------------------------------
test_sampler_single =: 3 : 0
  logits =. 5
  params =. <1
  sample =. params sampler_sample logits
  test_eq (<0), (<sample)
)

test_sampler_two_tokens =: 3 : 0
  logits =. 1 2
  params =. <1
  sample =. params sampler_sample logits
  sample = 0 +. sample = 1
)

test_sampler_negative_logits =: 3 : 0
  logits =. _5 _3 _1 _2
  params =. <1
  sample =. params sampler_sample logits
  0 <= sample *. sample < 4
)

test_sampler_mixed_sign =: 3 : 0
  logits =. _100 0 100
  params =. <1
  sample =. params sampler_sample logits
  0 <= sample *. sample < 3
)

NB. ----------------------------------------------------------------
NB. Run all tests
NB. ----------------------------------------------------------------
test_sampler_all =: 3 : 0
  suite_start =. 6!:1 ''
  echo '=== Sampler Test Suite ==='
  
  NB. Temperature tests
  'test_temp_basic' test_pass (test_temp_basic '')
  'test_temp_zero' test_pass (test_temp_zero '')
  'test_temp_high' test_pass (test_temp_high '')
  'test_temp_low' test_pass (test_temp_low '')
  
  NB. Softmax tests
  'test_softmax_basic' test_pass (test_softmax_basic '')
  'test_softmax_large' test_pass (test_softmax_large '')
  'test_softmax_uniform' test_pass (test_softmax_uniform '')
  'test_softmax_dominant' test_pass (test_softmax_dominant '')
  
  NB. Top-k tests
  'test_topk_indices_basic' test_pass (test_topk_indices_basic '')
  'test_topk_indices_more_than_vocab' test_pass (test_topk_indices_more_than_vocab '')
  'test_topk_indices_zero' test_pass (test_topk_indices_zero '')
  'test_topk_filter' test_pass (test_topk_filter '')
  
  NB. Top-p tests
  'test_topp_mask_basic' test_pass (test_topp_mask_basic '')
  'test_topp_mask_all' test_pass (test_topp_mask_all '')
  'test_topp_mask_strict' test_pass (test_topp_mask_strict '')
  'test_topp_filter' test_pass (test_topp_filter '')
  
  NB. Min-p tests
  'test_minp_mask_basic' test_pass (test_minp_mask_basic '')
  'test_minp_disabled' test_pass (test_minp_disabled '')
  'test_minp_strict' test_pass (test_minp_strict '')
  'test_minp_filter' test_pass (test_minp_filter '')
  
  NB. Weighted sample tests
  'test_weighted_sample_shape' test_pass (test_weighted_sample_shape '')
  'test_weighted_sample_rejects_invalid' test_pass (test_weighted_sample_rejects_invalid '')
  'test_weighted_sample_bias' test_pass (test_weighted_sample_bias '')
  
  NB. Full sampler tests
  'test_sampler_greedy' test_pass (test_sampler_greedy '')
  'test_sampler_temperature' test_pass (test_sampler_temperature '')
  'test_sampler_topk' test_pass (test_sampler_topk '')
  'test_sampler_topp' test_pass (test_sampler_topp '')
  'test_sampler_renormalization' test_pass (test_sampler_renormalization '')
  
  NB. Edge case tests
  'test_sampler_single' test_pass (test_sampler_single '')
  'test_sampler_two_tokens' test_pass (test_sampler_two_tokens '')
  'test_sampler_negative_logits' test_pass (test_sampler_negative_logits '')
  'test_sampler_mixed_sign' test_pass (test_sampler_mixed_sign '')
  
  echo ''
  echo '=== All 27 sampler tests complete ==='
  st =. ": (6!:1 '' - suite_start)
  echo 'Total: ' , ": tc
  echo 'Passed: ' , ": pc
  echo 'Failed: ' , ": fc
  echo 'Time:  ' , st , 's'
)

pm_start 1e8

test_sampler_all ''

pm_report ''
