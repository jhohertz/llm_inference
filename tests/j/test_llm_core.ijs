NB. ================================================================
NB. llm_core Helper Test Suite — direct unit coverage of the shared
NB. pure helpers in util/llm_core.ijs:
NB.   build_rope_expansions  (NEOX cos/sin table expansion, lead-generic)
NB.   gqa_attn_single        (GQA single-token attention)
NB.   gqa_attn_batch         (GQA batched/prompt attention)
NB. (load_gguf_base is verified end-to-end via the arch suites.)
NB. ================================================================
coclass 'inference'
load 'llm/inference/util/llm_core'
load './tests/j/test_harness.ijs'
load './tests/j/pm_fixture.ijs'

NB. ---- Main test runner ----
test_llm_core =: 3 : 0
  init_counters ''
  suite_start =. 6!:1 ''
  echo '========================================'
  echo 'llm_core Helper Test Suite'
  echo '========================================'

  NB. ---- build_rope_expansions ----
  echo ''
  echo '--- build_rope_expansions ---'
  cos_all =. 2 4 $ 1 2 3 4 5 6 7 8
  sin_all =. 2 4 $ 8 7 6 5 4 3 2 1
  exp =. (cos_all ; sin_all ; 4 ; 3 ; 2 ; 2) build_rope_expansions ''
  if. 2 3 4 -: $ (> 0 { exp) do. pass 'build_rope_expansions cos_expq shape (L,n_heads,half)'
  else. fail 'build_rope_expansions cos_expq shape' end.
  if. 2 2 4 -: $ (> 2 { exp) do. pass 'build_rope_expansions cos_expk shape (L,n_heads_kv,half)'
  else. fail 'build_rope_expansions cos_expk shape' end.
  if. ((3 4 $ 0 { cos_all) -: 0 { (> 0 { exp)) do. pass 'build_rope_expansions cos_expq head0 = cos_all[0] broadcast'
  else. fail 'build_rope_expansions cos_expq head0' end.
  if. ((3 4 $ 0 { sin_all) -: 0 { (> 1 { exp)) do. pass 'build_rope_expansions sin_expq head0 = sin_all[0] broadcast'
  else. fail 'build_rope_expansions sin_expq head0' end.
  if. ((2 4 $ 0 { cos_all) -: 0 { (> 2 { exp)) do. pass 'build_rope_expansions cos_expk head0 = cos_all[0] broadcast'
  else. fail 'build_rope_expansions cos_expk head0' end.

  NB. ---- gqa_attn_single ----
  echo ''
  echo '--- gqa_attn_single ---'
  Q =. 2 4 $ 1 0 0 0  0 0 0 1
  K =. 3 1 4 $ 1 0 0 0  0 1 0 0  0 0 0 1
  V =. 3 1 4 $ 1 0 0 0  0 1 0 0  0 0 0 1
  attn_o =. (6 8 $ i. 48) % 48
  out =. gqa_attn_single (Q ; K ; V ; attn_o ; 2 ; 1 ; 4 ; 2)
  if. 6 = # out do. pass 'gqa_attn_single shape (emb,)'
  else. fail 'gqa_attn_single shape' end.
  ref_s =. 0.141418 0.474751 0.808085 1.14142 1.47475 1.80808
  if. (1.0e_4 > +/ | , out - ref_s) do. pass 'gqa_attn_single matches reference'
  else. fail 'gqa_attn_single value' end.

  NB. ---- gqa_attn_batch (L=1 prefill) ----
  echo ''
  echo '--- gqa_attn_batch ---'
  Qb =. 1 2 4 $ 1 0 0 0  0 0 0 1
  Kb =. 1 1 4 $ 1 0 0 0
  Vb =. 1 1 4 $ 1 0 0 0
  maskb =. 2 1 $ 0 0
  ob =. gqa_attn_batch (Qb ; Kb ; Vb ; attn_o ; maskb ; 2 ; 1 ; 4 ; 2 ; 1)
  if. 1 6 -: $ ob do. pass 'gqa_attn_batch shape (L,emb)'
  else. fail 'gqa_attn_batch shape' end.
  ref_b =. 0.0833333 0.416667 0.75 1.08333 1.41667 1.75
  if. (1.0e_4 > +/ | , (, ob) - ref_b) do. pass 'gqa_attn_batch L=1 matches reference'
  else. fail 'gqa_attn_batch value' end.

  show_summary 1
)
pm_start 1e8

test_llm_core 0

pm_report ''
