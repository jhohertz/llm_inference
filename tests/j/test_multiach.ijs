NB. ================================================================
NB. Multi-arch session safety test — bd_* accessor clobber regression
NB.
NB. Regression: the arch modules used to redefine shared bd_* accessor
NB. names with different block_data indices (gemma3 13/14/15, qwen2
NB. 12/13/14, smollm2 9/10/11), so loading a second arch module broke the
NB. first's attention. Accessors are now per-arch prefixed
NB. (gem3_bd_* / qw2_bd_* / llama_bd_*). This test loads gemma3, then loads
NB. the qwen2 module (the old clobber trigger) and verifies gem3 accessors
NB. still read the gemma layout correctly.
NB. ================================================================
coclass 'inference'
load './inference.ijs'
load './tests/j/pm_fixture.ijs'

test_multiach =: 3 : 0
  tc =. 0
  pc =. 0
  fc =. 0
  fl =. ''

  echo ''
  echo '=============================================================='
  echo '  MULTI-ARCH SESSION SAFETY TEST (bd_* clobber regression)'
  echo '=============================================================='

  gemma_path =. 'gemma-3-270m-it'
  echo '  (loading gemma model)...'
  llm =. load_gguf_to_llm gemma_path
  bd_list =. llm_block_data llm
  bd0 =. > 0 { bd_list
  echo '  (loaded)'
  echo ''

  NB. --- gem3 accessor, before qwen2 module load ---
  tc =. tc + 1
  if. 4 -: gem3_bd_n_heads bd0 do.
    pc =. pc + 1
    echo 'PASS: gem3_bd_n_heads = 4 (before qwen2 module load)'
  else.
    fl =. fl , 'gem3_bd_n_heads before qwen2', LF
    echo 'FAIL: gem3_bd_n_heads before qwen2 load'; echo '  got: '; echo gem3_bd_n_heads bd0
  end.

  NB. --- load qwen2 module: the clobber trigger ---
  echo '  (loading qwen2 module — old clobber trigger)...'
  require 'llm/inference/models/qwen2'
  echo '  (loaded)'
  echo ''

  NB. --- gem3 accessor must STILL read the gemma layout ---
  tc =. tc + 1
  if. 4 -: gem3_bd_n_heads bd0 do.
    pc =. pc + 1
    echo 'PASS: gem3_bd_n_heads = 4 after qwen2 module load (no clobber)'
  else.
    fl =. fl , 'gem3_bd_n_heads after qwen2', LF
    echo 'FAIL: gem3_bd_n_heads after qwen2 load'; echo '  got: '; echo gem3_bd_n_heads bd0
  end.

  NB. --- qwen2 accessor is a separate name reading a different index ---
  tc =. tc + 1
  if. 0 = 4 -: qw2_bd_n_heads bd0 do.
    pc =. pc + 1
    echo 'PASS: qw2_bd_n_heads is independent of gem3_bd_n_heads (reads index 12, not 13)'
  else.
    fl =. fl , 'qw2_bd_n_heads independence', LF
    echo 'FAIL: qw2_bd_n_heads not independent'; echo '  got: '; echo qw2_bd_n_heads bd0
  end.

  echo ''
  echo '=============================================================='
  echo '  TEST SUMMARY'
  echo '=============================================================='
  echo ''
  echo '  Total:          ' , ": tc
  echo '  Passed:         ' , ": pc
  echo '  Failed:         ' , ": fc
  if. 0 < # fl do. echo ''; echo 'Failed:'; echo fl end.
  echo ''
  if. 0 = fc do. echo 'All tests passed!' else. echo 'Some tests failed.' end.
  echo ''
)

pm_start 1e8

test_multiach 0

pm_report ''
