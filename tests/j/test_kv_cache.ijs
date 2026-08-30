NB. ================================================================
NB. KV Cache Test Suite — tests kv_create, kv_write, kv_write_rows,
NB. kv_read, kv_reset (monadic, persistent kv_cache_g global)
NB. Cache: PRE-ALLOCATED flat arrays, one per kind —
NB. k_cache_g/v_cache_g: (n_layers*eff_seq, n_heads_kv*head_dim). kv_pos_g
NB. tracks the used length; kv_reset reuses the buffer (no realloc).
NB. ================================================================

coclass 'inference'
load './util/kv_cache.ijs'
load './tests/j/pm_fixture.ijs'

test_kv_cache =: 3 : 0
  tc =. 0
  pc =. 0
  fc =. 0
  fl =. ''
  suite_start =. 6!:1 ''

  echo ''
  echo '========== KV CACHE TEST SUITE =========='
  echo ''

  NB. ================================================================
  echo '--- Section 1: kv_create ---'
  echo ''

  NB. Basic create (2 layers, 512 seq, 1 kv head, 256 hd)
  tc =. tc + 1
  kv_create ((<2) , (<512) , (<1) , (<256))
  if. (4 -: # kv_meta) *. (1024 256 -: $ k_cache_g) *. (1024 256 -: $ v_cache_g) do.
    pc =. pc + 1
    echo 'PASS: kv_create sets kv_meta (4), flat k_cache_g/v_cache_g (1024,256)'
  else.
    fc =. fc + 1
    fl =. fl , 'kv_create sets globals', LF
    echo 'FAIL: kv_create sets globals'
  end.

  NB. Flat buffer shape = (n_layers*eff_seq, n_heads_kv*head_dim) = (1024, 256)
  tc =. tc + 1
  if. (1024 256 -: $ k_cache_g) do.
    pc =. pc + 1
    echo 'PASS: flat buffer shape (1024,256)'
  else.
    fc =. fc + 1
    fl =. fl , 'buffer shape', LF
    echo 'FAIL: buffer shape'
  end.

  NB. Empty (used length 0) after create
  tc =. tc + 1
  if. (0 -: kv_pos_g) do.
    pc =. pc + 1
    echo 'PASS: cache empty after create (kv_pos_g 0)'
  else.
    fc =. fc + 1
    fl =. fl , 'cache empty', LF
    echo 'FAIL: cache empty'
  end.

  NB. ================================================================
  echo '--- Section 2: kv_write (n_heads_kv=1, layer isolation) ---'
  echo ''

  NB. Write layer0 pos0, layer1 pos0 — must NOT interfere
  kv_create ((<2) , (<10) , (<1) , (<4))
  k0 =. 1 4 $ 42
  v0 =. 1 4 $ 99
  kv_write ((<0) , (<0) , (<k0) , (<v0))
  k1 =. 1 4 $ 77
  v1 =. 1 4 $ 88
  kv_write ((<1) , (<0) , (<k1) , (<v1))

  NB. layer0 pos0 = 42, layer1 pos0 = 77 (layer isolation)
  tc =. tc + 1
  result =. kv_read ((<0) , <0)
  k_r =. > 0 { result
  if. (1 1 4 -: $ k_r) *. (42 42 42 42 -: , k_r) do.
    pc =. pc + 1
    echo 'PASS: layer0 pos0 → k[0,:]=42'
  else.
    fc =. fc + 1
    fl =. fl , 'layer0 pos0', LF
    echo 'FAIL: layer0 pos0 → k[0,:]=42'
  end.
  tc =. tc + 1
  result =. kv_read ((<1) , <0)
  k_r =. > 0 { result
  if. (77 77 77 77 -: , k_r) do.
    pc =. pc + 1
    echo 'PASS: layer1 pos0 → k[0,:]=77 (isolated from layer0)'
  else.
    fc =. fc + 1
    fl =. fl , 'layer1 pos0 isolation', LF
    echo 'FAIL: layer1 pos0 → k[0,:]=77 (isolated)'
  end.

  NB. Write layer0 pos1 (stride within a layer)
  k2 =. 1 4 $ 99
  v2 =. 1 4 $ 1
  kv_write ((<0) , (<1) , (<k2) , (<v2))
  tc =. tc + 1
  result =. kv_read ((<0) , <1)
  k_r =. > 0 { result
  if. (2 1 4 -: $ k_r) *. (42 42 42 42 99 99 99 99 -: , k_r) do.
    pc =. pc + 1
    echo 'PASS: layer0 pos1 → k[0,:]=42, k[1,:]=99'
  else.
    fc =. fc + 1
    fl =. fl , 'layer0 pos1 stride', LF
    echo 'FAIL: layer0 pos1 stride (k[0,:]=42, k[1,:]=99)'
  end.

  NB. layer1 pos0 unchanged after layer0 pos1 write
  tc =. tc + 1
  result =. kv_read ((<1) , <0)
  k_r =. > 0 { result
  if. (77 77 77 77 -: , k_r) do.
    pc =. pc + 1
    echo 'PASS: layer1 pos0 unchanged after layer0 pos1 write'
  else.
    fc =. fc + 1
    fl =. fl , 'layer1 unchanged', LF
    echo 'FAIL: layer1 pos0 unchanged'
  end.

  NB. ================================================================
  echo '--- Section 3: kv_write (n_heads_kv=2, multi-head) ---'
  echo ''

  kv_create ((<2) , (<10) , (<2) , (<4))
  k0b =. 2 4 $ 42
  v0b =. 2 4 $ 99
  kv_write ((<0) , (<0) , (<k0b) , (<v0b))
  tc =. tc + 1
  result =. kv_read ((<0) , <0)
  k_r =. > 0 { result
  if. (1 2 4 -: $ k_r) do.
    pc =. pc + 1
    echo 'PASS: multi-head read shape (1,2,4)'
  else.
    fc =. fc + 1
    fl =. fl , 'multi-head shape', LF
    echo 'FAIL: multi-head read shape'
  end.

  k1b =. 2 4 $ 77
  v1b =. 2 4 $ 88
  kv_write ((<0) , (<1) , (<k1b) , (<v1b))
  tc =. tc + 1
  result =. kv_read ((<0) , <1)
  k_r =. > 0 { result
  if. (42 42 42 42 42 42 42 42 77 77 77 77 77 77 77 77 -: , k_r) do.
    pc =. pc + 1
    echo 'PASS: multi-head write layer0 pos0=42x8, pos1=77x8'
  else.
    fc =. fc + 1
    fl =. fl , 'multi-head write pos 1', LF
    echo 'FAIL: multi-head write pos 1'
  end.

  NB. ================================================================
  echo '--- Section 4: kv_write_rows (bulk prefill) ---'
  echo ''

  kv_create ((<2) , (<10) , (<1) , (<4))
  rows0 =. 3 1 4 $ 42
  rows1 =. 3 1 4 $ 99
  kv_write_rows ((<0) , (<0) , (<0) , <rows0)
  kv_write_rows ((<0) , (<1) , (<0) , <rows1)
  tc =. tc + 1
  result =. kv_read ((<0) , <2)
  k_r =. > 0 { result
  if. (3 1 4 -: $ k_r) *. ((3 1 4 $ 42) -: k_r) do.
    pc =. pc + 1
    echo 'PASS: kv_write_rows writes L=3 rows at layer0'
  else.
    fc =. fc + 1
    fl =. fl , 'kv_write_rows layer0', LF
    echo 'FAIL: kv_write_rows layer0'
  end.
  tc =. tc + 1
  result =. kv_read ((<1) , <2)
  k_r =. > 0 { result
  if. ((3 1 4 $ 99) -: k_r) do.
    pc =. pc + 1
    echo 'PASS: kv_write_rows layer1 isolated'
  else.
    fc =. fc + 1
    fl =. fl , 'kv_write_rows layer1', LF
    echo 'FAIL: kv_write_rows layer1'
  end.

  NB. ================================================================
  echo '--- Section 5: kv_read ---'
  echo ''

  kv_create ((<2) , (<10) , (<1) , (<4))
  k0 =. 1 4 $ 42
  v0 =. 1 4 $ 99
  kv_write ((<0) , (<0) , (<k0) , (<v0))
  k1 =. 1 4 $ 77
  v1 =. 1 4 $ 88
  kv_write ((<0) , (<1) , (<k1) , (<v1))

  NB. Read layer0 up to pos 0
  result =. kv_read ((<0) , <0)
  k_r =. > 0 { result
  v_r =. > 1 { result
  tc =. tc + 1
  if. (1 1 4 -: $ k_r) *. (99 99 99 99 -: , v_r) do.
    pc =. pc + 1
    echo 'PASS: kv_read layer0 pos0 shape (1,1,4), v=99s'
  else.
    fc =. fc + 1
    fl =. fl , 'kv_read layer0 pos0 shape/v', LF
    echo 'FAIL: kv_read layer0 pos0 shape/v'
  end.

  NB. Read layer0 up to pos 1
  result =. kv_read ((<0) , <1)
  k_r =. > 0 { result
  v_r =. > 1 { result
  tc =. tc + 1
  if. (2 1 4 -: $ k_r) *. (42 42 42 42 77 77 77 77 -: , k_r) *. (99 99 99 99 88 88 88 88 -: , v_r) do.
    pc =. pc + 1
    echo 'PASS: kv_read layer0 pos1 returns 42s then 77s (K), 99s then 88s (V)'
  else.
    fc =. fc + 1
    fl =. fl , 'kv_read layer0 pos1 values', LF
    echo 'FAIL: kv_read layer0 pos1 values'
  end.

  NB. ================================================================
  echo '--- Section 6: kv_reset ---'
  echo ''

  kv_reset ''
  tc =. tc + 1
  if. (0 -: kv_pos_g) do.
    pc =. pc + 1
    echo 'PASS: kv_reset empties cache (kv_pos_g 0, buffer reused)'
  else.
    fc =. fc + 1
    fl =. fl , 'kv_reset', LF
    echo 'FAIL: kv_reset'
  end.

  NB. ================================================================
  echo '--- Section 7: Edge cases ---'
  echo ''

  NB. Different layer/head_dim combos; stride = n_kv*hd per layer
  kv_create ((<3) , (<256) , (<8) , (<64))
  tc =. tc + 1
  stride =. (> 2 { kv_meta) * (> 3 { kv_meta)
  if. 512 = stride do.
    pc =. pc + 1
    echo 'PASS: stride = n_heads_kv * head_dim = 512'
  else.
    fc =. fc + 1
    fl =. fl , 'stride calc', LF
    echo 'FAIL: stride = n_heads_kv * head_dim (got ' , ": stride , ')'
  end.

  k0 =. 8 64 $ 1
  v0 =. 8 64 $ 2
  kv_write ((<0) , (<0) , (<k0) , (<v0))
  k1 =. 8 64 $ 3
  v1 =. 8 64 $ 4
  kv_write ((<0) , (<1) , (<k1) , (<v1))
  tc =. tc + 1
  result =. kv_read ((<0) , <1)
  k_r =. > 0 { result
  if. (2 8 64 -: $ k_r) *. ((8 64 $ 1) -: 0 { k_r) *. ((8 64 $ 3) -: 1 { k_r) do.
    pc =. pc + 1
    echo 'PASS: layer0 pos0=1s unchanged after pos1 write (8 heads), pos1=3s'
  else.
    fc =. fc + 1
    fl =. fl , 'layer0 pos0 values (8 heads)', LF
    echo 'FAIL: layer0 pos0/pos1 values (8 heads)'
  end.

  NB. ================================================================
  echo '--- Section 8: Batched sequences (kv_batch_g) ---'
  echo ''

  kv_batch_g =: 2
  kv_create ((<2) , (<10) , (<1) , (<4))
  tc =. tc + 1
  if. 40 4 -: $ k_cache_g do. pc =. pc + 1
    echo 'PASS: batched flat shape (40,4) = (2 layers * 2 seq * 10, 4)'
  else. fc =. fc + 1
    fl =. fl , 'batched flat shape', LF; echo 'FAIL: batched flat shape'; echo '  got: '; echo $ k_cache_g end.

  NB. seq0 layer0 pos0 = 42, seq1 layer0 pos0 = 77 (sequence isolation)
  kv_write ((<0) , (<0) , (<(1 4 $ 42)) , (<(1 4 $ 99)) , (<0))
  kv_write ((<0) , (<0) , (<(1 4 $ 77)) , (<(1 4 $ 88)) , (<1))
  tc =. tc + 1
  result0 =. kv_read ((<0) , (<0) , (<0))
  result1 =. kv_read ((<0) , (<0) , (<1))
  k0_r =. > 0 { result0
  k1_r =. > 0 { result1
  if. (42 42 42 42 -: , k0_r) *. (77 77 77 77 -: , k1_r) do. pc =. pc + 1
    echo 'PASS: seq0 k=42, seq1 k=77 (isolated)'
  else. fc =. fc + 1
    fl =. fl , 'seq isolation', LF; echo 'FAIL: seq isolation' end.

  NB. seq1 layer1 pos0 = 55 (layer isolation across seq)
  kv_write ((<1) , (<0) , (<(1 4 $ 55)) , (<(1 4 $ 66)) , (<1))
  tc =. tc + 1
  result0b =. kv_read ((<0) , (<0) , (<0))
  result1b =. kv_read ((<1) , (<0) , (<1))
  k0b =. > 0 { result0b
  k1b =. > 0 { result1b
  if. (42 42 42 42 -: , k0b) *. (55 55 55 55 -: , k1b) do. pc =. pc + 1
    echo 'PASS: layer isolation across seq (seq0 L0=42, seq1 L1=55)'
  else. fc =. fc + 1
    fl =. fl , 'layer isolation across seq', LF; echo 'FAIL: layer isolation across seq' end.

  NB. reset to single-sequence (batch change must realloc)
  kv_batch_g =: 1
  kv_create ((<2) , (<10) , (<1) , (<4))
  tc =. tc + 1
  if. 20 4 -: $ k_cache_g do. pc =. pc + 1
    echo 'PASS: batch change reallocates (20,4)'
  else. fc =. fc + 1
    fl =. fl , 'batch change realloc', LF; echo 'FAIL: batch change realloc'; echo '  got: '; echo $ k_cache_g end.

  NB. ================================================================
  suite_time =. 6!:1 '' - suite_start
  st =. ": suite_time
  echo '========== KV CACHE TEST SUMMARY =========='
  echo 'Total: ' , ": tc
  echo 'Passed: ' , ": pc
  echo 'Failed: ' , ": fc
  echo 'Time: ' , st , 's'
  if. 0 < #fl do.
    echo ''
    echo 'Failures:'
    echo fl
  end.
  echo ''
  echo 'KV cache test suite complete!'

  ''
)
pm_start 1e8

test_kv_cache 0

pm_report ''
