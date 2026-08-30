NB. ================================================================
NB. Qwen3-0.6B-Instruct (qwen3 arch) Test Suite
NB. Verifies inference outputs against llama.cpp reference
NB. (llama-cpp-python on the same GGUF). Depends on qwen3.ijs.
NB.
NB. Arch vs qwen2: per-head RMSNorm on Q and K (attn_q_norm/attn_k_norm,
NB. shared weights size=head_dim) BEFORE RoPE; NO Q/K/V biases; qwen3.* KV
NB. prefix; chat template adds no default system message.
NB. ================================================================

coclass 'inference'
load './inference.ijs'
load './models/qwen3.ijs'
load './tests/j/pm_fixture.ijs'

NB. ---- Helpers for timing & memory ----
fmt_bytes =: 3 : 0
  b =. y
  if. b < 1024 do. (": b), ' B'
  elseif. b < 1048576 do. (": <. b % 1024), ' KB'
  elseif. b < 1073741824 do. (": <. b % 1048576), ' MB'
  else. (": <. b % 1073741824), ' GB'
  end.
)
fmt_time =: 3 : 0
  t =. y
  if. t < 0.001 do. (": <. t * 1e6), ' us'
  elseif. t < 1 do. (": <. t * 1000), ' ms'
  else. (": <. t), ' s'
  end.
)

NB. ---- Argmax of a logits vector ----
argmax =: 3 : '>./ I. y = >./ y'

test_qwen3 =: 3 : 0
  tc =. 0
  pc =. 0
  fc =. 0
  fl =. ''

  echo ''
  echo '=============================================================='
  echo '  QWEN3-0.6B (QWEN3 ARCH) TEST SUITE'
  echo '=============================================================='

  NB. ================================================================
  echo '--- Section 1: Model Loading & Structure ---'
  echo ''

  qw_path =. 'qwen3-0.6b'
  echo '  (loading model)...'
  mem_before_load =. 7!:0 ''
  load_time =. 5 (6!:2) 'llm =. load_gguf_to_llm qw_path'
  mem_after_load =. 7!:0 ''
  load_mem =. mem_after_load - mem_before_load
  echo '  (done, ' , (": <. load_time) , 's)'
  echo '  Load memory delta: ' , fmt_bytes load_mem
  echo '  Workspace after load: ' , fmt_bytes mem_after_load
  echo ''

  NB. --- llm structure: 9 core + block_data + arch = 10 ---
  tc =. tc + 1
  if. 10 = # llm do.
    pc =. pc + 1
    echo 'PASS: llm has 10 elements'
  else.
    fc =. fc + 1
    fl =. fl , 'llm has 10 elements', LF
    echo 'FAIL: llm has 10 elements'; echo '  got: '; echo # llm
  end.

  mi =. llm_mi llm

  NB. block_count = 28
  tc =. tc + 1
  if. 28 -: mi_block_count mi do. pc =. pc + 1
    echo 'PASS: block_count = 28'
  else. fc =. fc + 1
    fl =. fl , 'block_count = 28', LF; echo 'FAIL: block_count = 28'; echo '  got: '; echo mi_block_count mi end.

  NB. context_length = 40960
  tc =. tc + 1
  if. 40960 -: mi_context_len mi do. pc =. pc + 1
    echo 'PASS: context_length = 40960'
  else. fc =. fc + 1
    fl =. fl , 'context_length = 40960', LF; echo 'FAIL: context_length = 40960'; echo '  got: '; echo mi_context_len mi end.

  NB. embedding_length = 1024
  tc =. tc + 1
  if. 1024 -: mi_emb_len mi do. pc =. pc + 1
    echo 'PASS: embedding_length = 1024'
  else. fc =. fc + 1
    fl =. fl , 'embedding_length = 1024', LF; echo 'FAIL: embedding_length = 1024'; echo '  got: '; echo mi_emb_len mi end.

  NB. attention.head_count = 16
  tc =. tc + 1
  if. 16 -: mi_n_heads mi do. pc =. pc + 1
    echo 'PASS: attention.head_count = 16'
  else. fc =. fc + 1
    fl =. fl , 'attention.head_count = 16', LF; echo 'FAIL: attention.head_count = 16'; echo '  got: '; echo mi_n_heads mi end.

  NB. attention.head_count_kv = 8  (GQA 16:8 = 2:1)
  tc =. tc + 1
  if. 8 -: mi_n_heads_kv mi do. pc =. pc + 1
    echo 'PASS: attention.head_count_kv = 8'
  else. fc =. fc + 1
    fl =. fl , 'attention.head_count_kv = 8', LF; echo 'FAIL: attention.head_count_kv = 8'; echo '  got: '; echo mi_n_heads_kv mi end.

  NB. head_dim = 128 (qwen3.attention.key_length = 128)
  tc =. tc + 1
  if. 128 -: mi_head_dim mi do. pc =. pc + 1
    echo 'PASS: head_dim = 128'
  else. fc =. fc + 1
    fl =. fl , 'head_dim = 128', LF; echo 'FAIL: head_dim = 128'; echo '  got: '; echo mi_head_dim mi end.

  NB. rope.freq_base = 1000000
  tc =. tc + 1
  if. 1000000 -: mi_rope_freq mi do. pc =. pc + 1
    echo 'PASS: rope.freq_base = 1000000'
  else. fc =. fc + 1
    fl =. fl , 'rope.freq_base = 1000000', LF; echo 'FAIL: rope.freq_base = 1000000'; echo '  got: '; echo mi_rope_freq mi end.

  NB. feed_forward_length = 3072
  tc =. tc + 1
  if. 3072 -: mi_n_ff mi do. pc =. pc + 1
    echo 'PASS: feed_forward_length = 3072'
  else. fc =. fc + 1
    fl =. fl , 'feed_forward_length = 3072', LF; echo 'FAIL: feed_forward_length = 3072'; echo '  got: '; echo mi_n_ff mi end.

  NB. ================================================================
  echo '--- Section 2: Block Data (Q/K norms, no QKV biases) ---'
  echo ''

  bd_list =. llm_block_data llm
  tc =. tc + 1
  if. 28 = # bd_list do. pc =. pc + 1
    echo 'PASS: 28 block_data entries'
  else. fc =. fc + 1
    fl =. fl , '28 block_data entries', LF; echo 'FAIL: 28 block_data entries'; echo '  got: '; echo # bd_list end.

  bd0 =. > 0 { bd_list
  tc =. tc + 1
  if. 14 = # bd0 do. pc =. pc + 1
    echo 'PASS: block_data[0] has 14 elements (no bias slots, q/k norms)'
  else. fc =. fc + 1
    fl =. fl , 'block_data[0] has 14 elements', LF; echo 'FAIL: block_data[0] has 14 elements'; echo '  got: '; echo # bd0 end.

  NB. Q/K norms: per-head RMSNorm weights, size = head_dim = 128
  tc =. tc + 1
  if. 128 = # qw3_bd_q_norm bd0 do. pc =. pc + 1
    echo 'PASS: attn_q_norm length = 128'
  else. fc =. fc + 1
    fl =. fl , 'attn_q_norm length = 128', LF; echo 'FAIL: attn_q_norm length = 128'; echo '  got: '; echo # qw3_bd_q_norm bd0 end.

  tc =. tc + 1
  if. 128 = # qw3_bd_k_norm bd0 do. pc =. pc + 1
    echo 'PASS: attn_k_norm length = 128'
  else. fc =. fc + 1
    fl =. fl , 'attn_k_norm length = 128', LF; echo 'FAIL: attn_k_norm length = 128'; echo '  got: '; echo # qw3_bd_k_norm bd0 end.

  NB. No Q/K/V biases in qwen3: the bias tensors must be absent
  tc =. tc + 1
  if. 0 = # 'blk.0.attn_q.bias' get_tensor_cached_d llm do. pc =. pc + 1
    echo 'PASS: no attn_q.bias tensor (qwen3 has no QKV biases)'
  else. fc =. fc + 1
    fl =. fl , 'no attn_q.bias tensor', LF; echo 'FAIL: no attn_q.bias tensor'; echo '  got len: '; echo # 'blk.0.attn_q.bias' get_tensor_cached_d llm end.

  NB. attn_q weight: (2048, 1024) = [out=16*128, in=emb]
  attn_q =. qw3_bd_attn_q bd0
  tq =. $ attn_q
  tc =. tc + 1
  if. 2048 = {. tq do. pc =. pc + 1
    echo 'PASS: attn_q shape[0] = 2048 (out dim = 16*128)'
  else. fc =. fc + 1
    fl =. fl , 'attn_q shape[0] = 2048', LF; echo 'FAIL: attn_q shape[0] = 2048'; echo '  got: '; echo {. tq end.
  tc =. tc + 1
  if. 1024 = {: tq do. pc =. pc + 1
    echo 'PASS: attn_q shape[1] = 1024 (emb_len)'
  else. fc =. fc + 1
    fl =. fl , 'attn_q shape[1] = 1024', LF; echo 'FAIL: attn_q shape[1] = 1024'; echo '  got: '; echo {: tq end.

  NB. attn_k weight: (1024, 1024) = [out=8*128, in=emb]
  attn_k =. qw3_bd_attn_k bd0
  tk =. $ attn_k
  tc =. tc + 1
  if. 1024 = {. tk do. pc =. pc + 1
    echo 'PASS: attn_k shape[0] = 1024 (out dim = 8*128)'
  else. fc =. fc + 1
    fl =. fl , 'attn_k shape[0] = 1024', LF; echo 'FAIL: attn_k shape[0] = 1024'; echo '  got: '; echo {. tk end.

  NB. attn_output weight: (1024, 1024)
  ao =. qw3_bd_attn_o bd0
  ao_sh =. $ ao
  tc =. tc + 1
  if. 1024 = {. ao_sh do. pc =. pc + 1
    echo 'PASS: attn_output shape[0] = 1024 (out dim)'
  else. fc =. fc + 1
    fl =. fl , 'attn_output shape[0] = 1024', LF; echo 'FAIL: attn_output shape[0] = 1024'; echo '  got: '; echo {. ao_sh end.

  NB. ffn_gate: (3072, 1024)
  tc =. tc + 1
  if. 3072 = {. $ qw3_bd_ff_gate bd0 do. pc =. pc + 1
    echo 'PASS: ffn_gate shape[0] = 3072'
  else. fc =. fc + 1
    fl =. fl , 'ffn_gate shape[0] = 3072', LF; echo 'FAIL: ffn_gate shape[0] = 3072'; echo '  got: '; echo {. $ qw3_bd_ff_gate bd0 end.

  NB. ================================================================
  echo '--- Section 3: Single-Token Inference (vs llama.cpp) ---'
  echo ''

  NB. oracle from llama-cpp-python on same GGUF (BF16):
  NB. hello->14582, The->15846, Paris->38297. ('a' is a 0.004 near-tie
  NB. between 61832/21806 — F32-vs-double rounding flips it, so not pinned.)
  c_hello =. <((<'hello') , <14582)
  c_The   =. <((<'The')   , <15846)
  c_Paris =. <((<'Paris') , <38297)
  cases1 =. c_hello , c_The , c_Paris

  ci =. 0
  while. ci < # cases1 do.
    case =. > ci { cases1
    text =. > 0 { case
    expect =. > 1 { case
    tc =. tc + 1
    r =. llm qw3_infer_simple text
    lg =. > 3 { r
    got =. argmax lg
    if. expect -: got do.
      pc =. pc + 1
      echo 'PASS: ' , text , ' -> argmax ' , ": got
    else.
      fc =. fc + 1
      fl =. fl , ('single-token: ', text) , LF
      gs =. ": got
      es =. ": expect
      echo 'FAIL: ' , text , ' -> argmax ' , gs , ' (want ' , es , ')'
    end.
    ci =. ci + 1
  end.

  NB. ================================================================
  echo '--- Section 4: Multi-Token Inference (vs llama.cpp) ---'
  echo ''

  NB. hello world->198, The capital of france is->220, don't stop->279
  c_hw       =. <((<'hello world') , <198)
  c_france   =. <((<'The capital of france is') , <220)
  c_dont     =. <((<'don''t stop') , <279)
  cases2 =. c_hw , c_france , c_dont

  ci =. 0
  while. ci < # cases2 do.
    case =. > ci { cases2
    text =. > 0 { case
    expect =. > 1 { case
    tc =. tc + 1
    r =. llm qw3_infer_simple text
    lg =. > 3 { r
    got =. argmax lg
    if. expect -: got do.
      pc =. pc + 1
      echo 'PASS: ' , text , ' -> argmax ' , ": got
    else.
      fc =. fc + 1
      fl =. fl , ('multi-token: ', text) , LF
      gs =. ": got
      es =. ": expect
      echo 'FAIL: ' , text , ' -> argmax ' , gs , ' (want ' , es , ')'
    end.
    ci =. ci + 1
  end.

  NB. ================================================================
  echo '--- Section 5: Generation (greedy chat) ---'
  echo ''

  NB. Qwen3 is a thinking model: greedy chat starts with the reasoning
  NB. block ("thinking\nOkay, ..."). Verify generate returns non-empty
  NB. answer-only text and does not leak the <|im_end|> stop token.
  tc =. tc + 1
  g =. llm qw3_generate ('The capital of France is' ; 8 ; <<0 ; 0 ; 0.95 ; 0.0)
  if. (0 < # g) *. -. +./ 'im_end' E. g do.
    pc =. pc + 1
    echo 'PASS: greedy generate non-empty, no stop-token leak'
  else. fc =. fc + 1
    fl =. fl , 'greedy generate non-empty', LF
    echo 'FAIL: greedy generate non-empty, no stop-token leak'; echo '  got: [' , g , ']' end.

  echo ''
  echo '=============================================================='
  echo '  TEST SUMMARY'
  echo '=============================================================='
  echo ''
  echo '  Total:          ' , ": tc
  echo '  Passed:         ' , ": pc
  echo '  Failed:         ' , ": fc
  if. tc > 0 do.
    rate =. <. (pc % tc) * 100
    echo '  Pass rate:      ' , (": rate) , '%'
  else.
    echo '  Pass rate:      N/A'
  end.
  if. 0 < # fl do. echo ''; echo 'Failed:'; echo fl end.
  echo ''
  if. 0 = fc do. echo 'All tests passed!' else. echo 'Some tests failed.' end.
  echo ''
)

pm_start 1e8

test_qwen3 0

pm_report ''
