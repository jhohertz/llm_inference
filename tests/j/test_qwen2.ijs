NB. ================================================================
NB. Qwen2.5-Coder-0.5B-Instruct (qwen2 arch) Test Suite
NB. Verifies inference outputs against llama.cpp reference
NB. (llama-cpp-python on the same GGUF). Depends on qwen2.ijs.
NB. ================================================================

coclass 'inference'
load './inference.ijs'
load './models/qwen2.ijs'
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

test_qwen2 =: 3 : 0
  tc =. 0
  pc =. 0
  fc =. 0
  fl =. ''

  echo ''
  echo '=============================================================='
  echo '  QWEN2.5-CODER-0.5B (QWEN2 ARCH) TEST SUITE'
  echo '=============================================================='

  NB. ================================================================
  echo '--- Section 1: Model Loading & Structure ---'
  echo ''

  qw_path =. 'qwen2.5-coder-0.5b'
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

  NB. block_count = 24
  tc =. tc + 1
  if. 24 -: mi_block_count mi do. pc =. pc + 1
    echo 'PASS: block_count = 24'
  else. fc =. fc + 1
    fl =. fl , 'block_count = 24', LF; echo 'FAIL: block_count = 24'; echo '  got: '; echo mi_block_count mi end.

  NB. context_length = 32768
  tc =. tc + 1
  if. 32768 -: mi_context_len mi do. pc =. pc + 1
    echo 'PASS: context_length = 32768'
  else. fc =. fc + 1
    fl =. fl , 'context_length = 32768', LF; echo 'FAIL: context_length = 32768'; echo '  got: '; echo mi_context_len mi end.

  NB. embedding_length = 896
  tc =. tc + 1
  if. 896 -: mi_emb_len mi do. pc =. pc + 1
    echo 'PASS: embedding_length = 896'
  else. fc =. fc + 1
    fl =. fl , 'embedding_length = 896', LF; echo 'FAIL: embedding_length = 896'; echo '  got: '; echo mi_emb_len mi end.

  NB. attention.head_count = 14
  tc =. tc + 1
  if. 14 -: mi_n_heads mi do. pc =. pc + 1
    echo 'PASS: attention.head_count = 14'
  else. fc =. fc + 1
    fl =. fl , 'attention.head_count = 14', LF; echo 'FAIL: attention.head_count = 14'; echo '  got: '; echo mi_n_heads mi end.

  NB. attention.head_count_kv = 2  (GQA 14:2 = 7:1)
  tc =. tc + 1
  if. 2 -: mi_n_heads_kv mi do. pc =. pc + 1
    echo 'PASS: attention.head_count_kv = 2'
  else. fc =. fc + 1
    fl =. fl , 'attention.head_count_kv = 2', LF; echo 'FAIL: attention.head_count_kv = 2'; echo '  got: '; echo mi_n_heads_kv mi end.

  NB. head_dim = 64 (emb_len / n_heads fallback; no key_length KV)
  tc =. tc + 1
  if. 64 -: mi_head_dim mi do. pc =. pc + 1
    echo 'PASS: head_dim = 64'
  else. fc =. fc + 1
    fl =. fl , 'head_dim = 64', LF; echo 'FAIL: head_dim = 64'; echo '  got: '; echo mi_head_dim mi end.

  NB. rope.freq_base = 1000000
  tc =. tc + 1
  if. 1000000 -: mi_rope_freq mi do. pc =. pc + 1
    echo 'PASS: rope.freq_base = 1000000'
  else. fc =. fc + 1
    fl =. fl , 'rope.freq_base = 1000000', LF; echo 'FAIL: rope.freq_base = 1000000'; echo '  got: '; echo mi_rope_freq mi end.

  NB. feed_forward_length = 4864
  tc =. tc + 1
  if. 4864 -: mi_n_ff mi do. pc =. pc + 1
    echo 'PASS: feed_forward_length = 4864'
  else. fc =. fc + 1
    fl =. fl , 'feed_forward_length = 4864', LF; echo 'FAIL: feed_forward_length = 4864'; echo '  got: '; echo mi_n_ff mi end.

  NB. ================================================================
  echo '--- Section 2: Block Data (with Q/K/V biases) ---'
  echo ''

  bd_list =. llm_block_data llm
  tc =. tc + 1
  if. 24 = # bd_list do. pc =. pc + 1
    echo 'PASS: 24 block_data entries'
  else. fc =. fc + 1
    fl =. fl , '24 block_data entries', LF; echo 'FAIL: 24 block_data entries'; echo '  got: '; echo # bd_list end.

  bd0 =. > 0 { bd_list
  tc =. tc + 1
  if. 15 = # bd0 do. pc =. pc + 1
    echo 'PASS: block_data[0] has 15 elements'
  else. fc =. fc + 1
    fl =. fl , 'block_data[0] has 15 elements', LF; echo 'FAIL: block_data[0] has 15 elements'; echo '  got: '; echo # bd0 end.

  NB. Q bias: (n_heads * head_dim) = 896 flat vector
  qb =. qw2_bd_q_bias bd0
  tc =. tc + 1
  if. 896 = # qb do. pc =. pc + 1
    echo 'PASS: attn_q.bias length = 896 (14*64)'
  else. fc =. fc + 1
    fl =. fl , 'attn_q.bias length = 896', LF; echo 'FAIL: attn_q.bias length = 896'; echo '  got: '; echo # qb end.

  NB. K bias: (n_kv_heads * head_dim) = 128 flat vector
  kb =. qw2_bd_k_bias bd0
  tc =. tc + 1
  if. 128 = # kb do. pc =. pc + 1
    echo 'PASS: attn_k.bias length = 128 (2*64)'
  else. fc =. fc + 1
    fl =. fl , 'attn_k.bias length = 128', LF; echo 'FAIL: attn_k.bias length = 128'; echo '  got: '; echo # kb end.

  NB. V bias: (n_kv_heads * head_dim) = 128
  vb =. qw2_bd_v_bias bd0
  tc =. tc + 1
  if. 128 = # vb do. pc =. pc + 1
    echo 'PASS: attn_v.bias length = 128'
  else. fc =. fc + 1
    fl =. fl , 'attn_v.bias length = 128', LF; echo 'FAIL: attn_v.bias length = 128'; echo '  got: '; echo # vb end.

  NB. attn_q weight: (896, 896) = [out=14*64, in=emb]
  attn_q =. qw2_bd_attn_q bd0
  tq =. $ attn_q
  tc =. tc + 1
  if. 896 = {. tq do. pc =. pc + 1
    echo 'PASS: attn_q shape[0] = 896 (out dim = 14*64)'
  else. fc =. fc + 1
    fl =. fl , 'attn_q shape[0] = 896', LF; echo 'FAIL: attn_q shape[0] = 896'; echo '  got: '; echo {. tq end.
  tc =. tc + 1
  if. 896 = {: tq do. pc =. pc + 1
    echo 'PASS: attn_q shape[1] = 896 (emb_len)'
  else. fc =. fc + 1
    fl =. fl , 'attn_q shape[1] = 896', LF; echo 'FAIL: attn_q shape[1] = 896'; echo '  got: '; echo {: tq end.

  NB. attn_k weight: (128, 896) = [out=2*64, in=emb]
  attn_k =. qw2_bd_attn_k bd0
  tk =. $ attn_k
  tc =. tc + 1
  if. 128 = {. tk do. pc =. pc + 1
    echo 'PASS: attn_k shape[0] = 128 (out dim = 2*64)'
  else. fc =. fc + 1
    fl =. fl , 'attn_k shape[0] = 128', LF; echo 'FAIL: attn_k shape[0] = 128'; echo '  got: '; echo {. tk end.

  NB. attn_output weight: (896, 896)
  ao =. qw2_bd_attn_o bd0
  ao_sh =. $ ao
  tc =. tc + 1
  if. 896 = {. ao_sh do. pc =. pc + 1
    echo 'PASS: attn_output shape[0] = 896 (out dim)'
  else. fc =. fc + 1
    fl =. fl , 'attn_output shape[0] = 896', LF; echo 'FAIL: attn_output shape[0] = 896'; echo '  got: '; echo {. ao_sh end.

  NB. ffn_gate: (4864, 896)
  tc =. tc + 1
  if. 4864 = {. $ qw2_bd_ff_gate bd0 do. pc =. pc + 1
    echo 'PASS: ffn_gate shape[0] = 4864'
  else. fc =. fc + 1
    fl =. fl , 'ffn_gate shape[0] = 4864', LF; echo 'FAIL: ffn_gate shape[0] = 4864'; echo '  got: '; echo {. $ qw2_bd_ff_gate bd0 end.

  NB. ================================================================
  echo '--- Section 3: Single-Token Inference (vs llama.cpp) ---'
  echo ''

  NB. oracle from llama-cpp-python on same GGUF:
  NB. hello->82334, The->12433, a->495, Paris->144328
  c_hello =. <((<'hello') , <82334)
  c_The   =. <((<'The')   , <12433)
  c_a     =. <((<'a')     , <495)
  c_Paris =. <((<'Paris') , <144328)
  cases1 =. c_hello , c_The , c_a , c_Paris

  ci =. 0
  while. ci < # cases1 do.
    case =. > ci { cases1
    text =. > 0 { case
    expect =. > 1 { case
    tc =. tc + 1
    r =. llm qw2_infer_simple text
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

  NB. hello world->198, The capital of france is->12095, don't stop->752
  c_hw       =. <((<'hello world') , <198)
  c_france   =. <((<'The capital of france is') , <12095)
  c_dont     =. <((<'don''t stop') , <752)
  cases2 =. c_hw , c_france , c_dont

  ci =. 0
  while. ci < # cases2 do.
    case =. > ci { cases2
    text =. > 0 { case
    expect =. > 1 { case
    tc =. tc + 1
    r =. llm qw2_infer_simple text
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
  echo '--- Section 5: Generation ---'
  echo ''

  tc =. tc + 1
  g =. llm qw2_generate ('The capital of France is' ; 8 ; <<0 ; 0 ; 0.95 ; 0.0)
  NB. generate now frames the prompt with the chat template and returns answer-only.
  NB. Greedy chat answer matches llama-cli -st ("The capital of France is Paris.").
  NB. Regression pin: the byte-token clamp (tok - 65536) corrupted real tokens >= 65536
  NB. (incl. <|im_start|>/<|im_end|> 151644/151645) -> garbage ("heim heim"); removed.
  got_gen =. > g
  expect_gen =. 'The capital of France is Paris.'
  if. expect_gen -: got_gen do.
    pc =. pc + 1
    echo 'PASS: generation "The capital of France is Paris."'
  else.
    fc =. fc + 1
    fl =. fl , 'generation chat answer', LF
    echo 'FAIL: generation chat answer'
    echo '  got: ' ; echo got_gen
  end.

  echo ''
  echo '=============================================================='
  echo '  Total:          ' , ": tc
  echo '  Passed:         ' , ": pc
  echo '  Failed:         ' , ": fc
  if. 0 = fc do. echo '  All tests passed!'
  else.
    echo '  Failed checks:'
    echo fl
  end.
  echo '=============================================================='
)

pm_start 1e8

test_qwen2 ''

pm_report ''
