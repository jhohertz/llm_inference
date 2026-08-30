NB. ================================================================
NB. SmolLM2-360M-Instruct (llama arch) Test Suite
NB. Verifies inference outputs against llama.cpp reference
NB. (llama-cpp-python on the same GGUF). Depends on llama.ijs.
NB. ================================================================

coclass 'inference'
load './inference.ijs'
load './models/llama.ijs'
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

test_smollm2 =: 3 : 0
  tc =. 0
  pc =. 0
  fc =. 0
  fl =. ''

  echo ''
  echo '=============================================================='
  echo '  SMOLLM2-360M (LLAMA ARCH) TEST SUITE'
  echo '=============================================================='

  NB. ================================================================
  echo '--- Section 1: Model Loading & Structure ---'
  echo ''

  sm_path =. 'smollm2-360m'
  echo '  (loading model)...'
  mem_before_load =. 7!:0 ''
  load_time =. 5 (6!:2) 'llm =. load_gguf_to_llm sm_path'
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

  NB. block_count = 32
  tc =. tc + 1
  if. 32 -: mi_block_count mi do. pc =. pc + 1
    echo 'PASS: block_count = 32'
  else. fc =. fc + 1
    fl =. fl , 'block_count = 32', LF; echo 'FAIL: block_count = 32'; echo '  got: '; echo mi_block_count mi end.

  NB. context_length = 8192
  tc =. tc + 1
  if. 8192 -: mi_context_len mi do. pc =. pc + 1
    echo 'PASS: context_length = 8192'
  else. fc =. fc + 1
    fl =. fl , 'context_length = 8192', LF; echo 'FAIL: context_length = 8192'; echo '  got: '; echo mi_context_len mi end.

  NB. embedding_length = 960
  tc =. tc + 1
  if. 960 -: mi_emb_len mi do. pc =. pc + 1
    echo 'PASS: embedding_length = 960'
  else. fc =. fc + 1
    fl =. fl , 'embedding_length = 960', LF; echo 'FAIL: embedding_length = 960'; echo '  got: '; echo mi_emb_len mi end.

  NB. attention.head_count = 15
  tc =. tc + 1
  if. 15 -: mi_n_heads mi do. pc =. pc + 1
    echo 'PASS: attention.head_count = 15'
  else. fc =. fc + 1
    fl =. fl , 'attention.head_count = 15', LF; echo 'FAIL: attention.head_count = 15'; echo '  got: '; echo mi_n_heads mi end.

  NB. attention.head_count_kv = 5  (GQA 15:5)
  tc =. tc + 1
  if. 5 -: mi_n_heads_kv mi do. pc =. pc + 1
    echo 'PASS: attention.head_count_kv = 5'
  else. fc =. fc + 1
    fl =. fl , 'attention.head_count_kv = 5', LF; echo 'FAIL: attention.head_count_kv = 5'; echo '  got: '; echo mi_n_heads_kv mi end.

  NB. head_dim = 64 (key_length)
  tc =. tc + 1
  if. 64 -: mi_head_dim mi do. pc =. pc + 1
    echo 'PASS: head_dim = 64'
  else. fc =. fc + 1
    fl =. fl , 'head_dim = 64', LF; echo 'FAIL: head_dim = 64'; echo '  got: '; echo mi_head_dim mi end.

  NB. rope.freq_base = 100000
  tc =. tc + 1
  if. 100000 -: mi_rope_freq mi do. pc =. pc + 1
    echo 'PASS: rope.freq_base = 100000'
  else. fc =. fc + 1
    fl =. fl , 'rope.freq_base = 100000', LF; echo 'FAIL: rope.freq_base = 100000'; echo '  got: '; echo mi_rope_freq mi end.

  NB. feed_forward_length = 2560
  tc =. tc + 1
  if. 2560 -: mi_n_ff mi do. pc =. pc + 1
    echo 'PASS: feed_forward_length = 2560'
  else. fc =. fc + 1
    fl =. fl , 'feed_forward_length = 2560', LF; echo 'FAIL: feed_forward_length = 2560'; echo '  got: '; echo mi_n_ff mi end.

  NB. ================================================================
  echo '--- Section 2: Block Data ---'
  echo ''

  bd_list =. llm_block_data llm
  tc =. tc + 1
  if. 32 = # bd_list do. pc =. pc + 1
    echo 'PASS: 32 block_data entries'
  else. fc =. fc + 1
    fl =. fl , '32 block_data entries', LF; echo 'FAIL: 32 block_data entries'; echo '  got: '; echo # bd_list end.

  bd0 =. > 0 { bd_list
  tc =. tc + 1
  if. 12 = # bd0 do. pc =. pc + 1
    echo 'PASS: block_data[0] has 12 elements'
  else. fc =. fc + 1
    fl =. fl , 'block_data[0] has 12 elements', LF; echo 'FAIL: block_data[0] has 12 elements'; echo '  got: '; echo # bd0 end.

  attn_q =. llama_bd_attn_q bd0
  tq =. $ attn_q
  tc =. tc + 1
  if. 960 = {. tq do. pc =. pc + 1
    echo 'PASS: attn_q shape[0] = 960 (out dim = 15*64)'
  else. fc =. fc + 1
    fl =. fl , 'attn_q shape[0] = 960', LF; echo 'FAIL: attn_q shape[0] = 960'; echo '  got: '; echo {. tq end.
  tc =. tc + 1
  if. 960 = {: tq do. pc =. pc + 1
    echo 'PASS: attn_q shape[1] = 960 (emb_len)'
  else. fc =. fc + 1
    fl =. fl , 'attn_q shape[1] = 960', LF; echo 'FAIL: attn_q shape[1] = 960'; echo '  got: '; echo {: tq end.

  attn_k =. llama_bd_attn_k bd0
  tk =. $ attn_k
  tc =. tc + 1
  if. 320 = {. tk do. pc =. pc + 1
    echo 'PASS: attn_k shape[0] = 320 (out dim = 5*64)'
  else. fc =. fc + 1
    fl =. fl , 'attn_k shape[0] = 320', LF; echo 'FAIL: attn_k shape[0] = 320'; echo '  got: '; echo {. tk end.

  ao =. llama_bd_attn_o bd0
  ao_sh =. $ ao
  tc =. tc + 1
  if. 960 = {. ao_sh do. pc =. pc + 1
    echo 'PASS: attn_output shape[0] = 960 (out dim)'
  else. fc =. fc + 1
    fl =. fl , 'attn_output shape[0] = 960', LF; echo 'FAIL: attn_output shape[0] = 960'; echo '  got: '; echo {. ao_sh end.

  tc =. tc + 1
  if. 2560 = {. $ llama_bd_ff_gate bd0 do. pc =. pc + 1
    echo 'PASS: ffn_gate shape[0] = 2560'
  else. fc =. fc + 1
    fl =. fl , 'ffn_gate shape[0] = 2560', LF; echo 'FAIL: ffn_gate shape[0] = 2560'; echo '  got: '; echo {. $ llama_bd_ff_gate bd0 end.

  NB. ================================================================
  echo '--- Section 3: Single-Token Inference (vs llama.cpp) ---'
  echo ''

  NB. build <text; expected> pairs, each boxed as one case
  c_hello =. <((<'hello') , <28)
  c_The   =. <((<'The')   , <2)
  c_a     =. <((<'a')     , <30)
  c_Paris =. <((<'Paris') , <198)
  cases1 =. c_hello , c_The , c_a , c_Paris

  ci =. 0
  while. ci < # cases1 do.
    case =. > ci { cases1
    text =. > 0 { case
    expect =. > 1 { case
    tc =. tc + 1
    r =. llm llama_infer_simple text
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

  c_hw       =. <((<'hello world') , <18)
  c_france   =. <((<'The capital of france is') , <1280)
  c_dont     =. <((<'don''t stop') , <549)
  cases2 =. c_hw , c_france , c_dont

  ci =. 0
  while. ci < # cases2 do.
    case =. > ci { cases2
    text =. > 0 { case
    expect =. > 1 { case
    tc =. tc + 1
    r =. llm llama_infer_simple text
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
  echo '--- Section 5: Generation (vs llama.cpp) ---'
  echo ''

  tc =. tc + 1
  g =. llm llama_generate ('The capital of France is' ; 8 ; <<0 ; 0 ; 0.95 ; 0.0)
  NB. generate now frames the prompt with the chat template and returns answer-only.
  NB. SmolLM2's greedy chat answer is its inherent odd style (matches llama.cpp).
  got_gen =. > g
  expect_gen =. 'I''m sorry for any confusion, but'
  if. expect_gen -: got_gen do.
    pc =. pc + 1
    echo 'PASS: generation "I''m sorry for any confusion, but"'
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

test_smollm2 ''

pm_report ''
