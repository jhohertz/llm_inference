NB. ================================================================
NB. Qwen3.5-0.8B (qwen35 arch) Test Suite
NB. Verifies inference outputs against llama.cpp reference
NB. (llama-cpp-python on the same GGUF). Depends on qwen35.ijs.
NB.
NB. Arch: hybrid — 6 full-attention layers (il+1 % 4 == 0) + 18
NB. gated-delta-net (SSM) layers. Fused Q+GATE projection, per-head
NB. Q/K RMSNorm, sigmoid gate, conv1d, L2-norm q/k, sequential delta-net
NB. recurrence, norm-gated output, NEOX RoPE over first n_rot dims.
NB. MTP blk.24 is NOT in scope (block_count = 24 trunk layers).
NB. ================================================================

coclass 'inference'
load './inference.ijs'
load './models/qwen35.ijs'
load './tests/j/pm_fixture.ijs'

NB. Bound the KV cache: qwen3.5-0.8b has ctx=262144 (~51.5 GB of cache at
NB. full ctx — kv_create zero-allocates it every fresh infer/generate, ~8.5s
NB. each). The low-memory override keeps eff_seq small; prompts here are short.
kv_max_seq_g =: 2048

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

test_qwen35 =: 3 : 0
  tc =. 0
  pc =. 0
  fc =. 0
  fl =. ''

  echo ''
  echo '=============================================================='
  echo '  QWEN3.5-0.8B (QWEN35 ARCH) TEST SUITE'
  echo '=============================================================='

  NB. ================================================================
  echo '--- Section 1: Model Loading & Structure ---'
  echo ''

  qw_path =. 'qwen3.5-0.8b'
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

  NB. block_count = 24 (trunk; MTP blk.24 NOT in scope)
  tc =. tc + 1
  if. 24 -: mi_block_count mi do. pc =. pc + 1
    echo 'PASS: block_count = 24'
  else. fc =. fc + 1
    fl =. fl , 'block_count = 24', LF; echo 'FAIL: block_count = 24'; echo '  got: '; echo mi_block_count mi end.

  NB. context_length = 262144
  tc =. tc + 1
  if. 262144 -: mi_context_len mi do. pc =. pc + 1
    echo 'PASS: context_length = 262144'
  else. fc =. fc + 1
    fl =. fl , 'context_length = 262144', LF; echo 'FAIL: context_length = 262144'; echo '  got: '; echo mi_context_len mi end.

  NB. embedding_length = 1024
  tc =. tc + 1
  if. 1024 -: mi_emb_len mi do. pc =. pc + 1
    echo 'PASS: embedding_length = 1024'
  else. fc =. fc + 1
    fl =. fl , 'embedding_length = 1024', LF; echo 'FAIL: embedding_length = 1024'; echo '  got: '; echo mi_emb_len mi end.

  NB. attention.head_count = 8
  tc =. tc + 1
  if. 8 -: mi_n_heads mi do. pc =. pc + 1
    echo 'PASS: attention.head_count = 8'
  else. fc =. fc + 1
    fl =. fl , 'attention.head_count = 8', LF; echo 'FAIL: attention.head_count = 8'; echo '  got: '; echo mi_n_heads mi end.

  NB. attention.head_count_kv = 2  (GQA 8:2 = 4:1)
  tc =. tc + 1
  if. 2 -: mi_n_heads_kv mi do. pc =. pc + 1
    echo 'PASS: attention.head_count_kv = 2'
  else. fc =. fc + 1
    fl =. fl , 'attention.head_count_kv = 2', LF; echo 'FAIL: attention.head_count_kv = 2'; echo '  got: '; echo mi_n_heads_kv mi end.

  NB. head_dim = 256 (qwen35.attention.key_length = 256)
  tc =. tc + 1
  if. 256 -: mi_head_dim mi do. pc =. pc + 1
    echo 'PASS: head_dim = 256'
  else. fc =. fc + 1
    fl =. fl , 'head_dim = 256', LF; echo 'FAIL: head_dim = 256'; echo '  got: '; echo mi_head_dim mi end.

  NB. rope.freq_base = 10000000 (1e7)
  tc =. tc + 1
  if. 10000000 -: mi_rope_freq mi do. pc =. pc + 1
    echo 'PASS: rope.freq_base = 10000000'
  else. fc =. fc + 1
    fl =. fl , 'rope.freq_base = 10000000', LF; echo 'FAIL: rope.freq_base = 10000000'; echo '  got: '; echo mi_rope_freq mi end.

  NB. feed_forward_length = 3584
  tc =. tc + 1
  if. 3584 -: mi_n_ff mi do. pc =. pc + 1
    echo 'PASS: feed_forward_length = 3584'
  else. fc =. fc + 1
    fl =. fl , 'feed_forward_length = 3584', LF; echo 'FAIL: feed_forward_length = 3584'; echo '  got: '; echo mi_n_ff mi end.

  NB. vocab = 248320 (derived from token_embd dims; no qwen35.vocab_size KV)
  tc =. tc + 1
  if. 248320 -: mi_vocab_size mi do. pc =. pc + 1
    echo 'PASS: vocab_size = 248320'
  else. fc =. fc + 1
    fl =. fl , 'vocab_size = 248320', LF; echo 'FAIL: vocab_size = 248320'; echo '  got: '; echo mi_vocab_size mi end.

  NB. SSM hyperparams
  tc =. tc + 1
  if. 2048 -: qw35_mi_ssm_d_inner mi do. pc =. pc + 1
    echo 'PASS: ssm_d_inner = 2048'
  else. fc =. fc + 1
    fl =. fl , 'ssm_d_inner = 2048', LF; echo 'FAIL: ssm_d_inner = 2048'; echo '  got: '; echo qw35_mi_ssm_d_inner mi end.
  tc =. tc + 1
  if. 128 -: qw35_mi_ssm_d_state mi do. pc =. pc + 1
    echo 'PASS: ssm_d_state = 128'
  else. fc =. fc + 1
    fl =. fl , 'ssm_d_state = 128', LF; echo 'FAIL: ssm_d_state = 128'; echo '  got: '; echo qw35_mi_ssm_d_state mi end.

  NB. ================================================================
  echo '--- Section 1b: Non-MTP flavour (Qwen3.5-0.8B-GGUF) ---'
  echo ''

  NB. The non-MTP repo has no qwen35.nextn_predict_layers KV: block_count
  NB. is already 24 (trunk). The loader must NOT subtract the missing _1
  NB. (which would yield 25). Spec is non-catalog (HF-style) — resolves via
  NB. model_path to ~user/models and downloads if not cached.
  nm_path =. 'unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-BF16.gguf'
  echo '  (loading non-MTP flavour)...'
  llm_nm =. load_gguf_to_llm nm_path
  mi_nm =. llm_mi llm_nm
  tc =. tc + 1
  if. 24 -: mi_block_count mi_nm do. pc =. pc + 1
    echo 'PASS: non-MTP block_count = 24'
  else. fc =. fc + 1
    fl =. fl , 'non-MTP block_count = 24', LF; echo 'FAIL: non-MTP block_count = 24'; echo '  got: '; echo mi_block_count mi_nm end.
  tc =. tc + 1
  if. 'qwen35' -: llm_arch llm_nm do. pc =. pc + 1
    echo 'PASS: non-MTP arch = qwen35'
  else. fc =. fc + 1
    fl =. fl , 'non-MTP arch = qwen35', LF; echo 'FAIL: non-MTP arch = qwen35'; echo '  got: '; echo llm_arch llm_nm end.
  tc =. tc + 1
  r_nm =. llm_nm qw35_infer_simple 'hello'
  lg_nm =. > 3 { r_nm
  got_nm =. argmax lg_nm
  if. 11 -: got_nm do. pc =. pc + 1
    echo 'PASS: non-MTP hello -> argmax 11'
  else. fc =. fc + 1
    fl =. fl , 'non-MTP hello argmax', LF; echo 'FAIL: non-MTP hello -> argmax ', (": got_nm), ' (want 11)' end.

  NB. ================================================================
  echo '--- Section 2: Block Data (SSM + attention layers) ---'
  echo ''

  bd_list =. llm_block_data llm
  tc =. tc + 1
  if. 24 = # bd_list do. pc =. pc + 1
    echo 'PASS: 24 block_data entries'
  else. fc =. fc + 1
    fl =. fl , '24 block_data entries', LF; echo 'FAIL: 24 block_data entries'; echo '  got: '; echo # bd_list end.

  bd0 =. > 0 { bd_list
  tc =. tc + 1
  if. 20 = # bd0 do. pc =. pc + 1
    echo 'PASS: block_data[0] (SSM) has 20 elements'
  else. fc =. fc + 1
    fl =. fl , 'block_data[0] (SSM) has 20 elements', LF; echo 'FAIL: block_data[0] (SSM) has 20 elements'; echo '  got: '; echo # bd0 end.

  bd3 =. > 3 { bd_list
  tc =. tc + 1
  if. 15 = # bd3 do. pc =. pc + 1
    echo 'PASS: block_data[3] (attention) has 15 elements'
  else. fc =. fc + 1
    fl =. fl , 'block_data[3] (attention) has 15 elements', LF; echo 'FAIL: block_data[3] (attention) has 15 elements'; echo '  got: '; echo # bd3 end.

  NB. SSM layer weights (block 0)
  tc =. tc + 1
  if. 6144 = {. $ qw35_bd_s_wqkv bd0 do. pc =. pc + 1
    echo 'PASS: ssm wqkv shape[0] = 6144 (conv_dim = key*2+value)'
  else. fc =. fc + 1
    fl =. fl , 'ssm wqkv shape[0] = 6144', LF; echo 'FAIL: ssm wqkv shape[0] = 6144'; echo '  got: '; echo {. $ qw35_bd_s_wqkv bd0 end.
  tc =. tc + 1
  if. 2048 = {. $ qw35_bd_s_gate bd0 do. pc =. pc + 1
    echo 'PASS: ssm gate shape[0] = 2048 (value_dim)'
  else. fc =. fc + 1
    fl =. fl , 'ssm gate shape[0] = 2048', LF; echo 'FAIL: ssm gate shape[0] = 2048'; echo '  got: '; echo {. $ qw35_bd_s_gate bd0 end.
  tc =. tc + 1
  if. 6144 = {. $ qw35_bd_s_conv1d bd0 do. pc =. pc + 1
    echo 'PASS: ssm conv1d shape[0] = 6144'
  else. fc =. fc + 1
    fl =. fl , 'ssm conv1d shape[0] = 6144', LF; echo 'FAIL: ssm conv1d shape[0] = 6144'; echo '  got: '; echo {. $ qw35_bd_s_conv1d bd0 end.
  tc =. tc + 1
  if. 128 = # qw35_bd_s_norm bd0 do. pc =. pc + 1
    echo 'PASS: ssm_norm length = 128 (head_v_dim)'
  else. fc =. fc + 1
    fl =. fl , 'ssm_norm length = 128', LF; echo 'FAIL: ssm_norm length = 128'; echo '  got: '; echo # qw35_bd_s_norm bd0 end.

  NB. Attention layer weights (block 3): fused Q+GATE, per-head norms
  tc =. tc + 1
  if. 4096 = {. $ qw35_bd_a_q bd3 do. pc =. pc + 1
    echo 'PASS: attn_q shape[0] = 4096 (2*head_dim*n_head fused Q+GATE)'
  else. fc =. fc + 1
    fl =. fl , 'attn_q shape[0] = 4096', LF; echo 'FAIL: attn_q shape[0] = 4096'; echo '  got: '; echo {. $ qw35_bd_a_q bd3 end.
  tc =. tc + 1
  if. 512 = {. $ qw35_bd_a_k bd3 do. pc =. pc + 1
    echo 'PASS: attn_k shape[0] = 512 (n_heads_kv*head_dim)'
  else. fc =. fc + 1
    fl =. fl , 'attn_k shape[0] = 512', LF; echo 'FAIL: attn_k shape[0] = 512'; echo '  got: '; echo {. $ qw35_bd_a_k bd3 end.
  tc =. tc + 1
  if. 256 = # qw35_bd_a_q_norm bd3 do. pc =. pc + 1
    echo 'PASS: attn_q_norm length = 256 (head_dim)'
  else. fc =. fc + 1
    fl =. fl , 'attn_q_norm length = 256', LF; echo 'FAIL: attn_q_norm length = 256'; echo '  got: '; echo # qw35_bd_a_q_norm bd3 end.

  NB. ================================================================
  echo '--- Section 3: Single-Token Inference (vs llama.cpp) ---'
  echo ''

  NB. oracle from llama-cpp-python on same GGUF (BF16):
  NB. hello->11, The->2614, Paris->11
  c_hello =. <((<'hello') , <11)
  c_The   =. <((<'The')   , <2614)
  c_Paris =. <((<'Paris') , <11)
  cases1 =. c_hello , c_The , c_Paris

  ci =. 0
  while. ci < # cases1 do.
    case =. > ci { cases1
    text =. > 0 { case
    expect =. > 1 { case
    tc =. tc + 1
    r =. llm qw35_infer_simple text
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

  NB. hello world->271, The capital of france is->39509, don't stop->279
  c_hw       =. <((<'hello world') , <271)
  c_france   =. <((<'The capital of france is') , <39509)
  c_dont     =. <((<'don''t stop') , <279)
  cases2 =. c_hw , c_france , c_dont

  ci =. 0
  while. ci < # cases2 do.
    case =. > ci { cases2
    text =. > 0 { case
    expect =. > 1 { case
    tc =. tc + 1
    r =. llm qw35_infer_simple text
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

  NB. Qwen3.5 is a thinking model: the chat prompt adds the
  NB. "thinking\n\n response\n\n" prefix, so greedy generation starts
  NB. with the reasoning block. Verify generate returns non-empty
  NB. answer-only text and does not leak the <|im_end|> stop token.
  tc =. tc + 1
  g =. llm qw35_generate ('The capital of France is' ; 8 ; <<0 ; 0 ; 0.95 ; 0.0)
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

test_qwen35 0

pm_report ''
