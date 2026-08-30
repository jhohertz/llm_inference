NB. ================================================================
NB. Gemma 3 270M Test Suite — full model-agnostic tests
NB. Tests: GGUF parsing, KV extraction, model loading, inference
NB. Depends on: inference.ijs (loads all dependencies including gemma3)
NB. ================================================================

coclass 'inference'
load './inference.ijs'
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

test_gemma3 =: 3 : 0
  tc =. 0
  pc =. 0
  fc =. 0
  fl =. ''
  suite_start =. 6!:1 ''
  overall_start =. 6!:1 ''
  mem_start =. 7!:0 ''

  echo ''
  echo '=============================================================='
  echo '  GEMMA 3 270M TEST SUITE — with timing & memory'
  echo '=============================================================='

  NB. ================================================================
  echo '--- Section 1: Model Loading ---'
  echo ''

  gemma_path =. 'gemma-3-270m-it'
  echo '  (loading model)...'
  mem_before_load =. 7!:0 ''
  load_time =. 5 (6!:2) 'llm =. load_gguf_to_llm gemma_path'
  mem_after_load =. 7!:0 ''
  load_mem =. mem_after_load - mem_before_load
  echo '  (done, ' , (": <. load_time) , 's)'
  echo '  Load memory delta: ' , fmt_bytes load_mem
  echo '  Workspace after load: ' , fmt_bytes mem_after_load
  echo ''

  NB. --- Test llm structure ---
  tc =. tc + 1
  if. 10 = # llm do.
    pc =. pc + 1
    echo 'PASS: llm has 10 elements'
  else.
    fc =. fc + 1
    fl =. fl , 'llm has 10 elements', LF
    echo 'FAIL: llm has 10 elements'; echo '  got: '; echo # llm
  end.

  tc =. tc + 1
  lp =. llm_path llm
  if. 0 < # lp do.
    pc =. pc + 1
    echo 'PASS: llm_path returns non-empty path'
  else.
    fc =. fc + 1
    fl =. fl , 'llm_path returns non-empty path', LF
    echo 'FAIL: llm_path returns non-empty path'
  end.

  tc =. tc + 1
  if. 0 < # llm_ti llm do.
    pc =. pc + 1
    echo 'PASS: llm_ti returns tensor infos'
  else.
    fc =. fc + 1
    fl =. fl , 'llm_ti returns tensor infos', LF
    echo 'FAIL: llm_ti returns tensor infos'
  end.

  mi =. llm_mi llm

  NB. block_count = 18
  tc =. tc + 1
  if. 18 -: mi_block_count mi do. pc =. pc + 1
    echo 'PASS: block_count = 18'
  else. fc =. fc + 1
    fl =. fl , 'block_count = 18', LF; echo 'FAIL: block_count = 18'; echo '  got: '; echo mi_block_count mi end.

  NB. context_length = 32768
  tc =. tc + 1
  if. 32768 -: mi_context_len mi do. pc =. pc + 1
    echo 'PASS: context_length = 32768'
  else. fc =. fc + 1
    fl =. fl , 'context_length = 32768', LF; echo 'FAIL: context_length = 32768'; echo '  got: '; echo mi_context_len mi end.

  NB. embedding_length = 640
  tc =. tc + 1
  if. 640 -: mi_emb_len mi do. pc =. pc + 1
    echo 'PASS: embedding_length = 640'
  else. fc =. fc + 1
    fl =. fl , 'embedding_length = 640', LF; echo 'FAIL: embedding_length = 640'; echo '  got: '; echo mi_emb_len mi end.

  NB. attention.head_count = 4
  tc =. tc + 1
  if. 4 -: mi_n_heads mi do. pc =. pc + 1
    echo 'PASS: attention.head_count = 4'
  else. fc =. fc + 1
    fl =. fl , 'attention.head_count = 4', LF; echo 'FAIL: attention.head_count = 4'; echo '  got: '; echo mi_n_heads mi end.

  NB. attention.head_count_kv = 1
  tc =. tc + 1
  if. 1 -: mi_n_heads_kv mi do. pc =. pc + 1
    echo 'PASS: attention.head_count_kv = 1'
  else. fc =. fc + 1
    fl =. fl , 'attention.head_count_kv = 1', LF; echo 'FAIL: attention.head_count_kv = 1'; echo '  got: '; echo mi_n_heads_kv mi end.

  NB. head_dim = 256
  tc =. tc + 1
  if. 256 -: mi_head_dim mi do. pc =. pc + 1
    echo 'PASS: head_dim = 256'
  else. fc =. fc + 1
    fl =. fl , 'head_dim = 256', LF; echo 'FAIL: head_dim = 256'; echo '  got: '; echo mi_head_dim mi end.

  NB. rope.freq_base = 1e6
  tc =. tc + 1
  if. 1.0e6 -: mi_rope_freq mi do. pc =. pc + 1
    echo 'PASS: rope.freq_base = 1e6'
  else. fc =. fc + 1
    fl =. fl , 'rope.freq_base = 1e6', LF; echo 'FAIL: rope.freq_base = 1e6'; echo '  got: '; echo mi_rope_freq mi end.

  NB. sliding_window = 512
  tc =. tc + 1
  if. 512 -: mi_swa mi do. pc =. pc + 1
    echo 'PASS: sliding_window = 512'
  else. fc =. fc + 1
    fl =. fl , 'sliding_window = 512', LF; echo 'FAIL: sliding_window = 512'; echo '  got: '; echo mi_swa mi end.

  NB. feed_forward_length = 2048
  tc =. tc + 1
  if. 2048 -: mi_n_ff mi do. pc =. pc + 1
    echo 'PASS: feed_forward_length = 2048'
  else. fc =. fc + 1
    fl =. fl , 'feed_forward_length = 2048', LF; echo 'FAIL: feed_forward_length = 2048'; echo '  got: '; echo mi_n_ff mi end.

  NB. ================================================================
  echo '--- Section 2: Block Data ---'
  echo ''

  bd_list =. llm_block_data llm
  tc =. tc + 1
  if. 18 = # bd_list do. pc =. pc + 1
    echo 'PASS: 18 block_data entries'
  else. fc =. fc + 1
    fl =. fl , '18 block_data entries', LF; echo 'FAIL: 18 block_data entries'; echo '  got: '; echo # bd_list end.

  bd0 =. > 0 { bd_list
  tc =. tc + 1
  if. 21 = # bd0 do. pc =. pc + 1
    echo 'PASS: block_data[0] has 21 elements'
  else. fc =. fc + 1
    fl =. fl , 'block_data[0] has 21 elements', LF; echo 'FAIL: block_data[0] has 21 elements'; echo '  got: '; echo # bd0 end.

  attn_q =. gem3_bd_attn_q bd0
  tq =. $ attn_q
  tc =. tc + 1
  if. 1024 = {. tq do. pc =. pc + 1
    echo 'PASS: attn_q shape[0] = 1024 (out dim)'
  else. fc =. fc + 1
    fl =. fl , 'attn_q shape[0] = 1024', LF; echo 'FAIL: attn_q shape[0] = 1024'; echo '  got: '; echo {. tq end.
  tc =. tc + 1
  if. 640 = {: tq do. pc =. pc + 1
    echo 'PASS: attn_q shape[1] = 640 (emb_len)'
  else. fc =. fc + 1
    fl =. fl , 'attn_q shape[1] = 640', LF; echo 'FAIL: attn_q shape[1] = 640'; echo '  got: '; echo {: tq end.

  ao =. gem3_bd_attn_o bd0
  ao_sh =. $ ao
  tc =. tc + 1
  if. 640 = {. ao_sh do. pc =. pc + 1
    echo 'PASS: attn_output shape[0] = 640 (out dim)'
  else. fc =. fc + 1
    fl =. fl , 'attn_output shape[0] = 640', LF; echo 'FAIL: attn_output shape[0] = 640'; echo '  got: '; echo {. ao_sh end.
  tc =. tc + 1
  if. 1024 = {: ao_sh do. pc =. pc + 1
    echo 'PASS: attn_output shape[1] = 1024 (emb_len)'
  else. fc =. fc + 1
    fl =. fl , 'attn_output shape[1] = 1024', LF; echo 'FAIL: attn_output shape[1] = 1024'; echo '  got: '; echo {: ao_sh end.

  tc =. tc + 1
  if. 2048 = {. $ gem3_bd_ff_gate bd0 do. pc =. pc + 1
    echo 'PASS: ffn_gate shape[0] = 2048'
  else. fc =. fc + 1
    fl =. fl , 'ffn_gate shape[0] = 2048', LF; echo 'FAIL: ffn_gate shape[0] = 2048'; echo '  got: '; echo {. $ gem3_bd_ff_gate bd0 end.
  tc =. tc + 1
  if. 2048 = {. $ gem3_bd_ff_up bd0 do. pc =. pc + 1
    echo 'PASS: ffn_up shape[0] = 2048'
  else. fc =. fc + 1
    fl =. fl , 'ffn_up shape[0] = 2048', LF; echo 'FAIL: ffn_up shape[0] = 2048'; echo '  got: '; echo {. $ gem3_bd_ff_up bd0 end.

  fd =. gem3_bd_ff_down bd0
  fd_sh =. $ fd
  tc =. tc + 1
  if. 640 = {. fd_sh do. pc =. pc + 1
    echo 'PASS: ffn_down shape[0] = 640 (emb_len)'
  else. fc =. fc + 1
    fl =. fl , 'ffn_down shape[0] = 640', LF; echo 'FAIL: ffn_down shape[0] = 640'; echo '  got: '; echo {. fd_sh end.
  tc =. tc + 1
  if. 2048 = {: fd_sh do. pc =. pc + 1
    echo 'PASS: ffn_down shape[1] = 2048 (n_ff)'
  else. fc =. fc + 1
    fl =. fl , 'ffn_down shape[1] = 2048', LF; echo 'FAIL: ffn_down shape[1] = 2048'; echo '  got: '; echo {: fd_sh end.

  tc =. tc + 1
  if. 1 = > 14 { bd0 do. pc =. pc + 1
    echo 'PASS: block_data n_heads_kv = 1'
  else. fc =. fc + 1
    fl =. fl , 'block_data n_heads_kv = 1', LF; echo 'FAIL: block_data n_heads_kv = 1'; echo '  got: '; echo > 14 { bd0 end.

  NB. ================================================================
  echo '--- Section 3: KV Cache ---'
  echo ''

  tc =. tc + 1
  kv_create ((<18)) , ((<512)) , ((<1)) , ((<256))
  if. (4 -: # kv_meta) *. (9216 256 -: $ k_cache_g) *. (9216 256 -: $ v_cache_g) do. pc =. pc + 1
    echo 'PASS: kv_create shape'
  else. fc =. fc + 1
    fl =. fl , 'kv_create shape', LF; echo 'FAIL: kv_create shape'; echo '  got: '; echo $ k_cache_g end.

  k_test =. (1, 256) $ 1
  kv_write (((<0)) , ((<0)) , ((<k_test)) , ((<(256 $ 2))))
  kv_result =. kv_read ((<0) , <0)
  tc =. tc + 1
  if. 1 1 256 -: $ > 0 { kv_result do. pc =. pc + 1
    echo 'PASS: kv_read shape'
  else. fc =. fc + 1
    fl =. fl , 'kv_read shape', LF; echo 'FAIL: kv_read shape'; echo '  got: '; echo $ > 0 { kv_result end.

  tc =. tc + 1
  kv_reset ''
  kv_write (((<0)) , ((<0)) , (<(1 256 $ 7)) , (<(256 $ 3)))
  kv_result =. kv_read ((<0) , <0)
  k_z =. > 0 { kv_result
  if. (7 -: {. , k_z) *. (3 -: {. , > 1 { kv_result) do. pc =. pc + 1
    echo 'PASS: kv_reset reuses buffer (write-then-read)'
  else. fc =. fc + 1
    fl =. fl , 'kv_reset reuses buffer', LF; echo 'FAIL: kv_reset reuses buffer' end.

  NB. ================================================================
  echo '--- Section 4: Tokenizer ---'
  echo ''

  tc =. tc + 1
  tok =. llama3_tokenize ((<llm)) , (<'hello')
  if. 0 < # , > tok do. pc =. pc + 1
    echo 'PASS: tokenize "hello"'
  else. fc =. fc + 1
    fl =. fl , 'tokenize "hello"', LF; echo 'FAIL: tokenize "hello"' end.

  tc =. tc + 1
  detok =. llama3_detokenize ((<llm)) , (<, > tok)
  if. 0 < # detok do. pc =. pc + 1
    echo 'PASS: detokenize produces text'
  else. fc =. fc + 1
    fl =. fl , 'detokenize produces text', LF; echo 'FAIL: detokenize produces text' end.

  first_tok =. > 0 { , > tok
  tc =. tc + 1
  if. 2 = first_tok do. pc =. pc + 1
    echo 'PASS: BOS token = 2'
  else. fc =. fc + 1
    fl =. fl , 'BOS token = 2', LF; echo 'FAIL: BOS token = 2'; echo '  got: '; echo first_tok end.

  NB. ================================================================
  echo '--- Section 5: Inference (single token) ---'
  echo ''

  echo '  (running single-token inference with "hello")...'
  mem_infer_before =. 7!:0 ''
  tc =. tc + 1
  infer_time =. 3 (6!:2) 'llm infer_simple ''hello'''
  mem_infer_after =. 7!:0 ''
  ir =. llm infer_simple 'hello'
  it =. ": infer_time
  echo '  (infer time: ' , it , 's)'
  echo '  Inference memory delta: ' , fmt_bytes (mem_infer_after - mem_infer_before)
  if. 4 = # ir do. pc =. pc + 1
    echo 'PASS: infer returns 4 elements'
  else. fc =. fc + 1
    fl =. fl , 'infer returns 4 elements', LF; echo 'FAIL: infer returns 4 elements'; echo '  got: '; echo # ir end.

  pred_tok =. > 1 { ir
  decoded =. > 2 { ir
  logits =. > 3 { ir

  tc =. tc + 1
  if. 0 < pred_tok do. pc =. pc + 1
    echo 'PASS: infer produces prediction'
  else. fc =. fc + 1
    fl =. fl , 'infer produces prediction', LF; echo 'FAIL: infer produces prediction' end.
  tc =. tc + 1
  if. 0 < # decoded do. pc =. pc + 1
    echo 'PASS: infer produces decoded text'
  else. fc =. fc + 1
    fl =. fl , 'infer produces decoded text', LF; echo 'FAIL: infer produces decoded text' end.
  tc =. tc + 1
  if. 262144 = # logits do. pc =. pc + 1
    echo 'PASS: logits shape = 262144'
  else. fc =. fc + 1
    fl =. fl , 'logits shape = 262144', LF; echo 'FAIL: logits shape = 262144'; echo '  got: '; echo # logits end.

  NB. ================================================================
  echo '--- Section 6: Multi-token generation ---'
  echo ''

   echo '  (generate greedy "hello world", 5 steps)... '
  mem_gen1_before =. 7!:0 ''
  gen_start1 =. 6!:2 ''
  tc =. tc + 1
  gr =. llm generate_simple (('hello world') ; (5))
  mem_gen1_after =. 7!:0 ''
  gen_time1 =. (6!:2 '') - gen_start1
  if. 0 < # gr do.
    pc =. pc + 1
    gt1 =. ": gen_time1
    echo 'PASS: generate greedy produces output (' , gt1 , 's)'
    echo '    Memory delta: ' , fmt_bytes (mem_gen1_after - mem_gen1_before)
  else. fc =. fc + 1
    fl =. fl , 'generate greedy produces output', LF; echo 'FAIL: generate greedy produces output' end.

  NB. Model params from models/gemma-3-270m-it-GGUF/params:
  NB.   temperature: 1.0, min_p: 0.001, top_k: 64, top_p: 0.95
  echo '  (generate sampled temp=1.0, top_p=0.95, top_k=64, min_p=0.001)... '
  mem_gen2_before =. 7!:0 ''
  gen_start2 =. 6!:2 ''
  tc =. tc + 1
  fp =. (1.0) , (64) , (0.95) , (0.001)
  ga =. ((<'The quick brown')) , ((<8)) , ((<fp))
  gs =. llm generate ga
  mem_gen2_after =. 7!:0 ''
  gen_time2 =. (6!:2 '') - gen_start2
  if. 0 < # gs do.
    pc =. pc + 1
    gt2 =. ": gen_time2
    echo 'PASS: generate sampled produces output (' , gt2 , 's)'
     echo '    Memory delta: ' , fmt_bytes (mem_gen2_after - mem_gen2_before)
    echo '  output: '
    echo gs
  else.
    fc =. fc + 1
    fl =. fl , 'generate sampled produces output', LF
    echo 'FAIL: generate sampled produces output'
  end.

  NB. Greedy baseline
  echo '  (generate greedy "The quick brown", 8 steps, temp=0)... '
  mem_gen3_before =. 7!:0 ''
  gen_start3 =. 6!:2 ''
  gp =. (0) , (0) , (0.95) , (0)
  ga_g =. ((<'The quick brown')) , ((<8)) , ((<gp))
  gg =. llm generate ga_g
  mem_gen3_after =. 7!:0 ''
  gen_time3 =. (6!:2 '') - gen_start3
  tc =. tc + 1
  if. 0 < # gg do.
    pc =. pc + 1
    echo 'PASS: generate greedy baseline produces output'
     echo '    Memory delta: ' , fmt_bytes (mem_gen3_after - mem_gen3_before)
    echo '  output: '
    echo gg
  else.
    fc =. fc + 1
    fl =. fl , 'generate greedy baseline produces output', LF
    echo 'FAIL: generate greedy baseline produces output'
  end.

  NB. --- Chat generation mechanism (Phase 1.1) ---
  echo '  (chat generate greedy "The capital of France is", 20 steps)... '
  msg =. ('user') ; 'The capital of France is'
  messages =. <msg
  ga_chat =. llm chat_generate (messages ; 20 ; <0 0 0.95 0.0)
  tc =. tc + 1
  if. 0 < # > ga_chat do.
    pc =. pc + 1
    echo 'PASS: chat_generate produces output'
  else.
    fc =. fc + 1
    fl =. fl , 'chat_generate produces output', LF
    echo 'FAIL: chat_generate produces output'
  end.
  tc =. tc + 1
  if. -. 1 e. ('<end_of_turn>' E. > ga_chat) do.
    pc =. pc + 1
    echo 'PASS: chat answer has no stop token'
  else.
    fc =. fc + 1
    fl =. fl , 'chat answer has no stop token', LF
    echo 'FAIL: chat answer has no stop token'
  end.
  NB. Regression pin (gold: HF transformers + llama.cpp, exact): greedy chat answer
  NB. for "The capital of France is" is "The capital of France is Paris.". Before the
  NB. SWA-freq_base + byte-clamp fixes the first token was "Paris" (50429) -> garbage.
  tc =. tc + 1
  if. ('The capital of France is Paris.' -: > ga_chat) do.
    pc =. pc + 1
    echo 'PASS: chat greedy answer matches gold continuation'
  else.
    fc =. fc + 1
    fl =. fl , 'chat greedy answer matches gold continuation', LF
    echo 'FAIL: chat greedy answer matches gold continuation'
  end.
  echo '  chat answer: ' , ": > ga_chat

  NB. ================================================================
  suite_time =. 6!:1 '' - suite_start
  overall_time =. 6!:1 '' - overall_start
  mem_end =. 7!:0 ''
  mem_total =. mem_end - mem_start
  st =. ": suite_time

  echo ''
  echo '=============================================================='
  echo '  TEST SUMMARY'
  echo '=============================================================='
  echo ''
  echo '  Metric:          Value'
  echo '  -------------------------------  ------'
  echo '  Total:          ' , ": tc
  echo '  Passed:         ' , ": pc
  echo '  Failed:         ' , ": fc
  if. tc > 0 do.
    rate =. <. (pc % tc) * 100
    echo '  Pass rate:      ' , (": rate) , '%'
  else.
    echo '  Pass rate:      N/A'
  end.
  echo '  Total time:     ' , fmt_time overall_time
  echo '  Workspace delta: ' , fmt_bytes (mem_total)
  echo ''
  if. 0 < # fl do. echo ''; echo 'Failed:'; echo fl end.
  echo ''
  if. 0 = fc do. echo 'All tests passed!' else. echo 'Some tests failed.' end.
  echo ''
)

pm_start 1e8

test_gemma3 0

pm_report ''
