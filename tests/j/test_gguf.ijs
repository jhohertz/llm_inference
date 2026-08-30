NB. ================================================================
NB. GGUF Library Test Suite
NB. Tests: KV retrieval, tensor info, tensor loading, gguf_dump
NB. ================================================================

coclass 'inference'
load './inference.ijs'
load './gguf_dump.ijs'

ns =: 3 : '(": y)'

NB. Shared counter module
shared_tc =: 0
shared_pc =: 0
shared_fc =: 0
shared_fl =: ''

NB. Test model verb — x=mpath, y=mname
run_model_test =: 4 : 0
  mpath =. x
  mname =. y
  echo '--- Model: ' , mname , ' ---'
  echo '  path: ' , mpath

  NB. Header
  hdr =. parse_hdr mpath
  shared_tc =: shared_tc + 1
  if. 4 = # hdr do.
    shared_pc =: shared_pc + 1
    v0 =. ns > 0 { hdr
    v1 =. ns > 1 { hdr
    s =. '  PASS: header parsed (magic=' , v0 , ' magic, v=' , v1 , ')'
    echo s
  else.
    shared_fc =: shared_fc + 1
    shared_fl =: shared_fl , 'header parse: ' , mname , LF
    echo '  FAIL: header parse'
  end.

  n_tensors =. > 2 { hdr
  n_kv =. > 3 { hdr

  shared_tc =: shared_tc + 1
  if. 0 < n_tensors do.
    shared_pc =: shared_pc + 1
    s =. ns n_tensors
    echo '  PASS: tensor count > 0 (' , s , ')'
  else.
    shared_fc =: shared_fc + 1
    shared_fl =: shared_fl , 'tensor count > 0: ' , mname , LF
    echo '  FAIL: tensor count > 0'
  end.

  NB. Parse KV pairs
  kv_result =. parse_kv_pairs mpath
  kvs =. > 0 { kv_result
  raw =. > 1 { kv_result
  kv_count =. > 2 { kv_result
  kv_end =. > 3 { kv_result

  shared_tc =: shared_tc + 1
  if. 0 < kv_count do.
    shared_pc =: shared_pc + 1
    s =. ns kv_count
    echo '  PASS: KV count > 0 (' , s , ')'
  else.
    shared_fc =: shared_fc + 1
    shared_fl =: shared_fl , 'KV count > 0: ' , mname , LF
    echo '  FAIL: KV count > 0'
  end.

  NB. Parse tensor infos
  ti =. parse_tensor_infos (<raw) , (<kv_end) , (<n_tensors)

  shared_tc =: shared_tc + 1
  if. n_tensors = ((# ti) % 6) do.
    shared_pc =: shared_pc + 1
    echo '  PASS: tensor info count matches'
  else.
    shared_fc =: shared_fc + 1
    shared_fl =: shared_fl , 'tensor info count: ' , mname , LF
    echo '  FAIL: tensor info count'
  end.

  NB. KV: architecture
  box1 =. <kvs
  box2 =. <raw
  ctx_kv =. box1 , box2
  arch =. 'general.architecture' kv_string ctx_kv
  if. 0 = # arch do.
    arch =. 'llm.architecture' kv_string ctx_kv
  end.
  shared_tc =: shared_tc + 1
  if. 0 < # arch do.
    shared_pc =: shared_pc + 1
    echo '  PASS: architecture = ' , arch
  else.
    shared_fc =: shared_fc + 1
    shared_fl =: shared_fl , 'architecture key: ' , mname , LF
    echo '  FAIL: architecture key (empty)'
  end.

   NB. KV: vocab size
   vocab_sz =. 'lfm2.vocab_size' kv_uint ctx_kv
   if. vocab_sz <= 0 do.
     vocab_sz =. 'tokenizer.ggml.vocab_size' kv_uint ctx_kv
   end.
   if. vocab_sz <= 0 do.
     vocab_sz =. 'gemma3.vocab_size' kv_uint ctx_kv
   end.
   if. vocab_sz <= 0 do.
     vocab_sz =. 'llama.vocab_size' kv_uint ctx_kv
   end.
   if. vocab_sz <= 0 do.
     vocab_sz =. 'qwen3.vocab_size' kv_uint ctx_kv
   end.
   if. vocab_sz <= 0 do.
     vocab_sz =. 'qwen2.vocab_size' kv_uint ctx_kv
   end.
   if. vocab_sz <= 0 do.
     vocab_sz =. 'qwen.vocab_size' kv_uint ctx_kv
   end.
   if. vocab_sz <= 0 do.
     vocab_sz =. 'granite.vocab_size' kv_uint ctx_kv
   end.
   if. vocab_sz <= 0 do.
     vocab_sz =. 'bert.vocab_size' kv_uint ctx_kv
   end.
    if. vocab_sz <= 0 do.
      vocab_sz =. 'tokenizer.ggml.tokens_length' kv_uint ctx_kv
    end.
    NB. Fallback: derive vocab_size from token_embd.weight tensor shape
     if. vocab_sz <= 0 do.
       emb_idx =. 'token_embd.weight' find_tensor_idx ti
       if. 0 > emb_idx do.
         vocab_sz =. _1
       else.
          emb_dims =. > ((emb_idx*6)+1) { ti
         if. 2 = # emb_dims do.
           vocab_sz =. > 1 { emb_dims
         else.
           vocab_sz =. _1
         end.
       end.
     end.
   shared_tc =: shared_tc + 1
   if. vocab_sz > 0 do.
    shared_pc =: shared_pc + 1
    s =. ns vocab_sz
    echo '  PASS: vocab_size = ' , s
  else.
    shared_fc =: shared_fc + 1
    shared_fl =: shared_fl , 'vocab_size: ' , mname , LF
    s =. ns vocab_sz
    echo '  FAIL: vocab_size (got ' , s , ')'
  end.

  NB. KV: unknown key
  box1 =. <kvs
  box2 =. <raw
  ctx_kv3 =. box1 , box2
  unk =. 'nonexistent_key_xyz' kv_uint ctx_kv3
  shared_tc =: shared_tc + 1
  if. _1 -: unk do.
    shared_pc =: shared_pc + 1
    echo '  PASS: unknown key returns -1'
  else.
    shared_fc =: shared_fc + 1
    shared_fl =: shared_fl , 'unknown key returns -1: ' , mname , LF
    echo '  FAIL: unknown key returns -1'
  end.

  NB. Find first weight tensor
  shared_tc =: shared_tc + 1
  first_w_idx =. 0
  while. first_w_idx < n_tensors do.
    nm =. > (first_w_idx*6) { ti
    if. (7 < # nm) *. ('.weight' -: |. 7 {. |. nm) do.
      first_w_name =. nm
      first_w_idx =. n_tensors
    else.
      first_w_idx =. first_w_idx + 1
    end.
  end.
  if. first_w_idx <: n_tensors do.
    shared_pc =: shared_pc + 1
    echo '  PASS: found weight tensor: ' , first_w_name

    NB. get_tensor_shape
    shared_tc =: shared_tc + 1
  shape =. first_w_name get_tensor_shape ti
  if. 1 <: # shape do.
    shared_pc =: shared_pc + 1
    s =. ns shape
    echo '  PASS: shape = ' , s
  else.
    shared_fc =: shared_fc + 1
    shared_fl =: shared_fl , 'tensor shape: ' , mname , LF
    echo '  FAIL: tensor shape'
  end.

  NB. get_tensor_type
  shared_tc =: shared_tc + 1
  ttype =. first_w_name get_tensor_type ti
  if. (0 = ttype) +. (1 = ttype) +. (30 = ttype) do.
    shared_pc =: shared_pc + 1
    echo '  PASS: type = ' , elem_type_name (<ttype)
  else.
    shared_fc =: shared_fc + 1
    shared_fl =: shared_fl , 'tensor type: ' , mname , LF
    s =. ns ttype
    echo '  FAIL: tensor type (got ' , s , ')'
  end.

   NB. Tensor load
   NB. Last tensor's "next" offset is the end of tensor info section
    ti_end =. > ((n_tensors*6)-1) { ti
   tds =. 32 * <. (ti_end + 31) % 32
  d1 =. <mpath
  d2 =. <ti
  d3 =. <tds
  d4 =. <first_w_name
  d5 =. <raw
  ctx_load =. d1 , d2 , d3 , d4 , d5
  td =. load_tdata ctx_load

  shared_tc =: shared_tc + 1
  if. 0 < # td do.
    shared_pc =: shared_pc + 1
    s =. ns $ td
    echo '  PASS: tensor loaded, shape = ' , s
  else.
    shared_fc =: shared_fc + 1
    shared_fl =: shared_fl , 'tensor load: ' , mname , LF
    echo '  FAIL: tensor load'
  end.

  NB. Value range
  shared_tc =: shared_tc + 1
  flat =. , td
  if. (1e10 > >./ flat) do.
    shared_pc =: shared_pc + 1
    echo '  PASS: values < 1e10'
  else.
    shared_fc =: shared_fc + 1
    shared_fl =: shared_fl , 'value range: ' , mname , LF
    echo '  FAIL: values >= 1e10'
  end.

  NB. Shape consistency
  shared_tc =: shared_tc + 1
  expected_shape =. first_w_name get_tensor_shape ti
  actual_shape =. $ td
  NB. GGUF stores 2D weight data REVERSED vs the dims field: dims=[in,out],
  NB. data reshaped as |. dims (see tensor_reshape). 1D shapes are unchanged.
  if. actual_shape -: |. expected_shape do.
    shared_pc =: shared_pc + 1
    echo '  PASS: loaded shape matches tensor info'
  else.
    shared_fc =: shared_fc + 1
    shared_fl =: shared_fl , 'shape consistency: ' , mname , LF
    echo '  FAIL: shape mismatch'
  end.
  else.
    shared_fc =: shared_fc + 1
    shared_fl =: shared_fl , 'no weight tensor found: ' , mname , LF
    echo '  FAIL: no weight tensor found'
  end.

  NB. gguf_dump
  dump_result =. gguf_dump mpath
  shared_tc =: shared_tc + 1
  shared_pc =: shared_pc + 1
  echo '  PASS: gguf_dump ran without error'
  echo ''
)

NB. Main test function
test_gguf =: 3 : 0
  shared_tc =: 0
  shared_pc =: 0
  shared_fc =: 0
  shared_fl =: ''

  echo ''
  echo '========== GGUF TEST SUITE =========='
  echo ''

  NB. === Section 1: Element Type Names ===
  echo '--- Section 1: Element Type Names ---'
  shared_tc =: shared_tc + 1
  if. 'F32' -: elem_type_name 0 do.
    shared_pc =: shared_pc + 1
    echo '  PASS: elem_type_name F32(0)'
  else.
    shared_fc =: shared_fc + 1
    shared_fl =: shared_fl , 'elem_type_name F32(0)', LF
    echo '  FAIL: elem_type_name F32(0)'
  end.
  shared_tc =: shared_tc + 1
  if. 'F16' -: elem_type_name 1 do.
    shared_pc =: shared_pc + 1
    echo '  PASS: elem_type_name F16(1)'
  else.
    shared_fc =: shared_fc + 1
    shared_fl =: shared_fl , 'elem_type_name F16(1)', LF
    echo '  FAIL: elem_type_name F16(1)'
  end.
  shared_tc =: shared_tc + 1
  if. 'BF16' -: elem_type_name 30 do.
    shared_pc =: shared_pc + 1
    echo '  PASS: elem_type_name BF16(30)'
  else.
    shared_fc =: shared_fc + 1
    shared_fl =: shared_fl , 'elem_type_name BF16(30)', LF
    echo '  FAIL: elem_type_name BF16(30)'
  end.
  shared_tc =: shared_tc + 1
  if. '<unknown>' -: elem_type_name 99 do.
    shared_pc =: shared_pc + 1
    echo '  PASS: elem_type_name <unknown>(99)'
  else.
    shared_fc =: shared_fc + 1
    shared_fl =: shared_fl , 'elem_type_name <unknown>(99)', LF
    echo '  FAIL: elem_type_name <unknown>(99)'
  end.

  NB. === Section 2: Value Type Names ===
  echo ''
  echo '--- Section 2: Value Type Names ---'
  shared_tc =: shared_tc + 1
  if. 'uint32' -: val_type_name 4 do.
    shared_pc =: shared_pc + 1
    echo '  PASS: val_type_name uint32(4)'
  else.
    shared_fc =: shared_fc + 1
    shared_fl =: shared_fl , 'val_type_name uint32(4)', LF
    echo '  FAIL: val_type_name uint32(4)'
  end.
  shared_tc =: shared_tc + 1
  if. 'float32' -: val_type_name 6 do.
    shared_pc =: shared_pc + 1
    echo '  PASS: val_type_name float32(6)'
  else.
    shared_fc =: shared_fc + 1
    shared_fl =: shared_fl , 'val_type_name float32(6)', LF
    echo '  FAIL: val_type_name float32(6)'
  end.
  shared_tc =: shared_tc + 1
  if. 'string' -: val_type_name 8 do.
    shared_pc =: shared_pc + 1
    echo '  PASS: val_type_name string(8)'
  else.
    shared_fc =: shared_fc + 1
    shared_fl =: shared_fl , 'val_type_name string(8)', LF
    echo '  FAIL: val_type_name string(8)'
  end.
  shared_tc =: shared_tc + 1
  if. 'array' -: val_type_name 9 do.
    shared_pc =: shared_pc + 1
    echo '  PASS: val_type_name array(9)'
  else.
    shared_fc =: shared_fc + 1
    shared_fl =: shared_fl , 'val_type_name array(9)', LF
    echo '  FAIL: val_type_name array(9)'
  end.
  shared_tc =: shared_tc + 1
  if. '<unknown>' -: val_type_name 99 do.
    shared_pc =: shared_pc + 1
    echo '  PASS: val_type_name <unknown>(99)'
  else.
    shared_fc =: shared_fc + 1
    shared_fl =: shared_fl , 'val_type_name <unknown>(99)', LF
    echo '  FAIL: val_type_name <unknown>(99)'
  end.

  NB. === Section 3: Multi-Model Tests ===
  echo ''
  echo '--- Section 3: Multi-Model Tests ---'
  echo ''

  (model_path 'gemma-3-270m-it') run_model_test 'gemma3-270M-F16'
  (model_path 'granite-4.0-350m') run_model_test 'granite-4.0-350m-BF16'
  (model_path 'qwen3-0.6b') run_model_test 'Qwen3-0.6B-BF16'
  (model_path 'granite-4.0-h-350m') run_model_test 'granite-4.0-h-350m-BF16'
  (model_path 'qwen3.5-0.8b') run_model_test 'Qwen3.5-0.8B-BF16'
  (model_path 'ernie-4.5-0.3b') run_model_test 'ERNIE-4.5-0.3B-F16'
  (model_path 'qwen2.5-coder-0.5b') run_model_test 'Qwen2.5-Coder-0.5B-F16'
  (model_path 'smollm2-360m') run_model_test 'SmolLM2-360M-F16'

  NB. Summary
  echo ''
  echo '========== GGUF TEST SUMMARY =========='
  echo 'Total: ' , ns shared_tc
  echo 'Passed: ' , ns shared_pc
  echo 'Failed: ' , ns shared_fc
  if. 0 < # shared_fl do.
    echo ''
    echo 'Failures:'
    echo shared_fl
  end.
  echo ''
  echo 'GGUF test suite complete!'
)

test_gguf 0
