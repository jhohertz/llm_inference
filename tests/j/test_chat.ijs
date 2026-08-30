NB. ================================================================
NB. Chat Template Test Suite (Phase 1.1)
NB. Verifies chat-prompt rendering + special-token encoding against the
NB. llama-cpp-python create_chat_completion _input_ids references, for all
NB. three arches. Tokenizers are built standalone from the GGUF KV pairs
NB. (no weight loads — fast).
NB. ================================================================
coclass 'inference'
load './tokenizers/tokenizer_llama3.ijs'
load './tokenizers/tokenizer_gpt2.ijs'
load './models/gemma3.ijs'
load './models/qwen2.ijs'
load './models/qwen35.ijs'
load './models/llama.ijs'
load './util/chat.ijs'
load './util/models.ijs'
load './tests/j/test_harness.ijs'
load './tests/j/pm_fixture.ijs'

gemma_path =: model_path 'gemma-3-270m-it'
qwen2_path =: model_path 'qwen2.5-coder-0.5b'
smollm2_path =: model_path 'smollm2-360m'
qwen35_path =: model_path 'qwen3.5-0.8b'

NB. ---- Build a minimal llm noun with tokenizer at index 3 ----
mk_llm =: 3 : 0
  (<'') , (<'') , (<'') , <y
)

NB. ---- Prompt-token oracle check for one arch ----
NB. y = <arch; path; ref> — dispatch builder/tokenizer/chat_prompt by arch.
test_arch_prompt =: 3 : 0
  arch =. > 0 { y
  path =. > 1 { y
  ref =. > 2 { y
  msg =. ('user') ; 'The capital of France is'
  messages =. <msg
  kv =. parse_kv_pairs path
  select. arch
  case. 'gemma3' do.
    tk =. build_llama3_tokenizer kv
    prompt =. gem3_chat_prompt messages
    toks =. llama3_tokenize (<mk_llm tk) , <prompt
  case. 'qwen2' do.
    tk =. build_gpt2_tokenizer kv
    prompt =. qw2_chat_prompt messages
    toks =. gpt2_tokenize (<mk_llm tk) , <prompt
  case. 'qwen35' do.
    tk =. build_gpt2_tokenizer kv
    prompt =. qw35_chat_prompt messages
    toks =. gpt2_tokenize (<mk_llm tk) , <prompt
  case. 'llama' do.
    tk =. build_gpt2_tokenizer kv
    prompt =. llama_chat_prompt messages
    toks =. gpt2_tokenize (<mk_llm tk) , <prompt
  end.
  got =. , > toks
  echo '  ' , arch , ' prompt: ' , prompt
  echo '  ' , arch , ' tokens: ' , ": got
  assert_test (ref -: got) ; (arch , ' chat prompt tokens match llama.cpp')
  ''
)

test_chat_prompt =: 3 : 0
  init_counters ''
  echo '========================================'
  echo 'Chat Template Test Suite'
  echo '========================================'
  echo ''
  echo '--- Chat prompt tokenization vs llama-cpp-python _input_ids ---'

  gemma_ref =. 2 105 2364 107 818 5279 529 7001 563 106 107 105 4368 107
  qwen2_ref =. 151644 8948 198 2610 525 1207 16948 11 3465 553 54364 14817 13 1446 525 264 10950 17847 13 151645 198 151644 872 198 785 6722 315 9625 374 151645 198 151644 77091 198
  smollm2_ref =. 1 9690 198 2683 359 253 5356 5646 11173 3365 3511 308 34519 28 7018 411 407 19712 8182 2 198 1 4093 198 504 3575 282 4649 314 2 198 1 520 9531 198
  qwen35_ref =. 248045 846 198 760 6511 314 9338 369 248046 198 248045 74455 198 248068 271 248069 271

  test_arch_prompt ('gemma3' ; gemma_path ; gemma_ref)
  test_arch_prompt ('qwen2' ; qwen2_path ; qwen2_ref)
  test_arch_prompt ('llama' ; smollm2_path ; smollm2_ref)
  test_arch_prompt ('qwen35' ; qwen35_path ; qwen35_ref)

  echo ''
  show_summary 1
  ''
)

pm_start 1e8

test_chat_prompt 0

pm_report ''
