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
  NB. Render via the real GGUF jinja template (chat_prompt is pure jinja now).
  ct_tmpl_g =: 'tokenizer.chat_template' kv_string (0 1 { kv)
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

NB. ---- chat_tmpl_render tools wiring (upstream tool_use golden) ----
NB. Feeds the real llama-3.1 tool template a tool_use context through the
NB. shared renderer (ct_tools_g JSON -> template `tools` input, pre-built
NB. message Values for tool_calls/typed content), and checks the output
NB. matches the upstream golden exactly (chat_tmpl_render uses '' bos/eos,
NB. so the <|startoftext|> marker is prepended to the render for the oracle).
test_tools_render =: 3 : 0
  tmpl =. 1!:1 < 'tests/j/templates/meta-llama-Llama-3.1-8B-Instruct.jinja'
  ct_tmpl_g =: tmpl
  ct_now_g =: 1721952000
  c =. ct_parse_json_chatpl_ (1!:1 < 'reference/minja/tests/contexts/tool_use.json')
  msgs =. 'messages' obj_get_minja_ c
  tools =. 'tools' obj_get_minja_ c
  messages =. arr_items_minja_ msgs
  ks =. obj_keys_minja_ c
  drop =. ;: 'messages tools add_generation_prompt bos_token eos_token'
  extra =. mkobj_minja_ ''
  for_i. i. # ks do.
    k =. > (i { ks)
    if. -. ((<k) e. drop) do.
      extra =. ((<k) , <(k obj_get_minja_ c)) obj_set_minja_ extra
    end.
  end.
  ct_vars_g =: extra
  ct_tools_g =: dumpc_minja_ ((<tools) , (<_1) , (<0) , (<1))
  out =. chat_tmpl_render messages
  gold =. 1!:1 < 'tests/j/goldens/meta-llama-Llama-3.1-8B-Instruct-tool_use.txt'
  assert_test (((('<|startoftext|>') , out) -: gold)) ; 'chat_tmpl_render tools wiring == upstream tool_use golden'

  NB. regression: no tools -> the tool-dump section is skipped
  ct_tools_g =: ''
  out0 =. chat_tmpl_render messages
  assert_test ((-. ('You have access to the following functions' e. out0))) ; 'chat_tmpl_render no-tools drops tool dump'
  assert_test (('You have access to the following functions' e. out)) ; 'chat_tmpl_render tools adds tool dump'
  ct_tools_g =: ''
  ct_vars_g =: ''
  ct_tmpl_g =: ''
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
  echo '--- chat_tmpl_render tools wiring (upstream llama-3.1 tool_use golden) ---'
  test_tools_render ''

  echo ''
  show_summary 1
  ''
)

pm_start 1e8

test_chat_prompt 0

pm_report ''
