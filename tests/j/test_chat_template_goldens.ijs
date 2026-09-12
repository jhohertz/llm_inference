NB. ================================================================
NB. test_chat_template_goldens.ijs — real-template golden renders.
NB. Port of reference/minja/tests/test-supported-template.cpp: for each
NB. real .jinja template and each context (simple/system/tool_use),
NB. build the chat-template inputs from the context JSON (messages,
NB. tools, add_generation_prompt, bos/eos, extra), apply via ct_apply with
NB. pinned now=1721952000 (2024-07-26, the upstream TEST_DATE), and compare
NB. to the Python-generated golden. 30 cases = 10 templates x 3 contexts.
NB. ================================================================
coclass 'chatpl'

load './tests/j/test_harness.ijs'
load './util/minja.ijs'
load './util/chat_template.ijs'

init_counters_inference_ ''

streq =: 4 : '(x -: , y)'

NB. mkctxin jsonpath -> chat-template inputs (now pinned to 1721952000).
mkctxin =: 3 : 0
  c =. ct_parse_json y
  msgs =. 'messages' obj_get_minja_ c
  tools =. 'tools' obj_get_minja_ c
  addg =. to_bool_minja_ ('add_generation_prompt' obj_get_minja_ c)
  bos =. payload_minja_ ('bos_token' obj_get_minja_ c)
  eos =. payload_minja_ ('eos_token' obj_get_minja_ c)
  ks =. obj_keys_minja_ c
  drop =. ;: 'messages tools add_generation_prompt bos_token eos_token'
  extra =. mkobj_minja_ ''
  for_i. i. # ks do.
    k =. > (i { ks)
    if. -. ((<k) e. drop) do.
      extra =. ((<k) , <(k obj_get_minja_ c)) obj_set_minja_ extra
    end.
  end.
  ((<msgs) , (<tools) , (<addg) , (<extra) , (<1721952000) , (<bos) , (<eos))
)

NB. golden_check (tmpl_path ; ctx_path ; gold_path ; label)
golden_check =: 3 : 0
  'tmpl_p ctx_p gold_p label' =. y
  tmpl =. 1!:1 < tmpl_p
  ct_new (tmpl ; '' ; '')
  inputs =. mkctxin (1!:1 < ctx_p)
  try.
    out =. ct_apply ((<tmpl) , (<inputs) , (<ct_caps_g) , (<ct_tool_ex_g) , (<(mk_options '')))
    goldtxt =. 1!:1 < gold_p
    assert_test_inference_ (((out streq goldtxt)) ; label)
  catcht. 'minja'
    assert_test_inference_ ((<0) ; label)
  catch.
    assert_test_inference_ ((<0) ; label)
  end.
)

section_header_inference_ 'real-template golden renders (test-supported-template.cpp, 10 templates x 3 contexts)'
golden_check ('tests/j/templates/meta-llama-Llama-3.1-8B-Instruct.jinja' ; 'reference/minja/tests/contexts/simple.json'   ; 'tests/j/goldens/meta-llama-Llama-3.1-8B-Instruct-simple.txt'   ; 'Llama-3.1 simple')
golden_check ('tests/j/templates/meta-llama-Llama-3.1-8B-Instruct.jinja' ; 'reference/minja/tests/contexts/system.json'   ; 'tests/j/goldens/meta-llama-Llama-3.1-8B-Instruct-system.txt'   ; 'Llama-3.1 system')
golden_check ('tests/j/templates/meta-llama-Llama-3.1-8B-Instruct.jinja' ; 'reference/minja/tests/contexts/tool_use.json' ; 'tests/j/goldens/meta-llama-Llama-3.1-8B-Instruct-tool_use.txt' ; 'Llama-3.1 tool_use')
golden_check ('tests/j/templates/meta-llama-Llama-3.2-3B-Instruct.jinja' ; 'reference/minja/tests/contexts/simple.json'   ; 'tests/j/goldens/meta-llama-Llama-3.2-3B-Instruct-simple.txt'   ; 'Llama-3.2 simple')
golden_check ('tests/j/templates/meta-llama-Llama-3.2-3B-Instruct.jinja' ; 'reference/minja/tests/contexts/system.json'   ; 'tests/j/goldens/meta-llama-Llama-3.2-3B-Instruct-system.txt'   ; 'Llama-3.2 system')
golden_check ('tests/j/templates/meta-llama-Llama-3.2-3B-Instruct.jinja' ; 'reference/minja/tests/contexts/tool_use.json' ; 'tests/j/goldens/meta-llama-Llama-3.2-3B-Instruct-tool_use.txt' ; 'Llama-3.2 tool_use')
golden_check ('tests/j/templates/mistralai-Mistral-Nemo-Instruct-2407.jinja' ; 'reference/minja/tests/contexts/simple.json'   ; 'tests/j/goldens/mistralai-Mistral-Nemo-Instruct-2407-simple.txt'   ; 'Mistral-Nemo simple')
golden_check ('tests/j/templates/mistralai-Mistral-Nemo-Instruct-2407.jinja' ; 'reference/minja/tests/contexts/system.json'   ; 'tests/j/goldens/mistralai-Mistral-Nemo-Instruct-2407-system.txt'   ; 'Mistral-Nemo system')
golden_check ('tests/j/templates/mistralai-Mistral-Nemo-Instruct-2407.jinja' ; 'reference/minja/tests/contexts/tool_use.json' ; 'tests/j/goldens/mistralai-Mistral-Nemo-Instruct-2407-tool_use.txt' ; 'Mistral-Nemo tool_use')
golden_check ('tests/j/templates/mistralai-Ministral-3-14B-Reasoning-2512.jinja' ; 'reference/minja/tests/contexts/simple.json'   ; 'tests/j/goldens/mistralai-Ministral-3-14B-Reasoning-2512-simple.txt'   ; 'Ministral-3 simple')
golden_check ('tests/j/templates/mistralai-Ministral-3-14B-Reasoning-2512.jinja' ; 'reference/minja/tests/contexts/system.json'   ; 'tests/j/goldens/mistralai-Ministral-3-14B-Reasoning-2512-system.txt'   ; 'Ministral-3 system')
golden_check ('tests/j/templates/mistralai-Ministral-3-14B-Reasoning-2512.jinja' ; 'reference/minja/tests/contexts/tool_use.json' ; 'tests/j/goldens/mistralai-Ministral-3-14B-Reasoning-2512-tool_use.txt' ; 'Ministral-3 tool_use')
golden_check ('tests/j/templates/CohereForAI-c4ai-command-r-plus-tool_use.jinja' ; 'reference/minja/tests/contexts/simple.json'   ; 'tests/j/goldens/CohereForAI-c4ai-command-r-plus-tool_use-simple.txt'   ; 'Command-r+ simple')
golden_check ('tests/j/templates/CohereForAI-c4ai-command-r-plus-tool_use.jinja' ; 'reference/minja/tests/contexts/system.json'   ; 'tests/j/goldens/CohereForAI-c4ai-command-r-plus-tool_use-system.txt'   ; 'Command-r+ system')
golden_check ('tests/j/templates/CohereForAI-c4ai-command-r-plus-tool_use.jinja' ; 'reference/minja/tests/contexts/tool_use.json' ; 'tests/j/goldens/CohereForAI-c4ai-command-r-plus-tool_use-tool_use.txt' ; 'Command-r+ tool_use')
golden_check ('tests/j/templates/Qwen-QwQ-32B.jinja' ; 'reference/minja/tests/contexts/simple.json'   ; 'tests/j/goldens/Qwen-QwQ-32B-simple.txt'   ; 'QwQ-32B simple')
golden_check ('tests/j/templates/Qwen-QwQ-32B.jinja' ; 'reference/minja/tests/contexts/system.json'   ; 'tests/j/goldens/Qwen-QwQ-32B-system.txt'   ; 'QwQ-32B system')
golden_check ('tests/j/templates/Qwen-QwQ-32B.jinja' ; 'reference/minja/tests/contexts/tool_use.json' ; 'tests/j/goldens/Qwen-QwQ-32B-tool_use.txt' ; 'QwQ-32B tool_use')
golden_check ('tests/j/templates/Qwen-Qwen3-Coder-30B-A3B-Instruct.jinja' ; 'reference/minja/tests/contexts/simple.json'   ; 'tests/j/goldens/Qwen-Qwen3-Coder-30B-A3B-Instruct-simple.txt'   ; 'Qwen3-Coder simple')
golden_check ('tests/j/templates/Qwen-Qwen3-Coder-30B-A3B-Instruct.jinja' ; 'reference/minja/tests/contexts/system.json'   ; 'tests/j/goldens/Qwen-Qwen3-Coder-30B-A3B-Instruct-system.txt'   ; 'Qwen3-Coder system')
golden_check ('tests/j/templates/Qwen-Qwen3-Coder-30B-A3B-Instruct.jinja' ; 'reference/minja/tests/contexts/tool_use.json' ; 'tests/j/goldens/Qwen-Qwen3-Coder-30B-A3B-Instruct-tool_use.txt' ; 'Qwen3-Coder tool_use')
golden_check ('tests/j/templates/zai-org-GLM-4.6.jinja' ; 'reference/minja/tests/contexts/simple.json'   ; 'tests/j/goldens/zai-org-GLM-4.6-simple.txt'   ; 'GLM-4.6 simple')
golden_check ('tests/j/templates/zai-org-GLM-4.6.jinja' ; 'reference/minja/tests/contexts/system.json'   ; 'tests/j/goldens/zai-org-GLM-4.6-system.txt'   ; 'GLM-4.6 system')
golden_check ('tests/j/templates/zai-org-GLM-4.6.jinja' ; 'reference/minja/tests/contexts/tool_use.json' ; 'tests/j/goldens/zai-org-GLM-4.6-tool_use.txt' ; 'GLM-4.6 tool_use')
golden_check ('tests/j/templates/synthetic-deepseek-v3.2-dsml.jinja' ; 'reference/minja/tests/contexts/simple.json'   ; 'tests/j/goldens/synthetic-deepseek-v3.2-dsml-simple.txt'   ; 'deepseek-v3.2 simple')
golden_check ('tests/j/templates/synthetic-deepseek-v3.2-dsml.jinja' ; 'reference/minja/tests/contexts/system.json'   ; 'tests/j/goldens/synthetic-deepseek-v3.2-dsml-system.txt'   ; 'deepseek-v3.2 system')
golden_check ('tests/j/templates/synthetic-deepseek-v3.2-dsml.jinja' ; 'reference/minja/tests/contexts/tool_use.json' ; 'tests/j/goldens/synthetic-deepseek-v3.2-dsml-tool_use.txt' ; 'deepseek-v3.2 tool_use')
golden_check ('tests/j/templates/meetkai-functionary-medium-v3.2.jinja' ; 'reference/minja/tests/contexts/simple.json'   ; 'tests/j/goldens/meetkai-functionary-medium-v3.2-simple.txt'   ; 'functionary-v3.2 simple')
golden_check ('tests/j/templates/meetkai-functionary-medium-v3.2.jinja' ; 'reference/minja/tests/contexts/system.json'   ; 'tests/j/goldens/meetkai-functionary-medium-v3.2-system.txt'   ; 'functionary-v3.2 system')
golden_check ('tests/j/templates/meetkai-functionary-medium-v3.2.jinja' ; 'reference/minja/tests/contexts/tool_use.json' ; 'tests/j/goldens/meetkai-functionary-medium-v3.2-tool_use.txt' ; 'functionary-v3.2 tool_use')

show_summary_inference_ ''
