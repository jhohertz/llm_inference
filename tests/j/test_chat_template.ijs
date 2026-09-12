NB. ================================================================
NB. test_chat_template.ijs — chat-template engine cases (Phase 5Gb).
NB. Port of reference/minja/tests/test-polyfills.cpp inline cases
NB. (NoPolyFill, SystemRoleSupported, SystemRolePolyfill,
NB. ToolCallSupported, ToolCallPolyfill, ToolsPolyfill, ToolSupported,
NB. ToolPolyfill). Uses util/chat_template.ijs ct_new/ct_apply.
NB. ================================================================
coclass 'chatpl'

load './tests/j/test_harness.ijs'
load './util/minja.ijs'
load './util/chat_template.ijs'

init_counters_inference_ ''

NB. ravel-normalized string compare (LF is a rank-0 scalar)
streq =: 4 : '(x -: , y)'

NB. ---- template builders (C++ '\n' escapes to a newline; minja handles) ----
NB. q = single-quote char (J lexes ''' ambiguously)
q =: 39 { a.
T_CHATML =: '{%- for message in messages -%}' , LF , '  {{- ' , q , '<|im_start|>' , q , ' + message.role + ' , q , LF , q , ' + message.content + ' , q , '<|im_end|>' , LF , q , ' -}}' , LF , '{%- endfor -%}' , LF , '{%- if add_generation_prompt -%}' , LF , '  {{- ' , q , '<|im_start|>assistant' , LF , q , ' -}}' , LF , '{%- endif -%}'

T_CHATML_NOSYS =: '{%- for message in messages -%}' , LF , '  {%- if message.role == ' , q , 'system' , q , ' -%}' , LF , '    {{- raise_exception(' , q , 'System role not supported' , q , ') -}}' , LF , '  {%- endif -%}' , LF , '  {{- ' , q , '<|im_start|>' , q , ' + message.role + ' , q , LF , q , ' + message.content + ' , q , '<|im_end|>' , LF , q , ' -}}' , LF , '{%- endfor -%}' , LF , '{%- if add_generation_prompt -%}' , LF , '  {{- ' , q , '<|im_start|>assistant' , LF , q , ' -}}' , LF , '{%- endif -%}'

T_DUMMY =: '{%- for tool in tools -%}' , LF , '  {{- ' , q , 'tool: ' , q , ' + (tool | tojson(indent=2)) + ' , q , LF , q , '  -}}' , LF , '{%- endfor -%}' , LF , '{%- for message in messages -%}' , LF , '  {{- ' , q , 'message: ' , q , ' + (message | tojson(indent=2)) + ' , q , LF , q , ' -}}' , LF , '{%- endfor -%}' , LF , '{%- if add_generation_prompt -%}' , LF , '  {{- ' , q , 'message: ' , q , ' -}}' , LF , '{%- endif -%}'

NB. ---- message builders ----
ct_msg =: 3 : 0
  'role content' =. y
  mkobj_minja_ ((('role') pair_minja_ (mkstr_minja_ role)) , (('content') pair_minja_ content))
)

NB. ---- apply helper ----
NB. apply_ct (source ; caps ; messages ; tools ; addgen ; bos ; eos ; opts) -> prompt
apply_ct =: 3 : 0
  'src caps msgs tools addgen bos eos opts' =. y
  opts =. > > opts
  inputs =. ((<msgs) , (<tools) , (<addgen) , (<(mkobj_minja_ '')) , (<0) , (< > bos) , (< > eos))
  ct_apply ((<src) , (<inputs) , (<caps) , (<ct_tool_ex_g) , (<opts))
)

section_header_inference_ 'chat-template polyfill cases (test-polyfills.cpp)'

NB. ---- NoPolyFill (177) ----
tmpl =: T_CHATML
caps =: ct_new (tmpl ; '' ; '')
um =: ct_msg ((<'user') , <(mkstr_minja_ 'I need help'))
opts0 =: <(0 1 1 1 1 1 1 1 1 1 1)
msgs1 =: mkarr_minja_ (<um)
exp1 =: '<|im_start|>user' , LF , 'I need help<|im_end|>' , LF , '<|im_start|>assistant' , LF
got1 =: apply_ct ((<tmpl) , (<caps) , (<msgs1) , (<(mkarr_minja_ '')) , (<1) , (<'') , (<'')   , (<opts0))
assert_test_inference_ (((got1 streq exp1)) ; 'NoPolyFill add_gen=1')
NB. add_gen=false
got1b =: apply_ct ((<tmpl) , (<caps) , (<msgs1) , (<(mkarr_minja_ '')) , (<0) , (<'') , (<'')   , (<opts0))
exp1b =: '<|im_start|>user' , LF , 'I need help<|im_end|>' , LF
assert_test_inference_ (((got1b streq exp1b)) ; 'NoPolyFill add_gen=0')
NB. two messages
am =: ct_msg ((<'assistant') , <(mkstr_minja_ 'Hello, world!'))
msgs2 =: mkarr_minja_ ((<um) , <am)
got1c =: apply_ct ((<tmpl) , (<caps) , (<msgs2) , (<(mkarr_minja_ '')) , (<0) , (<'') , (<'')   , (<opts0))
exp1c =: '<|im_start|>user' , LF , 'I need help<|im_end|>' , LF , '<|im_start|>assistant' , LF , 'Hello, world!<|im_end|>' , LF
assert_test_inference_ (((got1c streq exp1c)) ; 'NoPolyFill two messages')

NB. ---- SystemRoleSupported (204) ----
sm =: ct_msg ((<'system') , <(mkstr_minja_ 'I am The System!'))
msgs3 =: mkarr_minja_ ((<sm) , (<um))
got2 =: apply_ct ((<tmpl) , (<caps) , (<msgs3) , (<(mkarr_minja_ '')) , (<1) , (<'') , (<'')   , (<(mk_options '')))
exp2 =: '<|im_start|>system' , LF , 'I am The System!<|im_end|>' , LF , '<|im_start|>user' , LF , 'I need help<|im_end|>' , LF , '<|im_start|>assistant' , LF
assert_test_inference_ (((got2 streq exp2)) ; 'SystemRoleSupported chatml')
NB. DUMMY template
dummy_caps =: ct_new (T_DUMMY ; '' ; '')
got2d =: apply_ct ((<T_DUMMY) , (<dummy_caps) , (<msgs3) , (<(mkarr_minja_ '')) , (<1) , (<'') , (<'')   , (<(mk_options '')))
exp2d =: 'message: {' , LF , '  "role": "system",' , LF , '  "content": "I am The System!"' , LF , '}' , LF , 'message: {' , LF , '  "role": "user",' , LF , '  "content": "I need help"' , LF , '}' , LF , 'message: '
assert_test_inference_ (((got2d streq exp2d)) ; 'SystemRoleSupported dummy')

NB. ---- SystemRolePolyfill (231) ----
nosys_caps =: ct_new (T_CHATML_NOSYS ; '' ; '')
got3 =: apply_ct ((<T_CHATML_NOSYS) , (<nosys_caps) , (<msgs3) , (<(mkarr_minja_ '')) , (<1) , (<'') , (<'')   , (<(mk_options '')))
exp3 =: '<|im_start|>user' , LF , 'I am The System!' , LF , 'I need help<|im_end|>' , LF , '<|im_start|>assistant' , LF
assert_test_inference_ (((got3 streq exp3)) ; 'SystemRolePolyfill')

NB. ---- ToolCallSupported (249) ----
fn1 =: mkobj_minja_ ((('name') pair_minja_ (mkstr_minja_ 'special_function')) , (('arguments') pair_minja_ (mkstr_minja_ '{"arg1": 1}')))
tc1 =: mkobj_minja_ ((('type') pair_minja_ (mkstr_minja_ 'function')) , (('function') pair_minja_ fn1) , (('id') pair_minja_ (mkstr_minja_ '123456789')))
am1 =: mkobj_minja_ ((('role') pair_minja_ (mkstr_minja_ 'assistant')) , (('content') pair_minja_ (mknull_minja_ '')) , (('tool_calls') pair_minja_ (mkarr_minja_ (<tc1))))
msgs4 =: mkarr_minja_ ((<um) , <am1)
got4 =: apply_ct ((<T_DUMMY) , (<dummy_caps) , (<msgs4) , (<(mkarr_minja_ '')) , (<1) , (<'') , (<'')   , (<(mk_options '')))
exp4 =: 'message: {' , LF , '  "role": "user",' , LF , '  "content": "I need help"' , LF , '}' , LF , 'message: {' , LF , '  "role": "assistant",' , LF , '  "content": null,' , LF , '  "tool_calls": [' , LF , '    {' , LF , '      "type": "function",' , LF , '      "function": {' , LF , '        "name": "special_function",' , LF , '        "arguments": {' , LF , '          "arg1": 1' , LF , '        }' , LF , '      },' , LF , '      "id": "123456789"' , LF , '    }' , LF , '  ]' , LF , '}' , LF , 'message: '
assert_test_inference_ (((got4 streq exp4)) ; 'ToolCallSupported')

NB. ---- ToolCallPolyfill (280) ----
got5 =: apply_ct ((<tmpl) , (<caps) , (<msgs4) , (<(mkarr_minja_ '')) , (<1) , (<'') , (<'')   , (<(mk_options '')))
exp5 =: '<|im_start|>user' , LF , 'I need help<|im_end|>' , LF , '<|im_start|>assistant' , LF , '{' , LF , '  "tool_calls": [' , LF , '    {' , LF , '      "name": "special_function",' , LF , '      "arguments": {' , LF , '        "arg1": 1' , LF , '      },' , LF , '      "id": "123456789"' , LF , '    }' , LF , '  ]' , LF , '}<|im_end|>' , LF , '<|im_start|>assistant' , LF
assert_test_inference_ (((got5 streq exp5)) ; 'ToolCallPolyfill')

NB. ---- ToolsPolyfill (305) ----
eos_caps =: ct_new (tmpl ; '' ; '<|im_end|>')
arg1p =: mkobj_minja_ ((('type') pair_minja_ (mkstr_minja_ 'integer')) , (('description') pair_minja_ (mkstr_minja_ 'The arg.')))
params2 =: mkobj_minja_ ((('type') pair_minja_ (mkstr_minja_ 'object')) , (('properties') pair_minja_ (mkobj_minja_ ((<'arg1') , <arg1p))) , (('required') pair_minja_ (mkarr_minja_ (<(mkstr_minja_ 'arg1')))))
fn2 =: mkobj_minja_ ((('name') pair_minja_ (mkstr_minja_ 'special_function')) , (('description') pair_minja_ (mkstr_minja_ 'I''m special')) , (('parameters') pair_minja_ params2))
tool2 =: mkobj_minja_ ((('type') pair_minja_ (mkstr_minja_ 'function')) , (('function') pair_minja_ fn2))
msgs5 =: mkarr_minja_ (<um)
tools5 =: mkarr_minja_ (<tool2)
got6 =: apply_ct ((<tmpl) , (<eos_caps) , (<msgs5) , (<tools5) , (<1) , (<'') , (<'<|im_end|>')   , (<(mk_options '')))
exp6 =: '<|im_start|>system' , LF , 'You can call any of the following tools to satisfy the user''s requests: [' , LF , '  {' , LF , '    "type": "function",' , LF , '    "function": {' , LF , '      "name": "special_function",' , LF , '      "description": "I''m special",' , LF , '      "parameters": {' , LF , '        "type": "object",' , LF , '        "properties": {' , LF , '          "arg1": {' , LF , '            "type": "integer",' , LF , '            "description": "The arg."' , LF , '          }' , LF , '        },' , LF , '        "required": [' , LF , '          "arg1"' , LF , '        ]' , LF , '      }' , LF , '    }' , LF , '  }' , LF , ']' , LF , '' , LF , 'Example tool call syntax:' , LF , '' , LF , '{' , LF , '  "tool_calls": [' , LF , '    {' , LF , '      "name": "tool_name",' , LF , '      "arguments": {' , LF , '        "arg1": "some_value"' , LF , '      },' , LF , '      "id": "call_1___"' , LF , '    }' , LF , '  ]' , LF , '}' , LF , '' , LF , '<|im_end|>' , LF , '<|im_start|>user' , LF , 'I need help<|im_end|>' , LF , '<|im_start|>assistant' , LF
assert_test_inference_ (((got6 streq exp6)) ; 'ToolsPolyfill')

NB. ---- ToolSupported (355) ----
tm2 =: mkobj_minja_ ((('role') pair_minja_ (mkstr_minja_ 'tool')) , (('content') pair_minja_ (mkobj_minja_ ((<'result') , <(mkint_minja_ 123)))) , (('tool_call_id') pair_minja_ (mkstr_minja_ '123456789')))
msgs6 =: mkarr_minja_ (<tm2)
got7 =: apply_ct ((<T_DUMMY) , (<dummy_caps) , (<msgs6) , (<(mkarr_minja_ '')) , (<1) , (<'') , (<'')   , (<(mk_options '')))
exp7 =: 'message: {' , LF , '  "role": "tool",' , LF , '  "content": {' , LF , '    "result": 123' , LF , '  },' , LF , '  "tool_call_id": "123456789"' , LF , '}' , LF , 'message: '
assert_test_inference_ (((got7 streq exp7)) ; 'ToolSupported')

NB. ---- ToolPolyfill (373) ----
got8 =: apply_ct ((<T_CHATML_NOSYS) , (<nosys_caps) , (<msgs6) , (<(mkarr_minja_ '')) , (<1) , (<'') , (<'')   , (<(mk_options '')))
exp8 =: '<|im_start|>user' , LF , '{' , LF , '  "tool_response": {' , LF , '    "content": {' , LF , '      "result": 123' , LF , '    },' , LF , '    "tool_call_id": "123456789"' , LF , '  }' , LF , '}<|im_end|>' , LF , '<|im_start|>assistant' , LF
assert_test_inference_ (((got8 streq exp8)) ; 'ToolPolyfill')

NB. ---- SystemRolePolyfill throws with no polyfills (237) ----
throw_test =: 3 : 0
  try.
    apply_ct ((<T_CHATML_NOSYS) , (<nosys_caps) , (<msgs3) , (<(mkarr_minja_ '')) , (<1) , (<'') , (<'')   , (<opts0))
    assert_test_inference_ ((<0) ; 'SystemRolePolyfill throws (no polyfills)')
  catcht. 'minja'
    assert_test_inference_ ((((err_msg_g_minja_ i. 'System role not supported') < # err_msg_g_minja_)) ; 'SystemRolePolyfill throws')
  end.
)
throw_test 0

NB. ---- strftime / strftime_now (5Gc) ----
section_header_inference_ 'strftime / strftime_now'

NB. direct strftime_s tests (minja locale verb; UTC; epoch seconds)
assert_test_inference_ (((strftime_s_minja_ ('%Y-%m-%d %H:%M:%S' ; 0)) streq '1970-01-01 00:00:00') ; 'strftime epoch0')
assert_test_inference_ (((strftime_s_minja_ ('%d %b %Y' ; 1721952000)) streq '26 Jul 2024') ; 'strftime %d %b %Y')
assert_test_inference_ (((strftime_s_minja_ ('%Y-%m-%d %H:%M:%S' ; 1721952000)) streq '2024-07-26 00:00:00') ; 'strftime full datetime')
assert_test_inference_ (((strftime_s_minja_ ('%a %A %b %B %e %j' ; 1721952000)) streq 'Fri Friday Jul July 26 208') ; 'strftime weekday/month/doy')
assert_test_inference_ (((strftime_s_minja_ ('%I %p' ; (12 * 3600))) streq '12 PM') ; 'strftime %I noon')
assert_test_inference_ (((strftime_s_minja_ ('%I %p' ; 0)) streq '12 AM') ; 'strftime %I midnight')
assert_test_inference_ (((strftime_s_minja_ ('%y %F %T %%' ; 1721952000)) streq '24 2024-07-26 00:00:00 %') ; 'strftime %y %F %T %%')

NB. strftime_now template render through ct_apply (pinned now=1721952000 = 2024-07-26 UTC)
T_SF =: '{{ strftime_now(' , q , '%Y-%m-%d %H:%M:%S' , q , ') }}'
T_SF2 =: '{{ strftime_now(' , q , '%d %b %Y' , q , ') }}'
caps_sf =: ct_new (T_SF ; '' ; '')
msgs0 =: mkarr_minja_ ''
inputs_sf =: ((<msgs0) , (<(mkarr_minja_ '')) , (<0) , (<(mkobj_minja_ '')) , (<1721952000) , (<'') , (<''))
got_sf =: ct_apply ((<T_SF) , (<inputs_sf) , (<caps_sf) , (<'') , (<opts0))
assert_test_inference_ (((got_sf streq '2024-07-26 00:00:00')) ; 'strftime_now %Y-%m-%d %H:%M:%S (pinned)')
got_sf2 =: ct_apply ((<T_SF2) , (<inputs_sf) , (<caps_sf) , (<'') , (<opts0))
assert_test_inference_ (((got_sf2 streq '26 Jul 2024')) ; 'strftime_now %d %b %Y (pinned)')

NB. ---- capability flags (test-capabilities.cpp style) ----
section_header_inference_ 'capability flags'
caps_c =: ct_new (T_CHATML ; '' ; '')
caps_n =: ct_new (T_CHATML_NOSYS ; '' ; '')
caps_d =: ct_new (T_DUMMY ; '' ; '')
assert_test_inference_ (((to_bool_minja_ ('supports_system_role' obj_get_minja_ caps_c)) = 1) ; 'caps CHATML supports_system_role')
assert_test_inference_ (((to_bool_minja_ ('supports_tools' obj_get_minja_ caps_c)) = 0) ; 'caps CHATML supports_tools')
assert_test_inference_ (((to_bool_minja_ ('supports_tool_calls' obj_get_minja_ caps_c)) = 0) ; 'caps CHATML supports_tool_calls')
assert_test_inference_ (((to_bool_minja_ ('supports_tool_responses' obj_get_minja_ caps_c)) = 0) ; 'caps CHATML supports_tool_responses')
assert_test_inference_ (((to_bool_minja_ ('requires_object_arguments' obj_get_minja_ caps_c)) = 0) ; 'caps CHATML requires_object_arguments')
assert_test_inference_ (((to_bool_minja_ ('supports_system_role' obj_get_minja_ caps_n)) = 0) ; 'caps CHATML_NOSYS supports_system_role')
assert_test_inference_ (((to_bool_minja_ ('supports_system_role' obj_get_minja_ caps_d)) = 1) ; 'caps DUMMY supports_system_role')
assert_test_inference_ (((to_bool_minja_ ('supports_tools' obj_get_minja_ caps_d)) = 1) ; 'caps DUMMY supports_tools')
assert_test_inference_ (((to_bool_minja_ ('supports_tool_calls' obj_get_minja_ caps_d)) = 1) ; 'caps DUMMY supports_tool_calls')
assert_test_inference_ (((to_bool_minja_ ('supports_tool_responses' obj_get_minja_ caps_d)) = 1) ; 'caps DUMMY supports_tool_responses')
assert_test_inference_ (((to_bool_minja_ ('requires_object_arguments' obj_get_minja_ caps_d)) = 1) ; 'caps DUMMY requires_object_arguments')

NB. ---- real-template detect_caps (test-capabilities.cpp port, 10 templates) ----
NB. caps_check (path ; expected) -> assert all 10 flags vs upstream expected bits.
NB. expected = 10-bit list: sys tools tc tcid tr parallel objargs nonnull nonempty typed.
section_header_inference_ 'real-template detect_caps (test-capabilities.cpp)'
caps_check =: 3 : 0
  'path exp' =. y
  src =. 1!:1 < path
  caps =. ct_new (src ; '' ; '')
  flds =. ;: 'supports_system_role supports_tools supports_tool_calls supports_tool_call_id supports_tool_responses supports_parallel_tool_calls requires_object_arguments requires_non_null_content requires_non_empty_content requires_typed_content'
  for_i. i. # flds do.
    got =. to_bool_minja_ (i { flds) obj_get_minja_ caps
    assert_test_inference_ ((got = (i { exp)) ; (path , ' ' , (> i { flds)))
  end.
)
caps_check ('tests/j/templates/meta-llama-Llama-3.1-8B-Instruct.jinja' ; 1 1 1 0 1 0 1 0 0 0)
caps_check ('tests/j/templates/meta-llama-Llama-3.2-3B-Instruct.jinja' ; 1 1 1 0 1 0 1 0 0 0)
caps_check ('tests/j/templates/mistralai-Mistral-Nemo-Instruct-2407.jinja' ; 1 1 1 1 1 1 1 0 0 0)
caps_check ('tests/j/templates/mistralai-Ministral-3-14B-Reasoning-2512.jinja' ; 1 1 1 0 1 1 0 1 1 0)
caps_check ('tests/j/templates/CohereForAI-c4ai-command-r-plus-tool_use.jinja' ; 1 1 1 0 1 1 1 0 0 0)
caps_check ('tests/j/templates/Qwen-QwQ-32B.jinja' ; 1 1 1 0 1 1 1 1 0 0)
caps_check ('tests/j/templates/Qwen-Qwen3-Coder-30B-A3B-Instruct.jinja' ; 1 1 1 0 1 1 1 0 0 0)
caps_check ('tests/j/templates/zai-org-GLM-4.6.jinja' ; 1 1 1 0 1 1 1 0 0 0)
caps_check ('tests/j/templates/synthetic-deepseek-v3.2-dsml.jinja' ; 1 0 1 0 1 1 1 0 0 0)
caps_check ('tests/j/templates/meetkai-functionary-medium-v3.2.jinja' ; 1 1 1 0 1 1 0 0 0 0)

NB. ---- ToolTest: real templates + single tool message (test-polyfills.cpp ToolTest) ----
NB. message_tool = {"role":"tool","content":{"result":123},"tool_call_id":"123456789"}.
NB. Default inputs (no tools, no generation prompt, no system); C++ default now=clock,
NB. so the recorded date-bearing outputs ("26 Jul 2024") require pinning now=1721952000.
section_header_inference_ 'ToolTest (real templates + tool message)'
ct_msg_tool =: 3 : 0
  content =. mkobj_minja_ ((<'result') , <(mkint_minja_ 123))
  mkobj_minja_ ((('role') pair_minja_ (mkstr_minja_ 'tool')) , (('content') pair_minja_ content) , (('tool_call_id') pair_minja_ (mkstr_minja_ '123456789')))
)
apply_tooltest =: 3 : 0
  'path exp' =. y
  src =. 1!:1 < path
  caps =. ct_new (src ; '' ; '')
  msgs =. mkarr_minja_ (<(ct_msg_tool ''))
  NB. C++ default inputs: tools = JSON null (not empty array), add_generation_prompt = true.
  inputs =. ((<msgs) , (<(mknull_minja_ '')) , (<1) , (<(mkobj_minja_ '')) , (<1721952000) , (<'') , (<''))
  got =. ct_apply ((<src) , (<inputs) , (<caps) , (<ct_tool_ex_g) , (<(mk_options '')))
  assert_test_inference_ ((got streq exp) ; path)
)
NB. MistralNemo (448): no date in output, but pin now for uniformity.
apply_tooltest ('tests/j/templates/mistralai-Mistral-Nemo-Instruct-2407.jinja' ; '[TOOL_RESULTS]{"content": {' , q , 'result' , q , ': 123}, "call_id": "123456789"}[/TOOL_RESULTS]')
NB. Llama3_3 (497) — template is Llama-3.1-8B-Instruct; date pinned.
apply_tooltest ('tests/j/templates/meta-llama-Llama-3.1-8B-Instruct.jinja' ; '<|start_header_id|>system<|end_header_id|>' , LF , LF , 'Cutting Knowledge Date: December 2023' , LF , 'Today Date: 26 Jul 2024' , LF , LF , '<|eot_id|><|start_header_id|>ipython<|end_header_id|>' , LF , LF , '{"result": 123}<|eot_id|><|start_header_id|>assistant<|end_header_id|>' , LF , LF)
NB. MeetkaiFunctionary3_2 (535)
apply_tooltest ('tests/j/templates/meetkai-functionary-medium-v3.2.jinja' ; '<|start_header_id|>system<|end_header_id|>' , LF , LF , 'You are capable of executing available function(s) if required.' , LF , 'Only execute function(s) when absolutely necessary.' , LF , 'Ask for the required input to:recipient==all' , LF , 'Use JSON for function arguments.' , LF , 'Respond in this format:' , LF , '>>>${recipient}' , LF , '${content}' , LF , 'Available functions:' , LF , '// Supported function definitions that should be called when necessary.' , LF , 'namespace functions {' , LF , LF , '} // namespace functions<|eot_id|><|start_header_id|>tool<|end_header_id|>' , LF , LF , '{' , q , 'result' , q , ': 123}<|eot_id|><|start_header_id|>assistant<|end_header_id|>' , LF , LF , '>>>')

NB. ---- DeepSeekR1 (393): UTF-8 tool-output markers (｜ U+FF5C, ▁ U+2581) ----
apply_tooltest ('tests/j/templates/deepseek-ai-DeepSeek-R1-Distill-Llama-70B.jinja' ; '<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>{' , q , 'result' , q , ': 123}<｜tool▁output▁end｜><｜tool▁outputs▁end｜>')

NB. ---- MeetkaiFunctionary3_1 (516): no "Today Date" line (3.1 template) ----
apply_tooltest ('tests/j/templates/meetkai-functionary-medium-v3.1.jinja' ; '<|start_header_id|>system<|end_header_id|>' , LF , LF , LF , 'Cutting Knowledge Date: December 2023' , LF , LF , '<|eot_id|><|start_header_id|>ipython<|end_header_id|>' , LF , LF , '{' , q , 'result' , q , ': 123}<|eot_id|><|start_header_id|>assistant<|end_header_id|>' , LF , LF)


NB. ---- NousResearchHermes3 (459) / NousResearchHermes2 (478): same template + message -> same prompt ----
apply_tooltest ('tests/j/templates/NousResearch-Hermes-3-Llama-3.1-70B-tool_use.jinja' ; '<|im_start|>system' , LF , 'You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don''t make assumptions about what values to plug into functions. Here are the available tools: <tools>  </tools>Use the following pydantic model json schema for each tool call you will make: {"properties": {"name": {"title": "Name", "type": "string"}, "arguments": {"title": "Arguments", "type": "object"}}, "required": ["name", "arguments"], "title": "FunctionCall", "type": "object"}}' , LF , 'For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:' , LF , '<tool_call>' , LF , '{"name": <function-name>, "arguments": <args-dict>}' , LF , '</tool_call><|im_end|>' , LF , '<tool_response>' , LF , '{''result'': 123}' , LF , '</tool_response><|im_end|><|im_start|>assistant' , LF)
show_summary_inference_ ''
