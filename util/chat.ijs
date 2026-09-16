NB. ================================================================
NB. chat.ijs — chat-template inference (Phase 1.1)
NB. Renders each arch's chat template, tokenizes with special-token encoding,
NB. generates until the arch's stop tokens, returns the model's answer.
NB. Single-turn and rudimentary multi-turn (re-render full history per turn).
NB.
NB. Depends on: llm_core.ijs + the arch modules (gemma3/qwen2/smollm2).
NB. Usage (after load 'llm/inference'):
NB.   llm chat_generate_inference_ (messages ; max_steps ; <temp;k;p;min_p>) -> text
NB.   llm chat_generate_simple_inference_ (messages ; max_steps)
NB.   messages = boxed list of message boxes; each message = <role ; content>
NB.   built with (role) ; content  — e.g. ('user') ; 'The capital of France is'
NB. ================================================================
coclass 'inference'
require 'llm/inference/util/minja'
require 'llm/inference/util/chat_template'
require 'convert/pjson'   NB. tool-call JSON extraction (dec/enc)

NB. ---- Real GGUF chat-template + template variables (chat layer globals) ----
NB. ct_tmpl_g: the real jinja template pulled from the GGUF by the arch
NB. loader ('' = none -> bespoke fallback). Reset per load in inference.ijs.
NB. ct_vars_g: minja obj (Value) holding extra template variables
NB. (enable_thinking etc.), passed as `extra` to ct_apply.
ct_tmpl_g =: ''
ct_vars_g =: ''
NB. ct_tools_g: JSON string of tool definitions (OpenAI-style function schemas),
NB. passed to the template as the `tools` input. '' = no tools (renders null).
NB. Reset per call (chat_generate) / per load (inference.ijs).
ct_tools_g =: ''
NB. Optional epoch-seconds override for the `now` template variable (0 = use
NB. current time). Lets tests pin a stable date (e.g. 1721952000 = 26 Jul 2024).
ct_now_g =: 0

NB. ---- Dispatch helpers (arch string -> arch verb) ----
chat_prompt =: 4 : 0
  select. x
  case. 'gemma3' do. gem3_chat_prompt y
  case. 'qwen2'  do. qw2_chat_prompt y
  case. 'qwen3'  do. qw3_chat_prompt y
  case. 'qwen35' do. qw35_chat_prompt y
  case. 'llama'  do. llama_chat_prompt y
  case. 'granite' do. granite_chat_prompt y
  case. 'ernie4_5' do. ernie_chat_prompt y
  case. 'lfm2' do. lf2_chat_prompt y
  end.
)
chat_tokenize =: 4 : 0
  select. x
  case. 'gemma3' do. llama3_tokenize y
  case. 'qwen2'  do. gpt2_tokenize y
  case. 'qwen3'  do. gpt2_tokenize y
  case. 'qwen35' do. gpt2_tokenize y
  case. 'llama'  do. llama_tokenize y
  case. 'granite' do. granite_tokenize y
  case. 'ernie4_5' do. ernie_tokenize y
  case. 'lfm2' do. lf2_tokenize y
  end.
)
chat_detokenize =: 4 : 0
  select. x
  case. 'gemma3' do. llama3_detokenize y
  case. 'qwen2'  do. gpt2_detokenize y
  case. 'qwen3'  do. gpt2_detokenize y
  case. 'qwen35' do. gpt2_detokenize y
  case. 'llama'  do. llama_detokenize y
  case. 'granite' do. granite_detokenize y
  case. 'ernie4_5' do. ernie_detokenize y
  case. 'lfm2' do. lf2_detokenize y
  end.
)
chat_gen_loop =: 4 : 0
  llm =. x
  tokens =. > 0 { y
  max_steps =. > 1 { y
  temp =. > 2 { y
  k =. > 3 { y
  p =. > 4 { y
  min_p =. > 5 { y
  stop_list =. > 6 { y
  llm gen_loop_core (tokens ; '' ; max_steps ; temp ; k ; p ; min_p ; <stop_list)
)

NB. ================================================================
NB. Streaming incremental detokenizer (Phase 6 item 2)
NB. Port of llama.cpp's streaming detokenizer: accumulate each token's raw
NB. bytes, hold any incomplete trailing UTF-8 sequence, emit only complete
NB. characters. State globals st_buf_g (held bytes) + st_arch_g.
NB. ================================================================
st_buf_g =: ''
st_arch_g =: ''

chat_stream_reset =: 3 : 0
  st_buf_g =: ''
  st_arch_g =: ''
  ''
)

NB. ---- Per-arch token -> raw bytes (presentation: ▁ -> space) ----
NB. gpt2 family (qwen2/qwen3/qwen35/llama/granite/lfm2): byte_tab decode.
gpt2_tok_bytes =: 3 : 0
  llm_data =. > 0 { y
  token =. > 1 { y
  tokenizer =. llm_tokenizer llm_data
  vocab =. tokenizer_vocab_g tokenizer
  byte_tab =. tokenizer_byte tokenizer
  ts =. > token { vocab
  if. 0 = # ts do. '' return. end.
  cpts =. 3 u: 7 u: ts
  a. {~ byte_tab {~ cpts
)

NB. llama3 family (gemma3): byte token -> raw byte, else vocab string with ▁->space.
llama3_tok_bytes =: 3 : 0
  llm_data =. > 0 { y
  token =. > 1 { y
  tokenizer =. llm_tokenizer llm_data
  vocab =. tokenizer_vocab tokenizer
  if. (token >: 65536) *. (token < 65792) do.
    a. {~ (token - 65536)
  else.
    tok_sp (> token { vocab)
  end.
)

NB. spm family (ernie4_5): vocab string with ▁->space.
spm_tok_bytes =: 3 : 0
  llm_data =. > 0 { y
  token =. > 1 { y
  tokenizer =. spm_llm_tokenizer llm_data
  vocab =. spm_vocab tokenizer
  tok_sp (> token { vocab)
)

NB. Replace ▁ (U+2581, 3 bytes E2 96 81) with a single space in a token string.
tok_sp =: 3 : 0
  s =. y
  m =. '▁' E. s
  if. -. 1 e. m do. s return. end.
  res =. ''
  i =. 0
  while. i < # s do.
    if. i { m do.
      res =. res , ' '
      i =. i + 3
    else.
      res =. res , (i { s)
      i =. i + 1
    end.
  end.
  res
)

chat_tok_bytes =: 4 : 0
  select. x
  case. 'gemma3' do. llama3_tok_bytes y
  case. 'qwen2';'qwen3';'qwen35';'llama';'granite';'lfm2' do. gpt2_tok_bytes y
  case. 'ernie4_5' do. spm_tok_bytes y
  end.
)

NB. ---- Number of trailing bytes forming an incomplete UTF-8 sequence ----
utf8_tail =: 3 : 0
  buf =. y
  n =. # buf
  if. 0 = n do. 0 return. end.
  b0 =. a. i. (n - 1) { buf
  if. b0 < 128 do. 0 return. end.
  NB. find the lead byte of the last multi-byte sequence
  j =. n - 1
  b =. b0
  while. (j >: 0) *. (b >: 128) *. (b < 192) do.
    j =. j - 1
    if. j < 0 do. break. end.
    b =. a. i. j { buf
  end.
  if. j < 0 do. 0 return. end.
  if. b < 192 do. 0 return. end.
  exp =. 1 + (b >: 224) + (b >: 240)
  have =. n - j
  if. have < exp do. have else. 0 end.
)

NB. ---- Emit the text delta for one token (streaming) ----
NB. x = arch; y = <llm ; token>. Appends the token's bytes to st_buf_g and
NB. returns the complete UTF-8 text emitted ('' if a char is still incomplete).
chat_stream_piece =: 4 : 0
  arch =. x
  llm =. > 0 { y
  token =. > 1 { y
  st_arch_g =: arch
  bytes =. arch chat_tok_bytes (llm ; token)
  st_buf_g =: st_buf_g , bytes
  buf =. st_buf_g
  hold =. utf8_tail buf
  emit_n =. (# buf) - hold
  if. emit_n > 0 do.
    out =. emit_n {. buf
    st_buf_g =: emit_n }. buf
    out
  else.
    ''
  end.
)

NB. ---- Streaming chat completion callback wiring ----
NB. chat_stream_cb is the per-token streaming callback the caller installs as
NB. gen_cb_g (with gen_cb_on_g=1). It reads arch/llm from globals, emits each
NB. token's text delta via chat_cb_g (a monadic verb on the delta string), and
NB. returns the token unchanged — gen_loop_core replaces pred with the return
NB. value, so the delta must NOT leak into the token stream.
NB.
NB. NOTE: do NOT capture a caller verb via `x =: gen_cb_g` then reassign
NB. gen_cb_g — J verb assignment is a dynamic ALIAS to the name, so rebinding
NB. gen_cb_g would make the "capture" track the new value (recursion).
chat_cb_g =: ]
chat_cb_arch_g =: ''
chat_cb_llm_g =: ''
chat_cb_stop_g =: ''

NB. y = token. Emits the token's text delta via chat_cb_g and returns the token
NB. unchanged. arch/llm come from the chat_cb_arch_g / chat_cb_llm_g globals.
NB. Stop tokens are suppressed: gen_loop_core breaks WITHOUT appending a stop
NB. token to output, so the batch detokenize excludes it — the streaming delta
NB. must too, or stream != batch (e.g. qwen3.5's <|im_end|> has a non-empty
NB. byte-encoded vocab string that would leak into the delta tail).
chat_stream_cb =: 3 : 0
  pred =. y
  if. (chat_cb_stop_g i. pred) < # chat_cb_stop_g do. pred return. end.
  delta =. chat_cb_arch_g chat_stream_piece (chat_cb_llm_g ; pred)
  if. 0 < # delta do. chat_cb_g delta end.
  pred
)
chat_default_params =: 3 : 0
  select. y
  case. 'gemma3' do. gem3_default_params
  case. 'qwen2'  do. qw2_default_params
  case. 'qwen3'  do. qw3_default_params
  case. 'qwen35' do. qw35_default_params
  case. 'llama'  do. llama_default_params
  case. 'granite' do. granite_default_params
  case. 'ernie4_5' do. ernie_default_params
  case. 'lfm2' do. lf2_default_params
  end.
)
chat_stop_tokens =: 3 : 0
  llm =. y
  arch =. llm_arch llm
  select. arch
  case. 'gemma3' do. gem3_stop_tokens llm
  case. 'qwen2'  do. qw2_stop_tokens llm
  case. 'qwen3'  do. qw3_stop_tokens llm
  case. 'qwen35' do. qw35_stop_tokens llm
  case. 'llama'  do. llama_stop_tokens llm
  case. 'granite' do. granite_stop_tokens llm
  case. 'ernie4_5' do. ernie_stop_tokens llm
  case. 'lfm2' do. lf2_stop_tokens llm
  end.
)

NB. ---- Chat arg parsing ----
NB. y = <messages ; max_steps ; <params> ; <tmpl_vars?> ; <tools?>  (<params> = <temp;k;p;min_p>, possibly double-boxed)
NB. Returns <messages; max_steps; temp; k; p; min_p; tmpl_vars; tools>
chat_args =: 3 : 0
  messages =. > 0 { y
  max_steps =. > 1 { y
  params =. > 2 { y
  if. 1 = # params do.
    flat =. > > params
  else.
    flat =. > params
  end.
  temp =. 0 { flat
  k =. 1 { flat
  p =. 2 { flat
  min_p =. 3 { flat
  tmpl_vars =. ''
  if. 3 < # y do. tmpl_vars =. > 3 { y end.
  tools =. ''
  if. 4 < # y do. tools =. > 4 { y end.
  (<messages) , (<max_steps) , (<temp) , (<k) , (<p) , (<min_p) , (<tmpl_vars) , (<tools)
)

NB. ---- Template variables: boxed list of (<key) ; <value -> minja obj ----
NB. Values may be strings (char), ints, or floats. Sets the chat-template
NB. variables (enable_thinking etc.) used by the real jinja render.
chat_vars_obj =: 3 : 0
  vars =. y
  o =. mkobj_minja_ ''
  for_i. i. # vars do.
    p =. > i { vars
    k =. > 0 { p
    v =. > 1 { p
    if. 2 = 3!:0 v do.
      mv =. mkstr_minja_ v
    elseif. 1 = 3!:0 v do.
      mv =. mkint_minja_ v
    else.
      mv =. mkfloat_minja_ v
    end.
    o =. ((<k) , <mv) obj_set_minja_ o
  end.
  o
)

NB. ---- Real GGUF chat-template render (shared across arches) ----
NB. y = messages: boxed list of <role ; content>. Renders the real jinja
NB. template stored in ct_tmpl_g (set by the arch loader from the GGUF) via the
NB. minja/chat_template port, with ct_vars_g as extra template variables.
NB. Returns the rendered prompt string ('' if no template).
days_from_civil =: 3 : 0
  'y m d' =. y
  y =. y - (m <: 2)
  era =. <. ((y - 399 * (y < 0)) % 400)
  yoe =. y - era * 400
  m2 =. m + 9 - 12 * (m > 2)
  doy =. (<. ((153 * m2) + 2) % 5) + d - 1
  doe =. ((yoe * 365) + (<. (yoe % 4))) - (<. (yoe % 100))
  doe =. doe + doy
  (era * 146097) + doe - 719468
)

chat_tmpl_render =: 3 : 0
  messages =. y
  if. 0 = # ct_tmpl_g do.
    throw. 'chat-template: model has no tokenizer.chat_template'
  end.
  vals =. ''
  for_i. i. # messages do.
    elem =. i { messages
    msg =. > elem
    if. ('obj') -: 0 {:: msg do.
      NB. pre-built minja message Value (tool_calls / tool_call_id / typed content)
      mv =. msg
    else.
      NB. <role ; content> simple message
      role =. 0 { msg
      content =. 1 { msg
      mv =. mkobj_minja_ ((('role') pair_minja_ (mkstr_minja_ role)) , (('content') pair_minja_ (mkstr_minja_ content)))
    end.
    vals =. vals , < mv
  end.
  msgs =. mkarr_minja_ vals
  extra =. ct_vars_g
  if. '' -: extra do. extra =. mkobj_minja_ '' end.
  now =. ct_now_g
  if. 0 = now do. now =. (days_from_civil (3 {. (6!:0 ''))) * 86400 end.
  tools =. ct_tools_g
  if. '' -: tools do. tools =. mknull_minja_ '' else. tools =. ct_parse_json_chatpl_ tools end.
  inputs =. ((<msgs) , (<tools) , (<1) , (<extra) , (<now) , (<'') , (<''))
  src =. ct_tmpl_g
  caps =. ct_new_chatpl_ (src ; '' ; '')
  ct_apply_chatpl_ ((<src) , (<inputs) , (<caps) , (<ct_tool_ex_g_chatpl_) , (<(mk_options_chatpl_ '')))
)

NB. ---- Chat generation ----
NB. llm chat_generate (messages ; max_steps ; <temp;k;p;min_p> ; <tmpl_vars?> ; <tools?>) -> answer text.
NB. tmpl_vars = boxed list of (<key) ; <value — extra jinja template variables
NB. (e.g. enable_thinking). tools = JSON string of tool definitions (the
NB. template's `tools` input). Renders the full message history (multi-turn),
NB. adds the generation prompt, and generates until the arch's stop tokens.
chat_generate =: 4 : 0
  llm =. x
  args =. chat_args y
  messages =. > 0 { args
  max_steps =. > 1 { args
  temp =. > 2 { args
  k =. > 3 { args
  p =. > 4 { args
  min_p =. > 5 { args
  tmpl_vars =. > 6 { args
  tools =. > 7 { args
  ct_vars_g =: chat_vars_obj tmpl_vars
  ct_tools_g =: tools

  arch =. llm_arch llm
  prompt =. arch chat_prompt messages
  tokens =. arch chat_tokenize (<llm) , <prompt
  stop =. chat_stop_tokens llm
  L =. # , > tokens

  output =. llm chat_gen_loop (tokens ; max_steps ; temp ; k ; p ; min_p ; <stop)
  gen =. L }. output   NB. drop the prompt tokens — answer only
  arch chat_detokenize (<llm) , <gen
)

NB. ---- Simple wrapper (per-arch default params) ----
NB. llm chat_generate_simple (messages ; max_steps)
chat_generate_simple =: 4 : 0
  llm =. x
  messages =. > 0 { y
  max_steps =. > 1 { y
  arch =. llm_arch llm
  params =. chat_default_params arch
  llm chat_generate (messages ; max_steps ; <params)
)

NB. ---- Tool-call extraction (Phase 6 item 3) ----
NB. chat_extract_tool_calls content -> boxed list of OpenAI-shaped tool_call
NB. minja Values: {type:'function'; function:<name; arguments-JSON-string>; id}.
NB. Detects each <tool_call>...</tool_call> region in the generated content and
NB. parses its body. Two generation formats are handled:
NB.   qwen35: <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>
NB.   qwen3 (JSON): <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>
NB. The JSON body is parsed with pjson (dec) and the arguments object is
NB. re-encoded to a JSON string (enc). Returns '' (empty) if no tool calls.
chat_parse_tool_call =: 3 : 0
  body =. y
  if. 1 e. '<function=' E. body do.   NB. if. reduces a boolean list with *./, so use 1 e.
    NB. qwen35 <function=> format
    s =. body I.@:E.~ '<function='
    rest =. (s + 10) }. body
    gt =. rest i. '>'
    name =. gt {. rest
    po =. body I.@:E.~ '<parameter='
    pc =. body I.@:E.~ '</parameter>'
    rows =. ''
    for_i. i. # po do.
      o =. i { po
      c =. i { pc
      seg =. (o + 11) }. (c {. body)
      g2 =. seg i. '>'
      key =. g2 {. seg
      nl =. seg i. LF
      val =. (nl + 1) }. seg
      val =. val -. LF
      val =. val -. CR
      rows =. rows , <(<key) , <val
    end.
    args =. enc_pjson_ (> rows)   NB. enc wants an n x 2 key/value table
  else.
    NB. qwen3 JSON format
    r =. dec_pjson_ body
    keys =. 0 {"1 r
    vals =. 1 {"1 r
    name =. > (keys i. <'name') { vals
    ai =. keys i. <'arguments'
    av =. > ai { vals
    if. 2 = 3!:0 av do.
      args =. av
    else.
      args =. enc_pjson_ av
    end.
  end.
  fn =. mkobj_minja_ ((('name') pair_minja_ (mkstr_minja_ name)) , (('arguments') pair_minja_ (mkstr_minja_ args)))
  tc =. mkobj_minja_ ((('type') pair_minja_ (mkstr_minja_ 'function')) , (('function') pair_minja_ fn) , (('id') pair_minja_ (mkstr_minja_ ('call_' , name))))
  tc
)

chat_extract_tool_calls =: 3 : 0
  content =. y
  opens =. content I.@:E.~ '<tool_call>'
  if. 0 = # opens do. '' return. end.
  closes =. content I.@:E.~ '</tool_call>'
  res =. ''
  for_i. i. # opens do.
    o =. i { opens
    c =. i { closes
    body =. ((c - (o + 11)) {. ((o + 11) }. content))   NB. '<tool_call>' is 11 chars
    tc =. chat_parse_tool_call body
    res =. res , < tc
  end.
  res
)

NB. ================================================================
NB. Streaming-first chat completion (OpenAI-shaped) — Phase 6 item 1
NB. The core formalization: request `messages` + `tools` + params, generate
NB. via the shared renderer/gen_loop_core, and return an OpenAI-shaped
NB. <content ; finish_reason ; tool_calls> response. Streaming is supported
NB. through the per-token callback global (gen_cb_g) — the same verb later
NB. serves a network OpenAI SSE endpoint.
NB. ================================================================
NB. llm chat_completion (messages ; tools ; max_steps ; stream ; <params>)
NB.   messages   = boxed list of <role ; content> (or pre-built minja Values)
NB.   tools      = JSON string of tool definitions ('' = none)
NB.   max_steps  = generation cap (token count)
NB.   stream     = 1 -> use the caller's gen_cb_g verb (set before calling) as
NB.                the per-token streaming callback; 0 -> collect the full answer
NB.                For text-delta streaming, set gen_cb_g = chat_stream_cb and
NB.                chat_cb_g to a monadic verb on each delta (chat_completion
NB.                provides the chat_cb_arch_g/chat_cb_llm_g globals). The caller
NB.                sets gen_cb_on_g = 1 itself — chat_completion never overwrites
NB.                the caller's callback (J verb assignment aliases by name).
NB.   <params>   = <temp;k;p;min_p> (possibly double-boxed). MUST be the last
NB.                operand: a pre-boxed `;` operand that isn't trailing nests
NB.                (J `;`-chain gotcha), so params is kept at the end.
NB. Returns <content ; finish_reason ; tool_calls>
NB.   content      = generated text (detokenized), '' when a tool call is pending
NB.   finish_reason= 'stop' | 'length' | 'tool_calls' (Phase 6 item 3: a
NB.                <tool_call> marker in the content classifies as 'tool_calls'
NB.                and the tool_calls are extracted; content is nulled)
NB.   tool_calls   = boxed list of OpenAI-shaped tool_call minja Values
NB.                ({type; function:<name; arguments-JSON>; id}), '' if none.
chat_completion =: 4 : 0
  llm =. x
  messages =. > 0 { y
  tools =. > 1 { y
  max_steps =. > 2 { y
  stream =. > 3 { y
  params =. > 4 { y
  if. 1 = # params do.
    flat =. > > params
  else.
    flat =. > params
  end.
  temp =. 0 { flat
  k =. 1 { flat
  p =. 2 { flat
  min_p =. 3 { flat
  ct_vars_g =: chat_vars_obj ''
  ct_tools_g =: tools

  arch =. llm_arch llm
  prompt =. arch chat_prompt messages
  tokens =. arch chat_tokenize (<llm) , <prompt
  stop =. chat_stop_tokens llm
  L =. # , > tokens

  if. stream do.
    NB. Streaming: the CALLER sets gen_cb_g (per-token callback) + gen_cb_on_g=1
    NB. before calling — e.g. gen_cb_g = chat_stream_cb with chat_cb_g as the
    NB. delta consumer. We provide the arch/llm globals chat_stream_cb needs and
    NB. reset the incremental detokenizer.
    chat_cb_arch_g =: arch
    chat_cb_llm_g =: llm
    chat_cb_stop_g =: stop
    chat_stream_reset ''
  else.
    gen_cb_on_g =: 0
  end.
  output =. llm gen_loop_core (tokens ; '' ; max_steps ; temp ; k ; p ; min_p ; <stop)
  gen_cb_on_g =: 0
  if. stream do.
    NB. flush any held incomplete UTF-8 so the delta stream is complete
    if. 0 < # st_buf_g do. chat_cb_g st_buf_g end.
    chat_stream_reset ''
  end.
  gen =. L }. output
  content =. arch chat_detokenize (<llm) , <gen
  finish =. 'stop'
  if. max_steps <: # gen do. finish =. 'length' end.
  NB. Phase 6 item 3: tool-call classification. If the generated content carries
  NB. a <tool_call> marker, classify as 'tool_calls', null the text content
  NB. (OpenAI convention), and extract the tool_calls (OpenAI-shaped).
  tcs =. chat_extract_tool_calls content
  if. 0 < # tcs do.
    finish =. 'tool_calls'
    content =. ''
  end.
  (<content) , (<finish) , (<tcs)
)

NB. ---- Simple wrapper (per-arch default params) ----
NB. llm chat_completion_simple (messages ; tools ; max_steps ; stream)
NB. Streaming callback via gen_cb_g (set before calling).
chat_completion_simple =: 4 : 0
  llm =. x
  messages =. > 0 { y
  tools =. > 1 { y
  max_steps =. > 2 { y
  stream =. > 3 { y
  arch =. llm_arch llm
  params =. chat_default_params arch
  llm chat_completion (messages ; tools ; max_steps ; stream ; <params)
)

NB. ---- Chat prompt helper: build one message = <role ; content> ----
NB. role chat_msg content
chat_msg =: 4 : 0
  (<x) , <y
)

NB. ================================================================
NB. Chat session (persistent multi-turn — option B)
NB. The crude console chat: `llm chat 'next message'` continues the
NB. conversation, reusing the KV cache + token stream across calls.
NB. ================================================================
NB. chat_session_g = '' (no session) or
NB.   <arch; messages; total_tokens; cur_pos; max_steps; params>
NB.   - messages: boxed list of <role ; content> message boxes
NB.   - total_tokens: boxed list of ALL tokens processed (prompt + generated)
NB.   - cur_pos: # total_tokens == KV write frontier
NB.   - params: <temp;k;p;min_p>
NB. The KV cache lives in kv_cache_g (the global); the session NEVER holds a
NB. cache reference — a second ref would defeat the in-place amend. One active
NB. session per J session; multi-session support can install a session's cache
NB. into kv_cache_g on switch later.
chat_session_g =: ''

NB. ---- path counters (tests: assert the resume path actually runs) ----
chat_resume_count =: 0
chat_fallback_count =: 0

NB. ---- Reset the chat session (clears session + KV cache) ----
chat_reset =: 3 : 0
  chat_session_g =: ''
  kv_reset ''
  ''
)

NB. ---- Fresh full-render chat turn (stateless helper) ----
NB. x = llm; y = <messages; max_steps; temp; k; p; min_p; flat>
NB. Renders the FULL message history, generates fresh, stores the session.
NB. Returns the answer text.
chat_fresh =: 4 : 0
  llm =. x
  messages =. > 0 { y
  max_steps =. > 1 { y
  temp =. > 2 { y
  k =. > 3 { y
  p =. > 4 { y
  min_p =. > 5 { y
  flat =. > 6 { y
  arch =. llm_arch llm
  stop =. chat_stop_tokens llm
  prompt =. arch chat_prompt messages
  tokens =. arch chat_tokenize (<llm) , <prompt
  L =. # , > tokens
  output =. llm gen_loop_core (tokens ; '' ; max_steps ; temp ; k ; p ; min_p ; <stop)
  gen =. L }. output
  answer =. arch chat_detokenize (<llm) , <gen
  NB. keep the assistant answer in the history (the next turn's re-render must
  NB. include it, else the prefix check fails and persistence can't engage)
  messages =. messages , <('assistant') ; answer
  chat_session_g =: (<arch) , (<messages) , (<output) , (<(# , > output)) , (<max_steps) , (<flat)
  answer
)

NB. ---- Core chat turn: persistent session if one exists ----
NB. x = llm; y = <msg; max_steps; <params>  (<params> = <temp;k;p;min_p>, possibly double-boxed)
chat_core =: 4 : 0
  llm =. x
  msg =. > 0 { y
  max_steps =. > 1 { y
  params =. > 2 { y
  if. 1 = # params do.
    flat =. > > params
  else.
    flat =. > params
  end.
  temp =. 0 { flat
  k =. 1 { flat
  p =. 2 { flat
  min_p =. 3 { flat
  NB. persistent chat takes no tmpl_vars/tools — clear any from chat_generate.
  ct_vars_g =: ''
  ct_tools_g =: ''
  arch =. llm_arch llm

  if. 0 = # chat_session_g do.
    NB. no session — start fresh with a single user message
    messages =. <('user') ; msg
    llm chat_fresh (messages ; max_steps ; temp ; k ; p ; min_p ; <flat)
  else.
    s =. chat_session_g
    s_arch =. > 0 { s
    if. -. s_arch -: arch do.
      NB. different model loaded — start over
      messages =. <('user') ; msg
      llm chat_fresh (messages ; max_steps ; temp ; k ; p ; min_p ; <flat)
    else.
      prev_messages =. > 1 { s
      prev_toks =. > 2 { s
      prev_len =. > 3 { s
      messages =. prev_messages , <('user') ; msg
      prompt =. arch chat_prompt messages
      tokens =. arch chat_tokenize (<llm) , <prompt
      tok_list =. , > tokens
      prev_flat =. , > prev_toks
      stop =. chat_stop_tokens llm
      if. (prev_len {. tok_list) -: prev_flat do.
        NB. re-render prefix matches the stored token stream -> resume from cache
        chat_resume_count =: chat_resume_count + 1
        seg =. prev_len }. tok_list
        L_seg =. # seg
        output =. llm gen_loop_core ((<"0 seg) ; prev_len ; max_steps ; temp ; k ; p ; min_p ; <stop)
        gen =. L_seg }. output
        answer =. arch chat_detokenize (<llm) , <gen
        total =. prev_toks , output
        messages =. messages , <('assistant') ; answer
        chat_session_g =: (<arch) , (<messages) , (<total) , (<(# , > total)) , (<max_steps) , (<flat)
        answer
      else.
        NB. tokenizer round-trip drift — fall back to a full fresh re-render
        NB. (correct, just slower); the session resets to the new stream.
        chat_fallback_count =: chat_fallback_count + 1
        llm chat_fresh (messages ; max_steps ; temp ; k ; p ; min_p ; <flat)
      end.
    end.
  end.
)

NB. ---- Crude console chat: llm chat 'next message' -> answer (default params) ----
NB. Persistent: the session + KV cache carry across calls until chat_reset ''.
chat =: 4 : 0
  llm =. x
  msg =. y
  arch =. llm_arch llm
  params =. chat_default_params arch
  llm chat_core (msg ; 100000 ; <params)
)

NB. ---- chat with explicit params: llm chat_p ('msg' ; <temp;k;p;min_p>) ----
chat_p =: 4 : 0
  llm =. x
  msg =. > 0 { y
  params =. > 1 { y
  llm chat_core (msg ; 100000 ; <params)
)
