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

NB. ---- Real GGUF chat-template + template variables ----
NB. ct_tmpl_g: the real jinja template pulled from the GGUF by the arch
NB. loader ('' = none -> bespoke fallback). Reset per load in inference.ijs.
NB. ct_vars/ct_tools now live in the session (fields 2/4); chat_tmpl_render
NB. reads them from sess_cur_g. ct_now_g is the pinned-date determinism knob
NB. (0 = live time; tests pin e.g. 1721952000 = 26 Jul 2024).
ct_tmpl_g =: ''
ct_now_g =: 0

NB. ---- Shared chat prompt ----
NB. chat_prompt: monadic verb; y = messages (boxed list of <role ; content>).
NB. The arch is implicit — each arch loader sets ct_tmpl_g (the real GGUF jinja
NB. template) from the model file, and chat_tmpl_render reads it. All 8 arch
NB. renderers are byte-identical, so one shared verb serves them; the per-arch
NB. names (gem3_chat_prompt etc.) are aliases so tests/direct callers keep
NB. working. (Called as `chat_prompt messages` — monadic; the arch is implicit.)
chat_prompt =: 3 : 0
  messages =. y
  chat_tmpl_render messages
)
gem3_chat_prompt =: chat_prompt
qw2_chat_prompt =: chat_prompt
qw3_chat_prompt =: chat_prompt
qw35_chat_prompt =: chat_prompt
llama_chat_prompt =: chat_prompt
granite_chat_prompt =: chat_prompt
ernie_chat_prompt =: chat_prompt
lf2_chat_prompt =: chat_prompt
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
NB. characters. The session's st_buf (field 5) holds the held bytes, st_arch
NB. (field 6) the arch.  sess_cur_g is the CURRENT session for the streaming
NB. callback; the serial path always sets it (via the _s verbs), so the
NB. session-aware piece_s is the only streaming path.
NB. ================================================================
sess_cur_g =: ''

NB. ---- Ensure the serial-path global session is a real session, synced from
NB. ---- the current kv globals (callers/tests set kv_max_seq_g etc.).  The
NB. ---- legacy globals (ct_*_g) are DEPRECATED shims
NB. ---- kept for tests/external callers; the session noun is the canonical state.
session_ensure =: 3 : 0
  if. 0 = # session do.
    session =: session_new ''
  end.
  session =: (<kv_seq_g) (9) } session
  session =: (<kv_pos_g) (10) } session
  session =: (<kv_batch_g) (11) } session
  session =: (<kv_max_seq_g) (12) } session
  session =: (<kv_meta) (13) } session
  ''
)

NB. ---- Session-aware streaming detokenizer (Stage 2) ----
NB.  A session-aware port of chat_stream_piece/chat_stream_reset that reads/writes
NB.  the session's st_buf (field 5) / st_arch (field 6).  The streaming CALLBACK
NB.  (chat_stream_cb) stays a global verb (verbs can't be boxed in the session);
NB.  the session-aware path is the building block for batching, where the caller
NB.  drives it per-session.  chat_stream_reset_s returns the updated session;
NB.  chat_stream_piece_s returns <delta ; updated_sess>.
chat_stream_reset_s =: 4 : 0
  sess =. x
  sess =. (<'') (5) } sess
  sess =. (<'') (6) } sess
  sess
)

NB. ---- Emit the text delta for one token, session-aware ----
NB.  x = sess; y = <arch ; llm ; token>.  Reads/writes the session's st_buf/st_arch
NB.  and returns <delta ; updated_sess> (delta = complete UTF-8 text, '' if a
NB.  char is still incomplete).
chat_stream_piece_s =: 4 : 0
  sess =. x
  arch =. > 0 { y
  llm =. > 1 { y
  token =. > 2 { y
  bytes =. arch chat_tok_bytes (llm ; token)
  buf =. (> 5 { sess) , bytes
  sess =. (<buf) (5) } sess
  sess =. (<arch) (6) } sess
  hold =. utf8_tail buf
  emit_n =. (# buf) - hold
  if. emit_n > 0 do.
    out =. emit_n {. buf
    sess =. (<(emit_n }. buf)) (5) } sess
    (<out) , <sess
  else.
    (<'') , <sess
  end.
)

NB. ---- Stream arming helpers (external-locale callers, e.g. the chat TUI) ----
NB. The callback globals gen_cb_on_g (noun) / gen_cb_g (verb) live in llm_core;
NB. the addon exports VERBS only (not nouns), so an external locale cannot set
NB. gen_cb_on_g via the _inference_ suffix. These verbs arm/disarm the globals.
NB. chat_stream_start sets gen_cb_on_g=1 + gen_cb_g=chat_stream_cb; the caller
NB. sets chat_cb_g (its delta consumer) separately — chat_stream_start never
NB. touches chat_cb_g (verb-alias: rebinding it would clobber the caller's).
chat_stream_start =: 3 : 0
  gen_cb_on_g =: 1
  gen_cb_g =: chat_stream_cb
  ''
)
chat_stream_stop =: 3 : 0
  gen_cb_on_g =: 0
  gen_cb_g =: ]
  chat_cb_g =: ]
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
  res =. sess_cur_g chat_stream_piece_s (chat_cb_arch_g ; chat_cb_llm_g ; pred)
  delta =. > 0 { res
  sess_cur_g =: > 1 { res
  if. 0 < # delta do. chat_cb_g delta end.
  pred
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
NB. minja/chat_template port, with session ct_vars/ct_tools (fields 2/4) as the
NB. extra template variables / tools input.
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
  extra =. ''
  tools =. ''
  if. 0 < # sess_cur_g do.
    extra =. sess_vars sess_cur_g
    tools =. sess_tools sess_cur_g
  end.
  now =. ct_now_g
  if. '' -: extra do. extra =. mkobj_minja_ '' end.
  if. 0 = now do. now =. (days_from_civil (3 {. (6!:0 ''))) * 86400 end.
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
  sess =. session_new ''
  sess =. (<(chat_vars_obj tmpl_vars)) (2) } sess
  sess =. (<tools) (4) } sess
  sess =. (<kv_seq_g) (9) } sess
  sess =. (<kv_pos_g) (10) } sess
  sess =. (<kv_batch_g) (11) } sess
  sess =. (<kv_max_seq_g) (12) } sess
  sess =. (<kv_meta) (13) } sess
  sess_cur_g =: sess

  arch =. llm_arch llm
  prompt =. chat_prompt messages
  tokens =. arch chat_tokenize (<llm) , <prompt
  stop =. chat_stop_tokens llm
  L =. # , > tokens

  output =. llm chat_gen_loop (tokens ; max_steps ; temp ; k ; p ; min_p ; <stop)
  sess_cur_g =: ''
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
  params =. llm_default_params llm
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
  fin =. mkobj_minja_ ((('name') pair_minja_ (mkstr_minja_ name)) , (('arguments') pair_minja_ (mkstr_minja_ args)))
  tc =. mkobj_minja_ ((('type') pair_minja_ (mkstr_minja_ 'function')) , (('function') pair_minja_ fin) , (('id') pair_minja_ (mkstr_minja_ ('call_' , name))))
  tc
)

chat_extract_tool_calls =: 3 : 0
  content =. y
  opens =. content I.@:E.~ '<tool_call>'
  if. 0 < # opens do.
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
  else.
    NB. No <tool_call> markers: try a BARE OpenAI-style JSON tool call. Some
    NB. models (qwen2.5-coder) emit {"name":..., "arguments":{...}} WITHOUT the
    NB. <tool_call></tool_call> wrapper their template asks for. If the whole
    NB. content is a JSON object carrying name + arguments keys, treat it as one
    NB. tool call (chat_parse_tool_call's JSON branch parses this same shape).
    if. (0 < # content) *. ('{' = {. content) do.
      r =. dec_pjson_ content
      keys =. 0 {"1 r
      if. ((<'name') e. keys) *. ((<'arguments') e. keys) do.
        < chat_parse_tool_call content
      else.
        ''
      end.
    else.
      ''
    end.
  end.
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
NB. ---- Shared post-processing (chat_completion / chat_completion_batch) ----
NB.  x = arch; y = <llm ; gen ; max_steps>.  Detokenizes gen (the post-prompt
NB.  token stream), sets finish_reason ('stop'/'length'/'tool_calls'), nulls
NB.  content on tool-calls.  Returns <content ; finish ; tcs>.  Both the serial
NB.  and batched chat completions use this identical block.
chat_postprocess =: 4 : 0
  arch =. x
  llm =. > 0 { y
  gen =. > 1 { y
  max_steps =. > 2 { y
  content =. arch chat_detokenize (<llm) , <gen
  finish =. 'stop'
  if. max_steps <: # gen do. finish =. 'length' end.
  NB. Phase 6 item 3: tool-call classification. A <tool_call> marker classifies
  NB. as 'tool_calls', nulls the text content, and extracts the tool_calls.
  tcs =. chat_extract_tool_calls content
  if. 0 < # tcs do.
    finish =. 'tool_calls'
    content =. ''
  end.
  (<content) , (<finish) , (<tcs)
)

chat_completion =: 4 : 0
  llm =. x
  messages =. > 0 { y
  tools =. > 1 { y
  max_steps =. > 2 { y
  stream =. > 3 { y
  params =. > 4 { y
  session_ensure ''
  res =. llm chat_completion_s ((<session) , (<messages) , (<tools) , (<max_steps) , (<stream) , (<params))
  session =: > 3 { res
  (<> 0 { res) , (<> 1 { res) , (<> 2 { res)
)

NB. ---- Session-aware chat_completion (Stage 2) ----
NB.  x = llm; y = <sess ; messages ; tools ; max_steps ; stream ; <params>.
NB.  Session-aware port of chat_completion: reads/writes the session's ct_*
NB.  (fields 1-4: ct_vars/ct_tools) + st_buf (field 5); the global callbacks
NB.  (gen_cb_g/chat_cb_g/chat_cb_arch_g/chat_cb_llm_g/chat_cb_stop_g) stay
NB.  global (verbs can't be boxed in the session).  Returns
NB.  <content ; finish_reason ; tool_calls ; updated_sess>.
chat_completion_s =: 4 : 0
  llm =. x
  sess =. > 0 { y
  messages =. > 1 { y
  tools =. > 2 { y
  max_steps =. > 3 { y
  stream =. > 4 { y
  params =. > 5 { y
  if. 1 = # params do.
    flat =. > > params
  else.
    flat =. > params
  end.
  temp =. 0 { flat
  k =. 1 { flat
  p =. 2 { flat
  min_p =. 3 { flat
  sess =. (<(chat_vars_obj '')) (2) } sess
  sess =. (<tools) (4) } sess

  arch =. llm_arch llm
  sess_cur_g =: sess
  prompt =. chat_prompt messages
  tokens =. arch chat_tokenize (<llm) , <prompt
  stop =. chat_stop_tokens llm
  L =. # , > tokens

  if. stream do.
    chat_cb_arch_g =: arch
    chat_cb_llm_g =: llm
    chat_cb_stop_g =: stop
    sess =. sess chat_stream_reset_s ''
    sess_cur_g =: sess
  else.
    gen_cb_on_g =: 0
  end.
  output =. llm gen_loop_core (tokens ; '' ; max_steps ; temp ; k ; p ; min_p ; <stop)
  gen_cb_on_g =: 0
  sess =. sess_cur_g
  sess_cur_g =: ''
  if. stream do.
    if. 0 < # sess_stbuf sess do. chat_cb_g sess_stbuf sess end.
    sess =. sess chat_stream_reset_s ''
  end.
  gen =. L }. output
  content =. arch chat_detokenize (<llm) , <gen
  finish =. 'stop'
  if. max_steps <: # gen do. finish =. 'length' end.
  tcs =. chat_extract_tool_calls content
  if. 0 < # tcs do.
    finish =. 'tool_calls'
    content =. ''
  end.
  (<content) , (<finish) , (<tcs) , <sess
)

NB. ---- Batched chat completion (Stage 3) ----
NB.  x = llm; y = boxed list of B request records, each =
NB.  <messages ; tools ; max_steps ; <params (temp;k;p;min_p)>>.
NB.  gen_loop_batch shares ONE (max_steps; temp; k; p; min_p) across a whole
NB.  batch, so records are grouped by IDENTICAL params; each group runs ONE
NB.  gen_loop_batch (one forward per decode step over the group's B sequences —
NB.  the B-axis KV cache).  Per record the prompt is rendered + tokenized with
NB.  the real jinja template (exactly as chat_completion) and the output
NB.  post-processed (strip prompt, detokenize, finish_reason, tool-call
NB.  classification).  Returns a boxed list of B <content ; finish ; tcs> in
NB.  INPUT ORDER.  kv_batch_g is reset to 1 on exit (the serial path expects it).
chat_completion_batch =: 4 : 0
  llm =. x
  recs =. y
  B =. # recs
  if. 0 = B do. '' return. end.
  arch =. llm_arch llm
  stop =. chat_stop_tokens llm
  NB. Pre-parse each record -> <msgs ; tools ; key ; temp ; k ; p ; min_p ; mx>.
  parsed =. ''
  i =. 0
  while. i < B do.
    r =. > i { recs
    msgs =. > 0 { r
    tools =. > 1 { r
    mx =. > 2 { r
    params =. > 3 { r
    if. 1 = # params do. flat =. > > params else. flat =. > params end.
    temp =. 0 { flat
    k =. 1 { flat
    p =. 2 { flat
    min_p =. 3 { flat
    key =. (": mx) , '|' , (": temp) , '|' , (": k) , '|' , (": p) , '|' , (": min_p)
    parsed =. parsed , <(<msgs) , (<tools) , (<key) , (<temp) , (<k) , (<p) , (<min_p) , (<mx)
    i =. i + 1
  end.
  NB. Group record indices by key (order-preserving); a group = <key ; idxs>.
  groups =. ''
  i =. 0
  while. i < B do.
    key =. > 2 { > i { parsed
    placed =. 0
    g =. 0
    while. g < # groups do.
      gkey =. > 0 { > g { groups
      same =. 0
      if. (# key) = # gkey do. same =. */ key = gkey end.
      if. same do.
        idxs =. > 1 { > g { groups
        idxs =. idxs , <i
        cell =. (<key) , <idxs
        groups =. (<cell) g} groups
        placed =. 1
        break.
      end.
      g =. g + 1
    end.
    if. 1 ~: placed do.
      idxs =. <i
      cell =. (<key) , <idxs
      groups =. groups , <cell
    end.
    i =. i + 1
  end.
  NB. Per group: render + tokenize each seq, ONE gen_loop_batch, post-process.
  results =. B $ <''
  g =. 0
  while. g < # groups do.
    idxs =. > 1 { > g { groups
    Bg =. # idxs
    i0 =. > {. idxs
    r0 =. > i0 { parsed
    temp =. > 3 { r0
    k =. > 4 { r0
    p =. > 5 { r0
    min_p =. > 6 { r0
    mx =. > 7 { r0
    prompts_tok =. ''
    Ls =. ''
    sess =. session_new ''
    j =. 0
    while. j < Bg do.
      i =. > j { idxs
      r =. > i { parsed
      sess =. (<(chat_vars_obj '')) (2) } sess
      sess =. (> 1 { r) (4) } sess
      sess_cur_g =: sess
      prompt =. chat_prompt > 0 { r
      tokens =. arch chat_tokenize (<llm) , <prompt
      tok_list =. , > tokens
      prompts_tok =. prompts_tok , <tok_list
      Ls =. Ls , <(# tok_list)
      j =. j + 1
    end.
    sess_cur_g =: ''
    kv_batch_g =: Bg
    output =. llm gen_loop_batch (prompts_tok ; mx ; temp ; k ; p ; min_p ; <stop)
    j =. 0
    while. j < Bg do.
      i =. > j { idxs
      L =. > j { Ls
      gen =. (L) }. (> j { output)
      cell =. arch chat_postprocess (<llm) , (<gen) , <mx
      results =. (<cell) i} results
      j =. j + 1
    end.
    g =. g + 1
  end.
  kv_batch_g =: 1
  results
)

NB. ---- Tool dispatch registry (Phase 6 item 4) ----
NB. chat_tool_fn_g is the global verb executed for each tool call. y = <name ;
NB. args-JSON-string>. Returns the result STRING (fed back as a 'tool' message
NB. content). Mirrors the gen_cb_g global-verb pattern: verbs can't be boxed
NB. into a list (noun-verb syntax error) and can't be distinguished from a noun
NB. by -:/3!:0, so a global verb gates the dispatch. Caller sets
NB. chat_tool_fn_g =: my_verb (monadic, y = <name ; args>). Default returns an
NB. error string (the model sees it and may correct the call).
chat_tool_fn_g =: 3 : 0
  'tool not registered: ' , > 0 { y
)

NB. Build an assistant message Value carrying the model's tool_calls.
NB. y = boxed list of tool_call minja Values (from chat_extract_tool_calls).
chat_build_tool_call_msg =: 3 : 0
  tcs =. y
  tcarr =. mkarr_minja_ tcs
  mkobj_minja_ ((('role') pair_minja_ (mkstr_minja_ 'assistant')) , (('content') pair_minja_ (mknull_minja_ '')) , (('tool_calls') pair_minja_ tcarr))
)

NB. Build a 'tool' role message Value carrying one tool result.
NB. y = <tool_call_id ; result-string>.
chat_build_tool_msg =: 3 : 0
  id =. > 0 { y
  res =. > 1 { y
  mkobj_minja_ ((('role') pair_minja_ (mkstr_minja_ 'tool')) , (('content') pair_minja_ (mkstr_minja_ res)) , (('tool_call_id') pair_minja_ (mkstr_minja_ id)))
)

NB. Execute one tool_call minja Value via chat_tool_fn_g -> <id ; result-string>.
chat_exec_tool =: 3 : 0
  tc =. y
  fin =. ('function') obj_get_minja_ tc
  name =. payload_minja_ ('name') obj_get_minja_ fin
  args =. payload_minja_ ('arguments') obj_get_minja_ fin
  id =. payload_minja_ ('id') obj_get_minja_ tc
  res =. chat_tool_fn_g (name ; args)
  (<id) , <res
)

NB. ---- Tool-use loop (Phase 6 item 4) ----
NB. llm chat_tool_loop (messages ; tools ; max_steps ; stream ; max_rounds ; <params>)
NB.   messages   = boxed list of <role ; content> (or pre-built minja Values)
NB.   tools      = JSON string of tool definitions ('' = none)
NB.   max_steps  = generation cap per round (token count)
NB.   stream     = as chat_completion (re-arms the streaming globals each round)
NB.   max_rounds = cap on tool-use rounds (executes tools, then re-generates)
NB.   <params>   = <temp;k;p;min_p> (MUST be last; a pre-boxed `;` operand that
NB.                isn't trailing nests — see chat_completion, so max_rounds
NB.                comes BEFORE <params>)
NB. Calls chat_completion; if finish_reason == 'tool_calls', executes each tool
NB. via chat_tool_fn_g, appends the assistant tool_calls + tool result messages,
NB. and re-calls until the model stops calling tools (cap max_rounds).
NB. Returns <content ; finish_reason ; tool_calls_made> — content is the FINAL
NB. text answer, finish_reason the final 'stop'/'length' (or 'tool_calls' if
NB. max_rounds was hit mid-call), tool_calls_made the accumulated boxed list of
NB. tool_call Values the model requested across rounds.
chat_tool_loop =: 4 : 0
  llm =. x
  messages =. > 0 { y
  tools =. > 1 { y
  max_steps =. > 2 { y
  stream =. > 3 { y
  max_rounds =. > 4 { y
  params =. > 5 { y
  hist =. messages
  all_tcs =. ''
  round =. 0
  res =. ''
  while. 1 do.
    round =. round + 1
    if. stream do.
      gen_cb_on_g =: 1
      gen_cb_g =: chat_stream_cb
      chat_cb_arch_g =: llm_arch llm
      chat_cb_llm_g =: llm
    end.
    res =. llm chat_completion (hist ; tools ; max_steps ; stream ; <params)
    finish =. > 1 { res
    if. finish -: 'tool_calls' do.
      tcs =. > 2 { res
      if. round < max_rounds do.
        all_tcs =. all_tcs , tcs
        hist =. hist , <(chat_build_tool_call_msg tcs)
        for_i. i. # tcs do.
          exec =. chat_exec_tool (> (i { tcs))
          hist =. hist , <(chat_build_tool_msg (> exec))
        end.
        continue.
      else.
        NB. max_rounds hit with a pending tool call — record it and stop
        all_tcs =. all_tcs , tcs
        break.
      end.
    end.
    break.
  end.
  content =. > 0 { res
  (<content) , (<finish) , (<all_tcs)
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
  params =. llm_default_params llm
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
NB. The serial-path chat state is the session's field 0 (`sess_chat session`):
NB.   <arch; messages; total_tokens; cur_pos; max_steps; params>
NB.   - messages: boxed list of <role ; content> message boxes
NB.   - total_tokens: boxed list of ALL tokens processed (prompt + generated)
NB.   - cur_pos: # total_tokens == KV write frontier
NB.   - params: <temp;k;p;min_p>
NB. The KV cache lives in kv_cache_g (the global); the session NEVER holds a
NB. cache reference — a second ref would defeat the in-place amend. One active
NB. session per J session; multi-session support can install a session's cache
NB. into kv_cache_g on switch later.

NB. ---- path counters (tests: assert the resume path actually runs) ----
chat_resume_count =: 0
chat_fallback_count =: 0

NB. ---- Reset the chat session (clears session + KV cache) ----
chat_reset =: 3 : 0
  session =: 0 $ <''
  kv_reset ''
  ''
)

NB. ---- Fresh full-render chat turn (stateless helper) ----
NB. x = llm; y = <messages; max_steps; temp; k; p; min_p; flat>
NB. Renders the FULL message history, generates fresh, stores the session.
NB. Returns the answer text.
NB. (chat_fresh is superseded by chat_fresh_s — the serial path uses the _s
NB. verbs via the wrappers; the global-based chat_fresh was removed.)

NB. ---- Core chat turn: persistent session if one exists ----
NB. x = llm; y = <msg; max_steps; <params>  (<params> = <temp;k;p;min_p>, possibly double-boxed)
chat_core =: 4 : 0
  llm =. x
  msg =. > 0 { y
  max_steps =. > 1 { y
  params =. > 2 { y
  session_ensure ''
  res =. llm chat_core_s ((<session) , (<msg) , (<max_steps) , (<params))
  session =: > 1 { res
  > 0 { res
)

NB. ================================================================
NB.  Session-aware stateful chat (Stage 2 tail).  These are session-aware
NB.  ports of chat_fresh / chat_core / chat_gen_stream / chat_fresh_stream /
NB.  chat_core_stream: they read/write the session's chat_session (field 0),
NB.  st_buf (field 5), and kv state (fields 9-13) instead of the globals, and
NB.  return <answer ; updated_sess> (the caller rebinds the session).  The
NB.  global callbacks (gen_cb_g / chat_cb_* / chat_cb_g) stay global — verbs
NB.  can't be boxed in the session.
NB. ================================================================

NB. ---- Session-aware fresh chat turn ----
NB.  x = llm; y = <sess ; messages ; max_steps ; temp ; k ; p ; min_p ; flat>.
NB.  Renders the full history, generates fresh at the session's seq, stores the
NB.  session's chat_session.  Returns <answer ; updated_sess>.
chat_fresh_s =: 4 : 0
  llm =. x
  sess =. > 0 { y
  messages =. > 1 { y
  max_steps =. > 2 { y
  temp =. > 3 { y
  k =. > 4 { y
  p =. > 5 { y
  min_p =. > 6 { y
  flat =. > 7 { y
  arch =. llm_arch llm
  stop =. chat_stop_tokens llm
  sess_cur_g =: sess
  prompt =. chat_prompt messages
  tokens =. arch chat_tokenize (<llm) , <prompt
  L =. # , > tokens
  output =. llm gen_loop_core (tokens ; '' ; max_steps ; temp ; k ; p ; min_p ; <stop)
  sess =. sess_cur_g
  sess_cur_g =: ''
  gen =. L }. output
  answer =. arch chat_detokenize (<llm) , <gen
  messages =. messages , <('assistant') ; answer
  cs =. (<arch) , (<messages) , (<output) , (<(# , > output)) , (<max_steps) , (<flat)
  sess =. (<cs) (0) } sess
  (<answer) , <sess
)

NB. ---- Session-aware core chat turn (stateful resume) ----
NB.  x = llm; y = <sess ; msg ; max_steps ; <params>.  Resumes the KV cache via
NB.  the session's chat_session (field 0) + kv state.  Returns <answer ;
NB.  updated_sess>.
chat_core_s =: 4 : 0
  llm =. x
  sess =. > 0 { y
  msg =. > 1 { y
  max_steps =. > 2 { y
  params =. > 3 { y
  if. 1 = # params do.
    flat =. > > params
  else.
    flat =. > params
  end.
  temp =. 0 { flat
  k =. 1 { flat
  p =. 2 { flat
  min_p =. 3 { flat
  sess =. (<(chat_vars_obj '')) (2) } sess
  sess =. (<'') (4) } sess
  sess_cur_g =: sess
  arch =. llm_arch llm
  cs =. > 0 { sess
  if. 0 = # cs do.
    NB. no session — start fresh with a single user message
    messages =. <('user') ; msg
    res =. llm chat_fresh_s ((<sess) , (<messages) , (<max_steps) , (<temp) , (<k) , (<p) , (<min_p) , (<flat))
    res
  else.
    s_arch =. > 0 { cs
    if. -. s_arch -: arch do.
      NB. different model loaded — start over
      messages =. <('user') ; msg
      res =. llm chat_fresh_s ((<sess) , (<messages) , (<max_steps) , (<temp) , (<k) , (<p) , (<min_p) , (<flat))
      res
    else.
      prev_messages =. > 1 { cs
      prev_toks =. > 2 { cs
      prev_len =. > 3 { cs
      messages =. prev_messages , <('user') ; msg
      prompt =. chat_prompt messages
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
        sess =. sess_cur_g
        sess_cur_g =: ''
        gen =. L_seg }. output
        answer =. arch chat_detokenize (<llm) , <gen
        total =. prev_toks , output
        messages =. messages , <('assistant') ; answer
        cs =. (<arch) , (<messages) , (<total) , (<(# , > total)) , (<max_steps) , (<flat)
        sess =. (<cs) (0) } sess
        (<answer) , <sess
      else.
        NB. tokenizer round-trip drift — fall back to a full fresh re-render
        chat_fallback_count =: chat_fallback_count + 1
        res =. llm chat_fresh_s ((<sess) , (<messages) , (<max_steps) , (<temp) , (<k) , (<p) , (<min_p) , (<flat))
        res
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
  params =. llm_default_params llm
  llm chat_core (msg ; 100000 ; <params)
)

NB. ---- chat with explicit params: llm chat_p ('msg' ; <temp;k;p;min_p>) ----
chat_p =: 4 : 0
  llm =. x
  msg =. > 0 { y
  params =. > 1 { y
  llm chat_core (msg ; 100000 ; <params)
)

NB. ================================================================
NB. Phase 6 item 4: STATEFUL chat with STREAMING (the stateful TUI path).
NB. chat_core (above) resumes the KV cache but never streams; chat_completion
NB. streams but re-renders the full history each turn. chat_core_stream combines
NB. both: persistent session (the global `session` noun) + KV-cache resume, and arms the
NB. per-token streaming callback (mirrors chat_completion's stream mode).
NB. The caller must chat_stream_start '' + set chat_cb_g (its delta consumer)
NB. before calling, and chat_stream_stop '' after. Returns the answer text.
NB. ================================================================

NB. ---- Generate with STREAMING armed; return the raw output token list ----
NB. x = llm; y = <tokens; start_pos; max_steps; <flat>; stop>
chat_gen_stream =: 4 : 0
  llm =. x
  tokens =. > 0 { y
  start_pos =. > 1 { y
  max_steps =. > 2 { y
  flat =. > 3 { y
  stop =. > 4 { y
  session_ensure ''
  res =. llm chat_gen_stream_s ((<session) , (<tokens) , (<start_pos) , (<max_steps) , (<flat) , (<stop))
  session =: > 1 { res
  > 0 { res
)

NB. ---- Fresh full-render chat turn with STREAMING (stateful helper) ----
NB. x = llm; y = <messages; max_steps; <flat>; stop>
chat_fresh_stream =: 4 : 0
  llm =. x
  messages =. > 0 { y
  max_steps =. > 1 { y
  flat =. > 2 { y
  stop =. > 3 { y
  session_ensure ''
  res =. llm chat_fresh_stream_s ((<session) , (<messages) , (<max_steps) , (<flat) , (<stop))
  session =: > 1 { res
  > 0 { res
)

NB. ---- Core stateful streaming chat turn ----
NB. x = llm; y = <msg; max_steps; <params>  (same interface as chat_core; msg is
NB. the NEW user message only — the session holds the history).
chat_core_stream =: 4 : 0
  llm =. x
  msg =. > 0 { y
  max_steps =. > 1 { y
  params =. > 2 { y
  session_ensure ''
  res =. llm chat_core_stream_s ((<session) , (<msg) , (<max_steps) , (<params))
  session =: > 1 { res
  > 0 { res
)

NB. ================================================================
NB.  Session-aware stateful chat with STREAMING (Stage 2 tail).
NB.  These mirror chat_gen_stream / chat_fresh_stream / chat_core_stream but
NB.  drive the session-aware streaming callback (chat_stream_cb reads sess_cur_g
NB.  — the session's st_buf/st_arch) and store the session's chat_session.  They
NB.  set sess_cur_g before generating and clear it after; return
NB.  <answer ; updated_sess> (or <output ; updated_sess> for gen_stream_s).
NB. ================================================================

NB. ---- Session-aware generate with STREAMING armed ----
NB.  x = llm; y = <sess ; tokens ; start_pos ; max_steps ; <flat> ; stop>.
NB.  Sets sess_cur_g so the streaming callback reads the session's st_buf.
NB.  Returns <output ; updated_sess>.
chat_gen_stream_s =: 4 : 0
  llm =. x
  sess =. > 0 { y
  tokens =. > 1 { y
  start_pos =. > 2 { y
  max_steps =. > 3 { y
  flat =. > 4 { y
  stop =. > 5 { y
  temp =. 0 { flat
  k =. 1 { flat
  p =. 2 { flat
  min_p =. 3 { flat
  sess_cur_g =: sess
  output =. llm gen_loop_core (tokens ; start_pos ; max_steps ; temp ; k ; p ; min_p ; <stop)
  sess =. sess_cur_g
  sess_cur_g =: ''
  if. 0 < # sess_stbuf sess do. chat_cb_g sess_stbuf sess end.
  sess =. sess chat_stream_reset_s ''
  (<output) , <sess
)

NB. ---- Session-aware fresh full-render chat turn with STREAMING ----
NB.  x = llm; y = <sess ; messages ; max_steps ; <flat> ; stop>.  Returns
NB.  <answer ; updated_sess>.
chat_fresh_stream_s =: 4 : 0
  llm =. x
  sess =. > 0 { y
  messages =. > 1 { y
  max_steps =. > 2 { y
  flat =. > 3 { y
  stop =. > 4 { y
  arch =. llm_arch llm
  prompt =. chat_prompt messages
  tokens =. arch chat_tokenize (<llm) , <prompt
  L =. # , > tokens
  out =. llm chat_gen_stream_s ((<sess) , (<tokens) , (<'') , (<max_steps) , (<flat) , (<stop))
  output =. > 0 { out
  sess =. > 1 { out
  gen =. L }. output
  answer =. arch chat_detokenize (<llm) , <gen
  messages =. messages , <('assistant') ; answer
  cs =. (<arch) , (<messages) , (<output) , (<(# , > output)) , (<max_steps) , (<flat)
  sess =. (<cs) (0) } sess
  (<answer) , <sess
)

NB. ---- Session-aware core stateful streaming chat turn ----
NB.  x = llm; y = <sess ; msg ; max_steps ; <params>.  Returns <answer ;
NB.  updated_sess>.
chat_core_stream_s =: 4 : 0
  llm =. x
  sess =. > 0 { y
  msg =. > 1 { y
  max_steps =. > 2 { y
  params =. > 3 { y
  if. 1 = # params do.
    flat =. > > params
  else.
    flat =. > params
  end.
  temp =. 0 { flat
  k =. 1 { flat
  p =. 2 { flat
  min_p =. 3 { flat
  sess =. (<(chat_vars_obj '')) (2) } sess
  sess =. (<'') (4) } sess
  arch =. llm_arch llm
  stop =. chat_stop_tokens llm
  chat_cb_arch_g =: arch
  chat_cb_llm_g =: llm
  chat_cb_stop_g =: stop
  sess_cur_g =: sess
  cs =. > 0 { sess
  if. 0 = # cs do.
    NB. no session — start fresh with a single user message, STREAMING
    messages =. <('user') ; msg
    res =. llm chat_fresh_stream_s ((<sess) , (<messages) , (<max_steps) , (<flat) , (<stop))
    res
  else.
    s_arch =. > 0 { cs
    if. -. s_arch -: arch do.
      NB. different model loaded — start over
      messages =. <('user') ; msg
      res =. llm chat_fresh_stream_s ((<sess) , (<messages) , (<max_steps) , (<flat) , (<stop))
      res
    else.
      prev_messages =. > 1 { cs
      prev_toks =. > 2 { cs
      prev_len =. > 3 { cs
      messages =. prev_messages , <('user') ; msg
      prompt =. chat_prompt messages
      tokens =. arch chat_tokenize (<llm) , <prompt
      tok_list =. , > tokens
      prev_flat =. , > prev_toks
      if. (prev_len {. tok_list) -: prev_flat do.
        NB. re-render prefix matches the stored token stream -> resume, STREAMING
        chat_resume_count =: chat_resume_count + 1
        seg =. prev_len }. tok_list
        L_seg =. # seg
        out =. llm chat_gen_stream_s ((<sess) , (<(<"0 seg)) , (<prev_len) , (<max_steps) , (<flat) , (<stop))
        output =. > 0 { out
        sess =. > 1 { out
        gen =. L_seg }. output
        answer =. arch chat_detokenize (<llm) , <gen
        total =. prev_toks , output
        messages =. messages , <('assistant') ; answer
        cs =. (<arch) , (<messages) , (<total) , (<(# , > total)) , (<max_steps) , (<flat)
        sess =. (<cs) (0) } sess
        (<answer) , <sess
      else.
        NB. tokenizer round-trip drift — fall back to a full fresh re-render
        chat_fallback_count =: chat_fallback_count + 1
        res =. llm chat_fresh_stream_s ((<sess) , (<messages) , (<max_steps) , (<flat) , (<stop))
        res
      end.
    end.
  end.
)
