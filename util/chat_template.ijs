NB. ================================================================
NB. util/chat_template.ijs — J port of minja/chat-template.hpp
NB.
NB. The HuggingFace-standard messages/tools -> prompt formatter that wraps a
NB. parsed Jinja template. Port of reference/minja/include/minja/chat-template.hpp
NB. (MIT, Google LLC). Standalone: `coclass 'chat_template'`, depends only on
NB. util/minja.ijs (the minja Value model). Not wired into inference.ijs /
NB. chat.ijs — integrates with the larger project only after the engine
NB. (minja Phase 5B/5C) + this layer land.
NB.
NB. See PLAN.md Phase 5G for the breakdown. Oracle: minja's own tests
NB. (tests/test-polyfills.cpp, tests/test-capabilities.cpp,
NB. tests/test-supported-template.cpp).
NB.
NB. Public verbs (call with _chatpl_ suffix, or simple names in the chatpl
NB. locale): mk_caps, mk_inputs, mk_options, add_system, scan_has,
NB. polyfill_flags, ct_apply (stub — needs engine).
NB. ================================================================
coclass 'chatpl'
require 'llm/inference/util/minja'

NB. ---- chat_template_caps (10 flags) ----
NB. boxed obj: keys = flag names, values = minja bool Values.
mk_caps =: 3 : 0
  NB. defaults: all false
  mkobj_minja_ ((<'supports_tools') , <(mkbool_minja_ 0)) , ((<'supports_tool_calls') , <(mkbool_minja_ 0)) , ((<'supports_tool_responses') , <(mkbool_minja_ 0)) , ((<'supports_system_role') , <(mkbool_minja_ 0)) , ((<'supports_parallel_tool_calls') , <(mkbool_minja_ 0)) , ((<'supports_tool_call_id') , <(mkbool_minja_ 0)) , ((<'requires_object_arguments') , <(mkbool_minja_ 0)) , ((<'requires_non_null_content') , <(mkbool_minja_ 0)) , ((<'requires_non_empty_content') , <(mkbool_minja_ 0)) , ((<'requires_typed_content') , <(mkbool_minja_ 0))
)
caps_set =: 4 : 0
  NB. (name ; bool) caps_set caps -> caps with flag set
  'name v' =. x
  (name (mkbool_minja_ v)) obj_set_minja_ y
)

NB. ---- chat_template_inputs ----
NB. boxed (messages ; tools ; add_generation_prompt ; extra_context ; now ; bos ; eos)
mk_inputs =: 3 : 0
  NB. mk_inputs messages -> inputs with defaults (tools empty, add_gen 1, bos/eos empty)
  NB. messages = minja Value array of message Value objs
  ((<y) , (<(mkarr_minja_ '')) , (<1) , (<(mkobj_minja_ '')) , <0 , <'' , <'')
)

NB. ---- chat_template_options ----
NB. boxed (apply_polyfills ; use_bos ; use_eos ; define_strftime_now ; per-polyfill toggles)
mk_options =: 3 : 0
  NB. defaults: all true
  (1 1 1 1 1 1 1 1 1 1 1)
)

NB. ---- add_system (static) ----
NB. (prompt) add_system messages -> new messages (system prepended/injected).
NB. messages = minja Value array of message Value objs.
NB. If first message role == system, append prompt to its content;
NB. else insert a fresh system message at the front.
add_system =: 4 : 0
  'p' =. x
  items =. arr_items_minja_ y
  if. 0 = # items do. mkarr_minja_ '' return. end.
  m0 =. > (0 { items)
  ks =. obj_keys_minja_ m0
  if. (<'role') e. ks do.
    r =. ('role') obj_get_minja_ m0
    if. (payload_minja_ r) -: 'system' do.
      NB. first message is system — append prompt to its content
      c =. ('content') obj_get_minja_ m0
      newm =. (('content') pair_minja_ (mkstr_minja_ ((payload_minja_ c) , LF , LF , p))) obj_set_minja_ m0
      mkarr_minja_ ((<newm) , (}. items))
      return.
    end.
  end.
  sm =. mkobj_minja_ ((('role') pair_minja_ (mkstr_minja_ 'system')) , (('content') pair_minja_ (mkstr_minja_ p)))
  mkarr_minja_ ((<sm) , items)
)

NB. ---- has_* scan + polyfill flags ----
NB. scan_has ((<messages) , <tools) -> boxed (has_tools ; has_tool_calls ;
NB. has_tool_responses ; has_string_content). messages/tools = minja Values.
scan_has =: 3 : 0
  'msgs tools' =. y
  has_tools =. 0 < arr_size_minja_ tools
  has_tc =. 0
  has_tr =. 0
  has_sc =. 0
  items =. arr_items_minja_ msgs
  n =. # items
  i =. 0
  while. i < n do.
    m =. > (i { items)
    ks =. obj_keys_minja_ m
    if. (<'tool_calls') e. ks do.
      tc =. ('tool_calls') obj_get_minja_ m
      if. -. is_null_minja_ tc do. has_tc =. 1 end.
    end.
    if. (<'role') e. ks do.
      r =. ('role') obj_get_minja_ m
      if. (payload_minja_ r) -: 'tool' do. has_tr =. 1 end.
    end.
    if. (<'content') e. ks do.
      c =. ('content') obj_get_minja_ m
      if. is_str_minja_ c do. has_sc =. 1 end.
    end.
    i =. i + 1
  end.
  ((<has_tools) , (<has_tc) , (<has_tr) , (<has_sc))
)

NB. polyfill_flags — derive the polyfill_* toggles from caps + scan + options.
NB. (caps ; scan ; options) -> boxed (polyfill_system_role ; polyfill_tools ;
NB. polyfill_tool_call_example ; polyfill_tool_calls ; polyfill_tool_responses ;
NB. polyfill_object_arguments ; polyfill_typed_content ; needs_polyfills)
polyfill_flags =: 3 : 0
  'caps scan opts' =. y
  'has_tools has_tc has_tr has_sc' =. scan
  'apply_pf use_bos use_eos def_sf pt ptec ptc ptr psr poa ptyped' =. > opts
  supports_tools =. to_bool_minja_ ('supports_tools') obj_get_minja_ caps
  supports_tc    =. to_bool_minja_ ('supports_tool_calls') obj_get_minja_ caps
  supports_tr    =. to_bool_minja_ ('supports_tool_responses') obj_get_minja_ caps
  supports_sys   =. to_bool_minja_ ('supports_system_role') obj_get_minja_ caps
  req_obj_args   =. to_bool_minja_ ('requires_object_arguments') obj_get_minja_ caps
  req_typed      =. to_bool_minja_ ('requires_typed_content') obj_get_minja_ caps
  psys =. psr *. -. supports_sys
  pt   =. pt *. has_tools *. -. supports_tools
  ptec =. pt *. ptec
  ptc  =. ptc *. has_tc *. -. supports_tc
  ptr  =. ptr *. has_tr *. -. supports_tr
  poa  =. poa *. has_tc *. req_obj_args
  ptyp =. ptyped *. has_sc *. req_typed
  needs =. apply_pf *. (psys +. pt +. ptc +. ptr +. poa +. ptyp)
  ((<psys) , (<pt) , (<ptec) , (<ptc) , (<ptr) , (<poa) , (<ptyp) , (<needs))
)

NB. ---- apply (5Gb: normalization + render) ----
NB. ct_apply (source ; inputs ; caps ; tool_call_example ; options) -> prompt string.
NB. Port of chat-template.hpp apply(): normalize messages per polyfill flags,
NB. bind bos/eos/tools/extra_context, render the template (trim+lstrip blocks).
NB. inputs = <messages ; tools ; add_generation_prompt ; extra_context ; now ; bos ; eos>.
NB. caps = mk_caps '' (detected). options = mk_options ''.
ct_apply =: 3 : 0
  'source inputs caps tool_ex opts' =. y
  'messages tools addgen extra now bos eos' =. inputs
  addgen =. > addgen
  bos =. > bos
  eos =. > eos
  'apply_pf use_bos use_eos def_sf pt ptec ptc ptr psr poa ptyped' =. > opts
  'has_tools has_tc has_tr has_sc' =. scan_has ((<messages) , <tools)
  NB. polyfill flags
  supports_tools =. to_bool_minja_ ('supports_tools') obj_get_minja_ caps
  supports_tc    =. to_bool_minja_ ('supports_tool_calls') obj_get_minja_ caps
  supports_tr    =. to_bool_minja_ ('supports_tool_responses') obj_get_minja_ caps
  supports_sys   =. to_bool_minja_ ('supports_system_role') obj_get_minja_ caps
  req_obj_args   =. to_bool_minja_ ('requires_object_arguments') obj_get_minja_ caps
  req_typed      =. to_bool_minja_ ('requires_typed_content') obj_get_minja_ caps
  psys =. psr *. -. supports_sys
  pt   =. pt *. has_tools *. -. supports_tools
  ptec =. pt *. ptec
  ptc  =. ptc *. has_tc *. -. supports_tc
  ptr  =. ptr *. has_tr *. -. supports_tr
  poa  =. poa *. has_tc *. req_obj_args
  ptyp =. ptyped *. has_sc *. req_typed
  needs =. apply_pf *. (psys +. pt +. ptc +. ptr +. poa +. ptyp)
  actual =. messages
  if. needs do.
    actual =. ct_normalize ((<messages) , (<tools) , (<caps) , (<tool_ex) , <(psys , pt , ptec , ptc , ptr , poa , ptyp))
  end.
  NB. build context with messages/add_generation_prompt/bos/eos/tools/extra
  cobj =. mkobj_minja_ ((('messages') pair_minja_ actual) , (('add_generation_prompt') pair_minja_ (mkbool_minja_ addgen)))
  ctx =. mkctx_minja_ cobj
  if. use_bos do. ctx =. ((<'bos_token') , <(mkstr_minja_ bos)) ctx_set_minja_ ctx end.
  if. use_eos do. ctx =. ((<'eos_token') , <(mkstr_minja_ eos)) ctx_set_minja_ ctx end.
  if. def_sf do.
    strftime_now_g_minja_ =: (> now)
    ctx =. ((<'strftime_now') , <(mkcall_minja_ 'strftime_now')) ctx_set_minja_ ctx
  end.
  if. -. (is_null_minja_ tools) do.
    ctx =. ((<'tools') , <tools) ctx_set_minja_ ctx
  end.
  if. -. (is_null_minja_ extra) do.
    ks =. obj_keys_minja_ extra
    for_k. ks do.
      k =. > k
      ctx =. ((<k) , <(k obj_get_minja_ extra)) ctx_set_minja_ ctx
    end.
  end.
  NB. render the template (trim_blocks + lstrip_blocks)
  ctx render_opt_minja_ (source ; 'tl')
)

NB. ct_normalize (messages ; tools ; caps ; tool_ex ; flags) -> actual_messages.
NB. Port of chat-template.hpp apply() normalization (lines 387-511).
ct_actual_g =: ''
ct_pending_g =: ''
ct_ptyp_g =: 0

ct_add_message =: 3 : 0
  NB. msg -> append to ct_actual_g (typed-content conversion if ct_ptyp_g)
  msg =. y
  if. ct_ptyp_g do.
    if. (<'content') e. (obj_keys_minja_ msg) do.
      c =. ('content') obj_get_minja_ msg
      if. -. (is_null_minja_ c) do.
        if. is_str_minja_ c do.
          txt =. mkobj_minja_ ((('type') pair_minja_ (mkstr_minja_ 'text')) , (('text') pair_minja_ c))
          m2 =. ((<'content') , <(mkarr_minja_ (<txt))) obj_set_minja_ msg
          ct_actual_g =: mkarr_minja_ ((arr_items_minja_ ct_actual_g) , <m2)
          return.
        end.
      end.
    end.
  end.
  ct_actual_g =: mkarr_minja_ ((arr_items_minja_ ct_actual_g) , <msg)
)

ct_flush_sys =: 3 : 0
  if. 0 < # ct_pending_g do.
    um =. mkobj_minja_ ((('role') pair_minja_ (mkstr_minja_ 'user')) , (('content') pair_minja_ (mkstr_minja_ ct_pending_g)))
    ct_actual_g =: mkarr_minja_ ((arr_items_minja_ ct_actual_g) , <um)
    ct_pending_g =: ''
  end.
)

ct_normalize =: 3 : 0
  'messages tools caps tool_ex flags' =. y
  'psys pt ptec ptc ptr poa ptyp' =. > flags
  ct_actual_g =: mkarr_minja_ ''
  ct_pending_g =: ''
  ct_ptyp_g =: ptyp
  adjusted =. messages
  if. pt do.
    tool_dump =. dumpc_minja_ ((<tools) , (<2) , (<0) , (<1))
    prompt =. 'You can call any of the following tools to satisfy the user''s requests: ' , tool_dump
    if. ptec *. (0 < # tool_ex) do.
      prompt =. prompt , LF , LF , 'Example tool call syntax:' , LF , LF , tool_ex , LF , LF
    end.
    adjusted =. prompt add_system messages
  end.
  items =. arr_items_minja_ adjusted
  for_i. i. # items do.
    message =. > (i { items)
    if. -. ((<'role') e. (obj_keys_minja_ message)) do. err_minja_ 'message must have a role field' end.
    if. -. (((<'content') e. (obj_keys_minja_ message)) +. ((<'tool_calls') e. (obj_keys_minja_ message))) do.
      err_minja_ 'message must have content or tool_calls'
    end.
    role =. ('role') obj_get_minja_ message
    role_s =. payload_minja_ role
    if. (<'tool_calls') e. (obj_keys_minja_ message) do.
      tcs =. ('tool_calls') obj_get_minja_ message
      if. poa +. ptc do.
        tcs =. ct_parse_args tcs
        message =. ((<'tool_calls') , <tcs) obj_set_minja_ message
      end.
      if. ptc do.
        tcs2 =. ct_tool_calls_polyfill tcs
        obj =. mkobj_minja_ ((('tool_calls') pair_minja_ tcs2))
        if. (<'content') e. (obj_keys_minja_ message) do.
          c =. ('content') obj_get_minja_ message
          if. -. (is_null_minja_ c) do.
            if. 0 < arr_size_minja_ c do.
              obj =. ((<'content') , <c) obj_set_minja_ obj
            end.
          end.
        end.
        content =. dumpc_minja_ ((<obj) , (<2) , (<0) , (<1))
        message =. (('content') pair_minja_ (mkstr_minja_ content)) obj_set_minja_ message
        message =. (('tool_calls') pair_minja_ (mknull_minja_ '')) obj_set_minja_ message
        message =. ((<'tool_calls') , <(mkarr_minja_ '')) obj_set_minja_ message
      end.
    end.
    if. ptr *. (role_s -: 'tool') do.
      message =. (('role') pair_minja_ (mkstr_minja_ 'user')) obj_set_minja_ message
      obj =. mkobj_minja_ ((('tool_response') pair_minja_ (mkobj_minja_ '')))
      tr =. ('tool_response') obj_get_minja_ obj
      if. (<'name') e. (obj_keys_minja_ message) do.
        nm =. ('name') obj_get_minja_ message
        tr =. (('tool') pair_minja_ nm) obj_set_minja_ tr
      end.
      tr =. (('content') pair_minja_ (('content') obj_get_minja_ message)) obj_set_minja_ tr
      if. (<'tool_call_id') e. (obj_keys_minja_ message) do.
        tid =. ('tool_call_id') obj_get_minja_ message
        tr =. (('tool_call_id') pair_minja_ tid) obj_set_minja_ tr
      end.
      obj =. ((<'tool_response') , <tr) obj_set_minja_ obj
      content =. dumpc_minja_ ((<obj) , (<2) , (<0) , (<1))
      message =. (('content') pair_minja_ (mkstr_minja_ content)) obj_set_minja_ message
    end.
    if. psys do.
      c =. ('content') obj_get_minja_ message
      if. -. (is_null_minja_ c) do.
        if. role_s -: 'system' do.
          if. 0 < # ct_pending_g do. ct_pending_g =: ct_pending_g , LF end.
          ct_pending_g =: ct_pending_g , (payload_minja_ c)
          continue.
        elseif. role_s -: 'user' do.
          if. 0 < # ct_pending_g do.
            pc =. payload_minja_ c
            if. 0 < # pc do.
              joined =. ct_pending_g , LF , pc
            else.
              joined =. ct_pending_g
            end.
            message =. (('content') pair_minja_ (mkstr_minja_ joined)) obj_set_minja_ message
            ct_pending_g =: ''
          end.
        else.
          ct_flush_sys 0
        end.
      end.
    end.
    ct_add_message message
  end.
  ct_flush_sys 0
  ct_actual_g
)

NB. ct_parse_args tool_calls -> tool_calls with string arguments parsed to objects.
ct_parse_args =: 3 : 0
  tcs =. y
  out =. mkarr_minja_ ''
  for_tc. arr_items_minja_ tcs do.
    tc =. > tc
    fn =. ('function') obj_get_minja_ tc
    args =. ('arguments') obj_get_minja_ fn
    if. is_str_minja_ args do.
      parsed =. ct_parse_json (payload_minja_ args)
      fn2 =. ((<'arguments') , <parsed) obj_set_minja_ fn
      tc2 =. ((<'function') , <fn2) obj_set_minja_ tc
    else.
      tc2 =. tc
    end.
    out =. mkarr_minja_ ((arr_items_minja_ out) , <tc2)
  end.
  out
)

NB. ---- tiny JSON parser (tool arguments: strings -> objects) ----
NB. ct_parse_json s -> minja Value. Recursive descent over a JSON string.
NB. Parser state lives in globals ct_jpc_g/ct_jsrc_g/ct_jlen_g; helpers are
NB. top-level verbs (locals don't nest across verb calls).
ct_jpc_g =: 0
ct_jsrc_g =: ''
ct_jlen_g =: 0

ct_jskip =: 3 : 0
  while. (ct_jpc_g < ct_jlen_g) *. ((ct_jpc_g { ct_jsrc_g) e. ' ',TAB,LF,CR) do.
    ct_jpc_g =: ct_jpc_g + 1
  end.
)

ct_jstr =: 3 : 0
  ct_jpc_g =: ct_jpc_g + 1   NB. consume opening quote
  s =. ''
  while. (ct_jpc_g { ct_jsrc_g) ~: '"' do.
    ch =. ct_jpc_g { ct_jsrc_g
    if. ch = '\' do.
      ct_jpc_g =: ct_jpc_g + 1
      e =. ct_jpc_g { ct_jsrc_g
      select. e
      case. 'n' do. s =. s , LF
      case. 't' do. s =. s , TAB
      case. '\\' do. s =. s , '\'
      case. '"' do. s =. s , '"'
      case. 'u' do.
        h =. (ct_jpc_g + 1) }. ct_jsrc_g
        cp =. 16 #. '0123456789abcdef' i. (h {~ 0 1 2 3)
        s =. s , (cp { a.)
      case. do. s =. s , e
      end.
      ct_jpc_g =: ct_jpc_g + 1
    else.
      s =. s , ch
      ct_jpc_g =: ct_jpc_g + 1
    end.
  end.
  ct_jpc_g =: ct_jpc_g + 1   NB. consume closing quote
  mkstr_minja_ s
)

ct_jval =: 3 : 0
  ct_jskip ''
  c =. ct_jpc_g { ct_jsrc_g
  select. c
  case. '{' do.
    ct_jpc_g =: ct_jpc_g + 1
    pl =. ''
    while. 1 do.
      ct_jskip ''
      if. (ct_jpc_g { ct_jsrc_g) = '}' do. ct_jpc_g =: ct_jpc_g + 1 break. end.
      k =. ct_jstr ''
      ct_jskip ''
      ct_jpc_g =: ct_jpc_g + 1   NB. consume ':'
      v =. ct_jval ''
      pl =. pl , (<(payload_minja_ k)) , <v
      ct_jskip ''
      if. (ct_jpc_g { ct_jsrc_g) = ',' do. ct_jpc_g =: ct_jpc_g + 1 continue.
      elseif. (ct_jpc_g { ct_jsrc_g) = '}' do. ct_jpc_g =: ct_jpc_g + 1 break.
      else. break. end.
    end.
    mkobj_minja_ pl
  case. '[' do.
    ct_jpc_g =: ct_jpc_g + 1
    res =. mkarr_minja_ ''
    while. 1 do.
      ct_jskip ''
      if. (ct_jpc_g { ct_jsrc_g) = ']' do. ct_jpc_g =: ct_jpc_g + 1 break. end.
      v =. ct_jval ''
      res =. mkarr_minja_ ((arr_items_minja_ res) , <v)
      ct_jskip ''
      if. (ct_jpc_g { ct_jsrc_g) = ',' do. ct_jpc_g =: ct_jpc_g + 1 continue.
      elseif. (ct_jpc_g { ct_jsrc_g) = ']' do. ct_jpc_g =: ct_jpc_g + 1 break.
      else. break. end.
    end.
    res
  case. '"' do. ct_jstr ''
  case. 't' do.
    ct_jpc_g =: ct_jpc_g + 4   NB. 'true' = 4 chars
    mkbool_minja_ 1
  case. 'f' do.
    ct_jpc_g =: ct_jpc_g + 5   NB. 'false' = 5 chars
    mkbool_minja_ 0
  case. 'n' do.
    ct_jpc_g =: ct_jpc_g + 4
    mknull_minja_ ''
  case. do.
    NB. number
    s =. ''
    while. (ct_jpc_g < ct_jlen_g) *. ((ct_jpc_g { ct_jsrc_g) e. '0123456789-.eE+') do.
      s =. s , (ct_jpc_g { ct_jsrc_g)
      ct_jpc_g =: ct_jpc_g + 1
    end.
    mkint_minja_ (0 ". s)
  end.
)

ct_parse_json =: 3 : 0
  ct_jpc_g =: 0
  ct_jsrc_g =: y
  ct_jlen_g =: # y
  ct_jval ''
)

NB. ct_tool_calls_polyfill tool_calls -> array of {name, arguments[, id]}.
ct_tool_calls_polyfill =: 3 : 0
  tcs =. y
  out =. mkarr_minja_ ''
  for_tc. arr_items_minja_ tcs do.
    tc =. > tc
    if. -. ((<'function') e. (obj_keys_minja_ tc)) do. continue. end.
    fn =. ('function') obj_get_minja_ tc
    nm =. ('name') obj_get_minja_ fn
    args =. ('arguments') obj_get_minja_ fn
    tc2 =. mkobj_minja_ ((('name') pair_minja_ nm) , (('arguments') pair_minja_ args))
    if. (<'id') e. (obj_keys_minja_ tc) do.
      tid =. ('id') obj_get_minja_ tc
      tc2 =. (('id') pair_minja_ tid) obj_set_minja_ tc2
    end.
    out =. mkarr_minja_ ((arr_items_minja_ out) , <tc2)
  end.
  out
)

NB. ---- capability detection (5Gb: try_raw_render probes) ----
NB. Chat-template state lives in globals ct_source_g/ct_bos_g/ct_eos_g;
NB. detect_caps probes the template and sets ct_caps_g + ct_tool_ex_g.
ct_source_g =: ''
ct_bos_g =: ''
ct_eos_g =: ''
ct_caps_g =: ''
ct_tool_ex_g =: ''

ct_contains =: 3 : 0
  NB. (haystack ; needle) -> boolean (substring via E.)
  'hay needle' =. y
  1 e. needle E. hay
)

ct_try_raw =: 3 : 0
  NB. (messages ; tools ; addgen) -> render with apply_polyfills=false, now=0.
  NB. Returns '' on any error (mirror try_raw_render).
  'messages tools addgen' =. y
  addgen =. > addgen
  extra =. mkobj_minja_ ''
  opts =. <(0 1 1 1 1 1 1 1 1 1 1)
  inputs =. ((<messages) , (<tools) , (<addgen) , (<extra) , (<0) , (<ct_bos_g) , (<ct_eos_g))
  try.
    ct_apply ((<ct_source_g) , (<inputs) , (<(mk_caps '')) , (<'') , (<opts))
  catcht. 'minja'
    ''
  catch.
    ''
  end.
)

NB. ---- message/tool JSON builders for probes ----
ct_msg =: 3 : 0
  NB. (role ; content) -> message obj
  'role content' =. y
  mkobj_minja_ ((('role') pair_minja_ (mkstr_minja_ role)) , (('content') pair_minja_ content))
)

ct_typed_msg =: 3 : 0
  NB. (role ; text) -> message obj with typed content [{type:text,text:...}]
  'role text' =. y
  txt =. mkobj_minja_ ((('type') pair_minja_ (mkstr_minja_ 'text')) , (('text') pair_minja_ (mkstr_minja_ text)))
  ct_msg ((<role) , <(mkarr_minja_ (<txt)))
)

ct_tool_obj =: 3 : 0
  NB. tool obj: {name, type, function:{name,description,parameters}}
  'nm desc params' =. y
  fn =. mkobj_minja_ ((('name') pair_minja_ (mkstr_minja_ nm)) , (('description') pair_minja_ (mkstr_minja_ desc)) , (('parameters') pair_minja_ params))
  mkobj_minja_ ((('name') pair_minja_ (mkstr_minja_ nm)) , (('type') pair_minja_ (mkstr_minja_ 'function')) , (('function') pair_minja_ fn))
)

ct_tool_call =: 3 : 0
  NB. (name ; arguments) -> tool_call obj {id,type,function:{arguments,name}}
  'nm args' =. y
  fn =. mkobj_minja_ ((('arguments') pair_minja_ args) , (('name') pair_minja_ (mkstr_minja_ nm)))
  mkobj_minja_ ((('id') pair_minja_ (mkstr_minja_ 'call_1___')) , (('type') pair_minja_ (mkstr_minja_ 'function')) , (('function') pair_minja_ fn))
)


NB. probe helpers (top-level; locals don't nest across verb calls)
ct_render_wc =: 3 : 0
  NB. y = content -> render 4 messages (user,assistant,user,assistant)
  NB. uses ct_dummy_user_g
  am =. ct_msg ((<'assistant') , <y)
  ct_try_raw ((<mkarr_minja_ ((<ct_dummy_user_g) , (<am) , (<ct_dummy_user_g) , (<am))) , (<mkarr_minja_ '') , <0)
)

ct_contains_arg_needle =: 3 : 0
  o =. y
  needle2 =. ('"' , 'argument_needle' , '''' , ':')
  (ct_contains ((<o) , <'<parameter=argument_needle>')) +. (ct_contains ((<o) , <'"argument_needle"')) +. (ct_contains ((<o) , <needle2)) +. (ct_contains ((<o) , <'>argument_needle<'))
)

ct_tc_msg =: 3 : 0
  NB. (content ; tool_calls) -> assistant msg
  'content tcs' =. y
  mkobj_minja_ ((('role') pair_minja_ (mkstr_minja_ 'assistant')) , (('content') pair_minja_ content) , (('tool_calls') pair_minja_ tcs))
)

detect_caps =: 3 : 0
  user_needle =. '<User Needle>'
  sys_needle =. '<System Needle>'
  dummy_str_user =. ct_msg ((<'user') , <(mkstr_minja_ user_needle))
  dummy_typed_user =. ct_typed_msg ((<'user') , <user_needle)
  r1 =. ct_try_raw ((<mkarr_minja_ (<dummy_str_user)) , (<mkarr_minja_ '') , <0)
  r2 =. ct_try_raw ((<mkarr_minja_ (<dummy_typed_user)) , (<mkarr_minja_ '') , <0)
  req_typed =. ((-. (ct_contains ((<r1) , <user_needle))) *. (ct_contains ((<r2) , <user_needle)))
  dummy_user =. dummy_str_user
  if. req_typed do. dummy_user =. dummy_typed_user end.
  sys_content =. mkstr_minja_ sys_needle
  if. req_typed do. sys_content =. mkarr_minja_ (<(mkobj_minja_ ((('type') pair_minja_ (mkstr_minja_ 'text')) , (('text') pair_minja_ (mkstr_minja_ sys_needle))))) end.
  sys_msg =. ct_msg ((<'system') , <sys_content)
  supports_sys =. ct_contains ((<ct_try_raw ((<mkarr_minja_ ((<sys_msg) , (<dummy_user))) , (<mkarr_minja_ '') , <0)) , <sys_needle)
  NB. supports_tools
  params =. mkobj_minja_ ((('type') pair_minja_ (mkstr_minja_ 'object')) , (('properties') pair_minja_ (mkobj_minja_ ((<'arg') , <(mkobj_minja_ ((('type') pair_minja_ (mkstr_minja_ 'string')) , (('description') pair_minja_ (mkstr_minja_ 'Some argument.'))))))) , (('required') pair_minja_ (mkarr_minja_ (<(mkstr_minja_ 'arg')))))
  tool =. ct_tool_obj ((<'some_tool') , (<'Some tool.') , <params)
  out =. ct_try_raw ((<mkarr_minja_ (<dummy_user)) , (<mkarr_minja_ (<tool)) , <0)
  supports_tools =. ct_contains ((<out) , <'some_tool')
  NB. requires_non_empty / non_null
  ct_dummy_user_g =: dummy_user
  out_empty =. ct_render_wc (mkstr_minja_ '')
  out_null =. ct_render_wc (mknull_minja_ '')
  out_nonempty =. ct_render_wc (mkstr_minja_ ' ')
  c_empty =. ct_contains ((<out_empty) , <user_needle)
  c_null =. ct_contains ((<out_null) , <user_needle)
  c_nonempty =. ct_contains ((<out_nonempty) , <user_needle)
  req_nonempty =. c_nonempty *. (-. c_empty) *. (-. c_null)
  req_nonnull =. req_nonempty +. (c_empty *. (-. c_null))
  NB. tool_calls probes
  dummy_args_obj =. mkobj_minja_ ((<'argument_needle') , <(mkstr_minja_ 'print(''Hello, World!'')'))
  ac_null =. mknull_minja_ ''
  if. req_nonnull do. ac_null =. mkstr_minja_ '' end.
  if. req_nonempty do. ac_null =. mkstr_minja_ ' ' end.
  str_tc =. ct_tc_msg ((<ac_null) , <(mkarr_minja_ (<(ct_tool_call ((<'ipython') , <(mkstr_minja_ (dumpc_minja_ ((<dummy_args_obj) , (<_1) , (<0) , (<1)))))))))
  out =. ct_try_raw ((<mkarr_minja_ ((<dummy_user) , <str_tc)) , (<mkarr_minja_ '') , <0)
  str_renders =. ct_contains_arg_needle out
  obj_tc =. ct_tc_msg ((<ac_null) , <(mkarr_minja_ (<(ct_tool_call ((<'ipython') , <dummy_args_obj)))))
  out =. ct_try_raw ((<mkarr_minja_ ((<dummy_user) , <obj_tc)) , (<mkarr_minja_ '') , <0)
  obj_renders =. ct_contains_arg_needle out
  supports_tc =. str_renders +. obj_renders
  req_obj_args =. (-. str_renders) *. obj_renders
  supports_parallel =. 0
  supports_tr =. 0
  supports_tcid =. 0
  if. supports_tc do.
    dargs =. dummy_args_obj
    if. -. req_obj_args do. dargs =. mkstr_minja_ (dumpc_minja_ ((<dummy_args_obj) , (<_1) , (<0) , (<1))) end.
    tc1 =. ct_tool_call ((<'test_tool1') , <dargs)
    tc2 =. ct_tool_call ((<'test_tool2') , <dargs)
    ptc_msg =. ct_tc_msg ((<ac_null) , <(mkarr_minja_ ((<tc1) , (<tc2))))
    out =. ct_try_raw ((<mkarr_minja_ ((<dummy_user) , <ptc_msg)) , (<mkarr_minja_ '') , <0)
    supports_parallel =. (ct_contains ((<out) , <'test_tool1')) *. (ct_contains ((<out) , <'test_tool2'))
    tool_resp =. mkobj_minja_ ((('role') pair_minja_ (mkstr_minja_ 'tool')) , (('name') pair_minja_ (mkstr_minja_ 'test_tool1')) , (('content') pair_minja_ (mkstr_minja_ 'Some response!')) , (('tool_call_id') pair_minja_ (mkstr_minja_ 'call_911_')))
    tr_msg =. ct_tc_msg ((<ac_null) , <(mkarr_minja_ (<tc1)))
    out =. ct_try_raw ((<mkarr_minja_ ((<dummy_user) , (<tr_msg) , (<tool_resp))) , (<mkarr_minja_ '') , <0)
    supports_tr =. ct_contains ((<out) , <'Some response!')
    supports_tcid =. ct_contains ((<out) , <'call_911_')
  end.
  NB. build caps object
  ct_caps_g =: mkobj_minja_ (((<'supports_tools') , <(mkbool_minja_ supports_tools)) , ((<'supports_tool_calls') , <(mkbool_minja_ supports_tc)) , ((<'supports_tool_responses') , <(mkbool_minja_ supports_tr)) , ((<'supports_system_role') , <(mkbool_minja_ supports_sys)) , ((<'supports_parallel_tool_calls') , <(mkbool_minja_ supports_parallel)) , ((<'supports_tool_call_id') , <(mkbool_minja_ supports_tcid)) , ((<'requires_object_arguments') , <(mkbool_minja_ req_obj_args)) , ((<'requires_non_null_content') , <(mkbool_minja_ req_nonnull)) , ((<'requires_non_empty_content') , <(mkbool_minja_ req_nonempty)) , ((<'requires_typed_content') , <(mkbool_minja_ req_typed)))
  NB. tool_call_example (if !supports_tools)
  ct_tool_ex_g =: ''
  if. -. supports_tools do.
    try.
      user_msg =. ct_msg ((<'user') , <(mkstr_minja_ 'Hey'))
      args =. mkobj_minja_ ((<'arg1') , <(mkstr_minja_ 'some_value'))
      tc_arg =. args
      if. -. req_obj_args do. tc_arg =. mkstr_minja_ (dumpc_minja_ ((<args) , (<_1) , (<0) , (<1))) end.
      tcm =. ct_tc_msg ((<ac_null) , <(mkarr_minja_ (<(ct_tool_call ((<'tool_name') , <tc_arg)))))
      inputs1 =. ((<(mkarr_minja_ (<user_msg))) , (<(mkarr_minja_ '')) , (<1) , (<(mkobj_minja_ '')) , (<0) , (<ct_bos_g) , (<ct_eos_g))
      prefix =. ct_apply ((<ct_source_g) , (<inputs1) , (<ct_caps_g) , (<ct_tool_ex_g) , (<(mk_options '')))
      inputs2 =. ((<(mkarr_minja_ ((<user_msg) , <tcm))) , (<(mkarr_minja_ '')) , (<0) , (<(mkobj_minja_ '')) , (<0) , (<ct_bos_g) , (<ct_eos_g))
      full =. ct_apply ((<ct_source_g) , (<inputs2) , (<ct_caps_g) , (<ct_tool_ex_g) , (<(mk_options '')))
      NB. strip trailing eos from full
      epos =. (# full) - (# ct_eos_g)
      if. (epos >: 0) *. ((epos }. full) -: ct_eos_g) do.
        full =. (epos) {. full
      elseif. (epos >: 1) *. (((epos - 1) }. full) -: (ct_eos_g , LF)) do.
        full =. (epos - 1) {. full
      end.
      NB. common prefix length (skip '<')
      cpl =. 0
      i =. 0
      while. (i < (# prefix)) *. (i < (# full)) do.
        if. -. ((i { prefix) -: (i { full)) do. break. end.
        if. -. ((i { prefix) = '<') do. cpl =. i + 1 end.
        i =. i + 1
      end.
      example =. (cpl) }. full
      if. -. ((1 e. 'tool_name' E. example) *. (1 e. 'some_value' E. example)) do.
        NB. no example inferred (template bug); leave '' 
      else.
        ct_tool_ex_g =: example
      end.
    catcht. 'minja'
      ct_tool_ex_g =: ''
    catch.
      ct_tool_ex_g =: ''
    end.
  end.
  ct_caps_g
)

NB. ---- chat_template constructor ----
NB. ct_new (source ; bos ; eos) -> sets globals + runs detect_caps.
ct_new =: 3 : 0
  'src bos eos' =. y
  ct_source_g =: src
  ct_bos_g =: bos
  ct_eos_g =: eos
  ct_caps_g =: detect_caps ''
  ct_caps_g
)
