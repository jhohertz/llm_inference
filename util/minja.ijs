NB. ================================================================
NB. util/minja.ijs — J port of minja.hpp (Jinja engine)
NB.
NB. Port of reference/minja/include/minja/minja.hpp (MIT, Google LLC).
NB. Self-contained: `coclass 'minja'` so it can be lifted out into its own
NB. repo. No dependency on the rest of llm/inference.
NB.
NB. Value representation: boxed `(kind ; payload)`:
NB.   ('null';'')  ('bool';1|0)  ('int';n)  ('float';n)  ('str';'text')
NB.   ('arr';<boxed list of Values>)   ('obj';<flat (boxed key ; boxed value) pairs>)
NB.   ('callable';<'name'>)   builtin function/filter callable
NB. Fields extracted with `{::` (fetch): kind = 0 {:: v (unboxed char),
NB. payload = 1 {:: v (unboxed scalar / boxed list).
NB.
NB. Public verbs (call with _minja_ suffix from any locale, or simple names in
NB. the minja locale): the Value model (mknull..mkobj, pair, kind, payload,
NB. is_*, to_bool/to_int/to_num/to_str, dump, dumpj, eq, in, arr_at,
NB. obj_get/set, obj_keys/values), Context (mkctx, ctx_get/set/contains),
NB. builtins, and the engine: `(ctx) render (template)` — tokenize + parse +
NB. render (Phases 5B-5E). Errors are thrown via `err` (sets err_msg_g and
NB. throws), caught by a caller's try./catcht. — see tests.
NB.
NB. NOTE: minja deliberately diverges from Python jinja2 in places (bare
NB. `{{ none }}` renders empty; dict-with-None dumps `null` not `None`). The
NB. authoritative oracle is minja's own test-syntax.cpp EXPECT_EQ strings;
NB. Python jinja2 is the cross-check oracle for the cases minja matches.
NB. ================================================================
coclass 'minja'

NB. ---- Error signaling ----
NB. err sets err_msg_g then throws; a caller try./catcht. reads err_msg_g.
err_msg_g =: ''
err =: 3 : 0
  err_msg_g =: y
  throw. 'minja'
)

NB. ---- Value constructors ----
mknull =: 3 : 0
  'null' ; ''
)
mkbool =: 3 : 0
  'bool' ; , y
)
mkint =: 3 : 0
  'int' ; , y
)
mkfloat =: 3 : 0
  'float' ; , y
)
mkstr =: 3 : 0
  'str' ; , y
)
mkarr =: 3 : 0
  NB. y = boxed list of single-boxed Values, e.g. ((<mkint 1) , (<mkint 2))
  'arr' ; < y
)
mkobj =: 3 : 0
  NB. y = boxed list of (boxed key ; boxed value) pairs, e.g. ((<'a') , <(mkstr 'b'))
  'obj' ; < y
)
mkcall =: 3 : 0
  NB. builtin callable: y = name char string
  'callable' ; < y
)
pair =: 4 : 0
  NB. (key ; value) -> (<key) , (<value)   (build an obj flat pair, avoiding the `;` gotcha)
  (<x) , (<y)
)
mknode =: 3 : 0
  NB. y = (type ; <payloads) -> node; payloads = boxed list of single-box payloads
  't pl' =. y
  n =. <t
  for_i. i. # pl do.
    p =. i {:: pl
    if. p -: '__nil__' do. n =. n , nil else. n =. n , <p end.
  end.
  n
)
nil =: <'__nil__'
mkp =: 3 : 0
  NB. box a payload; empty '' -> nil (non-collapsing marker so `,`/`;` chains don't merge empties)
  if. 0 = # y do. nil else. <y end.
)
mknum =: 4 : 0
  NB. (n ; isint) mknum -> int or float Value
  if. y do. mkint x else. mkfloat x end.
)

NB. ---- Accessors ({:: fetch -> unboxed) ----
kind    =: 3 : 0
  NB. Value kinds are 'null'|'bool'|'int'|'float'|'str'|'arr'|'obj'|'callable'|'dict'.
  NB. Tokens/nodes are boxed pairs whose 0th element is an unboxed kind/type char.
  NB. A boxed char (single box wrapping a char list) must read as 'str'.
  if. 2 = 3!:0 y do. 'str' return. end.
  if. 1 = # y do.
    t =. 0 {:: y
    if. 2 = 3!:0 t do. 'str' return. end.
    0 {:: t return.
  end.
  0 {:: y
)
payload =: 3 : 0
  NB. Payload of a Value (1st element); a raw char returns itself; a boxed char's
  NB. payload is the char it wraps.
  if. 2 = 3!:0 y do. y return. end.
  if. (1 = # y) *. (2 = 3!:0 (0 {:: y)) do. (> 0 { y) return. end.
  1 {:: y
)

norm =: 3 : 0
  NB. normalize a name string to scalar when it is a single char (strip/lexer return 1-vectors)
  if. 1 = # y do. {. y else. y end.
)

NB. ---- Predicates ----
is_null    =: 3 : '''null'' -: kind y'
is_bool    =: 3 : '''bool'' -: kind y'
is_int     =: 3 : '''int''  -: kind y'
is_float   =: 3 : '''float'' -: kind y'
is_str     =: 3 : '''str''  -: kind y'
is_arr     =: 3 : '''arr''  -: kind y'
is_obj     =: 3 : '''obj''  -: kind y'
is_call    =: 3 : '''callable'' -: kind y'
is_number          =: 3 : '(is_int y) +. (is_float y)'
is_number_integer  =: is_int
is_number_float    =: is_float
is_string          =: is_str
is_array           =: is_arr
is_object          =: is_obj
is_iterable        =: 3 : '(is_arr y) +. (is_obj y) +. (is_str y)'
is_hashable        =: 3 : '(is_str y) +. (is_int y) +. (is_float y) +. (is_bool y)'

to_num =: 3 : '{. payload y'

NB. ---- Truthiness (Python-like) ----
to_bool =: 3 : 0
  select. kind y
  case. 'null' do. 0
  case. 'bool' do. {. payload y
  case. 'int'  do. 0 ~: {. payload y
  case. 'float' do. 0 ~: {. payload y
  case. 'str'  do. 0 < # payload y
  case. 'arr'  do. 0 < arr_size y
  case. 'obj'  do. 1
  case. do. 1
  end.
)

NB. ---- to_int (Python int(); string parse fallback 0; float trunc toward 0) ----
to_int =: 3 : 0
  select. kind y
  case. 'null' do. 0
  case. 'bool' do. {. payload y
  case. 'int'  do. {. payload y
  case. 'float' do.
    n =. {. payload y
    if. n < 0 do. >. n else. <. n end.   NB. int() truncates toward zero
  case. 'str'  do.
    s =. payload y
    if. 0 = # s do. 0 else.
      n =. ". s
      if. 0 = # n do. 0 else.
        v =. {. n
        if. v < 0 do. >. v else. <. v end.   NB. int() truncates toward zero
      end.
    end.
  case. do. 0
  end.
)

NB. ---- to_str (Python str() semantics; arrays/objects -> dump) ----
to_str =: 3 : 0
  if. 2 = 3!:0 y do. y return. end.
  select. kind y
  case. 'str'  do. payload y
  case. 'int'  do. ": payload y
  case. 'float' do. ": payload y
  case. 'bool' do. if. payload y do. 'True' else. 'False' end.
  case. 'null' do. 'None'
  case. do. dump y
  end.
)

NB. ---- JSON string escaping (nlohmann-style double-quoted) ----
json_str =: 3 : 0
  NB. minimal JSON string literal: escape ", \, and control chars
  s =. y
  out =. ''
  i =. 0
  while. i < # s do.
    c =. i { s
    select. c
    case. '"'  do. out =. out , '\"'
    case. '\'  do. out =. out , '\\'
    case. LF   do. out =. out , '\n'
    case. TAB  do. out =. out , '\t'
    case. do. out =. out , c
    end.
    i =. i + 1
  end.
  '"' , out , '"'
)

NB. ---- Python-style string repr (single-quoted) ----
py_str =: 3 : 0
  NB. minja dump_string: if string contains a single quote, fall back to
  NB. double-quoted JSON; else wrap in single quotes, doubling backslashes.
  s =. y
  if. 0 < # (I. s = 39) do. json_str s
  else.
    esc =. ''
    i =. 0
    while. i < # s do.
      c =. i { s
      if. c = '\' do. esc =. esc , '\\' else. esc =. esc , c end.
      i =. i + 1
    end.
    '''' , esc , ''''
  end.
)

NB. ---- dump (Python repr) / dumpj (JSON) ----
fmt_num =: 3 : 0
  NB. convert J's "_10" negative display to Python's "-10"
  if. 0 < # y do. if. '_' -: 0 { y do. '-' , (1 }. y) else. y end. else. y end.
)
dump =: 3 : 'dumpc ((<y) , (<_1) , (<0) , (<0))'     NB. Python-style repr (single quotes, True/False)
dumpj =: 3 : 'dumpc ((<y) , (<_1) , (<0) , (<1))'    NB. to_json (double quotes, true/false/null)

dumpc =: 3 : 0
  'v indent level tojson' =. y
  select. kind v
  case. 'null' do. 'null'
  case. 'bool' do.
    if. tojson do. if. to_bool v do. 'true' else. 'false' end.
    else. if. to_bool v do. 'True' else. 'False' end. end.
  case. 'str' do.
    if. tojson do. json_str payload v else. py_str payload v end.
  case. 'int'  do. fmt_num ": payload v
  case. 'float' do. fmt_num ": payload v
  case. 'callable' do. 'error: cannot dump callable'
  case. 'arr' do.
    n =. arr_size v
    if. indent < 0 do.
      out =. '['
      i =. 0
      while. i < n do.
        if. i do. out =. out , ',' , ' ' end.   NB. compact: ", " separator
        out =. out , dumpc ((<((mkint i) arr_at v)) , (<indent) , (<(level + 1)) , (<tojson))
        i =. i + 1
      end.
      out , ']'
    else.
      out =. '['
      pad =. ((level + 1) * indent) # ' '
      i =. 0
      while. i < n do.
        out =. out , LF , pad , (dumpc ((<((mkint i) arr_at v)) , (<indent) , (<(level + 1)) , (<tojson)))
        if. i < n - 1 do. out =. out , ',' end.
        i =. i + 1
      end.
      out , LF , ((level * indent) # ' ') , ']'
    end.
  case. 'obj' do.
    ks =. obj_keys v
    vs =. obj_values v
    n =. # ks
    if. indent < 0 do.
      out =. '{'
      i =. 0
      while. i < n do.
        if. i do. out =. out , ',' , ' ' end.
        if. tojson do. kstr =. json_str (to_str (>(i { ks))) else. kstr =. py_str (to_str (>(i { ks))) end.
        out =. out , kstr , ': ' , (dumpc ((<(>(i { vs))) , (<indent) , (<(level + 1)) , (<tojson)))
        i =. i + 1
      end.
      out , '}'
    else.
      out =. '{'
      pad =. ((level + 1) * indent) # ' '
      i =. 0
      while. i < n do.
        out =. out , LF , pad
        if. tojson do. kstr =. json_str (to_str (>(i { ks))) else. kstr =. py_str (to_str (>(i { ks))) end.
        out =. out , kstr , ': ' , (dumpc ((<(>(i { vs))) , (<indent) , (<(level + 1)) , (<tojson)))
        if. i < n - 1 do. out =. out , ',' end.
        i =. i + 1
      end.
      out , LF , ((level * indent) # ' ') , '}'
    end.
  case. do. ''
  end.
)

NB. ---- Equality (numeric-tolerant; else structural via dump) ----
eq =: 4 : 0
  if. 2 = 3!:0 x do. x =. mkstr x end.
  if. 2 = 3!:0 y do. y =. mkstr y end.
  if. (is_number x) *. (is_number y) do. (to_num x) = (to_num y) return. end.
  if. (is_null x) *. (is_null y) do. 1 return. end.
  if. (is_null x) +. (is_null y) do. 0 return. end.   NB. null != any non-null
  if. (is_str x) *. (is_str y) do. (payload x) -: (payload y) return. end.
  if. (is_bool x) *. (is_bool y) do. (to_bool x) = (to_bool y) return. end.
  (dump x) -: (dump y)
)
neq =: 4 : '-. x eq y'

NB. ---- `in` (Python membership) ----
mem =: 4 : 0
  NB. x mem y -> 1 if x eq any element box of y
  for_i. i. # y do.
    if. x eq (i {:: y) do. 1 return. end.
  end.
  0
)
in =: 4 : 0
  if. (is_arr y) do. x mem arr_items y
  elseif. (is_obj y) do.
    x mem obj_keys y          NB. key in dict -> key exists
  elseif. (is_str x) *. (is_str y) do.
    ((payload y) i. (payload x)) < (# payload y)    NB. substring search (x in y)
  else. 0 end.
)

NB. ---- Array accessors (boxed list of single-boxed Values) ----
arr_items =: 3 : 'payload y'
arr_size  =: 3 : '# arr_items y'
arr_at    =: 4 : 0
  NB. x arr_at y  -> y[x] with negative wrap (x = int Value)
  elems =. arr_items y
  n =. # elems
  i =. {. to_int x
  if. i < 0 do. i =. i + n end.
  > (i { elems)
)

NB. ---- Object accessors (flat list of (boxed key ; boxed value) pairs) ----
obj_items  =: 3 : 'payload y'
obj_keys   =: 3 : 0
  it =. obj_items y
  if. 1 = # it do. it =. > it end.
  (2 * i. (>. # it) % 2) { it
)
obj_values =: 3 : 0
  it =. obj_items y
  if. 1 = # it do. it =. > it end.
  ((2 * i. (>. # it) % 2) + 1) { it
)

to_key =: 3 : 0
  NB. normalize a lookup key: char -> str Value; Value -> as-is
  if. 2 = 3!:0 y do. mkstr y else. y end.
)
key_find =: 3 : 0
  NB. y = (ks ; key) -> index of Value key in ks, or _1
  'ks key' =. y
  n =. # ks
  i =. 0
  while. i < n do.
    if. (i {:: ks) eq key do. i return. end.
    i =. i + 1
  end.
  _1
)
obj_get =: 4 : 0
  NB. x obj_get y -> y[x] (x = string key or Value key); null if missing.
  NB. For non-obj/array receivers (e.g. a string), return null (not index into them).
  if. -. (is_obj y) +. (is_arr y) do. mknull '' return. end.
  key =. to_key x
  it =. obj_items y
  ks =. obj_keys y
  i =. key_find ((<ks) , <key)
  if. i >: 0 do. > ((2 * i) + 1) { it else. mknull '' end.
)
obj_get_d =: 4 : 0
  NB. x obj_get_d y with default -> Value (default if missing); x = (key ; default)
  'k dflt' =. x
  key =. to_key k
  it =. obj_items y
  ks =. obj_keys y
  i =. key_find ((<ks) , <key)
  if. i >: 0 do. > ((2 * i) + 1) { it else. dflt end.
)
obj_set =: 4 : 0
  NB. x obj_set y -> y with key (string or Value) set to Value v; preserves order
  'k v' =. x
  key =. to_key k
  it =. obj_items y
  ks =. obj_keys y
  i =. key_find ((<ks) , <key)
  if. i >: 0 do.
    mkobj ((<v) ((2 * i) + 1)} it)
  else.
    mkobj ((it) , (<key) , (<v))
  end.
)

NB. ---- Context (scoped variable lookup, parent chain) ----
NB. Representation: ( <values_obj , <parent_ctx_or_null ) — both single-boxed items.
mkctx_np =: 3 : 0
  NB. mkctx_np values_obj  -> context with NO parent (root)
  (<y) , <(mknull '')
)
mkctx =: 3 : 0
  NB. mkctx values_obj  -> context with builtins parent
  (<y) , <(builtins '')
)
mkctx_child =: 3 : 0
  NB. mkctx_child values_obj  -> context with builtins parent (loop scopes)
  (<y) , <(builtins '')
)
ctx_values  =: 3 : '0 {:: y'
ctx_parent  =: 3 : '1 {:: y'

ctx_contains =: 4 : 0
  key =. to_key x
  'vals parent' =. y
  ks =. obj_keys vals
  i =. key_find ((<ks) , <key)
  i >: 0
)

ctx_get =: 4 : 0
  NB. ctx_get name -> Value; parent-chain lookup to builtins
  key =. to_key x
  'vals parent' =. y
  it =. obj_items vals
  ks =. obj_keys vals
  i =. key_find ((<ks) , <key)
  if. i >: 0 do. > ((2 * i) + 1) { it
  else.
    if. is_null parent do. mknull '' else. key ctx_get parent end.
  end.
)

ctx_set =: 4 : 0
  NB. (name ; value) ctx_set ctx -> new ctx with name set in the local frame
  'name value' =. x
  key =. to_key name
  'vals parent' =. y
  (<(((<key) , <value) obj_set vals)) , <parent
)

ctx_set_found =: 4 : 0
  NB. (name ; value) ctx_set_found ctx -> ctx with name updated where it
  NB. already lives in the parent chain (namespace mutation: the shared obj
  NB. must be updated so all referencing scopes — incl. the post-loop outer
  NB. ctx — see the new value).
  'name value' =. x
  key =. to_key name
  value ctx_set_found1 ((<key) , <y)
)

ctx_set_found1 =: 4 : 0
  NB. x = value; y = (key ; ctx) -> ctx with key updated at its found level
  value =. x
  'key ctx' =. y
  'vals parent' =. ctx
  ks =. obj_keys vals
  i =. key_find ((<ks) , <key)
  if. i >: 0 do.
    (<(((<key) , <value) obj_set vals)) , <parent
  elseif. is_null parent do.
    (<(((<key) , <value) obj_set vals)) , <parent
  else.
    (<vals) , <(value ctx_set_found1 ((<key) , <parent))
  end.
)

ctx_loop_outer =: 4 : 0
  NB. d ctx_loop_outer ctx -> parent (outer_ctx) of THIS loop's loopctx.
  NB. The loopctx's local obj carries the __loop_depth__ marker (== d);
  NB. nested loops carry deeper markers and are skipped. The returned parent
  NB. is the outer_ctx, updated with any namespace mutations made in the body.
  d =. x
  ctx =. y
  while. -. is_null ctx do.
    'vals parent' =. ctx
    it =. obj_items vals
    ks =. obj_keys vals
    mi =. key_find ((<ks) , <'__loop_depth__')
    if. mi >: 0 do.
      dv =. > ((2 * mi) + 1) { it
      if. (payload dv) = d do. parent return. end.
    end.
    ctx =. parent
  end.
  y
)

NB. ---- builtins context (globals/filters as callables) ----
builtins =: 3 : 0
  names =. ;: 'raise_exception strftime_now tojson items first last trim capitalize lower upper default escape e joiner count dictsort join namespace equalto length safe string int list in unique select reject selectattr rejectattr map indent range replace dbg_type'
  vals =. ''
  for_n. names do.
    vals =. vals , (<(mkstr (> n))) , <(mkcall (>n))
  end.
  mkctx_np (mkobj vals)
)

NB. ================================================================
NB. Phase 5B — Expression tokenizer / parser / evaluator
NB. ================================================================

NB. ---- string helpers ----
strip =: 3 : 0
  NB. y = (s ; chars) -> trim chars (default whitespace) from both ends
  if. 32 = 3!:0 y do.
    s =. 0 {:: y
    ws =. 1 {:: y
  else.
    s =. y
    ws =. ' ',TAB,LF,CR
  end.
  a =. I. -. s e. ws
  if. 0 = # a do. ''
  else.
    first =. {. a
    last =. {: a
    ((last - first) + 1) {. (first }. s)
  end.
)
lstrip =: 3 : 0
  s =. y
  ws =. ' ',TAB,LF,CR
  a =. I. -. s e. ws
  if. 0 = # a do. '' else. ({. a) }. s end.
)
rstrip =: 3 : 0
  s =. y
  ws =. ' ',TAB,LF,CR
  a =. I. -. s e. ws
  if. 0 = # a do. '' else. (1 + {: a) {. s end.
)
cap =: 3 : 0
  NB. capitalize: first char upper, rest lower
  s =. y
  if. 0 = # s do. s else. (toupper (0 { s)) , (tolower (1 }. s)) end.
)
title_s =: 3 : 0
  s =. y
  res =. ''
  prevspace =. 1
  for_c. s do.
    cc =. c
    if. prevspace do. res =. res , toupper cc else. res =. res , tolower cc end.
    prevspace =. cc e. ' ',TAB,LF,CR
  end.
  res
)
replace_all =: 4 : 0
  NB. (b ; a) replace_all s -> all occurrences of b replaced with a
  'b a' =. x
  s =. y
  if. 0 = # b do. s return. end.
  res =. ''
  rest =. s
  while. 0 < # rest do.
    i =. b find_s rest
    if. i >: # rest do. res =. res , rest break. end.
    res =. res , (i {. rest) , a
    rest =. (i + # b) }. rest
  end.
  res
)
replace_n =: 3 : 0
  NB. y = (b ; a ; n ; s) -> s with at most n occurrences of b replaced with a
  b =. 0 {:: y
  a =. 1 {:: y
  n =. 2 {:: y
  s =. 3 {:: y
  if. 0 = # b do. s return. end.
  res =. ''
  rest =. s
  cnt =. 0
  while. (0 < # rest) *. (cnt < n) do.
    i =. b find_s rest
    if. i >: # rest do. res =. res , rest break. end.
    res =. res , (i {. rest) , a
    rest =. (i + # b) }. rest
    cnt =. cnt + 1
  end.
  res , rest
)
split =: 3 : 0
  NB. y = (s ; sep) -> boxed list of substrings.
  NB. Mirrors C++ split(): always push the trailing substring even when empty,
  NB. so ''.split(sep) -> [''] and 'a sep'.split(' sep') -> ['a', ''].
  s =. 0 {:: y
  sep =. 1 {:: y
  res =. ''
  while. 0 < # s do.
    i =. sep find_s s
    if. i >: # s do. break. end.
    res =. res , <(i {. s)
    s =. (i + # sep) }. s
  end.
  res =. res , <s
  res
)
indent_s =: 3 : 0
  'txt n first' =. y
  pad =. n # ' '
  lines =. split (txt ; LF)
  if. (0 < # txt) *. (LF = _1 {. txt) do. lines =. }: lines end.
  out =. ''
  is_first =. 1
  for_l. lines do.
    line =. > l
    needs =. (-. is_first) +. first
    if. is_first do. is_first =. 0 end.
    if. needs do. out =. out , pad end.
    out =. out , line
    if. -. (l -: _1 { lines) do. out =. out , LF end.
  end.
  if. (0 < # txt) *. (LF = _1 {. txt) do. out =. out , LF end.
  out
)
endswith =: 3 : 0
  NB. y = (s ; suf) -> 1|0
  s =. 0 {:: y
  suf =. 1 {:: y
  if. (# suf) > # s do. 0 else. suf -: (((# s) - (# suf)) }. s) end.
)
startswith =: 3 : 0
  NB. y = (s ; pre) -> 1|0
  s =. 0 {:: y
  pre =. 1 {:: y
  if. (# pre) > # s do. 0 else. pre -: ((# pre) {. s) end.
)
find_s =: 4 : 0
  NB. x find_s y -> position of first occurrence of substring x in y, or (#y) if absent
  m =. x E. y
  if. 1 e. m do. m i. 1 else. # y end.
)
len_v =: 3 : 0
  select. kind y
  case. 'arr' do. arr_size y
  case. 'str' do. # payload y
  case. 'obj' do. # obj_keys y
  case. do. 0
  end.
)

NB. ---- expression tokenizer ----
NB. tokens: (kind ; text); kind = 'num'|'str'|'id'|'op'|'end'
lex_expr =: 3 : 0
  s =. y
  n =. # s
  toks =. ''
  i =. 0
  while. i < n do.
    c =. i { s
    if. c e. ' ',TAB,LF,CR do. i =. i+1 continue. end.
    NB. number
    isnum =. c e. '0123456789'
    if. (c = '.') do.
      if. ((i+1) < n) *. (((i+1) { s) e. '0123456789') do. isnum =. 1 end.
    end.
    if. isnum do.
      j =. i
      dp =. 0
      while. j < n do.
        cj =. j { s
        if. cj e. '0123456789' do.
        elseif. (cj = '.') *. (0 = dp) do. dp =. 1
        elseif. (cj = 'e') +. (cj = 'E') do.
          j =. j + 1
          if. j < n do.
            ej =. j { s
            if. (ej = '+') +. (ej = '-') do. j =. j + 1 end.
            while. j < n do.
              ej =. j { s
              if. ej e. '0123456789' do. j =. j + 1 else. break. end.
            end.
          end.
          break.
        else. break. end.
        j =. j + 1
      end.
      txt =. (j - i) {. (i }. s)
      toks =. toks , <(('num') ; txt)
      i =. j
      continue.
    end.
    NB. string
    if. (c = '''') +. (c = '"') do.
      q =. c
      j =. i + 1
      res =. ''
      esc =. 0
      while. j < n do.
        cj =. j { s
        if. esc do.
          esc =. 0
          select. cj
          case. 'n' do. res =. res , LF
          case. 'r' do. res =. res , CR
          case. 't' do. res =. res , TAB
          case. 'b' do. res =. res , 8{a.
          case. 'f' do. res =. res , 12{a.
          case. '\' do. res =. res , '\'
          case. do. if. cj = q do. res =. res , cj else. res =. res , cj end.
          end.
        elseif. cj = '\' do. esc =. 1
        elseif. cj = q do. j =. j + 1 break.
        else. res =. res , cj end.
        j =. j + 1
      end.
      toks =. toks , <(('str') ; res)
      i =. j
      continue.
    end.
    NB. identifier/keyword
    if. c e. 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_' do.
      j =. i
      while. j < n do.
        cj =. j { s
        if. -. (cj e. 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789') do. break. end.
        j =. j + 1
      end.
      txt =. (j - i) {. (i }. s)
      toks =. toks , <(('id') ; txt)
      i =. j
      continue.
    end.
    NB. multi-char operators
    if. (i+1) < n do.
      two =. (i { s) , ((i+1) { s)
      if. (<two) e. '==';'!=';'<=';'>=';'//';'**' do.
        toks =. toks , <(('op') ; two)
        i =. i + 2
        continue.
      end.
    end.
    NB. single char op/symbol
    toks =. toks , <(('op') ; c)
    i =. i + 1
  end.
  toks , <(('end') ; '')
)

NB. ---- expression parser (global cursor over toks_g) ----
toks_g =: ''
pc =: 0
n_toks =: 0

tk =: 3 : '0 {:: > (pc { toks_g)'
tx =: 3 : '1 {:: > (pc { toks_g)'
adv =: 3 : '(pc =: pc + 1)'
is_kw =: 3 : 0
  ('id' -: tk 0) *. (y -: tx 0)
)
is_op =: 3 : 0
  ('op' -: tk 0) *. (y -: tx 0)
)
reserved =: 3 : 0
  (<y) e. ;: 'not is and or del in'
)

parse_expr_str =: 3 : 0
  NB. y = expression body string -> expr AST node
  toks_g =: lex_expr y
  n_toks =: # toks_g
  pc =: 0
  node =. parse_expr 1
  node
)

parse_expr =: 3 : 0
  NB. y = allow_if_expr (1|0)
  left =. parse_logical_or ''
  if. pc >: n_toks do. left return. end.
  if. -. y do. left return. end.
  if. is_kw 'if' do.
    adv ''
    cond =. parse_logical_or ''
    else_e =. ''
    if. (pc < n_toks) *. (is_kw 'else') do.
      adv ''
      else_e =. parse_expr 1
    end.
left =. mknode ('ifexpr' ; <(((mkp cond)) , (mkp left)) , (mkp else_e))
  end.
  left
)

parse_logical_or =: 3 : 0
  left =. parse_logical_and ''
  while. (pc < n_toks) *. (is_kw 'or') do.
    adv ''
    right =. parse_logical_and ''
left =. mknode ('bin' ; <(((mkp 'or')) , (mkp left)) , (mkp right))
  end.
  left
)

parse_logical_and =: 3 : 0
  left =. parse_logical_not ''
  while. (pc < n_toks) *. (is_kw 'and') do.
    adv ''
    right =. parse_logical_not ''
left =. mknode ('bin' ; <(((mkp 'and')) , (mkp left)) , (mkp right))
  end.
  left
)

parse_logical_not =: 3 : 0
  if. (pc < n_toks) *. (is_kw 'not') do.
    adv ''
    sub =. parse_logical_not ''
mknode ('un' ; <((mkp 'not')) , (mkp sub))
  else.
    parse_logical_compare ''
  end.
)

parse_logical_compare =: 3 : 0
  left =. parse_string_concat ''
  while. pc < n_toks do.
    op =. ''
    if. is_kw 'not' do.
      if. ((pc+1) < n_toks) *. (('id') -: (0 {:: > (pc+1) { toks_g)) *. (('in') -: (1 {:: > (pc+1) { toks_g)) do.
        adv ''
        adv ''
        op =. 'notin'
      end.
    elseif. is_kw 'in' do. op =. 'in'
    elseif. is_kw 'is' do.
      adv ''
      neg =. 0
      if. (pc < n_toks) *. (is_kw 'not') do. adv '' ; neg =. 1 end.
      id =. parse_identifier ''
      if. ('' -: id) do. err 'Expected identifier after 'is' keyword' end.
      if. neg do. op =. 'isnot' else. op =. 'is' end.
left =. mknode ('bin' ; <(((mkp op)) , (mkp left)) , (mkp id))
      break.
    elseif. (is_op '==') do. op =. '=='
    elseif. (is_op '!=') do. op =. '!='
    elseif. (is_op '<') do. op =. '<'
    elseif. (is_op '>') do. op =. '>'
    elseif. (is_op '<=') do. op =. '<='
    elseif. (is_op '>=') do. op =. '>='
    end.
    if. 0 = # op do. break. end.
    if. -. (op -: 'notin') do. adv '' end.
    right =. parse_string_concat ''
left =. mknode ('bin' ; <(((mkp op)) , (mkp left)) , (mkp right))
  end.
  left
)

parse_string_concat =: 3 : 0
  left =. parse_math_pow ''
  if. (pc < n_toks) *. (is_op '~') do.
    adv ''
    right =. parse_logical_and ''
left =. mknode ('bin' ; <(((mkp 'concat')) , (mkp left)) , (mkp right))
  end.
  left
)

parse_math_pow =: 3 : 0
  left =. parse_math_plusminus ''
  while. (pc < n_toks) *. (is_op '**') do.
    adv ''
    right =. parse_math_plusminus ''
left =. mknode ('bin' ; <(((mkp '**')) , (mkp left)) , (mkp right))
  end.
  left
)

parse_math_plusminus =: 3 : 0
  left =. parse_math_muldiv ''
  while. (pc < n_toks) *. (is_op '+') +. (is_op '-') do.
    op =. tx ''
    adv ''
    right =. parse_math_muldiv ''
left =. mknode ('bin' ; <(((mkp op)) , (mkp left)) , (mkp right))
  end.
  left
)

parse_math_muldiv =: 3 : 0
  left =. parse_math_unary ''
  while. (pc < n_toks) *. (((is_op '*') +. (is_op '/') +. (is_op '//') +. (is_op '%'))) do.
    op =. tx ''
    adv ''
    right =. parse_math_unary ''
    left =. mknode ('bin' ; <(((mkp op)) , (mkp left)) , (mkp right))
  end.
  NB. filter chain '|'
  if. (pc < n_toks) *. (is_op '|') do.
    adv ''
    expr =. parse_math_muldiv ''
    if. ('filter' -: 0 {:: expr) do.
      parts =. 1 {:: expr
      left =. (('filter') ; <((<left) , parts))
    else.
      left =. (('filter') ; <((<left) , <expr))
    end.
  end.
  left
)

parse_math_unary =: 3 : 0
  op =. ''
  if. (pc < n_toks) *. (is_op '+') +. (is_op '-') do.
    op =. tx ''
    adv ''
  end.
  expr =. parse_expansion ''
  if. 0 = # op do. expr
  else. mknode ('un' ; <((mkp op) , (mkp expr))) end.
)

parse_expansion =: 3 : 0
  op =. ''
  if. (pc < n_toks) *. (is_op '*') +. (is_op '**') do.
    op =. tx ''
    adv ''
  end.
  expr =. parse_value_expr ''
  if. 0 = # op do. expr
  else.
    if. op -: '**' do. mknode ('un' ; <((mkp 'expansiondict') , (mkp expr)))
    else. mknode ('un' ; <((mkp 'expansion') , (mkp expr))) end.
  end.
)

parse_value_expr =: 3 : 0
  val =. parse_value ''
  while. pc < n_toks do.
    if. is_op '[' do.
      adv ''
      start =. ''
      end_e =. ''
      step =. ''
      has1 =. 0
      has2 =. 0
      if. -. (is_op ':') do.
        start =. parse_expr 1
      end.
      if. (pc < n_toks) *. (is_op ':') do.
        has1 =. 1
        adv ''
        if. -. (((is_op ':') +. (is_op ']'))) do.
          end_e =. parse_expr 1
        end.
        if. (pc < n_toks) *. (is_op ':') do.
          has2 =. 1
          adv ''
          if. -. (is_op ']') do.
            step =. parse_expr 1
          end.
        end.
      end.
      if. (has1 +. has2) do.
idx =. mknode ('slice' ; <(((mkp start)) , (mkp end_e)) , (mkp step))
      else.
        idx =. start
      end.
      if. ('' -: idx) do. err 'Empty index in subscript' end.
      if. -. (is_op ']') do. err 'Expected closing bracket in subscript' end.
      adv ''
val =. mknode ('sub' ; <((mkp val)) , (mkp idx))
    elseif. is_op '.' do.
      adv ''
      id =. parse_identifier ''
      if. ('' -: id) do. err 'Expected identifier in subscript' end.
      if. (pc < n_toks) *. (is_op '(') do.
        args =. parse_call_args ''
val =. mknode ('mcall' ; <(((mkp val)) , (mkp id)) , (mkp args))
      else.
      val =. mknode ('sub' ; <((mkp val) , (mkp ('lit' ; <(mkstr (1 {:: id))))))
      end.
    elseif. is_op '(' do.
      args =. parse_call_args ''
val =. mknode ('call' ; <((mkp val)) , (mkp args))
    else.
      break.
    end.
  end.
  val
)

parse_value =: 3 : 0
  if. (pc < n_toks) *. (('str') -: tk '') do.
    v =. mkstr (tx '')
    adv ''
    (('lit') ; <v) return.
  end.
  if. (pc < n_toks) *. (('id') -: tk '') do.
    t =. tx ''
    select. t
    case. 'true' do.
      adv ''
      (('lit') ; <(mkbool 1)) return.
    case. 'True' do.
      adv ''
      (('lit') ; <(mkbool 1)) return.
    case. 'false' do.
      adv ''
      (('lit') ; <(mkbool 0)) return.
    case. 'False' do.
      adv ''
      (('lit') ; <(mkbool 0)) return.
    case. 'none' do.
      adv ''
      (('lit') ; <(mknull '')) return.
    case. 'None' do.
      adv ''
      (('lit') ; <(mknull '')) return.
    case. 'null' do.
      adv ''
      (('lit') ; <(mknull '')) return.
    end.
  end.
  if. (pc < n_toks) *. (('num') -: tk '') do.
    t =. tx ''
    adv ''
    num =. ". t
    if. -. (1 e. (t e. '.eE')) do. v =. mkint num else. v =. mkfloat num end.
    (('lit') ; <v) return.
  end.
  id =. parse_identifier ''
  if. -. ('' -: id) do. id return. end.
  b =. parse_braced ''
  if. -. ('' -: b) do. b return. end.
  a =. parse_array ''
  if. -. ('' -: a) do. a return. end.
  d =. parse_dict ''
  if. -. ('' -: d) do. d return. end.
  err 'Expected value expression'
)

parse_identifier =: 3 : 0
  if. (pc < n_toks) *. (('id') -: tk '') do.
    t =. tx ''
    if. -. (reserved t) do.
      adv ''
      (('var') ; t) return.
    end.
  end.
  ''
)

parse_braced =: 3 : 0
  if. -. (is_op '(') do. '' return. end.
  adv ''
  expr =. parse_expr 1
  if. is_op ')' do.
    adv ''
    expr return.
  end.
  tup =. <expr
  while. 1 do.
    if. -. (is_op ',') do. err 'Expected comma in tuple' end.
    adv ''
    nx =. parse_expr 1
    tup =. tup , <nx
    if. is_op ')' do.
      adv ''
      (('arr') ; <tup) return.
    end.
  end.
)

parse_array =: 3 : 0
  if. -. (is_op '[') do. '' return. end.
  adv ''
  els =. ''
  if. is_op ']' do.
    adv ''
    (('arr') ; <els) return.
  end.
  els =. <(parse_expr 1)
  while. 1 do.
    if. is_op ',' do.
      adv ''
      els =. els , <(parse_expr 1)
    elseif. is_op ']' do.
      adv ''
      (('arr') ; <els) return.
    else.
      err 'Expected comma or closing bracket in array'
    end.
  end.
)

parse_dict_pair =: 3 : 0
  k =. parse_expr 1
  if. -. (is_op ':') do. err 'Expected colon between key & value in dictionary' end.
  adv ''
  v =. parse_expr 1
  (<k) , <v
)

parse_dict =: 3 : 0
  if. -. (is_op '{') do. '' return. end.
  adv ''
  els =. ''
  if. is_op '}' do.
    adv ''
    (('dict') ; <els) return.
  end.
  els =. <(parse_dict_pair '')
  while. 1 do.
    if. is_op ',' do.
      adv ''
      els =. els , <(parse_dict_pair '')
    elseif. is_op '}' do.
      adv ''
      (('dict') ; <els) return.
    else.
      err 'Expected comma or closing brace in dictionary'
    end.
  end.
)

parse_call_args =: 3 : 0
  adv ''   NB. consume '('
  pos =. ''
  kw =. ''
  while. pc < n_toks do.
    if. is_op ')' do.
      adv ''
      break.
    end.
    expr =. parse_expr 1
    if. ('var' -: 0 {:: expr) do.
      nm =. 1 {:: expr
      if. is_op '=' do.
        adv ''
        val =. parse_expr 1
        kw =. kw , <((<nm) , <val)
      else.
        pos =. pos , <expr
      end.
    else.
      pos =. pos , <expr
    end.
    if. is_op ',' do.
      adv ''
    elseif. is_op ')' do.
      adv ''
      break.
    else.
      err 'Expected closing parenthesis in call args'
    end.
  end.
mknode ('args' ; <((mkp pos)) , (mkp kw))
)

NB. ---- expression evaluator (global ctx_g) ----
ctx_g =: ''

eval_expr =: 3 : 0
  node =. y
  select. 0 {:: node
  case. 'lit' do. 1 {:: node
  case. 'var' do.
    (1 {:: node) ctx_get ctx_g
  case. 'bin' do.
    op =. 1 {:: node
    l =. 2 {:: node
    r =. 3 {:: node
    eval_bin (((<op) , <l) , <r)
  case. 'un' do.
    op =. 1 {:: node
    e =. 2 {:: node
    v =. eval_expr e
    select. op
    case. 'not' do. mkbool (0 = to_bool v)
    case. '-' do. neg_v v
    case. '+' do. v
    case. 'expansion' do. err 'Expansion operator is only supported in function calls and collections'
    case. 'expansiondict' do. err 'Expansion operator is only supported in function calls and collections'
    end.
  case. 'arr' do.
    els =. 1 {:: node
    vals =. ''
    for_e. els do. vals =. vals , <(eval_expr (> e)) end.
    mkarr vals
  case. 'dict' do.
    pairs =. 1 {:: node
    objpairs =. ''
    for_p. pairs do.
      kv =. > p
      k =. eval_expr (0 {:: kv)
      v =. eval_expr (1 {:: kv)
      objpairs =. objpairs , <((<k) , <v)
    end.
    mkobj (; objpairs)
  case. 'sub' do.
    base =. 1 {:: node
    idx =. 2 {:: node
    v =. eval_expr base
    if. ('slice' -: 0 {:: idx) do.
      slice_sub ((<idx) , <v)
    else.
      iv =. eval_expr idx
      sub_at (((<iv) , <v) , <base)
    end.
  case. 'call' do.
    target =. 1 {:: node
    args =. 2 {:: node
    callable =. eval_expr target
    'posv kwv' =. eval_args args
    ((<posv) , <kwv) call callable
  case. 'mcall' do.
    obj =. 1 {:: node
    m =. 2 {:: node
    args =. 3 {:: node
    mname =. 1 {:: m
    objv =. eval_expr obj
    'posv kwv' =. eval_args args
    res =. ((<objv) , <mname) m_call ((<posv) , <kwv)
    if. (<mname) e. ;: 'append pop insert' do.
      res =. mut_write ((<obj) , <res)
    end.
    res
  case. 'filter' do.
    parts =. 1 {:: node
    res =. eval_expr (0 {:: parts)
    i =. 1
    while. i < # parts do.
      pe =. i {:: parts
      if. ('call' -: 0 {:: pe) do.
        target =. 1 {:: pe
        args =. 2 {:: pe
        callable =. eval_expr target
        'posv kwv' =. eval_args args
        res =. ((<((<res) , posv)) , <kwv) call callable
      elseif. ('var' -: 0 {:: pe) do.
        callable =. eval_expr pe
        res =. ((<(<res)) , <'') call callable
      elseif. ('bin' -: 0 {:: pe) do.
        op =. 1 {:: pe
        lpart =. 2 {:: pe
        rpart =. 3 {:: pe
        if. ('var' -: 0 {:: lpart) do.
          callable =. eval_expr lpart
          r1 =. ((<(<res)) , <'') call callable
        else.
          r1 =. eval_expr lpart
        end.
        rv =. eval_expr rpart
        res =. binop (((<op) , <r1) , <rv)
      else.
        err 'Unsupported filter part'
      end.
      i =. i + 1
    end.
    res
  case. 'ifexpr' do.
    cond =. 1 {:: node
    t =. 2 {:: node
    e =. 3 {:: node
    if. to_bool (eval_expr cond) do. eval_expr t
    else.
      if. ('__nil__' -: e) do. mknull '' else. eval_expr e end.
    end.
  case. do. err 'Unknown expression node'
  end.
)

eval_args =: 3 : 0
  NB. y = args node ('args'; pos_exprs ; kw_pairs) -> boxed (posv ; kwv)
  pos =. 1 {:: y
  kw =. 2 {:: y
  if. ('__nil__' -: pos) do. pos =. '' end.
  if. ('__nil__' -: kw) do. kw =. '' end.
  posv =. ''
  for_p. pos do.
    pe =. > p
    if. ('un' -: 0 {:: pe) *. ('expansion' -: 1 {:: pe) do.
      arr =. eval_expr (2 {:: pe)
      posv =. posv , arr_items arr
    else.
      posv =. posv , <(eval_expr pe)
    end.
  end.
  kwv =. ''
  for_k. kw do.
    kv =. > k
    kwv =. kwv , <((<0 {:: kv) , <(eval_expr (1 {:: kv)))
  end.
  (<posv) , <kwv
)

eval_bin =: 3 : 0
  op =. 0 {:: y
  l =. 1 {:: y
  r =. 2 {:: y
  lv =. eval_expr l
  select. op
  case. 'and' do.
    if. 0 = to_bool lv do. mkbool 0 else. mkbool (to_bool (eval_expr r)) end.
  case. 'or' do.
    if. to_bool lv do. lv else. eval_expr r end.
  case. 'is' ; 'isnot' do.
    res =. (1 {:: r) test_type lv
    if. op -: 'is' do. mkbool res else. mkbool (0 = res) end.
  case. do.
    rv =. eval_expr r
    binop (((<op) , <lv) , <rv)
  end.
)

test_type =: 4 : 0
  NB. (name) test_type (value) -> 1|0
  select. x
  case. 'none' do. is_null y
  case. 'boolean' do. is_bool y
  case. 'integer' do. is_int y
  case. 'float' do. is_float y
  case. 'number' do. is_number y
  case. 'string' do. is_str y
  case. 'mapping' do. is_obj y
  case. 'iterable' do. is_iterable y
  case. 'sequence' do. is_arr y
  case. 'defined' do. -. (is_null y)
  case. 'true' do. to_bool y
  case. 'false' do. 0 = to_bool y
  case. do. err 'Unknown type for ''is'' operator: ' , x
  end.
)

binop =: 3 : 0
  'op lv rv' =. y
  select. op
  case. 'concat' do. mkstr ((to_str lv) , (to_str rv))
  case. '+' do.
    if. (is_arr lv) *. (is_arr rv) do.
      mkarr ((arr_items lv) , (arr_items rv))
    elseif. (is_number lv) *. (is_number rv) do.
      ((to_num lv) + (to_num rv)) mknum ((is_int lv) *. (is_int rv))
    elseif. (is_str lv) +. (is_str rv) do.
      mkstr ((to_str lv) , (to_str rv))
    else. err 'Cannot add' end.
  case. '-' do.
    if. (is_number lv) *. (is_number rv) do.
      ((to_num lv) - (to_num rv)) mknum ((is_int lv) *. (is_int rv))
    else. err 'Cannot subtract' end.
  case. '*' do.
    if. (is_number lv) *. (is_number rv) do.
      ((to_num lv) * (to_num rv)) mknum ((is_int lv) *. (is_int rv))
    elseif. (is_str lv) *. (is_int rv) do.
      mkstr (; ((to_int rv) # <(payload lv)))
    elseif. (is_arr lv) *. (is_int rv) do.
      mkarr (; ((to_int rv) # <(arr_items lv)))
    else. err 'Cannot multiply' end.
  case. '/' do.
    if. (is_number lv) *. (is_number rv) do.
      mkfloat ((to_num lv) % (to_num rv))
    else. err 'Cannot divide' end.
  case. '**' do.
    if. (is_number lv) *. (is_number rv) do.
      mkfloat ((to_num lv) ^ (to_num rv))
    else. err 'Cannot power' end.
  case. '//' do.
    if. (is_number lv) *. (is_number rv) do.
      mkint (<. ((to_num lv) % (to_num rv)))
    else. err 'Cannot floor-divide' end.
  case. '%' do.
    if. (is_int lv) *. (is_int rv) do.
      mkint ((to_int lv) - (to_int rv) * (<. ((to_int lv) % (to_int rv))))
    else. err 'Cannot mod' end.
  case. '==' do. mkbool (lv eq rv)
  case. '!=' do. mkbool (0 = (lv eq rv))
  case. '<' do. mkbool (lv compare_lt rv)
  case. '>' do. mkbool (rv compare_lt lv)
  case. '<=' do. mkbool (-. (rv compare_lt lv))
  case. '>=' do. mkbool (-. (lv compare_lt rv))
  case. 'in' do. mkbool (lv in rv)
  case. 'notin' do. mkbool (0 = (lv in rv))
  case. do. err 'Unknown binary operator'
  end.
)

compare_lt =: 4 : 0
  if. (is_number x) *. (is_number y) do. (to_num x) < (to_num y)
  elseif. (is_str x) *. (is_str y) do.
    if. (payload x) -: (payload y) do. 0 else. {. ((payload x) < (payload y)) end.
  else. err 'Cannot compare' end.
)

neg_v =: 3 : 0
  if. is_int y do. mkint (- to_int y)
  elseif. is_float y do. mkfloat (- to_num y)
  else. err 'Cannot negate' end.
)

wrap =: 3 : 0
  NB. wrap (i ; len) -> wrapped i
  'i len' =. y
  if. i < 0 do. i + len else. i end.
)

sub_at =: 3 : 0
  'idxv target base' =. y
  if. is_null target do. err 'Cannot subscript null' end.
  select. kind target
  case. 'arr' do. idxv arr_at target
  case. 'str' do.
    NB. only integer subscripts index into a string's chars; a string key
    NB. (e.g. "abc".content) is undefined -> null (C++ at() throws on non-obj).
    if. -. (is_int idxv) do. mknull '' return. end.
    i =. to_int idxv
    n =. # payload target
    if. i < 0 do. i =. i + n end.
    mkstr (i { payload target)
  case. 'obj' do.
    idxv obj_get target
  case. do. err 'Subscripting only supported on arrays and strings'
  end.
)

mut_write =: 3 : 0
  NB. y = (path_node ; new_value): write new_value back through a var/sub path
  'node nv' =. y
  if. ('var' -: 0 {:: node) do.
    name =. 1 {:: node
    ctx_g =: ((<name) , <nv) ctx_set ctx_g
  elseif. ('sub' -: 0 {:: node) do.
    base =. 1 {:: node
    idx =. 2 {:: node
    bv =. eval_expr base
    if. ('lit' -: 0 {:: idx) do.
      kk =. payload (1 {:: idx)
      nb =. ((<kk) , <nv) obj_set bv
      mut_write ((<base) , <nb)
    else.
      err 'Cannot mutate subscripted target'
    end.
  else. err 'Cannot mutate target' end.
  nv
)

slice_sub =: 3 : 0
  'snode target' =. y
  s =. 1 {:: snode
  e =. 2 {:: snode
  st =. 3 {:: snode
  len =. len_v target
  if. ('__nil__' -: st) do. step =. 1 else. step =. to_int (eval_expr st) end.
  if. 0 = step do. err 'slice step cannot be zero' end.
  if. ('__nil__' -: s) do.
    if. step < 0 do. start =. len - 1 else. start =. 0 end.
  else.
    start =. wrap ((to_int (eval_expr s)) ; len)
  end.
  if. ('__nil__' -: e) do.
    if. step < 0 do. endv =. _1 else. endv =. len end.
  else.
    endv =. wrap ((to_int (eval_expr e)) ; len)
  end.
  if. is_str target do.
    str =. payload target
    res =. ''
    if. (start < endv) *. (1 = step) do.
      res =. (endv - start) {. (start }. str)
    else.
      i =. start
      while. if. step > 0 do. i < endv else. i > endv end. do.
        res =. res , (i { str)
        i =. i + step
      end.
    end.
    mkstr res
  elseif. is_arr target do.
    els =. arr_items target
    res =. 0 $ <''
    i =. start
    while. if. step > 0 do. i < endv else. i > endv end. do.
      res =. res , (i { els)
      i =. i + step
    end.
    mkarr res
  else. err 'Subscripting only supported on arrays and strings' end.
)

m_call =: 4 : 0
  NB. x = <objv ; mname) ; y = <posv ; kwv)
  objv =. 0 {:: x
  mname =. 1 {:: x
  posv =. 0 {:: y
  kwv =. 1 {:: y
  select. kind objv
  case. 'str' do.
    s =. payload objv
    select. mname
    case. 'strip' do.
      if. 0 < # posv do. mkstr (strip (s ; (to_str (0 {:: posv)))) else. mkstr (strip s) end.
    case. 'lstrip' do. mkstr (lstrip s)
    case. 'rstrip' do. mkstr (rstrip s)
    case. 'split' do.
      parts =. split (s ; (to_str (0 {:: posv)))
      arr =. ''
      for_p. parts do.
        arr =. arr , <(mkstr (> p))
      end.
      mkarr arr
    case. 'capitalize' do. mkstr (cap s)
    case. 'upper' do. mkstr (toupper s)
    case. 'lower' do. mkstr (tolower s)
    case. 'endswith' do. mkbool (endswith (s ; (to_str (0 {:: posv))))
    case. 'startswith' do. mkbool (startswith (s ; (to_str (0 {:: posv))))
    case. 'title' do. mkstr (title_s s)
    case. 'replace' do.
      rb =. to_str (0 {:: posv)
      ra =. to_str (1 {:: posv)
      if. 2 < # posv do.
        mkstr (replace_n (((((<rb) , <ra) , <(to_int (2 {:: posv))) , <s)))
      else.
        mkstr (((<rb) , <ra) replace_all s)
      end.
    case. do. err 'Unknown method: ' , mname
    end.
  case. 'arr' do.
    els =. arr_items objv
    select. mname
    case. 'append' do. mkarr ((arr_items objv) , <(0 {:: posv))
    case. 'pop' do.
      if. 0 = # els do. err 'pop from empty list' end.
      if. 0 = # posv do.
        mkarr (}: els)
      else.
        i =. to_int (0 {:: posv)
        mkarr ((i {. els) , ((i + 1) }. els))
      end.
    case. 'insert' do.
      i =. to_int (0 {:: posv)
      mkarr ((i {. els) , (<(1 {:: posv)) , ((i) }. els))
    case. do. err 'Unknown method: ' , mname
    end.
  case. 'obj' do.
    select. mname
    case. 'items' do.
      ks =. obj_keys objv
      res =. 0 $ <''
      for_k. ks do.
        k =. > k
        res =. res , <(mkarr ((<k) , <(k obj_get objv)))
      end.
      mkarr res
    case. 'keys' do.
      ks =. obj_keys objv
      res =. 0 $ <''
      for_k. ks do. res =. res , <(mkstr (to_str (> k))) end.
      mkarr res
    case. 'get' do.
      k =. 0 {:: posv
      if. 1 < # posv do. ((<k) , <(1 {:: posv)) obj_get_d objv else. (k) obj_get objv end.
    case. 'cycle' do.
      ci =. loop_cycle_i_g
      n =. # posv
      if. 0 = n do. mknull ''
      else.
        res =. ci {:: posv
        loop_cycle_i_g =: n | (ci + 1)
        res
      end.
    case. 'pop' do.
      it =. obj_items objv
      ks =. obj_keys objv
      if. 0 = # posv do.
        if. 0 = # ks do. err 'pop from empty dict' end.
        i =. 0
      else.
        k =. to_key (0 {:: posv)
        i =. key_find ((<ks) , <k)
        if. i < 0 do. err 'Key not found: ' , (to_str k) end.
      end.
      mkobj (((2 * i) {. it) , (((2 * i) + 2) }. it))   NB. drop the key/value pair at i
    case. do. err 'Unknown method: ' , mname
    end.
  case. 'null' do. err 'Trying to call method on null'
  case. do. err 'Unknown method: ' , mname
  end.
)

NB. ================================================================
NB. ---- strftime (separable: self-contained, lift-out-able) ----
NB. strftime_s (format ; epoch_seconds) -> formatted string (UTC).
NB. Supports the directives used by chat templates (minja strftime_now):
NB.   %Y %y %m %d %e %H %M %S %b %B %a %A %I %p %j %F %T %%
NB. civil_from_days + seconds decomposition via Howard Hinnant's
NB. date algorithms (proleptic Gregorian, UTC).
NB. ================================================================
strftime_now_g =: 0   NB. epoch seconds captured by strftime_now callable

civil_from_days =: 3 : 0
  NB. y = days since Unix epoch (UTC) -> y m d (proleptic Gregorian)
  z =. y + 719468
  era =. <. (z % 146097)
  doe =. z - era * 146097
  yoe =. <. ((doe - (<. (doe % 1460)) + (<. (doe % 36524)) - (<. (doe % 146096))) % 365)
  yy =. yoe + era * 400
  doy =. doe - ((365 * yoe) + (<. (yoe % 4)) - (<. (yoe % 100)))
  mp =. <. ((5 * doy) + 2) % 153
  d =. (doy - (<. ((153 * mp) + 2) % 5)) + 1
  m =. mp + (3 * (mp < 10)) - (9 * (mp >: 10))
  yy =. yy + (m <: 2)
  yy , m , d
)

strftime_mon_ab =: 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'
strftime_mon_full =: 'January February March April May June July August September October November December'
strftime_dow_ab =: 'Sun Mon Tue Wed Thu Fri Sat'
strftime_dow_full =: 'Sunday Monday Tuesday Wednesday Thursday Friday Saturday'

strftime_dow =: 3 : 0
  NB. y = days since epoch -> weekday 0=Sun
  7 | (y + 4)
)

strftime_td =: 3 : 0
  NB. y = seconds-of-day -> h mi s
  s =. y
  h =. <. (s % 3600)
  mi =. <. (s - h * 3600) % 60
  se =. s - (h * 3600) + (mi * 60)
  h , mi , se
)

strftime_pad2 =: 3 : 0
  if. y < 10 do. '0' , ": y else. ": y end.
)

strftime_pad4 =: 3 : 0
  s =. ": y
  while. 4 > # s do. s =. '0' , s end.
  s
)

strftime_s =: 3 : 0
  'fmt ts' =. y
  days =. <. (ts % 86400)
  'y m d' =. civil_from_days days
  'h mi s' =. strftime_td (ts - days * 86400)
  wd =. strftime_dow days
  out =. ''
  i =. 0
  while. i < # fmt do.
    c =. i { fmt
    if. c ~: '%' do.
      out =. out , c
      i =. i + 1
      continue.
    end.
    i =. i + 1
    if. i >: # fmt do. out =. out , '%' break. end.
    c2 =. i { fmt
    select. c2
    case. 'Y' do. out =. out , strftime_pad4 y
    case. 'y' do. out =. out , strftime_pad2 (100 | y)
    case. 'm' do. out =. out , strftime_pad2 m
    case. 'd' do. out =. out , strftime_pad2 d
    case. 'e' do. if. d < 10 do. out =. out , ' ' , ": d else. out =. out , ": d end.
    case. 'H' do. out =. out , strftime_pad2 h
    case. 'M' do. out =. out , strftime_pad2 mi
    case. 'S' do. out =. out , strftime_pad2 s
    case. 'b' do. out =. out , ((m - 1) {:: (;: strftime_mon_ab))
    case. 'B' do. out =. out , ((m - 1) {:: (;: strftime_mon_full))
    case. 'a' do. out =. out , (wd {:: (;: strftime_dow_ab))
    case. 'A' do. out =. out , (wd {:: (;: strftime_dow_full))
    case. 'I' do.
      ih =. 12 | h
      if. ih = 0 do. ih =. 12 end.
      out =. out , strftime_pad2 ih
    case. 'p' do. if. h < 12 do. out =. out , 'AM' else. out =. out , 'PM' end.
    case. 'j' do.
      leap =. ((0 = 4 | y) *. -. 0 = 100 | y) +. 0 = 400 | y
      mdays =. 31 28 31 30 31 30 31 31 30 31 30 31
      if. leap do. mdays =. 31 29 31 30 31 30 31 31 30 31 30 31 end.
      doy2 =. d + (+/ (m - 1) {. mdays)
      out =. out , strftime_pad2 doy2
    case. 'F' do. out =. out , (strftime_pad4 y) , '-' , (strftime_pad2 m) , '-' , (strftime_pad2 d)
    case. 'T' do. out =. out , (strftime_pad2 h) , ':' , (strftime_pad2 mi) , ':' , (strftime_pad2 s)
    case. '%' do. out =. out , '%'
    case. do. out =. out , '%' , c2
    end.
    i =. i + 1
  end.
  out
)

call =: 4 : 0
  NB. x = <posv ; kwv) ; y = callable ('callable'; name)
  posv =. 0 {:: x
  kwv =. 1 {:: x
  if. ('macro' -: 0 {:: y) do.
    pl0 =. 1 {:: y
    mname =. 0 {:: pl0
    mparams =. 1 {:: pl0
    mbody =. 2 {:: pl0
    defctx =. 3 {:: pl0
    exc =. (<(mkobj '')) , <defctx
    NB. propagate caller from current call ctx
    if. 'caller' ctx_contains ctx_g do.
      ccall =. 'caller' ctx_get ctx_g
      exc =. ((<'caller') , <ccall) ctx_set exc

    end.
    np =. # mparams
    if. ('__nil__' -: mparams) do. np =. 0 end.
    for_i. i. np do.
      if. i < # posv do.
        exc =. ((<0 {:: (i {:: mparams)) , <(i {:: posv)) ctx_set exc
      else.
        pdef =. 1 {:: (i {:: mparams)
        if. -. ('' -: pdef) do.
          oldctx =. ctx_g
          ctx_g =: defctx
          dv =. eval_expr pdef
          ctx_g =: oldctx
          exc =. ((<0 {:: (i {:: mparams)) , <dv) ctx_set exc
        end.
      end.
    end.
    for_k. kwv do.
      kv =. > k
      exc =. ((<0 {:: kv) , <(1 {:: kv)) ctx_set exc
    end.
    oldctx =. ctx_g
    ctx_g =: exc
    out =. rnode mbody
    ctx_g =: oldctx
    mkstr out
  elseif. ('caller' -: 0 {:: y) do.
    cpl0 =. 1 {:: y
    cbody =. 0 {:: cpl0
    cctx =. 1 {:: cpl0
    oldctx =. ctx_g
    ctx_g =: cctx
    out =. rnode cbody
    ctx_g =: oldctx
    mkstr out
  else.
  name =. 1 {:: y
  select. name
  case. (,'e') do. mkstr (html_esc (to_str (0 {:: posv)))
  case. (,'escape') do. mkstr (html_esc (to_str (0 {:: posv)))
  case. 'tojson' do.
    ind =. _1
    for_k. kwv do.
      kv =. > k
      if. (0 {:: kv) -: 'indent' do. ind =. to_int (1 {:: kv) end.
    end.
    if. ind < 0 do.
      mkstr (dumpj (0 {:: posv))
    else.
      mkstr (dumpc ((< (0 {:: posv)) , (<ind) , (<0) , (<1)))
    end.
  case. 'join' do.
    items =. 0 {:: posv
    if. 1 < # posv do. sep =. to_str (1 {:: posv) else. sep =. '' end.
    if. -. (is_arr items) do. err 'object is not iterable' end.
    mkstr (join_s ((<arr_items items) , <sep))
  case. 'joiner' do.
    joiner_sep_g =: to_str (0 {:: posv)
    joiner_first_g =: 1
    mkcall 'joiner_inst'
  case. 'joiner_inst' do.
    if. joiner_first_g do.
      joiner_first_g =: 0
      mkstr ''
    else.
      mkstr joiner_sep_g
    end.
  case. 'first' do.
    items =. 0 {:: posv
    if. -. (is_arr items) do. err 'object is not a list' end.
    if. 0 = arr_size items do. mknull '' else. (mkint 0) arr_at items end.
  case. 'last' do.
    items =. 0 {:: posv
    if. -. (is_arr items) do. err 'object is not a list' end.
    if. 0 = arr_size items do. mknull '' else. (mkint _1) arr_at items end.
  case. 'length' do. mkint (len_v (0 {:: posv))
  case. 'dbg_type' do. mkstr (kind (> 0 { posv))
  case. 'namespace' do.
    ns =. mkobj ''
    for_k. kwv do.
      kv =. > k
      ns =. ((<0 {:: kv) , <(1 {:: kv)) obj_set ns
    end.
    ns
  case. 'indent' do.
    txt =. to_str (0 {:: posv)
    if. 1 < # posv do. n =. to_int (1 {:: posv) else. n =. 0 end.
    first =. 0
    if. 0 < # kwv do. first =. to_bool (1 {:: (0 {:: kwv)) end.
    mkstr (indent_s (txt ; n ; first))
  case. 'list' do.
    items =. 0 {:: posv
    if. -. (is_arr items) do. err 'object is not iterable' end.
    items
  case. 'map' do.
    arr =. 0 {:: posv
    res =. 0 $ <''
    if. 0 < # kwv do.
      attr =. to_str (1 {:: (0 {:: kwv))
      for_e. (1 {:: arr) do.
        e =. > e
        res =. res , <(attr obj_get e)
      end.
    else.
      fname =. to_str (1 {:: posv)
      for_e. (1 {:: arr) do.
        e =. > e
        res =. res , <((<(<e)) , <'') call (mkcall fname)
      end.
    end.
    mkarr res
  case. 'upper' do. mkstr (toupper (to_str (0 {:: posv)))
  case. 'lower' do. mkstr (tolower (to_str (0 {:: posv)))
  case. 'capitalize' do. mkstr (cap (to_str (0 {:: posv)))
  case. 'trim' do. if. is_null (0 {:: posv) do. 0 {:: posv else. mkstr (strip (to_str (0 {:: posv))) end.
  case. 'replace' do.
    rb =. to_str (1 {:: posv)
    ra =. to_str (2 {:: posv)
    if. 3 < # posv do.
      mkstr (replace_n (((((<rb) , <ra) , <(to_int (3 {:: posv))) , <(to_str (0 {:: posv)))))
    else.
      mkstr (((<rb) , <ra) replace_all (to_str (0 {:: posv)))
    end.
  case. 'default' do.
    v =. 0 {:: posv
    d =. 1 {:: posv
    if. 2 < # posv do. boo =. to_bool (2 {:: posv) else. boo =. 0 end.
    if. boo do. if. to_bool v do. v else. d end. else. if. is_null v do. d else. v end. end.
  case. 'items' do.
    obj =. 0 {:: posv
    if. -. (is_obj obj) do. err 'Can only get item pairs from a mapping' end.
    ks =. obj_keys obj
    res =. ''
    for_k. ks do.
      k =. > k
      res =. res , <(mkarr ((<k) , <(k obj_get obj)))
    end.
    mkarr res
  case. 'dictsort' do.
    obj =. 0 {:: posv
    ks =. obj_keys obj
    ks =. (/:(to_str each ks)) { ks
    res =. ''
    for_k. ks do.
      k =. > k
      res =. res , <(mkarr ((<k) , <(k obj_get obj)))
    end.
    mkarr res
  case. 'range' do.
    if. 1 = # posv do.
      s0 =. 0
      e0 =. to_int (0 {:: posv)
      st0 =. 1
    else.
      s0 =. to_int (0 {:: posv)
      e0 =. to_int (1 {:: posv)
      if. 2 < # posv do. st0 =. to_int (2 {:: posv) else. st0 =. 1 end.
    end.
    res =. ''
    if. st0 > 0 do.
      i =. s0
      while. i < e0 do.
        res =. res , <(mkint i)
        i =. i + st0
      end.
    else.
      i =. s0
      while. i > e0 do.
        res =. res , <(mkint i)
        i =. i + st0
      end.
    end.
    mkarr res
  case. 'string' do. mkstr (to_str (> 0 { posv))
  case. 'int' do. mkint (to_int (0 {:: posv))
  case. 'safe' do. 0 {:: posv
  case. 'count' do. mkint (len_v (0 {:: posv))
  case. 'in' do. mkbool ((0 {:: posv) in (1 {:: posv))
  case. 'equalto' do. mkbool ((0 {:: posv) eq (1 {:: posv))
  case. 'unique' do.
    arr =. 0 {:: posv
    seen =. 0 $ <''
    res =. 0 $ <''
    for_e. (1 {:: arr) do.
      e =. > e
      if. -. (e mem seen) do.
        seen =. seen , <e
        res =. res , <e
      end.
    end.
    mkarr res
  case. 'select' ; 'reject' do.
    arr =. 0 {:: posv
    fname =. to_str (1 {:: posv)
    rest =. 2 }. posv
    res =. 0 $ <''
    for_e. (1 {:: arr) do.
      e =. > e
      xi =. ((<((<e) , rest)) , <'') call (mkcall fname)
      if. (to_bool xi) = (name -: 'select') do. res =. res , <e end.
    end.
    mkarr res
  case. 'selectattr' ; 'rejectattr' do.
    arr =. 0 {:: posv
    attr =. to_str (1 {:: posv)
    fname =. to_str (2 {:: posv)
    rest =. 3 }. posv
    res =. 0 $ <''
    for_e. (1 {:: arr) do.
      e =. > e
      av =. (attr) obj_get e
      xi =. ((<((<av) , rest)) , <'') call (mkcall fname)
      if. (to_bool xi) = (name -: 'selectattr') do. res =. res , <e end.
    end.
    mkarr res
  case. 'raise_exception' do. err (to_str (0 {:: posv))
  case. 'strftime_now' do. mkstr (strftime_s ((to_str (0 {:: posv)) ; strftime_now_g))
  case. do. err 'Unknown function/filter: ' , name
  end.
  end.
)

html_esc =: 3 : 0
  NB. HTML-escape a string: & < > " '
  out =. ''
  for_c. y do.
    ch =. > c
    select. ch
    case. '&' do. out =. out , ('&') , ('a') , ('m') , ('p') , (';')
    case. '<' do. out =. out , ('&') , ('l') , ('t') , (';')
    case. '>' do. out =. out , ('&') , ('g') , ('t') , (';')
    case. '"' do. out =. out , '&#34;'
    case. (39 { a.) do. out =. out , ('&') , ('a') , ('p') , ('o') , ('s') , (';')
    case. do. out =. out , ch
    end.
  end.
  out
)

join_s =: 3 : 0
  NB. y = (items ; sep) -> str ; items = boxed list of Values
  items =. 0 {:: y
  sep =. 1 {:: y
  out =. ''
  first =. 1
  for_i. i. # items do.
    v =. i {:: items
    if. first do. first =. 0 else. out =. out , sep end.
    out =. out , (to_str v)
  end.
  out
)

NB. ================================================================
NB. Phase 5C — Template tokenizer / parser / renderer
NB. ================================================================

find_close =: 3 : 0
  NB. y = (from ; needle ; text) -> absolute index or _1
  NB. Skips needle occurrences inside TERMINATED '...'/\"...\" string literals
  NB. (comment closers #} are literal text). An unterminated string does not
  NB. hide the needle — it acts as the expression/block terminator.
  from =. 0 {:: y
  nd =. 1 {:: y
  text =. 2 {:: y
  is_comment =. nd -: '#}'
  sub =. from }. text
  n =. # sub
  i =. 0
  while. i < n do.
    c =. i { sub
    if. -. is_comment do.
      if. (c = '''') +. (c = '"') do.
        q =. c
        j =. i + 1
        closed =. 0
        while. j < n do.
          cj =. j { sub
          if. cj = '\' do. j =. j + 2
          elseif. cj = q do. closed =. 1 break.
          else. j =. j + 1 end.
        end.
        if. closed do. i =. j + 1 continue.
        else. i =. i + 1 continue.
        end.
      end.
    end.
    if. (i + 1) < n do.
      two =. (i { sub) , ((i+1) { sub)
      if. two -: nd do. from + i return. end.
    end.
    i =. i + 1
  end.
  _1
)

lex_template =: 3 : 0
  s =. y
  n =. # s
  toks =. ''
  tflags =. ''
  i =. 0
  while. i < n do.
    rest =. i }. s
    p1 =. ('{{') E. rest
    p2 =. ('{%') E. rest
    p3 =. ('{#') E. rest
    dpos =. n
    if. 1 e. p1 do. dpos =. dpos <. (p1 i. 1) end.
    if. 1 e. p2 do. dpos =. dpos <. (p2 i. 1) end.
    if. 1 e. p3 do. dpos =. dpos <. (p3 i. 1) end.
    if. dpos = n do.
      toks =. toks , <(('text') ; rest)
      tflags =. tflags , <0
      i =. n
      continue.
    end.
    if. dpos > 0 do. toks =. toks , <(('text') ; (dpos {. rest)) end.
    if. dpos > 0 do. tflags =. tflags , <0 end.
    d2 =. (dpos + 1) { rest
    lt =. 0
    rt =. 0
    if. (dpos + 2) < # rest do. if. (dpos + 2) { rest = '-' do. lt =. 1 end. end.
    if. d2 = '{' do.
      NB. expression
      body_start =. dpos + 2 + lt
      close =. find_close (body_start ; '}}' ; rest)
      if. close < 0 do. err 'Expected closing expression tag' end.
      body_end =. close
      if. close > body_start do. if. (close - 1) { rest = '-' do.
        rt =. 1
        body_end =. close - 1
      end. end.
      body =. (body_end - body_start) {. (body_start }. rest)
      node =. parse_expr_str body
      toks =. toks , <(('expr') ; <node)
      tflags =. tflags , <(lt + 2 * rt)
      i =. i + body_start + (close - body_start) + 2
      continue.
    elseif. d2 = '%' do.
      NB. block
      body_start =. dpos + 2 + lt
      close =. find_close (body_start ; '%}' ; rest)
      if. close < 0 do. err 'Expected closing block tag' end.
      body_end =. close
      if. close > body_start do. if. (close - 1) { rest = '-' do.
        rt =. 1
        body_end =. close - 1
      end. end.
      body =. (body_end - body_start) {. (body_start }. rest)
      lb =. lex_block body
      toks =. toks , <lb
      tflags =. tflags , <(lt + 2 * rt)
      i =. i + body_start + (close - body_start) + 2
      continue.
    elseif. d2 = '#' do.
      NB. comment
      body_start =. dpos + 2 + lt
      close =. find_close (body_start ; '#}' ; rest)
      if. close < 0 do. err 'Missing end of comment tag' end.
      toks =. toks , <(('comment') ; '')
      tflags =. tflags , <(lt + 2 * rt)
      i =. i + body_start + (close - body_start) + 2
      continue.
    end.
  end.
  NB. whitespace control: left-trim strips trailing ws of preceding text;
  NB. right-trim strips leading ws of following text.
  i =. 0
  while. i < # toks do.
    f =. > i { tflags
    lt2 =. 2 | f
    rt2 =. f >: 2
    if. lt2 = 1 do.
      if. i > 0 do.
        prev =. > (i - 1) { toks
        if. ('text' -: > 0 { prev) do.
          s =. > 1 { prev
          j =. # s
          while. j > 0 do.
            if. -. ((a. i. ((j - 1) { s)) e. 9 10 13 32) do. break. end.
            j =. j - 1
          end.
          toks =. (<(('text') ; (j {. s))) (i - 1) } toks
        end.
      end.
    end.
    if. rt2 = 1 do.
      if. i < (# toks) - 1 do.
        nxt =. > (i + 1) { toks
        if. ('text' -: > 0 { nxt) do.
          s =. > 1 { nxt
          j =. 0
          while. j < # s do.
            if. -. ((a. i. (j { s)) e. 9 10 13 32) do. break. end.
            j =. j + 1
          end.
          toks =. (<(('text') ; (j }. s))) (i + 1) } toks
        end.
      end.
    end.
    i =. i + 1
  end.
  NB. Options (trim_blocks/lstrip_blocks) applied to text tokens, in addition
  NB. to the per-tag `-` markers above. opts_g = '' | 't' | 'l' | 'tl' (+ 'k').
  if. 0 < # opts_g do.
    tb =. 't' e. opts_g
    lb =. 'l' e. opts_g
    i =. 0
    n2 =. # toks
    while. i < n2 do.
      tok =. > i { toks
      if. ('text' -: > 0 { tok) do.
        s =. > 1 { tok
        if. lb *. (i + 1) < n2 do.
          NB. text preceding a block: strip the leading whitespace of the line
          NB. containing the block. i0 = trailing-ws length of s (C++ `i`).
          i0 =. # s
          while. (i0 > 0) do.
            c =. (i0 - 1) { s
            if. -. ((' ' = c) +. (TAB = c)) do. break. end.
            i0 =. i0 - 1
          end.
          if. i0 = 0 do.
            NB. all whitespace: strip only if this is the first token.
            if. 0 = i do. s =. '' end.
          elseif. (LF = (i0 - 1) { s) do.
            NB. last non-ws char is a newline: keep up to it.
            s =. i0 {. s
          end.
        end.
        if. tb *. (i > 0) do.
          prev =. > (i - 1) { toks
          if. -. ('expr' -: > 0 { prev) do.
            if. 0 < # s do.
              if. LF = 0 { s do. s =. (1) }. s end.
            end.
          end.
        end.
        toks =. (<(('text') ; s)) i } toks
      end.
      i =. i + 1
    end.
  end.
  NB. keep_trailing_newline=false: strip the trailing newline of the FINAL
  NB. token only if it is a text node at the top level (C++ Parser, `it==end`).
  NB. Nested text (if/for bodies) must NOT be stripped — matches C++.
  if. -. ('k' e. opts_g) do.
    if. 0 < # toks do.
      ltok =. > (_1) { toks
      if. ('text' -: > 0 { ltok) do.
        s =. > 1 { ltok
        if. 0 < # s do.
          if. LF = _1 {. s do.
            s =. }: s
            if. 0 < # s do.
              if. CR = _1 {. s do. s =. }: s end.
            end.
          end.
        end.
        toks =. (<(('text') ; s)) (_1) } toks
      end.
    end.
  end.
  toks
)

parse_varnames =: 3 : 0
  NB. y = 'x' or 'a, b' -> boxed list of names
  parts =. split (y ; ',')
  res =. ''
  for_p. parts do.
    w =. strip (> p)
    if. 1 = # w do. w =. {. w end.
    res =. res , <w
  end.
  res
)

parse_macro_params =: 3 : 0
  NB. y = "x, z, w=10" -> boxed list of (name ; default-expr-or-'') pairs
  if. 0 = # y do. '' return. end.
  parts =. split (y ; ',')
  res =. ''
  for_p. parts do.
    w =. strip (> p)
    eqp =. w i. '='
    if. eqp < # w do.
      pname =. strip (eqp {. w)
      pdef =. parse_expr_str ((eqp + 1) }. w)
      res =. res , <((<pname) , <pdef)
    else.
      res =. res , <((<w) , <'')
    end.
  end.
  res
)

lex_block =: 3 : 0
  NB. y = block body string (between {% and %}) -> template token
  btrim =. strip y
  sp =. btrim i. ' '
  if. sp < # btrim do. kw =. sp {. btrim else. kw =. btrim end.
  if. sp < # btrim do. rest =. (sp + 1) }. btrim else. rest =. '' end.
  select. kw
  case. 'if' do. (('if') ; <(parse_expr_str rest))
  case. 'elif' do. (('elif') ; <(parse_expr_str rest))
  case. 'else' do. (('else') ; '')
  case. 'endif' do. (('endif') ; '')
  case. 'for' do.
    inpos =. ' in ' find_s rest
    if. inpos >: # rest do. err 'Expected 'in' keyword in for block' end.
    vns =. parse_varnames (inpos {. rest)
    iter_str =. (inpos + 4) }. rest
    rec =. 0
    if. endswith (iter_str ; 'recursive') do.
      rec =. 1
      iter_str =. (((# iter_str) - 9) {. iter_str)
    end.
    cond =. ''
    ifp =. ' if ' find_s iter_str
    if. ifp < # iter_str do.
      cond =. parse_expr_str ((ifp + 4) }. iter_str)
      iter_str =. ifp {. iter_str
    end.
    iter =. parse_expr_str iter_str
mknode ('for' ; <((((mkp vns)) , (mkp iter)) , (mkp cond)) , (mkp rec))
  case. 'endfor' do. (('endfor') ; '')
  case. 'set' do.
    eqpos =. ' = ' find_s rest
    if. eqpos < # rest do.
      lhs =. eqpos {. rest
      rhs =. (eqpos + 3) }. rest
      dotp =. lhs i. '.'
      if. dotp < # lhs do.
        ns =. strip (dotp {. lhs)
        vns =. <(strip ((dotp + 1) }. lhs))
        v =. parse_expr_str rhs
mknode ('set' ; <(((mkp ns)) , (mkp vns)) , (mkp v))
      else.
        vns =. parse_varnames lhs
        v =. parse_expr_str rhs
mknode ('set' ; <(((mkp '')) , (mkp vns)) , (mkp v))
      end.
    else.
      vns =. parse_varnames rest
mknode ('set' ; <(((mkp '')) , (mkp vns)) , (mkp ''))
    end.
  case. 'endset' do. (('endset') ; '')
  case. 'break' do. (('break') ; '')
  case. 'continue' do. (('continue') ; '')
  case. 'generation' do. (('generation') ; '')
  case. 'endgeneration' do. (('endgeneration') ; '')
  case. 'filter' do. (('filterblk') ; <(parse_expr_str rest))
  case. 'endfilter' do. (('endfilter') ; '')
  case. 'macro' do.
    mp =. rest i. '('
    if. mp >: # rest do. err 'Expected '(' in macro block' end.
    mname =. strip (mp {. rest)
    mparams =. ((rest i. ')') - (mp + 1)) {. ((mp + 1) }. rest)
    mpnode =. parse_macro_params mparams
    mknode ('macro' ; <(((mkp mname)) , (mkp mpnode)) , (mkp ''))
  case. 'endmacro' do. (('endmacro') ; '')
  case. 'call' do.
    mknode ('call' ; <((mkp (parse_expr_str rest))) , (mkp ''))
  case. 'endcall' do. (('endcall') ; '')
  case. do. err 'Unexpected block: ' , kw
  end.
)

NB. ---- template parser (global cursor over ttoks_g) ----
ttoks_g =: ''
opts_g =: ''
tpc =: 0
nt_toks =: 0

tkt =: 3 : '> (tpc { ttoks_g)'

tkt_safe =: 3 : 0
  if. tpc < nt_toks do.
    > (tpc { ttoks_g)
  else.
    ('' ) ; ''
  end.
)

parse_template =: 3 : 0
  NB. y = fully flag (1 at root)
  ch =. ''
  while. tpc < nt_toks do.
    tok =. tkt ''
    type =. 0 {:: tok
    select. type
    case. 'if' do.
      tpc =: tpc + 1
      cond =. 1 {:: tok
      body =. parse_template 0
      cascade =. <( (<cond) ; body )
      while. (tpc < nt_toks) *. (('elif') -: 0 {:: tkt_safe '') do.
        etok =. tkt ''
        tpc =: tpc + 1
        ebody =. parse_template 0
        cascade =. cascade , <((<1 {:: etok) ; ebody)
      end.
      if. (tpc < nt_toks) *. (('else') -: 0 {:: tkt_safe '') do.
        tpc =: tpc + 1
        cascade =. cascade , <(nil ; (parse_template 0))
      end.
      if. (tpc >: nt_toks) +. -. (('endif') -: 0 {:: tkt_safe '') do.
        err 'Unterminated if'
      end.
      tpc =: tpc + 1
      ch =. ch , <(('if') ; <cascade)
    case. 'for' do.
      tpc =: tpc + 1
      vns =. 1 {:: tok
      iter =. 2 {:: tok
      cond =. 3 {:: tok
      rec =. 4 {:: tok
      body =. parse_template 0
      else_body =. ''
      if. (tpc < nt_toks) *. (('else') -: 0 {:: tkt_safe '') do.
        tpc =: tpc + 1
        else_body =. parse_template 0
      end.
      if. (tpc >: nt_toks) +. -. (('endfor') -: 0 {:: tkt_safe '') do.
        err 'Unterminated for'
      end.
      tpc =: tpc + 1
      ch =. ch , <(mknode ('for' ; <((((((mkp vns)) , (mkp iter)) , (mkp cond)) , (mkp body)) , (mkp rec)) , (mkp else_body)))
    case. 'generation' do.
      tpc =: tpc + 1
      body =. parse_template 0
      if. (tpc >: nt_toks) +. -. (('endgeneration') -: 0 {:: tkt_safe '') do.
        err 'Unterminated generation'
      end.
      tpc =: tpc + 1
      ch =. ch , body
    case. 'text' do.
      tpc =: tpc + 1
      ch =. ch , <(('text') ; (1 {:: tok))
    case. 'expr' do.
      tpc =: tpc + 1
      ch =. ch , <(('expr') ; <(1 {:: tok))
    case. 'set' do.
      tpc =: tpc + 1
      ns =. 1 {:: tok
      vns =. 2 {:: tok
      value =. 3 {:: tok
      if. ('__nil__' -: value) do.
        body =. parse_template 0
        if. (tpc >: nt_toks) +. -. (('endset') -: 0 {:: tkt_safe '') do.
          err 'Unterminated set'
        end.
        tpc =: tpc + 1
        ch =. ch , <(mknode ('settmpl' ; <((mkp (0 {:: vns)) , (mkp body))))
      else.
      ch =. ch , <(mknode ('set' ; <(((mkp ns)) , (mkp vns)) , (mkp value)))
      end.
    case. 'break' do.
      tpc =: tpc + 1
      ch =. ch , <(('break') ; '')
    case. 'continue' do.
      tpc =: tpc + 1
      ch =. ch , <(('continue') ; '')
    case. 'comment' do.
      tpc =: tpc + 1
    case. 'filterblk' do.
      tpc =: tpc + 1
      fname =. 1 {:: tok
      body =. parse_template 0
      if. (tpc >: nt_toks) +. -. (('endfilter') -: 0 {:: tkt_safe '') do.
        err 'Unterminated filter'
      end.
      tpc =: tpc + 1
      ch =. ch , <(mknode ('filterblk' ; <((mkp fname) , (mkp body))))
    case. 'macro' do.
      tpc =: tpc + 1
      mname =. 1 {:: tok
      mparams =. 2 {:: tok
      body =. parse_template 0
      if. (tpc >: nt_toks) +. -. (('endmacro') -: 0 {:: tkt_safe '') do.
        err 'Unterminated macro'
      end.
      tpc =: tpc + 1
      ch =. ch , <(mknode ('macro' ; <(((mkp mname)) , (mkp mparams)) , (mkp body)))
    case. 'call' do.
      tpc =: tpc + 1
      expr =. 1 {:: tok
      body =. parse_template 0
      if. (tpc >: nt_toks) +. -. (('endcall') -: 0 {:: tkt_safe '') do.
        err 'Unterminated call'
      end.
      if. -. ('call' -: 0 {:: expr) do.
        err 'Invalid call block syntax - expected function call'
      end.
      tpc =: tpc + 1
      ch =. ch , <(mknode ('call' ; <((mkp expr)) , (mkp body)))
    case. 'endmacro' ; 'endcall' do. break.
    case. 'endfor' ; 'endset' ; 'endif' ; 'else' ; 'elif' ; 'endgeneration' ; 'endfilter' do.
      break.
    case. do. err 'Unexpected ' , type
    end.
  end.
  if. 0 = # ch do.
    <(('text') ; '')
  elseif. 1 = # ch do.
    0 { ch
  else.
    <(('seq') ; <ch)
  end.
)

NB. ---- renderer (global ctx_g, loop_depth_g, ctrl_g) ----
loop_depth_g =: 0
ctrl_g =: ''

assign_d =: 3 : 0
  NB. y = (vns ; item ; ctx) -> new ctx (destructuring assign)
  vns =. 0 {:: y
  item =. 1 {:: y
  ctx =. 2 {:: y
  if. 1 = # vns do.
    ((<0 {:: vns) , <item) ctx_set ctx
  else.
    if. -. (is_arr item) *. (# vns) = arr_size item do.
      err 'Mismatched number of variables and items in destructuring assignment'
    end.
    ctx2 =. ctx
    for_i. i. # vns do.
      ctx2 =. ((<i {:: vns) , <((mkint i) arr_at item)) ctx_set ctx2
    end.
    ctx2
  end.
)

rnode =: 3 : 0
  node =. y
  select. 0 {:: node
  case. 'seq' do.
    ch =. 1 {:: node
    out =. ''
    for_i. i. # ch do.
      out =. out , rnode (i {:: ch)
      if. (ctrl_g -: 'break') +. (ctrl_g -: 'continue') do. break. end.
    end.
    out
  case. 'text' do.
    t =. 1 {:: node
    if. 0 < # t do. last_text_g =: 1 end.
    t
  case. 'expr' do.
    last_text_g =: 0
    v =. eval_expr (1 {:: node)
    select. kind v
    case. 'str' do. payload v
    case. 'bool' do. if. to_bool v do. 'True' else. 'False' end.
    case. 'null' do. ''
    case. do. dump v
    end.
  case. 'filterblk' do.
    fname =. 1 {:: node
    body =. > 2 {:: node
    rendered =. rnode body
    callable =. eval_expr fname
    sv =. mkstr rendered
    res =. ((<(<sv)) , <'') call callable
    select. kind res
    case. 'str' do. payload res
    case. 'bool' do. if. to_bool res do. 'True' else. 'False' end.
    case. 'null' do. ''
    case. do. dump res
    end.
  case. 'if' do.
    cas =. 1 {:: node
    dbg_i =. 0
    for_b. cas do.
      pair =. > b
      cond =. > 0 {:: pair
      body =. 1 {:: pair
      if. ('__nil__' -: cond) do. rnode body return. end.
      if. to_bool (eval_expr cond) do. rnode body return. end.
    end.
    ''
  case. 'for' do.
    vns =. 1 {:: node
    iter =. 2 {:: node
    cond =. 3 {:: node
    body =. > 4 {:: node
    rec =. 5 {:: node
    elseb =. > 6 {:: node
    outer_ctx =. ctx_g
    iterable =. eval_expr iter
    items =. ''
    if. -. (is_null iterable) do.
      if. is_str iterable do.
        src =. <"0 (mkstr &> payload iterable)
      elseif. is_obj iterable do.
        src =. obj_keys iterable
      elseif. is_arr iterable do.
        src =. arr_items iterable
      else. err 'For loop iterable must be iterable' end.
      for_it. src do.
        item =. > it
        ctx_g =: assign_d (((<vns) , <item) , <ctx_g)
        if. ('__nil__' -: cond) do.
          items =. items , <item
        else.
          if. to_bool (eval_expr cond) do. items =. items , <item end.
        end.
      end.
      ctx_g =: outer_ctx
    end.
    if. 0 = # items do.
      if. ('__nil__' -: elseb) do. '' else. rnode elseb end.
    else.
      NB. outer_ctx captured before the filtering pass (no loop-var leak)
      loop_depth_g =: loop_depth_g + 1
      mydepth =. loop_depth_g
      ctrl_g =: ''
      loop_cycle_i_g =: 0
      n =. # items
      loopobj =. mkobj ''
      loopobj =. ((<'__loop_depth__') , <(mkint mydepth)) obj_set loopobj
      loopctx =. (<loopobj) , <outer_ctx
      ctx_g =: loopctx
      out =. ''
      for_i. i. n do.
        item =. i {:: items
        ctx_g =: assign_d (((<vns) , <item) , <ctx_g)
        lo =. mkobj ''
        lo =. ((<'index') , <(mkint (i + 1))) obj_set lo
        lo =. ((<'index0') , <(mkint i)) obj_set lo
        lo =. ((<'revindex') , <(mkint (n - i))) obj_set lo
        lo =. ((<'revindex0') , <(mkint ((n - i) - 1))) obj_set lo
        lo =. ((<'length') , <(mkint n)) obj_set lo
        lo =. ((<'first') , <(mkbool (i = 0))) obj_set lo
        lo =. ((<'last') , <(mkbool (i = (n - 1)))) obj_set lo
        if. i > 0 do. prev =. (i - 1) {:: items else. prev =. mknull '' end.
        if. i < n - 1 do. next =. (i + 1) {:: items else. next =. mknull '' end.
        lo =. ((<'previtem') , <prev) obj_set lo
        lo =. ((<'nextitem') , <next) obj_set lo
        ctx_g =: ((<'loop') , <lo) ctx_set ctx_g
        ctrl_g =: ''
        out =. out , rnode body
        if. ctrl_g -: 'break' do. break.
        elseif. ctrl_g -: 'continue' do. continue. end.
      end.
      loop_depth_g =: loop_depth_g - 1
      NB. C++ keeps ONE shared loop_context whose parent is the outer ctx, so
      NB. mutations to pre-existing outer vars persist after the loop. J contexts
      NB. are immutable: ctx_loop_outer gives the outer chain updated with any
      NB. ctx_set_found/namespace mutations; merge the loop-local-frame vars that
      NB. shadow pre-existing outer vars (e.g. res.append) into it as well.
      outer2 =. 1 {:: ctx_g
      lkeys =. (<'loop') , (<'__loop_depth__') , vns
      lframe =. 0 {:: ctx_g
      for_k. (obj_keys lframe) do.
        k =. > k
        if. (-. (<(to_str k)) e. lkeys) *. (k ctx_contains outer2) do.
          outer2 =. ((<k) , <(k ctx_get ctx_g)) ctx_set_found outer2
        end.
      end.
      ctx_g =: outer2
      out
    end.
  case. 'set' do.
    ns =. 1 {:: node
    vns =. 2 {:: node
    value =. 3 {:: node
    v =. eval_expr value
    if. ('__nil__' -: ns) do.
      ctx_g =: assign_d (((<vns) , <v) , <ctx_g)
    else.
      nsv =. ns ctx_get ctx_g
      if. -. (is_obj nsv) do. err 'Namespace is not an object' end.
      nsv2 =. ((<0 {:: vns) , <v) obj_set nsv
      ctx_g =: ((<ns) , <nsv2) ctx_set_found ctx_g
    end.
    ''
  case. 'macro' do.
    mname =. 1 {:: node
    mparams =. 2 {:: node
    body =. > 3 {:: node
    NB. Bind a provisional macro first so the captured defctx (execution
    NB. parent) includes the macro itself — required for recursion.
    pl0 =. ''
    pl0 =. pl0 , <mname
    pl0 =. pl0 , <mparams
    pl0 =. pl0 , <body
    pl0 =. pl0 , <ctx_g
    mc0 =. ('macro') ; <pl0
    ctx_g =: ((<mname) , <mc0) ctx_set ctx_g
    NB. Now capture defctx (contains the macro) and rebind with the final pair.
    pl =. ''
    pl =. pl , <mname
    pl =. pl , <mparams
    pl =. pl , <body
    pl =. pl , <ctx_g
    mc =. ('macro') ; <pl
    ctx_g =: ((<mname) , <mc) ctx_set ctx_g
    ''
  case. 'call' do.
    expr =. 1 {:: node
    body =. > 2 {:: node
    cpl =. ''
    cpl =. cpl , <body
    cpl =. cpl , <ctx_g
    caller =. ('caller') ; <cpl
    ctx_g =: ((<'caller') , <caller) ctx_set ctx_g
    res =. eval_expr expr
    select. kind res
    case. 'str' do. payload res
    case. 'bool' do. if. to_bool res do. 'True' else. 'False' end.
    case. 'null' do. ''
    case. do. dump res
    end.
  case. 'settmpl' do.
    name =. 1 {:: node
    body =. > 2 {:: node
    val =. mkstr (rnode body)
    ctx_g =: ((<name) , <val) ctx_set ctx_g
    ''
  case. 'break' do.
    if. 0 = loop_depth_g do. err 'break outside of a loop' end.
    ctrl_g =: 'break'
    ''
  case. 'continue' do.
    if. 0 = loop_depth_g do. err 'continue outside of a loop' end.
    ctrl_g =: 'continue'
    ''
  case. 'generation' do.
    rnode (1 {:: node)
  case. 'comment' do. ''
  case. do. err 'Unknown template node'
  end.
)

render =: 4 : 0
  NB. x = Context (or '' for empty); y = template string -> rendered string
  opts_g =: ''
  x render_core (y ; '')
)

render_opt =: 4 : 0
  NB. x = Context (or ''); y = <template ; options> -> rendered string.
  NB. options = char flags: 't' trim_blocks, 'l' lstrip_blocks, 'k' keep_trailing_newline.
  'tmpl opts' =. y
  opts_g =: opts
  x render_core (tmpl ; opts)
)

render_core =: 4 : 0
  NB. x = Context (or ''); y = <template ; options> (opts_g set) -> rendered string
  'tmpl opts' =. y
  ttoks_g =: lex_template tmpl
  nt_toks =: # ttoks_g
  tpc =: 0
  root =. parse_template 1
  if. tpc < nt_toks do. err 'Unexpected ' , (0 {:: > (tpc { ttoks_g)) end.
  if. ('' -: x) do. ctx_g =: mkctx (mkobj '') else. ctx_g =: x end.
  loop_depth_g =: 0
  ctrl_g =: ''
  last_text_g =: 0
  out =. rnode (> root)
  out
)
