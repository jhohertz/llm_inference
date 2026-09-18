NB. ============================================================
NB.  builders.ijs - OpenAI chat.completions response builders for
NB.  the jsocket server (molded from reference/j-http-handoff).
NB.
NB.  Builds the OpenAI shapes from OUR chat_completion output
NB.  (<content ; finish_reason ; tool_calls>), byte-verified against
NB.  enc_json/dec_json. Uses the convert/json addon (enc_json_json_).
NB.  Locale-independent; loaded by server.ijs into the inference locale.
NB.
NB.  convert/json contract (proven in the handoff):
NB.    object = (2,k) box array: row0 keys, row1 values (cells boxed)
NB.    ; boxes each argument;  ,:  appends rows
NB.    1 $ <x = one-element box array holding x  (single choice)
NB.    mixed-type value row: box each value, then catenate the boxes
NB.    true/false/null = json_true / json_false / json_null
NB.    empty object = 0 0 $ <''   (NOT 2 0 $ 0 # <'' -> bare '}')
NB.  Args arrive as box lists; unpack with >n{y .
NB. ============================================================
NB. Capture the CALLER's locale before convert/json loads (it `coclass 'json'`
NB. switches the current locale), then restore it so the builders land in the
NB. caller's locale.  The bare `cocurrent <'...'>` fails to parse at top level
NB. in this build (unexecutable fragment) but works inside a def, so restore
NB. through a helper.
curloc =: 18!:5 ''
require 'convert/json'
loc_restore =: 3 : 0
  NB. y is the caller's locale NAME (boxed, from 18!:5 '') — already boxed,
  NB. so pass to cocurrent directly (cocurrent <y would double-box -> domain err).
  cocurrent y
)
loc_restore curloc
enc_json =: enc_json_json_

NB. ============================================================
NB.  y mkobj  ->  a (2,k) box object from a boxed key/value pair.
NB.  y = <keys ; values>  (keys = row of key atoms, values = row of
NB.  boxed value cells).  Returns keys ,: values.
mkobj =: 3 : 0
  'k v' =. y
  k ,: v
)

NB. ============================================================
NB.  y respbody  ->  chars of the FULL non-streaming response body.
NB.  y = <id ; model ; created ; content ; finish ; tcs>
NB.  tcs = boxed list of tool-call minja Values ({type; function:<name;
NB.  arguments>; id}), possibly empty.  finish = 'stop'|'length'|'tool_calls'.
respbody =: 3 : 0
  cid =. >0{ y
  cm  =. >1{ y
  cr  =. >2{ y
  ct  =. >3{ y
  fn  =. >4{ y
  tcs =. >5{ y
  NB. message object
  if. 0 < # tcs do.
    NB. assistant message with tool_calls, content null
    mk =. ('role';'content';'tool_calls')
    tca =. tcs_to_json tcs
    mv =. ('assistant';'json_null';tca)
    msg =. mk ,: mv
  else.
    mk =. ('role';'content')
    mv =. ('assistant';ct)
    msg =. mk ,: mv
  end.
  ck =. ('index';'message';'logprobs';'finish_reason')
  cv =. (0;msg;'json_null';fn)
  ch =. ck ,: cv
  A =. 1 $ <ch
  uk =. ('prompt_tokens';'completion_tokens';'total_tokens')
  uv =. (0;0;0)
  u =. uk ,: uv
  b1 =. <cid
  b2 =. <'chat.completion'
  b3 =. <cr
  b4 =. <cm
  b5 =. <A
  b6 =. <u
  h =. ('id';'object';'created';'model';'choices';'usage')
  v =. b1 , b2 , b3 , b4 , b5 , b6
  O =. h ,: v
  enc_json O
)

NB. ============================================================
NB.  tcs tcs_to_json  ->  a convert/json ARRAY of tool-call objects
NB.  (choices[0].message.tool_calls) built from the chat_completion
NB.  tool-call minja Values.  Each tc is {type; function:<name;
NB.  arguments>; id} with arguments a JSON STRING.
tcs_to_json =: 3 : 0
  tcs =. y
  out =. ''
  for_tc. tcs do.
    tc =. > tc
    id  =. payload_minja_ ('id') obj_get_minja_ tc
    ty  =. payload_minja_ ('type') obj_get_minja_ tc
    fn0 =. ('function') obj_get_minja_ tc
    nm  =. payload_minja_ ('name') obj_get_minja_ fn0
    ar  =. payload_minja_ ('arguments') obj_get_minja_ fn0
    fk =. ('name';'arguments')
    fv =. (nm;ar)
    fo =. fk ,: fv
    k =. ('id';'type';'function')
    v =. (id;ty;fo)
    o =. k ,: v
    out =. out , <o
  end.
  out
)

NB. ============================================================
NB.  y chunkbody  ->  chars of ONE SSE chunk body: role+content delta,
NB.  finish_reason null.   y = <id ; model ; created ; content>
chunkbody =: 3 : 0
  cid =. >0{ y
  cm  =. >1{ y
  cr  =. >2{ y
  ct  =. >3{ y
  dk =. ('role';'content')
  dv =. ('assistant';ct)
  d =. dk ,: dv
  ck =. ('index';'delta';'finish_reason')
  cv =. (0;d;'json_null')
  ch =. ck ,: cv
  A =. 1 $ <ch
  b1 =. <cid
  b2 =. <'chat.completion.chunk'
  b3 =. <cr
  b4 =. <cm
  b5 =. <A
  h =. ('id';'object';'created';'model';'choices')
  v =. b1 , b2 , b3 , b4 , b5
  O =. h ,: v
  enc_json O
)

NB. ============================================================
NB.  tok midchunkbody params  ->  chars of an SSE chunk body with a
NB.  CONTENT-ONLY delta (1-key object).  params = <id ; model ; created>
midchunkbody =: 4 : 0
  cid =. >0{ y
  cm  =. >1{ y
  cr  =. >2{ y
  dk =. <'content'
  dv =. <x
  d =. dk ,: dv
  ck =. ('index';'delta';'finish_reason')
  cv =. (0;d;'json_null')
  ch =. ck ,: cv
  A =. 1 $ <ch
  b1 =. <cid
  b2 =. <'chat.completion.chunk'
  b3 =. <cr
  b4 =. <cm
  b5 =. <A
  h =. ('id';'object';'created';'model';'choices')
  v =. b1 , b2 , b3 , b4 , b5
  O =. h ,: v
  enc_json O
)

NB. ============================================================
NB.  finish endchunkbody params  ->  chars of the FINAL SSE chunk
NB.  body: empty delta {} + finish_reason <finish>.
NB.  params = <id ; model ; created>
endchunkbody =: 4 : 0
  cid =. >0{ y
  cm  =. >1{ y
  cr  =. >2{ y
  e =. 0 0 $ <''
  ck =. ('index';'delta';'finish_reason')
  cv =. (0;e;x)
  ch =. ck ,: cv
  A =. 1 $ <ch
  b1 =. <cid
  b2 =. <'chat.completion.chunk'
  b3 =. <cr
  b4 =. <cm
  b5 =. <A
  h =. ('id';'object';'created';'model';'choices')
  v =. b1 , b2 , b3 , b4 , b5
  O =. h ,: v
  enc_json O
)

NB. ============================================================
NB.  Streaming frame builders (called by the server's SSE sender,
NB.  one per text delta).  Each returns the FULL chunked SSE frame
NB.  bytes for one data: line:  h11_chunk('data: ' , body , LF , LF).
NB.  params = <id ; model ; created>
NB.  x = text delta.
NB.  y frame_first / frame_mid / frame_end -> bytes.
frame_first =: 4 : 0
  fr =. 'data: ' , (chunkbody (y ; x)) , LF , LF
  h11_chunk fr
)
frame_mid =: 4 : 0
  fr =. 'data: ' , (x midchunkbody y) , LF , LF
  h11_chunk fr
)
frame_end =: 4 : 0
  fr =. 'data: ' , (x endchunkbody y) , LF , LF
  h11_chunk fr
)
frame_done =: 3 : 0
  r =. h11_chunk (h11_done '')
  r , h11_chunkend ''
)

NB. ============================================================
NB.  y plainres  ->  full bytes of a non-streaming HTTP response
NB.  (Content-Length form).  y = <id ; model ; created ; content ;
NB.  finish ; tcs>
plainres =: 3 : 0
  body =. respbody y
  hd =. 'Content-Type: application/json' , CRLF
  lst =. '200' ; 'OK' ; hd ; body
  h11_simple lst
)

NB. ============================================================
NB.  y streamhead  ->  the streaming response head bytes: status line
NB.  + SSE headers + chunked-transfer header + CRLFCRLF.
streamhead =: 3 : 0
  hd =. 'Content-Type: text/event-stream' , CRLF , 'Cache-Control: no-cache' , CRLF
  h11_chunkhead '200' ; 'OK' ; hd
)

NB. ============================================================
NB.  y notfound  ->  bytes of a 404 response.
notfound =: 3 : 0
  hd =. 'Content-Type: text/plain' , CRLF
  lst =. '404' ; 'Not Found' ; hd ; 'unknown path'
  h11_simple lst
)
