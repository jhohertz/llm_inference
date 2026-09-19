NB. ============================================================
NB.  server.ijs - J9.7 non-blocking HTTP/1.1 server exposing the
NB.  OpenAI-compatible POST /v1/chat/completions endpoint, driven by
NB.  OUR llm_inference (real generation: plain + streamed SSE).
NB.  Molded from reference/j-http-handoff/server.ijs (the mock).
NB.
NB.  run:  ~/j9.7/bin/jconsole http/server.ijs [MODEL]   (listens 8790)
NB.  shell helper:  scripts/llm_server.sh [MODEL]
NB.
NB.  CONCURRENCY MODEL (from the handoff, proven):
NB.  - EVERY socket is NON-BLOCKING:  sdcheck sdioctl fd,FIONBIO,1
NB.  - Event loop: z=: sdcheck sdselect FSET;FSET;FSET;200  (200ms timeout)
NB.    sdcheck drops the status cell -> <read;write;error>; READ fds at cell 0.
NB.  - One op per ready fd per cycle: listener -> sdaccept, conn -> sdrecv
NB.    (4096), accumulate in CBF until h11_complete.
NB.  - sdaccept -> <0;newfd> | <11;''> EAGAIN; sdrecv -> <0;DATA> | <0;''> EOF
NB.    | <11;''> EAGAIN.  Inspect error codes directly (never block).
NB.
NB.  Locale: runs in the INFERENCE locale (+ coinsert 'jsocket') so the
NB.  streaming callback (chat_cb_g -> sse_sender) and chat_completion resolve
NB.  where chat_stream_cb CALLS them (mirrors the TUI's locale fix).
NB. ============================================================
require 'socket'
require 'llm/inference'
cocurrent <'inference'
coinsert 'jsocket'
require 'llm/inference/http/protocol'
require 'llm/inference/http/builders'

stderr=: 1!:2&5
say=: 3 : 0
  stderr y , CRLF
)
PORT=: 8790
SKLISTEN=: _1
NB. rank-1 empty lists: (0 0 $ x) is RANK 2 (shape 0 0) - appending
NB. scalars yields 2-D shapes.  0 # 0 is the rank-1 empty list.
CFD=: 0 # 0
CBF=: 0 # <''

NB. ---- Model (first ARGV arg or default) + boot-time response id/created ----
get_model =: 3 : 0
  mdl =. 'qwen3-0.6b'
  if. 3 <: #ARGV do.
    m =. > 2 { ARGV
    if. 0 < # m do. mdl =. m end.
  end.
  mdl
)
MODEL =: get_model ''
LLM =: load_gguf_to_llm_inference_ MODEL

NB. ---- Pure-J unix epoch (no shell; local time treated as UTC) ----
NB. OpenAI `created` is unix seconds; compute from 6!:0 'YYYY-MM-DD hh:mm:ss'
NB. via Gregorian day-count.  J gotchas: % is divide (not modulus), the +/-
NB. chain groups right-to-left so must be parenthesized, and months index
NB. moff 1-based (subtract 1).
dig =: 3 : 0
  10 #. ((a. i. y) - 48)
)
moff =: 0 31 59 90 120 151 181 212 243 273 304 334
leap =: 3 : 0
  ((0 = 4 | y) *. (0 ~: 100 | y)) +. (0 = 400 | y)
)
dby =: 3 : 0
  ((((365 * y) + (<. y % 4)) - (<. y % 100)) + (<. y % 400)) - 719527
)
unix_time =: 3 : 0
  'ymd hms' =. ' ' cut y
  yr =. dig 4 {. ymd
  mn =. dig 5 6 { ymd
  dy =. dig 8 9 { ymd
  hr =. dig 2 {. hms
  mi =. dig 3 4 { hms
  se =. dig 6 7 { hms
  doy =. ((mn - 1) { moff) + dy - 1
  if. (2 <: mn) *. (leap yr) do. doy =. doy + 1 end.
  days =. (dby yr) + doy
  (days * 86400) + (hr * 3600) + (mi * 60) + se
)
created_now =: 3 : 0
  unix_time (6!:0 'YYYY-MM-DD hh:mm:ss')
)
SID =: 'chatcmpl-' , (": created_now '')

NB. ============================================================
NB.  addconn fd  ->  register fd, make it NON-BLOCKING, empty buffer.
addconn =: 3 : 0
  sdcheck sdioctl y , FIONBIO , 1
  CFD=: CFD , y
  CBF=: CBF , <''
  say 'conn open fd=', (": y) , ' total=', (": # CFD)
)

NB. ============================================================
NB.  rmconn fd  ->  drop fd from both lists (while.-loop, scalar ops).
rmconn =: 3 : 0
  k=: 0
  n=: # CFD
  keepc=: 0 # 0
  keepb=: 0 # <''
  while. k < n do.
    if. y = k { CFD do.
    else.
      keepc=: keepc , k { CFD
      keepb=: keepb , k { CBF
    end.
    k=: k + 1
  end.
  CFD=: keepc
  CBF=: keepb
  say 'conn closed fd=', (": y) , ' total=', (": # CFD)
)

NB. ============================================================
NB.  fd idx  ->  index of fd in CFD (scalar), or _1 if absent.
idx =: 3 : 0
  k=: 0
  while. k < # CFD do.
    if. y = k { CFD do. return. k end.
    k=: k + 1
  end.
  _1
)

NB. ============================================================
NB.  fd bufget  ->  buffered chars for fd (positionally aligned with CFD).
bufget =: 3 : 0
  k=: idx y
  if. _1 = k do. '' return. end.
  > k { CBF
)

NB. ============================================================
NB.  fd bufput b  ->  replace fd's buffer cell with <b> in place.
bufput =: 4 : 0
  k=: idx x
  cell=: <y
  CBF=: (k {. CBF) , cell , (k + 1) }. CBF
)

NB. ============================================================
NB.  fd finish resp  ->  send response bytes, close, deregister.
finish =: 4 : 0
  try.
    sdcheck y sdsend x , 0
  catch.
    say 'senderr fd=', (": x)
  end.
  sdclose x
  rmconn x
)

NB. ============================================================
NB.  a ceq b  ->  1 if a and b are EXACTLY equal char vectors, else 0.
NB.  "-:" is unreliable for 1-char strings in this build; ceq uses
NB.  length guard + */ x = y.
ceq =: 4 : 0
  nx=: #x
  ny=: #y
  if. nx = ny do.
    */ x = y
  else.
    0
  end.
)

NB. ============================================================
NB.  JSON request parsing (convert/pjson — preserves numbers/bools).
NB.  dec_pjson_ returns a (2,k) table: row0 keys, row1 values (boxed).
NB.  key getv r  ->  unboxed value, or _1 if absent.
NB.  y = the pjson table r (n x 2): each row = <key ; value>.  Column 0 =
NB.  keys (strings), column 1 = values (boxed list of mixed types — never
NB.  > of the whole column, which would domain-error on mixed types).
getv =: 4 : 0
  ks=. 0 {"1 y
  vs=. 1 {"1 y
  i=. ks i. <x
  if. i = # ks do. _1 return. end.
  > vs {~ i
)

NB. ============================================================
NB.  lst enc_arr  ->  JSON array string of the boxed list of object
NB.  tables (each table row0 keys/row1 values; enc_pjson_ wants n x 2,
NB.  so transpose).  Used to pass `tools` to the chat template.
enc_arr =: 3 : 0
  lst=: y
  parts=: ''
  i=: 0
  while. i < # lst do.
    ti=: > i { lst
    parts=: parts , enc_pjson_ (|: ti)
    if. (i + 1) < # lst do. parts=: parts , ',' end.
    i=: i + 1
  end.
  '[' , parts , ']'
)

NB. ============================================================
NB.  lst mk_msgs  ->  boxed list of <role ; content> message boxes from
NB.  the pjson-decoded messages array (each a {role;content} table).
mk_msgs =: 3 : 0
  lst=: y
  out=: ''
  i=: 0
  while. i < # lst do.
    mt=: > i { lst
    k2=: 0 {"1 mt
    v2=: 1 {"1 mt
    role=: > v2 {~ k2 i. <'role'
    ct =: > v2 {~ k2 i. <'content'
    out=: out , < (role) ; ct
    i=: i + 1
  end.
  out
)

NB. ============================================================
NB.  Streaming SSE sender — installed as chat_cb_g.  Each text delta
NB.  becomes one SSE chunk (first frame carries role+content, rest are
NB.  content-only), sent over SFD (the current connection).  Reads the
NB.  global SFD/SID/MODEL/CREATED set by v1_chat before generating.
sse_sender =: 3 : 0
  if. 0 < # y do.
    if. SFIRST do.
      fr=: y frame_first (SID ; MODEL ; CREATED)
      SFIRST=: 0
    else.
      fr=: y frame_mid (SID ; MODEL ; CREATED)
    end.
    try.
      sdcheck fr sdsend SFD , 0
    catch.
      say 'sse send err'
    end.
  end.
  ''
)

NB. ============================================================
NB.  x stream_chat y  ->  serve a STREAMING /v1/chat/completions request.
NB.  x = connection fd; y = <msgs ; tools_json ; temp ; top_p ; max_steps>.
NB.  Sends the SSE head, arms streaming (chat_cb_g -> sse_sender), generates
NB.  (blocks the loop for the generation), sends the final empty-delta chunk
NB.  + "data: [DONE]", closes.  Returns '' (serve skips finish).
stream_chat =: 4 : 0
  fd=: x
  msgs=: > 0 { y
  tools_json=: > 1 { y
  temp=: > 2 { y
  top_p=: > 3 { y
  max_steps=: > 4 { y
  params=: < temp ; 0 ; top_p ; 0
  head=: streamhead ''
  try.
    sdcheck head sdsend fd , 0
  catch.
    say 'stream head err'
  end.
  SFD=: fd
  CREATED=: created_now ''
  SID=: 'chatcmpl-' , (": CREATED)
  SFIRST=: 1
  chat_stream_start ''
  chat_cb_g =: sse_sender
  res=: LLM chat_completion (msgs ; tools_json ; max_steps ; 1 ; <params)
  chat_stream_stop ''
  fin=: > 1 { res
  fr=: 'data: ' , (fin endchunkbody (SID ; MODEL ; CREATED)) , LF , LF
  try.
    sdcheck (h11_chunk fr) sdsend fd , 0
  catch.
    say 'stream end err'
  end.
  try.
    sdcheck (frame_done '') sdsend fd , 0
  catch.
    say 'stream done err'
  end.
  sdclose fd
  rmconn fd
  ''
)

NB. ============================================================
NB.  x v1_chat y  ->  the POST /v1/chat/completions handler.
NB.  x = connection fd; y = request body (complete).  Parses the OpenAI
NB.  request with convert/pjson, maps params, calls OUR chat_completion.
NB.  Returns full plain response bytes, or '' for a stream (already sent).
v1_chat =: 4 : 0
  fd=: x
  body=: y
  r=: dec_pjson_ body
  ks=: 0 {"1 r
  vs=: 1 {"1 r
  stream=: 'stream' getv r
  if. _1 -: stream do. stream=: 0 end.
  msgs=: 'messages' getv r
  if. _1 -: msgs do. msgs=: <('user') ; 'Hi' else. msgs=: mk_msgs msgs end.
  tools=: 'tools' getv r
  tools_json=: ''
  if. -. _1 -: tools do. tools_json=: enc_arr tools end.
  temp=: 'temperature' getv r
  if. _1 -: temp do. temp=: 0 end.
  top_p=: 'top_p' getv r
  if. _1 -: top_p do. top_p=: 0.95 end.
  mx=: 'max_tokens' getv r
  if. _1 -: mx do. mx=: 200 end.
  cid=: 'chatcmpl-' , (": created_now '')
  if. 1 = stream do.
    x stream_chat (msgs ; tools_json ; temp ; top_p ; mx)
    ''
  else.
    params=: < temp ; 0 ; top_p ; 0
    res=: LLM chat_completion (msgs ; tools_json ; mx ; 0 ; <params)
    ct=: > 0 { res
    fin=: > 1 { res
    tcs=: > 2 { res
    bdy=: respbody (cid ; MODEL ; CREATED ; ct ; fin ; tcs)
    hd=: 'Content-Type: application/json' , CRLF
    lst=: '200' ; 'OK' ; hd ; bdy
    h11_simple lst
  end.
)

NB. ============================================================
NB.  fd serve buf  ->  parse the request and dispatch.
serve =: 4 : 0
  fd=: x
  raw=: y
  'm p v b'=. h11_parse raw
  resp=: notfound ''
  if. 'GET' ceq m do.
    if. '/' ceq p do.
      hd=: 'Content-Type: text/plain' , CRLF
      lst=: '200' ; 'OK' ; hd ; 'J9.7 OpenAI-style LLM server running'
      resp=: h11_simple lst
    end.
  else.
    if. 'POST' ceq m do.
      if. '/v1/chat/completions' ceq p do.
        resp=: fd v1_chat b
      end.
    end.
  end.
  say 'respond fd=', (": fd) , ' bytes=', (": # resp)
  if. 0 < # resp do.
    fd finish resp
  end.
)

NB. ============================================================
NB.  onaccept  ->  accept ONE pending connection (or EAGAIN: skip).
onaccept =: 3 : 0
  try.
    ns=: > 0 pick sdcheck sdaccept SKLISTEN
    addconn ns
  catch.
    say 'accept raise (skip) - nothing registered this cycle'
  end.
)

NB. ============================================================
NB.  onread fd  ->  ONE non-blocking read; accumulate; serve on complete.
NB.  sdrecv: success <0;data> (data may be '' on EOF), error <'';errno>.
onread =: 3 : 0
  y=. 0 { y
  raw=: sdrecv y , 4096 , 0
  if. 0 = > 0 { raw do.
    d=: > 1 { raw
    if. 0 = # d do.
      oneof y
      return.
    end.
    b=: bufget y
    b=: b , d
    y bufput b
    if. 1 = h11_complete b do.
      y serve b
    end.
  else.
    e=: > 1 { raw
    if. -. 11 = e do.
      oneof y
    end.
  end.
)

NB. ============================================================
NB.  oneof fd  ->  peer closed; drop the connection (guard: may be gone).
oneof =: 3 : 0
  if. y e. CFD do.
    sdclose y
    rmconn y
  end.
)

NB. ============================================================
NB.  selectloop  ->  the non-blocking event loop (never returns).
selectloop =: 3 : 0
  while. 1 do.
    fset=: (SKLISTEN) , CFD
    z=: sdcheck sdselect fset ; fset ; fset ; 200
    rd=: > 0 { z
    while. 0 < # rd do.
      fd=: {. rd
      rd=: }. rd
      try.
        if. fd = SKLISTEN do.
          onaccept ''
        else.
          onread fd
        end.
      catch.
        say 'LOOPERR fd=', (": fd) , ' err: ' , 13!:12 ''
        oneof fd
      end.
    end.
  end.
)

NB. ============================================================
NB.  server_run  ->  DORMANT launch verb.  server.ijs is loaded as a LIBRARY
NB.  (defines verbs only); launching is an explicit call so a second load
NB.  never re-binds the socket (double-load was causing EADDRINUSE).  The
NB.  entry point (http/run.ijs / scripts/llm_server.sh) calls server_run ''.
server_run =: 3 : 0
  sdcleanup ''
  SKLISTEN=: 0 pick sdcheck sdsocket ''
  sdcheck sdioctl SKLISTEN , FIONBIO , 1
  sdcheck sdbind SKLISTEN ; AF_INET ; '' ; PORT
  sdcheck sdlisten SKLISTEN , 5 , 1
  say 'listening on port ', (": PORT) , ' SKLISTEN=', (": SKLISTEN)
  selectloop ''
  say 'server stopped'
)
