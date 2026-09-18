NB. ============================================================
NB.  protocol.ijs - HTTP/1.1 protocol layer for the jsocket server.
NB.  (molded from the j-http-handoff: pure protocol helpers, NO socket
NB.  calls): request framing / parse, response build, chunked
NB.  transfer-encoding, SSE events. The driver (server.ijs) owns the
NB.  sockets and calls these. Locale-independent; loaded by server.ijs
NB.  into the inference locale.
NB.  Load:   load 'llm/inference/http/protocol'  (addon) or './http/protocol.ijs'
NB.
NB.  Operator code page (standard J9.7, confirmed against
NB.  system/main/stdlib.ijs):
NB.    -  13{a. / 10{a. = at (index-fetch) -> CR / LF bytes
NB.       (matches stdlib.ijs:207  '...CR...'=: ...13{a. )
NB.    -  n{. y = take (first n)      n}. y = drop (all but first n)
NB.    -  n{  y = at   (fetch ONE element, 0-based, negative wraps)
NB.    -  x E. y = pattern-beginnings mask (x=pattern y=haystack)
NB.       (decapdot.htm)  ->  1s at each occurrence start
NB.    -  I. y = interval index = indices of the 1s in y (dicapdot)
NB.    -  NO operator precedence (strictly right-to-left); every
NB.       count / slice / compare is parenthesized.
NB.    - Control structures live INSIDE explicit defs; loops use
NB.       while. with a manual index.  "x>0" boolean idiom: 0<x.
NB.    - Comparison:  x = y exact equal;  x -: y shape ONLY.
NB.    - number->string:  s =: \": n  (named var FIRST, then comma).
NB.       Hex via stdlib hfd/dfh.  Decimal parse: 10 #. ((a. i. s)-48).
NB.    - cutopen returns a LIST OF BOXED items: fetch with >n{list .
NB. ============================================================

NB. --- line terminators (1 REAL byte each; at-fetch by code) ---
CR   =: 13{a.
LF   =: 10{a.
TAB  =: 9{a.
CRLF =: CR , LF
EOM  =: CRLF , CRLF        NB. CR LF CR LF  (end-of-headers marker)

NB. ============================================================
NB.  y h11_eom  ->  start pos of the FIRST empty line (CRLFCRLF),
NB.  else _1.  (The empty line that separates headers from body.)
h11_eom=: 3 : 0
  p =: I. (EOM E. y)
  if. 0 = #p do. _1 return. end.
  0{p
)

NB. ============================================================
NB.  y h11_complete  ->  1 if y is a COMPLETE request: the empty
NB.  line is present AND the body has at least Content-Length bytes.
NB.  NOTE: "0 <= x" misbehaves here (0<=-20 -> 1); "x >= n" likewise.
NB.  Only  "0 < x"  /  "x > 0"  verified reliable -> compute the
NB.  shortfall first, then test  0 < d  (right-to-left: precompute!).
h11_complete=: 3 : 0
  e =: h11_eom y
  if. e = _1 do. 0 return. end.
  cl =: h11_clen y
  bd =: h11_body y
  d =: cl - (#bd)
  if. 0 < d do. 0 return. end.
  1
)

NB. ============================================================
NB.  y h11_body  ->  raw body chars (after the empty line), else ''.
h11_body=: 3 : 0
  e =: h11_eom y
  if. e = _1 do. '' return. end.
  (e + 4) }. y
)

NB. ============================================================
NB.  y h11_hb  ->  header block (request line + header lines, up to
NB.  but not including the empty line), else ''.
h11_hb=: 3 : 0
  e =: h11_eom y
  if. e = _1 do. '' return. end.
  e {. y
)

NB. ============================================================
NB.  y h11_rline  ->  the request line (up to the first CRLF), ''.
h11_rline=: 3 : 0
  p1 =: I. (CRLF E. y)
  if. 0 = #p1 do. '' return. end.
  (0{p1) {. y
)

NB. ============================================================
NB.  y h11_method / y h11_path / y h11_ver
NB.  ->  the 3 request-line fields (split on space).
h11_method=: 3 : 0
  >0{' ' cutopen (h11_rline y)
)
h11_path=: 3 : 0
  >1{' ' cutopen (h11_rline y)
)
h11_ver=: 3 : 0
  >2{' ' cutopen (h11_rline y)
)

NB. ============================================================
NB.  name h11_hval  y  ->  value of the header "name" in y (a header
NB.  block), else ''.  Matches  "name:"  case-sensitively; value runs
NB.  to the next CR.
h11_hval=: 4 : 0
  pref =: x , ':'
  m =: pref E. y
  p =: I. m
  if. 0 = #p do. '' return. end.
  q =: 0{p
  v0 =: (q + #pref) }. y
  if. 0 = #v0 do. '' return. end.
  if. ' ' = 0{v0 do. v0 =: 1}. v0 end.
  cm =: CR E. v0
  c1 =: I. cm
  if. 0 = #c1 do. v0 return. end.
  c0 =: 0{c1
  (c0) {. v0
)

NB. ============================================================
NB.  y h11_clen  ->  Content-Length as a number (0 if absent).
NB.  y is a full request; searches only the header block.
h11_clen=: 3 : 0
  v =: 'Content-Length' h11_hval (h11_hb y)
  if. 0 = #v do. 0 return. end.
  10 #. ((a. i. v) - 48)
)

NB. ============================================================
NB.  y h11_parse  ->  m ; p ; v ; b  (4-item box list, Link).
NB.  (y is a COMPLETE request buffer.)  Caller unboxes with >n{r .
NB.  NOTE: locals renamed mm/pp/vv/bb to avoid colliding with the
NB.  local  "p"  inside h11_rline (which h11_path calls).
h11_parse=: 3 : 0
  mm =: h11_method y
  pp =: h11_path y
  vv =: h11_ver y
  bb =: h11_body y
  mm ; pp ; vv ; bb
)

NB. ============================================================
NB.  status reason ->  'HTTP/1.1 <status> <reason>' + CRLF
h11_statusline=: 4 : 0
  'HTTP/1.1 ' , x , ' ' , y , CRLF
)

NB. ============================================================
NB.  hdrs h11_hdrstr  ->  header bytes, each line + CRLF.
NB.  hdrs is a 1D list of "Name: value" strings (one per row).
h11_hdrstr=: 3 : 0
  r =: ''
  nl =: #y
  idx =: 0
  while. 0 < nl - idx do.
    r =: r , (idx { y) , CRLF
    idx =: idx + 1
  end.
  r
)

NB. ============================================================
NB.  y h11_simple  ->  a COMPLETE response.
NB.  y is a 4-item  " ; "  list (char atoms, NO boxes):
NB.  <status , reason , hdrs , body>.  Caller builds with  ,  not  ; .
NB.  status line + hdrs + Content-Length + CRLFCRLF + body.
h11_simple=: 3 : 0
  'st rs hd bd'=. 0 1 2 3 { y
  r =: st h11_statusline rs
  r =: r , hd
  cl =: ": (#bd)
  r =: r , 'Content-Length: ' , cl , CRLF
  r =: r , CRLF
  r , bd
)

NB. ============================================================
NB.  y h11_chunkhead  ->  status line + hdrs
NB.  + 'Transfer-Encoding: chunked' + CRLFCRLF.
NB.  y is a 3-item  " ; "  list:  <status , reason , hdrs>.
h11_chunkhead=: 3 : 0
  'st rs hd'=. 0 1 2 { y
  r =: st h11_statusline rs
  r =: r , hd
  r =: r , 'Transfer-Encoding: chunked' , CRLF
  r , CRLF
)

NB. ============================================================
NB.  data h11_chunk  ->  one chunked piece:
NB.  <hexlen> CRLF data CRLF   (hex via stdlib hfd)
h11_chunk=: 3 : 0
  (hfd (#y)) , CRLF , y , CRLF
)

NB. ============================================================
NB.  h11_chunkend  ->  the terminating chunk: 0 CRLF CRLF
h11_chunkend=: 3 : 0
  '0' , CRLF , CRLF
)

NB. ============================================================
NB.  data h11_sse  ->  one SSE message: "data: <data>\n\n"
NB.  (OpenAI stream framing; data is already a JSON string.)
h11_sse=: 3 : 0
  'data: ' , y , LF , LF
)

NB. ============================================================
NB.  h11_done  ->  the SSE stream terminator: "data: [DONE]\n\n"
h11_done=: 3 : 0
  'data: [DONE]' , LF , LF
)
