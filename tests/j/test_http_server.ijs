NB. ================================================================
NB. test_http_server.ijs — HTTP server contract test (model-free
NB. surface). Verifies the OpenAI-compatible server's request parsing
NB. (h11_parse), pjson value extraction (getv), message conversion
NB. (mk_msgs), and response builders (v1_models / respbody / streaming
NB. frame_* / notfound) against exact output strings.
NB.
NB. NOTE: loading ./http/server.ijs loads the default model once
NB. (cached), but no generation or socket is exercised here — the live
NB. loop (res=/sdclose fix) is verified by scripts/llm_server.sh.
NB. ================================================================
coclass 'inference'

load './tests/j/test_harness.ijs'
load './http/server.ijs'

init_counters_inference_ ''

NB. ravel-normalized string compare
streq =: 4 : '(x -: , y)'

NB. CRLF is defined by protocol.ijs (loaded via server.ijs); do NOT
NB. redefine here — `13{a. , 10{a.}` parses right-to-left as
NB. `13{(a. , 10{a.)}` = CR only (1 char), NOT CRLF.

section_header_inference_ 'HTTP request parsing (h11_parse)'
req =: 'POST /v1/chat/completions HTTP/1.1' , CRLF , 'Host: localhost' , CRLF , 'Content-Length: 40' , CRLF , CRLF , '{"messages":[{"role":"user","content":"hi"}],"max_tokens":3}'
'm p v b'=. h11_parse req
assert_test ('POST' -: m) ; 'h11_parse method = POST'
assert_test ('/v1/chat/completions' -: p) ; 'h11_parse path = /v1/chat/completions'
assert_test ('HTTP/1.1' -: v) ; 'h11_parse version = HTTP/1.1'
assert_test ('{"messages":[{"role":"user","content":"hi"}],"max_tokens":3}' -: b) ; 'h11_parse body exact'

NB. GET root /v1/models request parses too
greq =: 'GET /v1/models HTTP/1.1' , CRLF , 'Host: localhost' , CRLF , CRLF
'gm gp gv gb'=. h11_parse greq
assert_test ('GET' -: gm) ; 'h11_parse GET method'
assert_test ('/v1/models' -: gp) ; 'h11_parse GET path'

section_header_inference_ 'pjson value extraction (getv)'
r=. dec_pjson_ b
assert_test (_1 -: 'stream' getv r) ; 'getv absent key -> _1'
assert_test (3 -: 'max_tokens' getv r) ; 'getv max_tokens = 3'

section_header_inference_ 'message conversion (mk_msgs)'
msgs=. 'messages' getv r
out=. mk_msgs msgs
assert_test (('user' -: > 0 { > 0 { out) *. ('hi' -: > 1 { > 0 { out)) ; 'mk_msgs first msg role+content'
assert_test (1 -: # out) ; 'mk_msgs one message'

section_header_inference_ 'response builders'
vm=. v1_models ''
assert_test (1 e. '{"object":"list","data":[{"id":"qwen3-0.6b","object":"model","created":' E. vm) ; 'v1_models id+object prefix'
assert_test (1 e. '"owned_by":"j"' E. vm) ; 'v1_models owned_by'
assert_test ('{"id":"cid1","object":"chat.completion","created":123,"model":"qwen3-0.6b","choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}' -: respbody ('cid1' ; 'qwen3-0.6b' ; 123 ; 'Hello' ; 'stop' ; '')) ; 'respbody exact'

section_header_inference_ 'streaming frame builders'
f1=. 'x' frame_first ('sid' ; 'qwen3-0.6b' ; 123)
assert_test ((1 e. '{"id":"sid","object":"chat.completion.chunk","created":123,"model":"qwen3-0.6b","choices":[{"index":0,"delta":{"role":"assistant","content":"x"},"finish_reason":null}]}' E. f1) *. (0 < # f1)) ; 'frame_first carries role+content delta'
f2=. 'y' frame_mid ('sid' ; 'qwen3-0.6b' ; 123)
assert_test ((1 e. '{"id":"sid","object":"chat.completion.chunk","created":123,"model":"qwen3-0.6b","choices":[{"index":0,"delta":{"content":"y"},"finish_reason":null}]}' E. f2) *. (0 < # f2)) ; 'frame_mid carries content-only delta'
f3=. 'stop' frame_end ('sid' ; 'qwen3-0.6b' ; 123)
assert_test ((1 e. 'finish_reason":"stop"}' E. f3) *. (0 < # f3)) ; 'frame_end carries finish_reason'
assert_test (0 < # frame_done '') ; 'frame_done emits done chunk'

section_header_inference_ 'error response (notfound)'
nf=. notfound ''
assert_test ((1 e. 'HTTP/1.1 404 Not Found' E. nf) *. (1 e. 'unknown path' E. nf)) ; 'notfound 404 + body'

show_summary_inference_ ''
