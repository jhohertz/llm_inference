NB. ================================================================
NB. Llama3 Tokenizer Test Suite
NB. Tests: pre-tokenizer, symbol lookup, tokenize, detokenize
NB. ================================================================
coclass 'inference'
load './tokenizers/tokenizer_llama3.ijs'
load './tests/j/test_harness.ijs'
load './tests/j/pm_fixture.ijs'

NB. ---- Pre-tokenizer tests ----
test_pre_tokenize =: 3 : 0
  echo ''
  echo '--- Pre-tokenizer ---'
  
  pieces =. llama3_pre_tokenize 'Hello world'
  if. 2 -: # pieces do. pass 'simple text splits' else. fail 'simple text' end.
  
  pieces =. llama3_pre_tokenize 'it''s'
  if. 2 <: # pieces do. pass 'contraction handled' else. fail 'contraction' end.
  
  pieces =. llama3_pre_tokenize 'test 123 numbers'
  if. 3 <: # pieces do. pass 'numbers handled' else. fail 'numbers' end.
  
  pieces =. llama3_pre_tokenize 'Hello, world!'
  if. 4 <: # pieces do. pass 'punctuation separated' else. fail 'punctuation' end.
  
  pieces =. llama3_pre_tokenize ''
  if. 0 = # pieces do. pass 'empty input' else. fail 'empty input' end.
  
  pieces =. llama3_pre_tokenize 'a'
  if. 1 = # pieces do. pass 'single char' else. fail 'single char' end.
  
  ''
)

NB. ---- Build a minimal llm noun with tokenizer at index 3 ----
NB. tokenizer = [vocab; bos_token_id; tk_len; sym; eos_token_id; specials]
NB. llama3_tokenize/detokenize read index 3
mk_llm =: 3 : 0
  (<'') , (<'') , (<'') , <y
)

NB. ---- Full tokenizer tests ----
test_tokenize =: 3 : 0
  echo ''
  echo '--- Full tokenizer ---'
  vocab =. (<'Hello') , (<'world') , (<'test') , (<'!') , (<'end')
  tokenizer =. (<vocab) , (<1) , (<s: vocab) , (<2) , (<'')
  llm =. mk_llm tokenizer

  tokens =. llama3_tokenize (<llm) , <'Hello world'
  if. 3 <: # tokens do. pass 'tokenize basic' else. fail 'tokenize basic' end.

  if. 1 = > 0 { tokens do. pass 'BOS prepended' else. fail 'BOS' end.

  tokens =. llama3_tokenize (<llm) , <'unknown word'
  if. 3 <: # tokens do. pass 'tokenize unknown words' else. fail 'tokenize unknown' end.

  tokens =. llama3_tokenize (<llm) , <''
  if. 1 = # tokens do. pass 'empty text gives BOS only' else. fail 'empty text' end.

  ''
)

NB. ---- Detokenizer tests ----
test_detokenize =: 3 : 0
  echo ''
  echo '--- Detokenizer ---'
  vocab =. (<'Hello') , (<'world') , (<'!')
  tokenizer =. (<vocab) , (<0) , (<s: vocab) , (<2) , (<'')
  llm =. mk_llm tokenizer

  tokens =. <0 1 2
  result =. llama3_detokenize (<llm) , <tokens
  if. 0 < # result do. pass 'detokenize produces text' else. fail 'detokenize' end.

  tokens =. <(65536 + 65) , (65536 + 66)
  result =. llama3_detokenize (<llm) , <tokens
  if. 'AB' -: result do. pass 'byte token decode' else. fail 'byte token' end.

  tokens =. ''
  result =. llama3_detokenize (<llm) , <tokens
  if. 0 = # result do. pass 'empty tokens' else. fail 'empty tokens' end.

  ''
)

NB. ---- Round-trip tests ----
test_roundtrip =: 3 : 0
  echo ''
  echo '--- Round-trip ---'
  vocab =. (<'Hello') , (<'world') , (<'!') , (<'test')
  tokenizer =. (<vocab) , (<1) , (<s: vocab) , (<2) , (<'')
  llm =. mk_llm tokenizer

  original =. 'Hello world!'
  tokens =. llama3_tokenize (<llm) , <original
  decoded =. llama3_detokenize (<llm) , <tokens
  if. 0 < # decoded do. pass 'round-trip produces output' else. fail 'round-trip' end.

  original =. 'foo bar baz'
  tokens =. llama3_tokenize (<llm) , <original
  decoded =. llama3_detokenize (<llm) , <tokens
  if. 0 < # decoded do. pass 'round-trip unknown words' else. fail 'round-trip unknown' end.

  ''
)

NB. ---- Main test runner ----
test_tokenizer_llama3 =: 3 : 0
  init_counters ''
  echo '========================================'
  echo 'Llama3 Tokenizer Test Suite'
  echo '========================================'
  
  test_pre_tokenize ''
  test_tokenize ''
  test_detokenize ''
  test_roundtrip ''
  
  show_summary 1
  
  ''
)

pm_start 1e8

test_tokenizer_llama3 0

pm_report ''
