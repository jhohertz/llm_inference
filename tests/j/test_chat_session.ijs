NB. ================================================================
NB. Chat Session Test Suite — crude console chat + KV persistence (option B)
NB. Verifies:
NB.   - turn-by-turn `chat` equals stateless chat_generate (full re-render)
NB.   - the RESUME path actually runs (chat_resume_count), not the fallback
NB.   - session state tracking + chat_reset
NB. qwen2 greedy is deterministic (default params 0 0 0.95 0.0); gemma3 uses
NB. explicit greedy params via chat_p.
NB. ================================================================
coclass 'inference'
load './inference.ijs'
load './models/qwen2.ijs'
load './models/gemma3.ijs'
load './util/chat.ijs'
load './tests/j/test_harness.ijs'
load './tests/j/pm_fixture.ijs'

qw_path =: 'qwen2.5-coder-0.5b'
gemma_path =: 'gemma-3-270m-it'

test_chat_session =: 3 : 0
  init_counters ''
  echo '========================================'
  echo 'Chat Session Test Suite (console chat + KV persistence)'
  echo '========================================'
  echo ''

  NB. ---- Section 1: qwen2 turn-by-turn persistence (greedy, deterministic) ----
  echo '--- qwen2: turn-by-turn chat == full-history chat_generate ---'
  llm =. load_gguf_to_llm qw_path

  NB. turn 1 (fresh)
  a1 =. llm chat 'The capital of France is'
  assert_test (a1 -: 'The capital of France is Paris.') ; 'qwen2 chat turn1 answer pin'

  NB. session state after turn 1
  assert_test ((> 0 { chat_session_g) -: 'qwen2') ; 'session arch = qwen2'
  assert_test (2 = # > 1 { chat_session_g) ; 'session has 2 messages (user + assistant)'
  pos1 =. > 3 { chat_session_g
  assert_test (pos1 > 0) ; 'session cur_pos > 0 after turn 1'

  NB. turn 2 (persistence)
  a2 =. llm chat 'And what is its population?'
  assert_test (1 = chat_resume_count) ; 'qwen2 turn2 used the RESUME path (not fallback)'
  NB. stateless reference: the FULL conversation (user1, assistant1, user2)
  msgs =. <('user') ; 'The capital of France is'
  msgs =. msgs , <('assistant') ; a1
  msgs =. msgs , <('user') ; 'And what is its population?'
  ref2 =. llm chat_generate (msgs ; 100000 ; <0 0 0.95 0.0)
  assert_test (a2 -: ref2) ; 'qwen2 turn2 (persisted) == full re-render'
  assert_test (pos1 < > 3 { chat_session_g) ; 'session cur_pos grew across turns'

  NB. chat_reset clears the session; a fresh turn == turn 1
  chat_reset ''
  assert_test (0 = # chat_session_g) ; 'chat_reset clears session'
  a1b =. llm chat 'The capital of France is'
  assert_test (a1b -: a1) ; 'qwen2 chat after reset == fresh turn 1'

  NB. ---- Section 2: gemma3 persistence (SentencePiece round-trip risk) ----
  echo '--- gemma3: turn-by-turn persistence (greedy params) ---'
  chat_resume_count =: 0
  chat_fallback_count =: 0
  llm2 =. load_gguf_to_llm gemma_path
  g1 =. llm2 chat_p ('The capital of France is' ; <0 0 0.95 0.0)
  assert_test (g1 -: 'The capital of France is Paris.') ; 'gemma3 chat turn1 answer pin'
  g2 =. llm2 chat_p ('And what is its population?' ; <0 0 0.95 0.0)
  assert_test (1 = chat_resume_count) ; 'gemma3 turn2 used the RESUME path (not fallback)'
  gmsgs =. <('user') ; 'The capital of France is'
  gmsgs =. gmsgs , <('assistant') ; g1
  gmsgs =. gmsgs , <('user') ; 'And what is its population?'
  gref2 =. llm2 chat_generate (gmsgs ; 100000 ; <0 0 0.95 0.0)
  assert_test (g2 -: gref2) ; 'gemma3 turn2 (persisted) == full re-render'

  echo ''
  show_summary 1
  ''
)

pm_start 1e8

test_chat_session 0

pm_report ''
