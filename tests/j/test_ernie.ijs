NB. ================================================================
NB. ERNIE-4.5-0.3B-PT (ernie4_5 arch) Test Suite
NB. Verifies structure, the SPM tokenizer, infer argmax, chat prompt tokens,
NB. and generation against the llama-cpp-python oracle.
NB. ERNIE is a standard decoder (byte-for-byte the llama arch: GQA, SwiGLU,
NB. interleaved RoPE, 1/sqrt(head_dim), tied embeddings, NO scaling KVs) with
NB. an SPM (SentencePiece bigram-merge) tokenizer and the ERNIE chat template
NB. (cls <|begin_of_sentence|>; User:/Assistant:/system; 'Assistant: ' prompt).
NB. ================================================================
coclass 'inference'
load './inference.ijs'
load './tests/j/test_harness.ijs'
load './tests/j/pm_fixture.ijs'

NB. ---- Argmax of a logits vector ----
argmax =: 3 : '>./ I. y = >./ y'

test_ernie =: 3 : 0
  init_counters ''
  section_header 'ERNIE-4.5-0.3B (ernie4_5 arch)'

  er_path =. 'ernie-4.5-0.3b'
  llm =. load_gguf_to_llm er_path

  NB. --- architecture ---
  assert_test ('ernie4_5' -: llm_arch llm) ; 'arch = ernie4_5'
  assert_test ('default' -: spm_pre (llm_tokenizer llm)) ; 'tokenizer pre = default (SPM)'
  mi =. llm_mi llm
  assert_test (18 -: mi_block_count mi) ; 'block_count = 18'
  assert_test (131072 -: mi_context_len mi) ; 'context_length = 131072'
  assert_test (1024 -: mi_emb_len mi) ; 'embedding_length = 1024'
  assert_test (16 -: mi_n_heads mi) ; 'head_count = 16'
  assert_test (2 -: mi_n_heads_kv mi) ; 'head_count_kv = 2 (GQA 16:2)'
  assert_test (128 -: mi_head_dim mi) ; 'head_dim = 128 (key_length)'
  assert_test ((| 500000 - mi_rope_freq mi) < 0.001) ; 'rope.freq_base = 5e5'
  assert_test (3072 -: mi_n_ff mi) ; 'feed_forward_length = 3072'
  assert_test (103424 -: mi_vocab_size mi) ; 'vocab_size = 103424 (token_embd dims)'

  NB. --- SPM tokenizer vs llama.cpp tokenize (add_bos=False) ---
  tok_cases =. (<'hello' ; 23013) , (<'The' ; 526) , (<'Paris' ; 11855) , (<'x' ; 843)
  ci =. 0
  while. ci < # tok_cases do.
    case =. > ci { tok_cases
    text =. > 0 { case
    expect =. , > 1 { case
    toks =. , > (ernie_tokenize (<llm) , <text)
    assert_test (expect -: toks) ; ('tokenize ' , text , ' = ' , ": expect)
    ci =. ci + 1
  end.
  NB. Multi-token cases (exact token streams)
  assert_test (23013 3135 -: , > (ernie_tokenize (<llm) , <'hello world')) ; 'tokenize hello world'
  assert_test (526 9689 315 10298 357 -: , > (ernie_tokenize (<llm) , <'The capital of France is')) ; 'tokenize The capital of France is'
  assert_test (1504 93968 93921 4038 -: , > (ernie_tokenize (<llm) , <'don''t stop')) ; 'tokenize don''t stop'
  assert_test (269 8143 15075 -: , > (ernie_tokenize (<llm) , <'  leading spaces')) ; 'tokenize leading spaces'

  NB. --- single-token infer argmax vs llama.cpp ---
  NB. Oracle ids = fresh-context greedy completion tokens from llama-cpp-python
  NB. (single-token gens: ' the' ' the' '\n' '.' ' located').
  cases =. (<'hello' ; 290) , (<'The' ; 290) , (<'Paris' ; 23) , (<'hello world' ; 93937) , (<'The capital of France is' ; 7365)
  ci =. 0
  while. ci < # cases do.
    case =. > ci { cases
    text =. > 0 { case
    expect =. > 1 { case
    r =. llm ernie_infer_simple text
    lg =. > 3 { r
    got =. argmax lg
    assert_test (expect -: got) ; ('infer argmax ' , text , ' = ' , ": expect)
    ci =. ci + 1
  end.

  NB. --- chat prompt tokens vs llama.cpp _input_ids (prompt-only prefix) ---
  messages =. <('user') ; 'The capital of France is'
  prompt =. ernie_chat_prompt messages
  toks =. ernie_tokenize (<llm) , <prompt
  got_toks =. , > toks
  exp_toks =. 100273 8933 93963 526 9689 315 10298 357 23 92267 93963 93919
  assert_test (exp_toks -: got_toks) ; 'chat prompt tokens match llama.cpp'

  NB. --- generation (chat-framed, greedy) ---
  g =. llm ernie_generate ('The capital of France is' ; 32 ; <<0 ; 0 ; 0.95 ; 0.0)
  assert_test ('The capital of France is Paris.' -: > g) ; 'generation chat answer'

  show_summary 1
)
pm_start 1e8

test_ernie 0

pm_report ''
