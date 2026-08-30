NB. ================================================================
NB. Llama-3.2-1B-Instruct (llama arch) Test Suite
NB. Verifies structure, infer argmax, chat-prompt tokens, and generation
NB. against the llama-cpp-python oracle. Llama-3.2 is the generic llama
NB. arch: dims from the GGUF, llama-bpe tokenizer (llama3 regex pre + gpt2
NB. BPE merges), llama3 chat template with the always-emitted system block.
NB. ================================================================
coclass 'inference'
load './inference.ijs'
load './tests/j/test_harness.ijs'
load './tests/j/pm_fixture.ijs'

NB. ---- Argmax of a logits vector ----
argmax =: 3 : '>./ I. y = >./ y'

test_llama32 =: 3 : 0
  init_counters ''
  section_header 'Llama-3.2-1B-Instruct (llama arch)'

  sm_path =. 'llama-3.2-1b'
  llm =. load_gguf_to_llm sm_path

  NB. Pin the chat template date for a stable oracle (llama-cpp-python
  NB. injects the current date; the test oracle uses 26 Jul 2024). Set
  NB. AFTER the load — the dispatch require of models/llama.ijs resets it.
  llama32_chat_date_g =: '26 Jul 2024'

  NB. --- architecture ---
  assert_test ('llama' -: llm_arch llm) ; 'arch = llama'
  assert_test ('llama-bpe' -: llama_tokenizer_pre_g) ; 'tokenizer pre = llama-bpe (llama3 template)'
  mi =. llm_mi llm
  assert_test (16 -: mi_block_count mi) ; 'block_count = 16'
  assert_test (131072 -: mi_context_len mi) ; 'context_length = 131072'
  assert_test (2048 -: mi_emb_len mi) ; 'embedding_length = 2048'
  assert_test (32 -: mi_n_heads mi) ; 'head_count = 32'
  assert_test (8 -: mi_n_heads_kv mi) ; 'head_count_kv = 8 (GQA 32:8)'
  assert_test (64 -: mi_head_dim mi) ; 'head_dim = 64'
  assert_test (500000 -: mi_rope_freq mi) ; 'rope.freq_base = 5e5'
  assert_test (8192 -: mi_n_ff mi) ; 'feed_forward_length = 8192'
  assert_test (128256 -: mi_vocab_size mi) ; 'vocab_size = 128256'

  NB. --- single/multi-token infer argmax vs llama.cpp ---
  NB. Oracle ids from llama-cpp-python on the same GGUF (logits_all argmax).
  cases =. (<'hello' ; 11) , (<'The' ; 2768) , (<'Paris' ; 11) , (<'hello world' ; 271) , (<'The capital of france is' ; 12366) , (<'don''t stop' ; 2888)
  ci =. 0
  while. ci < # cases do.
    case =. > ci { cases
    text =. > 0 { case
    expect =. > 1 { case
    r =. llm llama_infer_simple text
    lg =. > 3 { r
    got =. argmax lg
    assert_test (expect -: got) ; ('infer argmax ' , text , ' = ' , ": expect)
    ci =. ci + 1
  end.

  NB. --- chat prompt tokens vs llama.cpp _input_ids (fixed date) ---
  messages =. <('user') ; 'The capital of France is'
  prompt =. llama_chat_prompt messages
  toks =. llama_tokenize (<llm) , <prompt
  got_toks =. , > toks
  exp_toks =. 128000 128006 9125 128007 271 38766 1303 33025 2696 25 6790 220 2366 18 198 15724 2696 25 220 1627 10263 220 2366 19 271 128009 128006 882 128007 271 791 6864 315 9822 374 128009 128006 78191 128007 271
  assert_test (exp_toks -: got_toks) ; 'chat prompt tokens match llama.cpp (26 Jul 2024)'

  NB. --- generation (chat-framed, greedy) ---
  g =. llm llama_generate ('The capital of France is' ; 32 ; <<0 ; 0 ; 0.95 ; 0.0)
  assert_test ('The capital of France is Paris.' -: > g) ; 'generation chat answer'

  show_summary 1
)
pm_start 1e8

test_llama32 0

pm_report ''
