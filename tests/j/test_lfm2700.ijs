NB. ================================================================
NB. LFM2-700M (lfm2 arch) Test Suite
NB. Same arch as LFM2-350M (models/lfm2.ijs) — hybrid 10 shortconv + 6 attention
NB. layers, gpt2/llama3-regex tokenizer (same 65536 vocab), simple qwen-style
NB. chat template. Differences: emb 1536, n_ff 6912, heads 24 (GQA 24:8).
NB. ================================================================
coclass 'inference'
load './inference.ijs'
load './tests/j/test_harness.ijs'
load './tests/j/pm_fixture.ijs'

NB. ---- Argmax of a logits vector ----
argmax =: 3 : '>./ I. y = >./ y'

test_lfm2700 =: 3 : 0
  init_counters ''
  section_header 'LFM2-700M (lfm2 arch)'

  lf_path =. 'lfm2-700m'
  llm =. load_gguf_to_llm lf_path

  NB. --- architecture ---
  assert_test ('lfm2' -: llm_arch llm) ; 'arch = lfm2'
  assert_test ('lfm2' -: tokenizer_pre_g (llm_tokenizer llm)) ; 'tokenizer pre = lfm2 (llama3 regex)'
  mi =. llm_mi llm
  assert_test (16 -: mi_block_count mi) ; 'block_count = 16'
  assert_test (32768 -: mi_context_len mi) ; 'context_length = 32768'
  assert_test (1536 -: mi_emb_len mi) ; 'embedding_length = 1536'
  assert_test (24 -: mi_n_heads mi) ; 'head_count = 24'
  assert_test (8 -: mi_n_heads_kv mi) ; 'head_count_kv = 8 (GQA 24:8)'
  assert_test (64 -: mi_head_dim mi) ; 'head_dim = 64 (emb_len % n_heads)'
  assert_test ((| 1000000 - mi_rope_freq mi) < 0.001) ; 'rope.freq_base = 1e6'
  assert_test (6912 -: mi_n_ff mi) ; 'feed_forward_length = 6912'
  assert_test (65536 -: mi_vocab_size mi) ; 'vocab_size = 65536 (token_embd dims)'

  NB. --- tokenizer vs llama.cpp tokenize (add_bos=True, same vocab as LFM2) ---
  assert_test (1 52572 -: , > (lf2_tokenize (<llm) , <'hello')) ; 'tokenize hello'
  assert_test (1 597 -: , > (lf2_tokenize (<llm) , <'x')) ; 'tokenize x'
  assert_test (1 1098 5706 803 4481 856 -: , > (lf2_tokenize (<llm) , <'The capital of France is')) ; 'tokenize The capital of France is'
  assert_test (1 16203 1901 5910 -: , > (lf2_tokenize (<llm) , <'don''t stop')) ; 'tokenize don''t stop'
  assert_test (1 730 5845 12914 -: , > (lf2_tokenize (<llm) , <'  leading spaces')) ; 'tokenize leading spaces'

  NB. --- single-token infer argmax vs llama.cpp ---
  NB. Oracle ids = fresh-context greedy completion tokens ('!' ':\n' " '" 'ox').
  cases =. (<'hello' ; 510) , (<'The capital of France is' ; 1334) , (<'don''t stop' ; 1452) , (<'x' ; 1974)
  ci =. 0
  while. ci < # cases do.
    case =. > ci { cases
    text =. > 0 { case
    expect =. > 1 { case
    r =. llm lf2_infer_simple text
    lg =. > 3 { r
    got =. argmax lg
    assert_test (expect -: got) ; ('infer argmax ' , text , ' = ' , ": expect)
    ci =. ci + 1
  end.

  NB. --- chat prompt tokens vs llama.cpp (rendered template, special=True) ---
  messages =. <('user') ; 'The capital of France is'
  prompt =. lf2_chat_prompt messages
  toks =. lf2_tokenize (<llm) , <prompt
  got_toks =. , > toks
  exp_toks =. 1 6 6423 708 1098 5706 803 4481 856 7 708 6 64015 708
  assert_test (exp_toks -: got_toks) ; 'chat prompt tokens match llama.cpp'

  NB. --- generation (chat-framed, greedy) ---
  g =. llm lf2_generate ('The capital of France is' ; 64 ; <<0 ; 0 ; 0.95 ; 0.0)
  assert_test ('The capital of France is Paris.' -: > g) ; 'generation chat answer'

  show_summary 1
)
pm_start 1e8

test_lfm2700 0

pm_report ''
