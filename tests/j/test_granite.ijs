NB. ================================================================
NB. Granite-4.0-350m (granite arch) Test Suite
NB. Verifies structure, the Granite 4.0 scaling params, infer argmax, chat
NB. prompt tokens, and generation against the llama-cpp-python oracle.
NB. Granite is a standard decoder (like llama) plus the granite scaling:
NB. embed*12, residual*0.263 on attn/ffn outputs, scores*0.015625,
NB. logits/4, tied embeddings, dbrx pre-tokenizer (same regex as llama3).
NB. ================================================================
coclass 'inference'
load './inference.ijs'
load './tests/j/test_harness.ijs'
load './tests/j/pm_fixture.ijs'

NB. ---- Argmax of a logits vector ----
argmax =: 3 : '>./ I. y = >./ y'

test_granite =: 3 : 0
  init_counters ''
  section_header 'Granite-4.0-350m (granite arch)'

  gr_path =. 'granite-4.0-350m'
  llm =. load_gguf_to_llm gr_path

  NB. --- architecture ---
  assert_test ('granite' -: llm_arch llm) ; 'arch = granite'
  assert_test ('dbrx' -: tokenizer_pre_g (llm_tokenizer llm)) ; 'tokenizer pre = dbrx (llama3 regex)'
  mi =. llm_mi llm
  assert_test (28 -: mi_block_count mi) ; 'block_count = 28'
  assert_test (32768 -: mi_context_len mi) ; 'context_length = 32768'
  assert_test (1024 -: mi_emb_len mi) ; 'embedding_length = 1024'
  assert_test (16 -: mi_n_heads mi) ; 'head_count = 16'
  assert_test (4 -: mi_n_heads_kv mi) ; 'head_count_kv = 4 (GQA 16:4)'
  assert_test (64 -: mi_head_dim mi) ; 'head_dim = 64 (rope.dimension_count)'
  assert_test ((| 10000000 - mi_rope_freq mi) < 0.001) ; 'rope.freq_base = 1e7'
  assert_test (2048 -: mi_n_ff mi) ; 'feed_forward_length = 2048'
  assert_test (100352 -: mi_vocab_size mi) ; 'vocab_size = 100352'
  assert_test (12 -: granite_mi_embed_scale mi) ; 'embedding_scale = 12'
  assert_test ((| 0.263 - granite_mi_resid_scale mi) < 0.001) ; 'residual_scale = 0.263'
  assert_test (4 -: granite_mi_logit_scale mi) ; 'logit_scale = 4'
  assert_test (0.015625 -: granite_mi_attn_scale mi) ; 'attention.scale = 0.015625'

  NB. --- single/multi-token infer argmax vs llama.cpp ---
  NB. Oracle ids = fresh-context greedy completion tokens from llama-cpp-python
  NB. (single-token gens: ',' ' of' ' paris' ' until').
  cases =. (<'hello' ; 11) , (<'The' ; 11) , (<'Paris' ; 315) , (<'hello world' ; 11) , (<'The capital of france is' ; 41958) , (<'don''t stop' ; 3156)
  ci =. 0
  while. ci < # cases do.
    case =. > ci { cases
    text =. > 0 { case
    expect =. > 1 { case
    r =. llm granite_infer_simple text
    lg =. > 3 { r
    got =. argmax lg
    assert_test (expect -: got) ; ('infer argmax ' , text , ' = ' , ": expect)
    ci =. ci + 1
  end.

  NB. --- chat prompt tokens vs llama.cpp _input_ids ---
  messages =. <('user') ; 'The capital of France is'
  prompt =. granite_chat_prompt messages
  toks =. granite_tokenize (<llm) , <prompt
  got_toks =. , > toks
  exp_toks =. 100264 9125 100265 2675 527 264 11190 18328 13 5321 6106 14847 527 6721 11 13687 11 323 6220 13 100257 198 100264 882 100265 791 6864 315 9822 374 100257 198 100264 78191 100265
  assert_test (exp_toks -: got_toks) ; 'chat prompt tokens match llama.cpp'

  NB. --- generation (chat-framed, greedy) ---
  g =. llm granite_generate ('The capital of France is' ; 32 ; <<0 ; 0 ; 0.95 ; 0.0)
  assert_test ('The capital of France is Paris.' -: > g) ; 'generation chat answer'

  show_summary 1
)
pm_start 1e8

test_granite 0

pm_report ''
