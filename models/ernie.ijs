NB. ================================================================
NB. ERNIE arch — standard-decoder transformer (GQA, SwiGLU,
NB. interleaved RoPE, separate QKV/O weights, TIED embeddings).
NB. The forward pass is byte-for-byte the generic llama arch
NB. (llama.cpp src/models/ernie4-5.cpp: no embedding/residual/logit/
NB. attention scales, Q scaled 1/sqrt(head_dim), RoPE NORM/interleaved,
NB. n_rot = head_dim). ERNIE reuses llama.ijs forward verbs; only the
NB. loader, tokenizer (SPM), and chat template differ.
NB. Tokenizer: SentencePiece (llama.cpp llm_tokenizer_spm bigram merge),
NB. tokenizer.ggml.model='llama' -> SPM; add_space_prefix=true default
NB. (leading ▁); byte tokens are <0xXX> strings; NO bos (add_bos=false).
NB. Chat template: cls <|begin_of_sentence|>; User:/Assistant:/system
NB. messages; generation prompt 'Assistant: '; stop on </s> or
NB. <|end_of_sentence|>.
NB. Depends on: llm_core.ijs, kernels.ijs, gguf.ijs, kv_cache.ijs, sampler.ijs,
NB.           tokenizer_spm.ijs, llama.ijs (forward verbs)
NB. ================================================================
coclass 'inference'
require 'llm/inference/util/llm_core'
require 'llm/inference/tokenizers/tokenizer_spm'
require 'llm/inference/models/llama'
require 'llm/inference/util/minja'
require 'llm/inference/util/chat_template'

NB. ---- Extract ERNIE model info from KV pairs ----
NB. mi = <block_count; context_len; emb_len; n_heads; n_heads_kv; head_dim;
NB.      rope_freq; vocab_size; rms_eps; n_ff>  (load appends rope tables at
NB.      10/11 and fixes vocab_size at 7 from token_embd dims — ERNIE has
NB.      no vocab_size KV)
ernie_extract_hparams =: 3 : 0
  data =. y
  block_count =. 'ernie4_5.block_count' kv_uint data
  context_length =. 'ernie4_5.context_length' kv_uint data
  emb_len =. 'ernie4_5.embedding_length' kv_uint data
  n_heads =. 'ernie4_5.attention.head_count' kv_uint data
  n_heads_kv =. 'ernie4_5.attention.head_count_kv' kv_uint data
  rope_freq =. 'ernie4_5.rope.freq_base' kv_float data
  vocab_size =. 0
  rms_eps =. 'ernie4_5.attention.layer_norm_rms_epsilon' kv_float data
  n_ff =. 'ernie4_5.feed_forward_length' kv_uint data
  key_len =. 'ernie4_5.attention.key_length' kv_uint data
  if. key_len <: 0 do. key_len =. emb_len % n_heads end.
  head_dim =. key_len
  (<"0) block_count , context_length , emb_len , n_heads , n_heads_kv , head_dim , rope_freq , vocab_size , rms_eps , n_ff
)

NB. ---- Load ERNIE GGUF into llm noun ----
ernie_load =: 3 : 0
  NB. y = <path; raw> — raw is the memory-mapped file (mapped by
  NB. load_gguf_to_llm; unmap'd after load — kvs_ctx is load-time only).
  data =. y
  path =. > 0 { data
  raw =. > 1 { data
  header =. parse_hdr_raw raw
  n_tensors =. > 2 { header
  kv_result =. parse_kv_pairs_raw raw
  kvs =. > 0 { kv_result
  kv_end =. > 3 { kv_result
  ti =. parse_tensor_infos (<raw) , (<kv_end) , (<n_tensors)
  ti_end_offset =. > ((n_tensors * 6) - 1) { ti
  tds =. 32 * <. (ti_end_offset + 31) % 32
  kvs_ctx =. (<kvs) , (<raw)
  kv_data =. build_kv_dict kvs_ctx
  mi =. ernie_extract_hparams kvs_ctx
  mi =. build_mi_dict mi
  rope_tables =. build_rope_tables ((< mi_context_len mi) , (< mi_head_dim mi) , (< mi_rope_freq mi))
  NB. Default attention/residual scales: 1/sqrt(hd) and 1. ERNIE is a
  NB. byte-for-byte llama arch (aliases llama's forward verbs), so it sets the
  NB. shared mi_attn_scale/mi_resid_scale defaults (granite overrides with GGUF).
  NB. ONE multi-put: the jdict single-put (1-key) path is broken in this J9.8
  NB. build (domain/length error); multi-put (2+ keys) and get/has work.
  NB. ERNIE has no vocab_size KV — derive from token_embd dims (dims = [emb, vocab]).
  head_dim =. mi_head_dim mi
  tok_info =. 'token_embd.weight' get_tensor_info ti
  dims =. ti_dims tok_info
  vocab_size =. 1 { dims
  ((> 0 { rope_tables) ; (> 1 { rope_tables) ; (1 % head_dim ^ 0.5) ; 1 ; vocab_size) put__mi 'cos_tab' ; 'sin_tab' ; 'attn_scale' ; 'resid_scale' ; 'vocab_size'
  NB. Real chat template from the GGUF ('' if absent → bespoke fallback).
  ct_tmpl =. 'tokenizer.chat_template' kv_string (0 1 { kv_result)
  tokenizer =. build_spm_tokenizer kv_result
  stop_tokens =. (spm_eos tokenizer) , 100272
  all_tensors =. ''
  tensor_idx =. 0
  while. tensor_idx < n_tensors do.
    tname =. > (tensor_idx * 6) { ti
    tdata =. (<path) , (<ti) , (<tds) , (<tname) , (<raw)
    td =. load_tdata tdata
    if. 0 < # td do.
      ti_row =. tname get_tensor_info ti
      dims =. ti_dims ti_row
      etype =. ti_etype ti_row
      all_tensors =. all_tensors , (<tname) , (<td) , (<dims) , (<etype)
    end.
    tensor_idx =. tensor_idx + 1
  end.
  n_heads_kv =. mi_n_heads_kv mi
  head_dim =. mi_head_dim mi
  ctx_len =. mi_context_len mi
  block_count =. mi_block_count mi
  p  =. <path
  t  =. <ti
  ze =. <0 0 0.95 0.0   NB. default_params (chat sampling defaults)
  tk =. <tokenizer
  mi_b =. <mi
  kc_b =. <(block_count ; ctx_len ; n_heads_kv ; head_dim)   NB. kv cache config (per-model, authoritative; mirror globals are the hot path)
  td =. <tds
  all_tensors =. emb_canonical all_tensors
  at =. <all_tensors
  llm =. p , t , ze , tk , mi_b , kc_b , td , at
  block_data =. llama_pre_build_block_data llm
  llm =. llm , <block_data
  llm =. llm , <kv_data
  llm =. llm , <stop_tokens
  llm =. llm , <ct_tmpl
)

NB. ---- Tokenize/detokenize (SPM) ----
NB. No BOS (add_bos_token=false): raw infer and chat both start with the text.
ernie_tokenize =: llm_tokenize
ernie_detokenize =: llm_detokenize

NB. ---- Forward verbs: ERNIE reuses the generic llama arch ----
ernie_run_blocks =: llama_run_blocks
ernie_run_blocks_b =: llama_run_blocks_b
ernie_run_blocks_bd =: llama_run_blocks_bd
ernie_run_blocks_bp =: llama_run_blocks_bp



NB. ---- ERNIE chat template ----
NB. cls_token '<|begin_of_sentence|>'; user 'User: <content>\n'; assistant
NB. 'Assistant: <content><|end_of_sentence|>'; system '<content>\n'; then the
NB. generation prompt 'Assistant: '. No BOS (add_bos_token=false) and no trim.


NB. Stop tokens: EOS (</s> = 2) and the chat sep token <|end_of_sentence|> (100272).
NB. Computed once in ernie_load (dict member stop_tokens).
ernie_stop_tokens =: llm_stop_tokens
