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

NB. ---- KV helpers ----
ernie_kv_uint =: 4 : 0
  key =. x
  data =. y
  key kv_uint data
)

ernie_kv_float =: 4 : 0
  key =. x
  data =. y
  key kv_float data
)

NB. ---- Extract ERNIE model info from KV pairs ----
NB. mi = <block_count; context_len; emb_len; n_heads; n_heads_kv; head_dim;
NB.      rope_freq; vocab_size; rms_eps; n_ff>  (load appends rope tables at
NB.      10/11 and fixes vocab_size at 7 from token_embd dims — ERNIE has
NB.      no vocab_size KV)
ernie_extract_hparams =: 3 : 0
  data =. y
  block_count =. 'ernie4_5.block_count' ernie_kv_uint data
  context_length =. 'ernie4_5.context_length' ernie_kv_uint data
  emb_len =. 'ernie4_5.embedding_length' ernie_kv_uint data
  n_heads =. 'ernie4_5.attention.head_count' ernie_kv_uint data
  n_heads_kv =. 'ernie4_5.attention.head_count_kv' ernie_kv_uint data
  rope_freq =. 'ernie4_5.rope.freq_base' ernie_kv_float data
  vocab_size =. 0
  rms_eps =. 'ernie4_5.attention.layer_norm_rms_epsilon' ernie_kv_float data
  n_ff =. 'ernie4_5.feed_forward_length' ernie_kv_uint data
  key_len =. 'ernie4_5.attention.key_length' ernie_kv_uint data
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
  mi =. ernie_extract_hparams kvs_ctx
  rope_tables =. build_rope_tables ((< mi_context_len mi) , (< mi_head_dim mi) , (< mi_rope_freq mi))
  mi =. mi , rope_tables
  NB. ERNIE has no vocab_size KV — derive from token_embd dims (dims = [emb, vocab]).
  tok_info =. 'token_embd.weight' get_tensor_info ti
  dims =. ti_dims tok_info
  vocab_size =. 1 { dims
  mi =. (<vocab_size) 7} mi
  tokenizer =. build_spm_tokenizer kv_result
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
  ze =. <$0
  tk =. <tokenizer
  mi_b =. <mi
  kc_b =. <''   NB. kv cache is the kv_cache_g global, not stored in the llm
  td =. <tds
  all_tensors =. emb_canonical all_tensors
  at =. <all_tensors
  llm =. p , t , ze , tk , mi_b , kc_b , td , at
  block_data =. llama_pre_build_block_data llm
  llm =. llm , <block_data
)

NB. ---- Tokenize/detokenize (SPM) ----
NB. No BOS (add_bos_token=false): raw infer and chat both start with the text.
ernie_tokenize =: 3 : 0
  spm_tokenize y
)
ernie_detokenize =: 3 : 0
  spm_detokenize y
)

NB. ---- Forward verbs: ERNIE reuses the generic llama arch ----
ernie_run_blocks =: llama_run_blocks
ernie_run_blocks_b =: llama_run_blocks_b
ernie_run_blocks_bd =: llama_run_blocks_bd

NB. ---- Single-token inference ----
NB. Usage: llm ernie_infer (text ; <temp;k;p;min_p>)
NB.        Simple (default params): llm ernie_infer_simple text
ernie_infer =: 4 : 0
  llm =. x
  args =. infer_args y
  text =. > 0 { args
  temp =. > 1 { args
  k =. > 2 { args
  p =. > 3 { args
  min_p =. > 4 { args

  tokens =. ernie_tokenize (<llm) , <text
  mi =. llm_mi llm
  emb_len =. mi_emb_len mi
  scale =. 1
  block_count =. mi_block_count mi
  ctx_len =. mi_context_len mi
  n_heads_kv =. mi_n_heads_kv mi
  head_dim =. mi_head_dim mi
  emb_w =. 'token_embd.weight' get_tensor_cached_d llm
  output_norm_w =. 'output_norm.weight' get_tensor_cached_d llm
  tok_list =. , > tokens
  n_tokens =. # tok_list
  kv_create ((<block_count) , (<ctx_len) , (<n_heads_kv) , (<head_dim))
  if. 1 = n_tokens do.
    tok =. 0 { tok_list
    hidden =. scale * |: (tok {"1 emb_w)
    pre_s =. 6!:2 'result =. hidden ernie_run_blocks (<llm) , <0'
    hidden =. > 0 { result
  else.
    emb_all =. scale * |: (tok_list {"1 emb_w)
    pre_s =. 6!:2 'result_b =. emb_all ernie_run_blocks_b ((<llm) , <0)'
    h_b =. > 0 { result_b
    hidden =. > (n_tokens - 1) { h_b
  end.
  logits =. output_head ((< mi_rms_eps mi) , (<output_norm_w) , (<emb_w) , <hidden)
  report_prefill (pre_s , n_tokens)
  pred_tok =. sample_from ((<temp) , (<k) , (<p) , (<min_p) , <logits)
  decoded =. ernie_detokenize (<llm) , <pred_tok
  tokens ; pred_tok ; decoded ; logits
)

NB. ---- Multi-token generation ----
NB. Usage: llm ernie_generate (text ; max_steps ; <temp;k;p;min_p>)
NB.        Simple (default params): llm ernie_generate_simple (text ; max_steps)
NB. Generation is the UNIFIED gen_loop_core (llm_core.ijs) — per-arch
NB. differences (embedding scale, logit scale, run_blocks/run_blocks_b) are
NB. dispatched by llm_arch. Fresh mode: kv_create + batched prefill; resume
NB. mode (chat sessions): incremental prefill of the new segment. Stop token
NB. not appended.
ernie_generate =: 4 : 0
  llm =. x
  args =. gen_args y
  text =. > 0 { args
  max_steps =. > 1 { args
  temp =. > 2 { args
  k =. > 3 { args
  p =. > 4 { args
  min_p =. > 5 { args

  NB. Frame the prompt with the chat template (single user message) so the
  NB. instruct model emits its stop tokens; stop on the arch stop list.
  messages =. <('user') ; text
  prompt =. ernie_chat_prompt messages
  tokens =. ernie_tokenize (<llm) , <prompt
  stop =. ernie_stop_tokens llm
  L =. # , > tokens
  output =. llm gen_loop_core (tokens ; '' ; max_steps ; temp ; k ; p ; min_p ; <stop)
  gen =. L }. output
  ernie_detokenize (<llm) , <gen
)

NB. ---- Batched generation: B independent prompts in parallel ----
NB. Usage: llm ernie_generate_batch (prompts ; max_steps ; <temp;k;p;min_p>)
NB. Forward = llama_run_blocks_bd alias; SPM tokenizer + ERNIE chat template.
ernie_generate_batch =: 4 : 0
  llm =. x
  args =. gen_args y
  prompts =. > 0 { args
  max_steps =. > 1 { args
  temp =. > 2 { args
  k =. > 3 { args
  p =. > 4 { args
  min_p =. > 5 { args
  B =. # prompts
  prompts_tok =. ''
  prompts_len =. ''
  i =. 0
  while. i < B do.
    text =. > i { prompts
    messages =. <('user') ; text
    prompt =. ernie_chat_prompt messages
    tokens =. ernie_tokenize (<llm) , <prompt
    tok_list =. , > tokens
    prompts_tok =. prompts_tok , <tok_list
    prompts_len =. prompts_len , <(# tok_list)
    i =. i + 1
  end.
  stop =. ernie_stop_tokens llm
  kv_batch_g =: B
  output =. llm gen_loop_batch (prompts_tok ; max_steps ; temp ; k ; p ; min_p ; <stop)
  answers =. ''
  i =. 0
  while. i < B do.
    L =. > i { prompts_len
    gen =. (L) }. (> i { output)
    answers =. answers , <(ernie_detokenize (<llm) , <gen)
    i =. i + 1
  end.
  answers
)

NB. ---- Simple wrappers (default greedy/top-p params) ----
NB. llm ernie_infer_simple text            | llm ernie_generate_simple (text ; n)
ernie_infer_simple =: ernie_infer (] ; (<0 0 0.95 0.0)"_)
ernie_generate_simple =: ernie_generate (0&{ , 1&{ , (<0 0 0.95 0.0)"_)

NB. ---- ERNIE chat template ----
NB. cls_token '<|begin_of_sentence|>'; user 'User: <content>\n'; assistant
NB. 'Assistant: <content><|end_of_sentence|>'; system '<content>\n'; then the
NB. generation prompt 'Assistant: '. No BOS (add_bos_token=false) and no trim.
ernie_chat_prompt =: 3 : 0
  messages =. y
  res =. '<|begin_of_sentence|>'
  for_i. i. # messages do.
    msg =. > i { messages
    role =. > 0 { msg
    content =. > 1 { msg
    if. 'user' -: role do.
      res =. res , 'User: ' , content , LF
    elseif. 'assistant' -: role do.
      res =. res , 'Assistant: ' , content , '<|end_of_sentence|>'
    elseif. 'system' -: role do.
      res =. res , content , LF
    end.
  end.
  res =. res , 'Assistant: '
  res
)

ernie_default_params =: 0 0 0.95 0.0
NB. Stop tokens: EOS (</s> = 2) and the chat sep token <|end_of_sentence|> (100272).
ernie_stop_tokens =: 3 : 0
  tk =. llm_tokenizer y
  (spm_eos tk) , 100272
)
