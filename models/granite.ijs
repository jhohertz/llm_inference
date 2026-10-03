NB. ================================================================
NB. Granite arch — standard-decoder transformer (GQA, SwiGLU,
NB. interleaved RoPE, separate QKV/O weights, tied embeddings) with the
NB. Granite 4.0 scaling scheme:
NB.   embedding_scale:  input embeddings * 12 (after token_embd lookup)
NB.   residual_scale:   per layer, attn_out * 0.263 + input, then
NB.                     ffn_out * 0.263 + (attn_scaled + input)
NB.   attention_scale:  Q*K^T scores * 0.015625 (NOT 1/sqrt(head_dim))
NB.   logit_scale:      lm_head logits / 4
NB. All dims are read from the GGUF (granite.* KVs); tokenizer is gpt2
NB. byte-level BPE with the dbrx pre-tokenizer (SAME regex as llama3).
NB. Chat template: granite 4.0 (always-emitted default system message,
NB. <|start_of_role|>/<|end_of_role|>/<|end_of_text|> markers).
NB. Depends on: llm_core.ijs, kernels.ijs, gguf.ijs, kv_cache.ijs, sampler.ijs,
NB.           tokenizer_gpt2.ijs (which pulls in tokenizer_llama3.ijs)
NB. ================================================================
coclass 'inference'
require 'llm/inference/util/llm_core'
require 'llm/inference/models/llama'
require 'llm/inference/tokenizers/tokenizer_gpt2'
require 'llm/inference/util/minja'
require 'llm/inference/util/chat_template'

NB. ---- KV helpers ----
granite_kv_uint =: 4 : 0
  key =. x
  data =. y
  key kv_uint data
)

granite_kv_float =: 4 : 0
  key =. x
  data =. y
  key kv_float data
)

NB. ---- Extract granite model info from KV pairs ----
NB. mi = <block_count; context_len; emb_len; n_heads; n_heads_kv; head_dim;
NB.      rope_freq; vocab_size; rms_eps; n_ff>  (load appends rope tables at
NB.      10/11 and the scale fields at 12..15)
granite_extract_hparams =: 3 : 0
  data =. y
  block_count =. 'granite.block_count' granite_kv_uint data
  context_length =. 'granite.context_length' granite_kv_uint data
  emb_len =. 'granite.embedding_length' granite_kv_uint data
  n_heads =. 'granite.attention.head_count' granite_kv_uint data
  n_heads_kv_arr =. 'granite.attention.head_count_kv' kv_array data
  NB. granite-4.0 stores head_count_kv as an ARRAY (vt=9, all-equal) -> take
  NB. first; granite-4.2+ stores it as a scalar UINT (vt=4) -> kv_array is empty,
  NB. so fall back to kv_uint. (kv_array only decodes vt=9 arrays.)
  if. 0 < # n_heads_kv_arr do.
    n_heads_kv =. {. n_heads_kv_arr
  else.
    n_heads_kv =. 'granite.attention.head_count_kv' kv_uint data
  end.
  rope_freq =. 'granite.rope.freq_base' granite_kv_float data
  vocab_size =. 'granite.vocab_size' granite_kv_uint data
  rms_eps =. 'granite.attention.layer_norm_rms_epsilon' granite_kv_float data
  n_ff =. 'granite.feed_forward_length' granite_kv_uint data
  head_dim =. 'granite.rope.dimension_count' granite_kv_uint data
  (<"0) block_count , context_length , emb_len , n_heads , n_heads_kv , head_dim , rope_freq , vocab_size , rms_eps , n_ff
)

NB. ---- granite-specific mi accessors ----
NB. Shared mi_* cover indices 0..11; mi_attn_scale/mi_resid_scale (12/13) are the
NB. COMMON attn/resid scales the shared forward verbs read; embed/logit stay
NB. granite-specific (indices 14/15).
granite_mi_embed_scale =: 3 : 0
  d =. y
  > get__d (<'embed_scale')
)
granite_mi_logit_scale =: 3 : 0
  d =. y
  > get__d (<'logit_scale')
)

NB. ---- block_data accessors (granite: separate weights, like llama) ----
NB. block_data = <attn_norm; attn_q; attn_k; attn_v; attn_o; ffn_norm; ffn_gate; ffn_up; ffn_down; n_heads; head_dim; rope_freq; n_heads_kv; n_ff; block_idx>
granite_bd_attn_norm =: >@(0&{)
granite_bd_attn_q    =: >@(1&{)
granite_bd_attn_k    =: >@(2&{)
granite_bd_attn_v    =: >@(3&{)
granite_bd_attn_o    =: >@(4&{)
granite_bd_ff_norm   =: >@(5&{)
granite_bd_ff_gate   =: >@(6&{)
granite_bd_ff_up     =: >@(7&{)
granite_bd_ff_down   =: >@(8&{)
granite_bd_n_heads   =: >@(9&{)
granite_bd_head_dim  =: >@(10&{)

granite_bd_n_heads_kv=: >@(11&{)
NB. ---- Pre-build block data for granite arch ----
NB. ---- Build one block's data (rank-1; x=llm, y=block index) ----
granite_build_block =: 4 : 0
  llm =. x
  b =. y
  mi =. llm_mi llm
  n_heads =. mi_n_heads mi
  n_heads_kv =. mi_n_heads_kv mi
  p =. 'blk.' , (": b) , '.'
  attn_norm =. (p , 'attn_norm.weight') get_tensor_cached_d llm
  attn_q =. (p , 'attn_q.weight') get_tensor_cached_d llm
  attn_k =. (p , 'attn_k.weight') get_tensor_cached_d llm
  attn_v =. (p , 'attn_v.weight') get_tensor_cached_d llm
  attn_o_w =. (p , 'attn_output.weight') get_tensor_cached_d llm
  ffn_norm =. (p , 'ffn_norm.weight') get_tensor_cached_d llm
  ff_gate =. (p , 'ffn_gate.weight') get_tensor_cached_d llm
  ff_up =. (p , 'ffn_up.weight') get_tensor_cached_d llm
  ff_down =. (p , 'ffn_down.weight') get_tensor_cached_d llm
  head_dim =. (0 { $ attn_q) % n_heads
  (<attn_norm),(<attn_q),(<attn_k),(<attn_v),(<attn_o_w),(<ffn_norm),(<ff_gate),(<ff_up),(<ff_down),(<n_heads),(<head_dim),(<n_heads_kv)
)

granite_pre_build_block_data =: 3 : 0
  llm =. y
  block_count =. mi_block_count (llm_mi llm)
  (<llm) granite_build_block each i. block_count
)

NB. ---- Expand KV axis (n_kv) -> n_heads (each KV head repeated n_groups times) ----

NB. ---- Single-token attention (granite: GQA, interleaved RoPE) ----
NB. x = hidden; y = <block_data; pos; mi; layer>
granite_attention =: llama_attention

NB. ---- Batched attention (prompt prefill, GQA, causal) ----
NB. x = hidden (L, emb); y = <block_data; mi; layer; start_pos>
granite_attention_b =: llama_attention_b

NB. ---- Single block forward (granite: residual scale on attn/ffn outputs) ----
NB. x = hidden; y = <block_data; pos; mi; layer>
granite_block_forward =: llama_block_forward

NB. ---- Batched block forward ----
NB. x = hidden (L, emb); y = <block_data; mi; layer; start_pos; rope>
granite_block_forward_b =: llama_block_forward_b

NB. ---- Run all blocks (single token; cache lives in kv_cache_g global) ----
NB. x = hidden; y = <llm; pos>
granite_run_blocks =: llama_run_blocks

NB. ---- Batched run all blocks (prompt prefill) ----
NB. x = hidden (L, emb); y = <llm; start_pos>  (positions start_pos..start_pos+L-1)
granite_run_blocks_b =: llama_run_blocks_b

NB. ---- Batched-prefill run all blocks (granite = llama + mi scaling) ----
NB. Granite reuses the llama bp verbs: the residual/attention scaling comes from
NB. the mi dict (mi_attn_scale=0.015625, mi_resid_scale per layer), so the llama
NB. batched-prefill forward math applies unchanged.  Embed/logit scales are
NB. handled in gen_loop_batch (granite_mi_embed_scale / granite_mi_logit_scale).
granite_run_blocks_bp =: llama_run_blocks_bp

NB. ---- Batched-DECODE attention (B sequences, ONE token each at pos[b]) ----
NB. Granite = llama + attention_scale (scores*0.015625, NOT 1/sqrt(hd)).
NB. x = hidden (B, emb); y = <block_data; pos; mi; layer>
granite_attention_bd =: llama_attention_bd

NB. ---- Batched-decode block forward (granite: residual scale on attn/ffn outputs) ----
granite_block_forward_bd =: llama_block_forward_bd

NB. ---- Run all blocks for B sequences (one token each at pos[b]) ----
granite_run_blocks_bd =: llama_run_blocks_bd

granite_load =: 3 : 0
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
  mi =. granite_extract_hparams kvs_ctx
  mi =. build_mi_dict mi
  rope_tables =. build_rope_tables ((< mi_context_len mi) , (< mi_head_dim mi) , (< mi_rope_freq mi))
  NB. granite-specific config fields, read from the kv_data dict by name.
  NB. attn_scale/resid_scale are the COMMON mi accessors (mi_attn_scale/mi_resid_scale)
  NB. the shared forward verbs read; embed/logit stay granite-specific.
  NB. ONE multi-put: the jdict single-put (1-key) path is broken in this J9.8
  NB. build (domain/length error); multi-put (2+ keys) and get/has work.
  attn_scale =. 'granite.attention.scale' dict_get kv_data
  resid_scale =. 'granite.residual_scale' dict_get kv_data
  embed_scale =. 'granite.embedding_scale' dict_get kv_data
  logit_scale =. 'granite.logit_scale' dict_get kv_data
  ((> 0 { rope_tables) ; (> 1 { rope_tables) ; attn_scale ; resid_scale ; embed_scale ; logit_scale) put__mi 'cos_tab' ; 'sin_tab' ; 'attn_scale' ; 'resid_scale' ; 'embed_scale' ; 'logit_scale'
  NB. Real chat template from the GGUF ('' if absent → bespoke fallback).
  ct_tmpl_g =: 'tokenizer.chat_template' kv_string (0 1 { kv_result)
  tokenizer =. build_gpt2_tokenizer kv_result
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
  kc_b =. <''   NB. kv cache is the kv_cache_g global, not stored in the llm
  td =. <tds
  all_tensors =. emb_canonical all_tensors
  at =. <all_tensors
  llm =. p , t , ze , tk , mi_b , kc_b , td , at
  block_data =. granite_pre_build_block_data llm
  llm =. llm , <block_data
  llm =. llm , <kv_data
)

NB. ---- Generic tokenize/detokenize (granite arch) ----
NB. gpt2 byte-level BPE with dbrx pre (same regex as llama3). Granite has
NB. add_bos_token=false and bos=eos=100257 (<|end_of_text|>), so NO bos is
NB. prepended — raw infer and chat both start with the prompt text.
granite_tokenize =: 3 : 0
  gpt2_tokenize y
)
granite_detokenize =: 3 : 0
  gpt2_detokenize y
)

NB. ---- Single-token inference ----
NB. Usage: llm granite_infer (text ; <temp;k;p;min_p>)
NB.        Simple (default params): llm granite_infer_simple text
granite_infer =: 4 : 0
  llm =. x
  args =. infer_args y
  text =. > 0 { args
  temp =. > 1 { args
  k =. > 2 { args
  p =. > 3 { args
  min_p =. > 4 { args

  tokens =. granite_tokenize (<llm) , <text
  mi =. llm_mi llm
  emb_len =. mi_emb_len mi
  scale =. granite_mi_embed_scale mi
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
    pre_s =. 6!:2 'result =. hidden granite_run_blocks (<llm) , <0'
    hidden =. > 0 { result
  else.
    emb_all =. scale * |: (tok_list {"1 emb_w)
    pre_s =. 6!:2 'result_b =. emb_all granite_run_blocks_b ((<llm) , <0)'
    h_b =. > 0 { result_b
    hidden =. > (n_tokens - 1) { h_b
  end.
  logits =. output_head ((< mi_rms_eps mi) , (<output_norm_w) , (<emb_w) , <hidden)
  logits =. logits % granite_mi_logit_scale mi
  report_prefill (pre_s , n_tokens)
  pred_tok =. sample_from ((<temp) , (<k) , (<p) , (<min_p) , <logits)
  decoded =. granite_detokenize (<llm) , <pred_tok
  tokens ; pred_tok ; decoded ; logits
)

NB. ---- Multi-token generation ----
NB. Usage: llm granite_generate (text ; max_steps ; <temp;k;p;min_p>)
NB.        Simple (default params): llm granite_generate_simple (text ; max_steps)
NB. Generation is the UNIFIED gen_loop_core (llm_core.ijs) — per-arch
NB. differences (embedding scale, logit scale, run_blocks/run_blocks_b) are
NB. dispatched by llm_arch. Fresh mode: kv_create + batched prefill; resume
NB. mode (chat sessions): incremental prefill of the new segment. Stop token
NB. not appended.
granite_generate =: 4 : 0
  llm =. x
  args =. gen_args y
  text =. > 0 { args
  max_steps =. > 1 { args
  temp =. > 2 { args
  k =. > 3 { args
  p =. > 4 { args
  min_p =. > 5 { args

  NB. Frame the prompt with the chat template (single user message) so the
  NB. model emits its stop tokens; stop on the arch stop list.
  messages =. <('user') ; text
  prompt =. granite_chat_prompt messages
  tokens =. granite_tokenize (<llm) , <prompt
  stop =. granite_stop_tokens llm
  L =. # , > tokens
  output =. llm gen_loop_core (tokens ; '' ; max_steps ; temp ; k ; p ; min_p ; <stop)
  gen =. L }. output
  granite_detokenize (<llm) , <gen
)

NB. ---- Batched generation: B independent prompts in parallel ----
NB. Usage: llm granite_generate_batch (prompts ; max_steps ; <temp;k;p;min_p>)
granite_generate_batch =: 4 : 0
  llm =. x
  args =. gen_args y
  prompts =. > 0 { args
  max_steps =. > 1 { args
  temp =. > 2 { args
  k =. > 3 { args
  p =. > 4 { args
  min_p =. > 5 { args
  B =. # prompts
  llm_box =. <llm
  prompts_tok =. ''
  prompts_len =. ''
  i =. 0
  while. i < B do.
    text =. > i { prompts
    messages =. <('user') ; text
    prompt =. granite_chat_prompt messages
    tokens =. granite_tokenize (llm_box , <prompt)
    tok_list =. , > tokens
    prompts_tok =. prompts_tok , <tok_list
    prompts_len =. prompts_len , <(# tok_list)
    i =. i + 1
  end.
  stop =. granite_stop_tokens llm
  kv_batch_g =: B
  output =. llm gen_loop_batch (prompts_tok ; max_steps ; temp ; k ; p ; min_p ; <stop)
  answers =. ''
  i =. 0
  while. i < B do.
    L =. > i { prompts_len
    gen =. (L) }. (> i { output)
    answers =. answers , <(granite_detokenize (llm_box , <gen))
    i =. i + 1
  end.
  answers
)

NB. ---- Simple wrappers (default greedy/top-p params) ----
NB. llm granite_infer_simple text            | llm granite_generate_simple (text ; n)
granite_infer_simple =: granite_infer (] ; (<0 0 0.95 0.0)"_)
granite_generate_simple =: granite_generate (0&{ , 1&{ , (<0 0 0.95 0.0)"_)

NB. ---- Granite 4.0 chat template ----
NB. The template ALWAYS emits a system block — with no system message it uses
NB. the default: 'You are a helpful assistant. Please ensure responses are
NB. professional, accurate, and safe.' Each message renders
NB. '<|start_of_role|>role<|end_of_role|>content<|end_of_text|>\n'; the
NB. assistant generation prompt is '<|start_of_role|>assistant<|end_of_role|>'
NB. (no trailing newline). No BOS (add_bos_token=false) and no trimming.


NB. Stop token: EOS (granite <|end_of_text|> = 100257).
granite_stop_tokens =: 3 : 0
  tk =. llm_tokenizer y
  tokenizer_eos_g tk
)
