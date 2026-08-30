NB. ================================================================
NB. chat.ijs — chat-template inference (Phase 1.1)
NB. Renders each arch's chat template, tokenizes with special-token encoding,
NB. generates until the arch's stop tokens, returns the model's answer.
NB. Single-turn and rudimentary multi-turn (re-render full history per turn).
NB.
NB. Depends on: llm_core.ijs + the arch modules (gemma3/qwen2/smollm2).
NB. Usage (after load 'llm/inference'):
NB.   llm chat_generate_inference_ (messages ; max_steps ; <temp;k;p;min_p>) -> text
NB.   llm chat_generate_simple_inference_ (messages ; max_steps)
NB.   messages = boxed list of message boxes; each message = <role ; content>
NB.   built with (role) ; content  — e.g. ('user') ; 'The capital of France is'
NB. ================================================================
coclass 'inference'

NB. ---- Dispatch helpers (arch string -> arch verb) ----
chat_prompt =: 4 : 0
  select. x
  case. 'gemma3' do. gem3_chat_prompt y
  case. 'qwen2'  do. qw2_chat_prompt y
  case. 'qwen3'  do. qw3_chat_prompt y
  case. 'qwen35' do. qw35_chat_prompt y
  case. 'llama'  do. llama_chat_prompt y
  case. 'granite' do. granite_chat_prompt y
  case. 'ernie4_5' do. ernie_chat_prompt y
  case. 'lfm2' do. lf2_chat_prompt y
  end.
)
chat_tokenize =: 4 : 0
  select. x
  case. 'gemma3' do. llama3_tokenize y
  case. 'qwen2'  do. gpt2_tokenize y
  case. 'qwen3'  do. gpt2_tokenize y
  case. 'qwen35' do. gpt2_tokenize y
  case. 'llama'  do. llama_tokenize y
  case. 'granite' do. granite_tokenize y
  case. 'ernie4_5' do. ernie_tokenize y
  case. 'lfm2' do. lf2_tokenize y
  end.
)
chat_detokenize =: 4 : 0
  select. x
  case. 'gemma3' do. llama3_detokenize y
  case. 'qwen2'  do. gpt2_detokenize y
  case. 'qwen3'  do. gpt2_detokenize y
  case. 'qwen35' do. gpt2_detokenize y
  case. 'llama'  do. llama_detokenize y
  case. 'granite' do. granite_detokenize y
  case. 'ernie4_5' do. ernie_detokenize y
  case. 'lfm2' do. lf2_detokenize y
  end.
)
chat_gen_loop =: 4 : 0
  llm =. x
  tokens =. > 0 { y
  max_steps =. > 1 { y
  temp =. > 2 { y
  k =. > 3 { y
  p =. > 4 { y
  min_p =. > 5 { y
  stop_list =. > 6 { y
  llm gen_loop_core (tokens ; '' ; max_steps ; temp ; k ; p ; min_p ; <stop_list)
)
chat_default_params =: 3 : 0
  select. y
  case. 'gemma3' do. gem3_default_params
  case. 'qwen2'  do. qw2_default_params
  case. 'qwen3'  do. qw3_default_params
  case. 'qwen35' do. qw35_default_params
  case. 'llama'  do. llama_default_params
  case. 'granite' do. granite_default_params
  case. 'ernie4_5' do. ernie_default_params
  case. 'lfm2' do. lf2_default_params
  end.
)
chat_stop_tokens =: 3 : 0
  llm =. y
  arch =. llm_arch llm
  select. arch
  case. 'gemma3' do. gem3_stop_tokens llm
  case. 'qwen2'  do. qw2_stop_tokens llm
  case. 'qwen3'  do. qw3_stop_tokens llm
  case. 'qwen35' do. qw35_stop_tokens llm
  case. 'llama'  do. llama_stop_tokens llm
  case. 'granite' do. granite_stop_tokens llm
  case. 'ernie4_5' do. ernie_stop_tokens llm
  case. 'lfm2' do. lf2_stop_tokens llm
  end.
)

NB. ---- Chat arg parsing ----
NB. y = <messages ; max_steps ; <params>  (<params> = <temp;k;p;min_p>, possibly double-boxed)
NB. Returns <messages; max_steps; temp; k; p; min_p>
chat_args =: 3 : 0
  messages =. > 0 { y
  max_steps =. > 1 { y
  params =. > 2 { y
  if. 1 = # params do.
    flat =. > > params
  else.
    flat =. > params
  end.
  temp =. 0 { flat
  k =. 1 { flat
  p =. 2 { flat
  min_p =. 3 { flat
  (<messages) , (<max_steps) , (<temp) , (<k) , (<p) , (<min_p)
)

NB. ---- Chat generation ----
NB. llm chat_generate (messages ; max_steps ; <temp;k;p;min_p>) -> answer text.
NB. Renders the full message history (multi-turn), adds the generation prompt,
NB. and generates until the arch's stop tokens. The stop token is not included.
chat_generate =: 4 : 0
  llm =. x
  args =. chat_args y
  messages =. > 0 { args
  max_steps =. > 1 { args
  temp =. > 2 { args
  k =. > 3 { args
  p =. > 4 { args
  min_p =. > 5 { args

  arch =. llm_arch llm
  prompt =. arch chat_prompt messages
  tokens =. arch chat_tokenize (<llm) , <prompt
  stop =. chat_stop_tokens llm
  L =. # , > tokens

  output =. llm chat_gen_loop (tokens ; max_steps ; temp ; k ; p ; min_p ; <stop)
  gen =. L }. output   NB. drop the prompt tokens — answer only
  arch chat_detokenize (<llm) , <gen
)

NB. ---- Simple wrapper (per-arch default params) ----
NB. llm chat_generate_simple (messages ; max_steps)
chat_generate_simple =: 4 : 0
  llm =. x
  messages =. > 0 { y
  max_steps =. > 1 { y
  arch =. llm_arch llm
  params =. chat_default_params arch
  llm chat_generate (messages ; max_steps ; <params)
)

NB. ---- Chat prompt helper: build one message = <role ; content> ----
NB. role chat_msg content
chat_msg =: 4 : 0
  (<x) , <y
)

NB. ================================================================
NB. Chat session (persistent multi-turn — option B)
NB. The crude console chat: `llm chat 'next message'` continues the
NB. conversation, reusing the KV cache + token stream across calls.
NB. ================================================================
NB. chat_session_g = '' (no session) or
NB.   <arch; messages; total_tokens; cur_pos; max_steps; params>
NB.   - messages: boxed list of <role ; content> message boxes
NB.   - total_tokens: boxed list of ALL tokens processed (prompt + generated)
NB.   - cur_pos: # total_tokens == KV write frontier
NB.   - params: <temp;k;p;min_p>
NB. The KV cache lives in kv_cache_g (the global); the session NEVER holds a
NB. cache reference — a second ref would defeat the in-place amend. One active
NB. session per J session; multi-session support can install a session's cache
NB. into kv_cache_g on switch later.
chat_session_g =: ''

NB. ---- path counters (tests: assert the resume path actually runs) ----
chat_resume_count =: 0
chat_fallback_count =: 0

NB. ---- Reset the chat session (clears session + KV cache) ----
chat_reset =: 3 : 0
  chat_session_g =: ''
  kv_reset ''
  ''
)

NB. ---- Fresh full-render chat turn (stateless helper) ----
NB. x = llm; y = <messages; max_steps; temp; k; p; min_p; flat>
NB. Renders the FULL message history, generates fresh, stores the session.
NB. Returns the answer text.
chat_fresh =: 4 : 0
  llm =. x
  messages =. > 0 { y
  max_steps =. > 1 { y
  temp =. > 2 { y
  k =. > 3 { y
  p =. > 4 { y
  min_p =. > 5 { y
  flat =. > 6 { y
  arch =. llm_arch llm
  stop =. chat_stop_tokens llm
  prompt =. arch chat_prompt messages
  tokens =. arch chat_tokenize (<llm) , <prompt
  L =. # , > tokens
  output =. llm gen_loop_core (tokens ; '' ; max_steps ; temp ; k ; p ; min_p ; <stop)
  gen =. L }. output
  answer =. arch chat_detokenize (<llm) , <gen
  NB. keep the assistant answer in the history (the next turn's re-render must
  NB. include it, else the prefix check fails and persistence can't engage)
  messages =. messages , <('assistant') ; answer
  chat_session_g =: (<arch) , (<messages) , (<output) , (<(# , > output)) , (<max_steps) , (<flat)
  answer
)

NB. ---- Core chat turn: persistent session if one exists ----
NB. x = llm; y = <msg; max_steps; <params>  (<params> = <temp;k;p;min_p>, possibly double-boxed)
chat_core =: 4 : 0
  llm =. x
  msg =. > 0 { y
  max_steps =. > 1 { y
  params =. > 2 { y
  if. 1 = # params do.
    flat =. > > params
  else.
    flat =. > params
  end.
  temp =. 0 { flat
  k =. 1 { flat
  p =. 2 { flat
  min_p =. 3 { flat
  arch =. llm_arch llm

  if. 0 = # chat_session_g do.
    NB. no session — start fresh with a single user message
    messages =. <('user') ; msg
    llm chat_fresh (messages ; max_steps ; temp ; k ; p ; min_p ; <flat)
  else.
    s =. chat_session_g
    s_arch =. > 0 { s
    if. -. s_arch -: arch do.
      NB. different model loaded — start over
      messages =. <('user') ; msg
      llm chat_fresh (messages ; max_steps ; temp ; k ; p ; min_p ; <flat)
    else.
      prev_messages =. > 1 { s
      prev_toks =. > 2 { s
      prev_len =. > 3 { s
      messages =. prev_messages , <('user') ; msg
      prompt =. arch chat_prompt messages
      tokens =. arch chat_tokenize (<llm) , <prompt
      tok_list =. , > tokens
      prev_flat =. , > prev_toks
      stop =. chat_stop_tokens llm
      if. (prev_len {. tok_list) -: prev_flat do.
        NB. re-render prefix matches the stored token stream -> resume from cache
        chat_resume_count =: chat_resume_count + 1
        seg =. prev_len }. tok_list
        L_seg =. # seg
        output =. llm gen_loop_core ((<"0 seg) ; prev_len ; max_steps ; temp ; k ; p ; min_p ; <stop)
        gen =. L_seg }. output
        answer =. arch chat_detokenize (<llm) , <gen
        total =. prev_toks , output
        messages =. messages , <('assistant') ; answer
        chat_session_g =: (<arch) , (<messages) , (<total) , (<(# , > total)) , (<max_steps) , (<flat)
        answer
      else.
        NB. tokenizer round-trip drift — fall back to a full fresh re-render
        NB. (correct, just slower); the session resets to the new stream.
        chat_fallback_count =: chat_fallback_count + 1
        llm chat_fresh (messages ; max_steps ; temp ; k ; p ; min_p ; <flat)
      end.
    end.
  end.
)

NB. ---- Crude console chat: llm chat 'next message' -> answer (default params) ----
NB. Persistent: the session + KV cache carry across calls until chat_reset ''.
chat =: 4 : 0
  llm =. x
  msg =. y
  arch =. llm_arch llm
  params =. chat_default_params arch
  llm chat_core (msg ; 100000 ; <params)
)

NB. ---- chat with explicit params: llm chat_p ('msg' ; <temp;k;p;min_p>) ----
chat_p =: 4 : 0
  llm =. x
  msg =. > 0 { y
  params =. > 1 { y
  llm chat_core (msg ; 100000 ; <params)
)
