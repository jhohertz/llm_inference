NB. ================================================================
NB. Chat launcher — load a model and enter the interactive console.
NB. Usage (see scripts/chat.sh for the shell helper):
NB.   jconsole chat_launch.ijs MODEL
NB.
NB. Loads 'llm/inference', loads MODEL (any supported arch), and exposes the
NB. console verbs + model noun in the GLOBAL (base) locale so the REPL can
NB. use simple names as-is (a script-file cocurrent does NOT persist to the
NB. REPL — the prompt resumes in base). Prints howto instructions, then
NB. leaves the console interactive (does NOT exit) — type J at the prompt:
NB.   llm chat 'your message'              -> answer (per-arch default params)
NB.   llm chat_p ('msg' ; <temp;k;p;min_p) -> explicit params
NB.   chat_reset ''                        -> clear session + KV cache
NB.   exit ''                              -> leave the console
NB. The session persists across calls until chat_reset '' (or exit).
NB.
NB. Depends on: load 'llm/inference' (addon must be installed via
NB. scripts/install_local.sh --force first).
NB. ================================================================
model =. > 2 { ARGV
load 'llm/inference'
NB. Define the console names in the global locale (name_z_ =: value) so
NB. they resolve at the base-locale REPL prompt. chat/chat_p/chat_reset
NB. live in the inference locale; alias them here. llm is the model noun.
llm_z_ =: load_gguf_to_llm_inference_ model
chat_z_ =: chat_inference_
chat_p_z_ =: chat_p_inference_
chat_reset_z_ =: chat_reset_inference_
sq =. ''''   NB. the single-quote character
echo ''
echo '============================================================'
echo '  Model loaded: ' , model
echo ('  Chat with:    llm chat ' , sq , 'your message' , sq)
echo ('  Params:       llm chat_p (' , sq , 'msg' , sq , ' ; <temp;k;p;min_p)')
echo ('  Reset:        chat_reset ' , sq , sq)
echo ('  Quit:         exit ' , sq , sq)
echo '============================================================'
