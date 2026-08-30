NB. ================================================================
NB. LLM CLI — one-shot generation from the command line
NB. Usage (see scripts/llm.sh for the shell helper):
NB.   jconsole llm_cli.ijs MODEL "PROMPT"              NB. until stop tokens
NB.   jconsole llm_cli.ijs MODEL "PROMPT" MAX_STEPS   NB. optional cap
NB.   jconsole llm_cli.ijs MODEL "PROMPT" MAX_STEPS 1 NB. chat mode (single turn)
NB.   jconsole llm_cli.ijs MODEL @PROMPTFILE          NB. read prompt from a file
NB.
NB. MODEL is a GGUF file path, PROMPT the input text. If the PROMPT arg starts
NB. with '@', the rest is a file path and its contents are read as the prompt
NB. (curl-style; handy for long/multi-line prompts). The prompt is wrapped in
NB. the model's chat template (single user message) so instruct models emit
NB. their stop tokens; generation stops at the arch's stop list (llama-cli -st
NB. behavior). Without MAX_STEPS it runs until a stop token (capped at a
NB. generous safety bound). Sampling params: temp 1.0, top_k 64, top_p 0.95,
NB. min_p 0.001.
NB.
NB. Chat mode (4th arg = 1): same single-turn chat, but uses chat_generate_simple
NB. with the arch's default sampler params (gemma 1.0 64 0.95 0.001; qwen2 /
NB. smollm2 greedy 0 0 0.95 0.0). Now effectively redundant for single turn.
NB.
NB. Depends on: load 'llm/inference' (addon must be installed via
NB. scripts/install_local.sh --force first).
NB. ================================================================

require 'llm/inference'

main =: 3 : 0
  NB. ARGV = bin/jconsole ; script ; model ; prompt ; [max_steps] ; [chat]
  model =. > 2 { ARGV
  prompt =. > 3 { ARGV
  NB. @file flag: read the prompt from a file (curl-style) instead of inline.
  if. '@' = 0 { prompt do.
    file =. }. prompt
    content =. fread file
    if. 2 ~: 3!:0 content do.            NB. fread returns 0 (int) on failure
      echo 'llm_cli: cannot read prompt file: ' , file
      exit 1
    end.
    prompt =. content
  end.
  max_steps =. 100000
  if. 4 < # ARGV do. max_steps =. ". > 4 { ARGV end.
  chat_mode =. 0
  if. 5 < # ARGV do. chat_mode =. ". > 5 { ARGV end.

  NB. load_gguf_to_llm detects the arch and requires that arch's module itself.
  NB. (The arch modules redefine shared bd_* accessors, so only one arch module
  NB. must be loaded per session — the CLI loads just the one for the model.)
  llm =. load_gguf_to_llm_inference_ model
  if. chat_mode do.
    msg =. ('user') ; prompt
    messages =. <msg
    out =. llm chat_generate_simple_inference_ (messages ; max_steps)
  else.
    out =. llm generate_inference_ (prompt ; max_steps ; <1.0 64 0.95 0.001)
  end.
  echo > out
  exit 0
)

main ''
