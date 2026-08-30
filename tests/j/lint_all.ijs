NB. tests/j/lint_all.ijs — jlinter over the checkout's runtime .ijs files.
NB. debug/lint checks each file STANDALONE, so cross-file "undefined name"
NB. findings are expected (multi-file project); the reliable signal is the
NB. LOAD-PROBE (actually loads the file, catching real syntax/load regressions).
NB. Test files are NOT linted: debug/lint loads a file to check it, and the
NB. test files execute their suites on load (minutes each).
NB. Entry-point scripts (llm_cli, chat_launch) execute main on load — their
NB. load=0 is expected and not a gate.
NB.
NB. Use: load this, then  lint_all_z_ ''   -> prints report, sets LINT_EXIT_z_

coclass 'inference'
require 'tmcguire/jlinter'

ENTRYPOINTS =: 'llm_cli.ijs';'chat_launch.ijs'

NB. Runtime files (the addon's installed ITEMS — same set as install_local.sh)
RUNTIME =: 'manifest.ijs';'inference.ijs';'gguf_dump.ijs';'llm_cli.ijs';'chat_launch.ijs';'models/gemma3.ijs';'models/llama.ijs';'models/granite.ijs';'models/qwen2.ijs';'models/qwen3.ijs';'models/qwen35.ijs';'models/ernie.ijs';'models/lfm2.ijs';'tokenizers/tokenizer_llama3.ijs';'tokenizers/tokenizer_gpt2.ijs';'tokenizers/tokenizer_spm.ijs';'kernels/jfloat.ijs';'gguf/gguf.ijs';'gguf/quant.ijs';'gguf/quant_tables.ijs';'util/kv_cache.ijs';'util/llm_core.ijs';'util/sampler.ijs';'util/chat.ijs';'util/models.ijs';'util/llmobj.ijs'

LINT_EXIT =: 0

lint_all =: 3 : 0
  out =. 'J lint report' , LF
  out =. out , '--- runtime files (lint + load-probe gate) ---' , LF
  i =. 0
  while. i < # RUNTIME do.
    fn =. > i { RUNTIME
    'ems lok lmsg' =. lint_file_jlinter_ < fn
    n =. # ems
    line =. fn , ': ' , (": n) , ' finding(s), load=' , (": lok)
    if. (0 = lok) *. -. (< fn) e. ENTRYPOINTS do.
      LINT_EXIT =: 1
      line =. line , '  <-- LOAD FAIL (gate)'
    end.
    out =. out , line , LF
    i =. i + 1
  end.
  out =. out , LF , 'lint exit: ' , (": LINT_EXIT)
  out
)

lint_all_z_ =: 3 : 0
  echo lint_all_inference_ ''
  LINT_EXIT_z_ =: LINT_EXIT_inference_
)
