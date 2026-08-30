NB. ================================================================
NB. Batched decode test — *_generate_batch must be token-identical to
NB. the single *_generate for every architecture (qwen2, qwen3, llama,
NB. granite, ernie, gemma3, lfm2, qwen35). B=2 (and B=3 for qwen2)
NB. with copies of the same prompt must match the single output.
NB. ================================================================

coclass 'inference'
load './inference.ijs'
load './models/qwen2.ijs'
load './models/qwen3.ijs'
load './models/llama.ijs'
load './models/granite.ijs'
load './models/ernie.ijs'
load './models/gemma3.ijs'
load './models/lfm2.ijs'
load './models/qwen35.ijs'
load './tests/j/pm_fixture.ijs'

NB. Bound the KV cache: qwen3.5-0.8b has ctx=262144 (~51 GB/seq at full
NB. ctx) — the low-memory override keeps the suite tractable.
kv_max_seq_g =: 2048

NB. y = <arch; llm; prompt; max_steps> -> single answer text
run_single =: 3 : 0
  arch =. > 0 { y
  llm =. > 1 { y
  prompt =. > 2 { y
  max_steps =. > 3 { y
  select. arch
  case. 'qwen2'   do. > llm qw2_generate_simple   (prompt ; max_steps)
  case. 'qwen3'   do. > llm qw3_generate_simple   (prompt ; max_steps)
  case. 'llama'   do. > llm llama_generate_simple (prompt ; max_steps)
  case. 'granite' do. > llm granite_generate_simple (prompt ; max_steps)
  case. 'ernie'   do. > llm ernie_generate_simple (prompt ; max_steps)
  case. 'gemma3'  do. > llm gem3_generate_simple  (prompt ; max_steps)
  case. 'lfm2'    do. > llm lf2_generate_simple   (prompt ; max_steps)
  case. 'qwen35'  do. > llm qw35_generate_simple  (prompt ; max_steps)
  end.
)

NB. y = <arch; llm; prompts; max_steps> -> batched answer list
run_batch =: 3 : 0
  arch =. > 0 { y
  llm =. > 1 { y
  prompts =. > 2 { y
  max_steps =. > 3 { y
  select. arch
  case. 'qwen2'   do. llm qw2_generate_batch   (prompts ; max_steps ; <0 0 0.95 0.0)
  case. 'qwen3'   do. llm qw3_generate_batch   (prompts ; max_steps ; <0 0 0.95 0.0)
  case. 'llama'   do. llm llama_generate_batch (prompts ; max_steps ; <0 0 0.95 0.0)
  case. 'granite' do. llm granite_generate_batch (prompts ; max_steps ; <0 0 0.95 0.0)
  case. 'ernie'   do. llm ernie_generate_batch (prompts ; max_steps ; <0 0 0.95 0.0)
  case. 'gemma3'  do. llm gem3_generate_batch  (prompts ; max_steps ; <0 0 0.95 0.0)
  case. 'lfm2'    do. llm lf2_generate_batch   (prompts ; max_steps ; <0 0 0.95 0.0)
  case. 'qwen35'  do. llm qw35_generate_batch  (prompts ; max_steps ; <0 0 0.95 0.0)
  end.
)

test_batched =: 3 : 0
  tc =. 0
  pc =. 0
  fc =. 0
  fl =. ''

  echo ''
  echo '=============================================================='
  echo '  BATCHED DECODE TEST SUITE (batch == single across arches)'
  echo '=============================================================='

  NB. Each case: <arch; catalog id; prompt; max_steps>
  cases =. 0$0
  cases =. cases , <'qwen2'    ; 'qwen2.5-coder-0.5b'    ; 'Write a single-page HTML flappy bird game' ; 24
  cases =. cases , <'qwen3'    ; 'qwen3-0.6b'            ; 'Write a single-page HTML flappy bird game' ; 24
  cases =. cases , <'llama'    ; 'smollm2-135m'          ; 'The capital of France is'                   ; 24
  cases =. cases , <'granite'  ; 'granite-4.0-350m'      ; 'What is the capital of France?'             ; 24
  cases =. cases , <'ernie'    ; 'ernie-4.5-0.3b'        ; 'Hello'                                     ; 24
  cases =. cases , <'gemma3'   ; 'gemma-3-270m-it'       ; 'hello world'                               ; 24
  cases =. cases , <'lfm2'     ; 'lfm2-350m'             ; 'The capital of France is'                   ; 24
  cases =. cases , <'qwen35'   ; 'qwen3.5-0.8b'          ; 'Hello'                                     ; 20

  n_cases =. # cases
  ci =. 0
  while. ci < n_cases do.
    c =. > ci { cases
    arch =. > 0 { c
    mid =. > 1 { c
    prompt =. > 2 { c
    max_steps =. > 3 { c

    echo ''
    echo '--- ' , arch , ' (' , mid , ') ---'
    echo '  (loading model)...'
    llm =. load_gguf_to_llm mid
    echo '  (single generate)...'
    single =. run_single (arch ; llm ; prompt ; max_steps)
    echo '  (batched generate B=2)...'
    p2 =. (<prompt) , (<prompt)
    batched =. run_batch (arch ; llm ; p2 ; max_steps)

    tc =. tc + 1
    if. (single -: > 0 { batched) *. (single -: > 1 { batched) do.
      pc =. pc + 1
      echo 'PASS: B=2 [0] and [1] match single'
    else.
      fc =. fc + 1
      fl =. fl , arch , ' batch B=2 != single', LF
      echo 'FAIL: B=2 [0] and [1] match single'
      echo '  single: ' ; echo single
      echo '  batch0: ' ; echo > 0 { batched
      echo '  batch1: ' ; echo > 1 { batched
    end.

    NB. B=3 cross-check on the first arch (qwen2)
    if. 0 = ci do.
      echo '  (batched generate B=3)...'
      p3 =. (<prompt) , (<prompt) , (<prompt)
      batched3 =. run_batch (arch ; llm ; p3 ; max_steps)
      tc =. tc + 1
      if. (single -: > 0 { batched3) *. (single -: > 1 { batched3) *. (single -: > 2 { batched3) do.
        pc =. pc + 1
        echo 'PASS: B=3 all match single'
      else.
        fc =. fc + 1
        fl =. fl , arch , ' batch B=3 != single', LF
        echo 'FAIL: B=3 all match single'
      end.
    end.

    ci =. ci + 1
  end.

  echo ''
  echo '=============================================================='
  echo '  Total:          ' , ": tc
  echo '  Passed:         ' , ": pc
  echo '  Failed:         ' , ": fc
  if. 0 = fc do. echo '  All tests passed!'
  else.
    echo '  Failed checks:'
    echo fl
  end.
  echo '=============================================================='
)

pm_start 1e8

test_batched ''

pm_report ''
