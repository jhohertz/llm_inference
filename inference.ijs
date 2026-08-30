NB. ================================================================
NB. LLM Inference — Addon Entry Point (model-agnostic)
NB. Usage (after ./scripts/install_local.sh):
NB.   load 'llm/inference'
NB.   llm =. load_gguf_to_llm_inference_ '/path/to/model.gguf'
NB.   result =. llm infer_simple_inference_ 'hello world'        NB. greedy defaults
NB.   generated =. llm generate_simple_inference_ ('hello world' ; 20)
NB.   sampled =. llm infer_inference_ ('hello world' ; <1.0 ; 64 ; 0.95 ; 0.001)
NB. ================================================================

coclass 'inference'

NB. Output control (9!:37): raise the session's max line length from the
NB. 256 default so long generation answers aren't chopped with "...".
NB. y = <eol; max_line_len; max_lines_before; max_lines_after>.
9!:37 (0 1000000 0 222)

require 'llm/inference/gguf/gguf'
require 'llm/inference/gguf/quant_tables'
require 'llm/inference/gguf/quant'
require 'llm/inference/kernels/jfloat'
require 'llm/inference/tokenizers/tokenizer_llama3'
require 'llm/inference/util/kv_cache'
require 'llm/inference/util/sampler'
require 'llm/inference/util/models'

NB. Default arch: gemma3. The other arch modules (models/llama.ijs,
NB. models/qwen2.ijs) load separately; the generic entry below selects the
NB. arch-specific loader by reading the model's GGUF header, so load order
NB. no longer matters.
require 'llm/inference/models/gemma3'

NB. Chat-template inference (Phase 1.1): renders per-arch chat templates,
NB. tokenizes with special-token encoding, generates until per-arch stop tokens.
require 'llm/inference/util/chat'

NB. ---- Architecture detection from GGUF header ----
NB. Reads the 'general.architecture' KV string. Known: gemma3 | llama | qwen2 | qwen3.
NB. (jforc Ch 30: this check happens once, at load time — playsound-style.)
NB. y = <path; raw> — the mapped raw (mapped once by load_gguf_to_llm).
detect_arch =: 3 : 0
  raw =. > 1 { y
  kv_result =. parse_kv_pairs_raw raw
  'general.architecture' kv_string kv_result
)

NB. ---- Generic model loader: detect arch -> arch-specific loader ----
NB. Loads the GGUF, finds its architecture, hands off to that arch's loader,
NB. then maps the generic infer/generate entry points onto the arch's verbs —
NB. so the inference path has no per-call dispatch. The llm noun carries its
NB. arch at index 9 (llm_arch) for later OOP/interface work.
NB. The file is memory-mapped ONCE here (mmap_gguf); detect_arch and the
NB. arch loader parse from the mapped raw. The mapping is unmap'd after
NB. load (one model per load) — the llm noun never holds the mapped-raw ref,
NB. so unmap frees it.
load_gguf_to_llm =: 3 : 0
  NB. Accept a model spec (catalog id / HF path / URL / ~models path) or a
  NB. plain filesystem path; model_path downloads to ~user/models if needed.
  y =. model_path y
  raw =. mmap_gguf y
  arch =. detect_arch (y ; raw)
  select. arch
  case. 'gemma3' do.
    require 'llm/inference/models/gemma3'
    llm =. gem3_load (y ; raw)
    infer =: gem3_infer
    generate =: gem3_generate
  case. 'llama' do.
    require 'llm/inference/models/llama'
    llm =. llama_load (y ; raw)
    infer =: llama_infer
    generate =: llama_generate
  case. 'qwen2' do.
    require 'llm/inference/models/qwen2'
    llm =. qw2_load (y ; raw)
    infer =: qw2_infer
    generate =: qw2_generate
  case. 'qwen3' do.
    require 'llm/inference/models/qwen3'
    llm =. qw3_load (y ; raw)
    infer =: qw3_infer
    generate =: qw3_generate
  case. 'qwen35' do.
    require 'llm/inference/models/qwen35'
    llm =. qw35_load (y ; raw)
    infer =: qw35_infer
    generate =: qw35_generate
  case. 'granite' do.
    require 'llm/inference/models/granite'
    llm =. granite_load (y ; raw)
    infer =: granite_infer
    generate =: granite_generate
  case. 'ernie4_5' do.
    require 'llm/inference/models/ernie'
    llm =. ernie_load (y ; raw)
    infer =: ernie_infer
    generate =: ernie_generate
  case. 'lfm2' do.
    require 'llm/inference/models/lfm2'
    llm =. lf2_load (y ; raw)
    infer =: lf2_infer
    generate =: lf2_generate
  case. do.
    echo 'load_gguf_to_llm: unknown architecture: ', arch
    raw =. ''
    unmap_gguf ''
    return.
  end.
  NB. One model per load: the mapping is load-time only. The llm noun never
  NB. holds a mapped-raw ref, so unmap frees it.
  raw =. ''
  unmap_gguf ''
  llm =. llm , <arch
)

NB. ---- Dyadic infer/generate: canonical boxed form ----
NB. These are playsound-style: load_gguf_to_llm re-maps them to the loaded
NB. arch's verbs. Initial aliases = default arch (gemma3).
infer =: gem3_infer
generate =: gem3_generate

NB. ---- Simple wrappers: default greedy/top-p params ----
NB. llm infer_simple text            |  llm generate_simple (text ; n)
infer_simple =: 4 : 'x infer (y ; <0 0 0.95 0.0)'
generate_simple =: 4 : 'x generate (0{y) , (1{y) , <0 0 0.95 0.0'
