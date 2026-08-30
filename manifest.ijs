NB. ================================================================
NB. manifest for llm/inference addon
NB. ================================================================

CAPTION=: 'GGUF-based language model inference in J'

DESCRIPTION=: 0 : 0
Generic GGUF-based language model inference written in J (J9.7), multi-model.
A model-agnostic GGUF parser loads weights; each architecture has its own
module (gemma3, llama, granite, ernie, qwen2, qwen3, qwen35, lfm2)
implementing the forward pass.
Inference logits are verified exact vs llama-cpp-python.

All code lives in the 'inference' locale. Call verbs with the _inference_
suffix from any locale, e.g. load_gguf_to_llm_inference_ 'model.gguf'.

Models are not bundled; point load_gguf_to_llm at any supported GGUF file.
)

VERSION=: '0.1.0'

RELEASE=: 'j9.7'

FOLDER=: 'llm/inference'

PLATFORMS=: 'linux'

DEPENDS=: 0 : 0
web/gethttp
)

FILES=: 0 : 0
README.md
inference.ijs
gguf_dump.ijs
llm_cli.ijs
chat_launch.ijs
models/gemma3.ijs
models/llama.ijs
models/granite.ijs
models/qwen2.ijs
models/qwen3.ijs
models/qwen35.ijs
models/ernie.ijs
models/lfm2.ijs
tokenizers/tokenizer_llama3.ijs
tokenizers/tokenizer_gpt2.ijs
tokenizers/tokenizer_spm.ijs
kernels/jfloat.ijs
gguf/gguf.ijs
gguf/quant.ijs
gguf/quant_tables.ijs
util/kv_cache.ijs
util/llm_core.ijs
util/sampler.ijs
util/chat.ijs
util/models.ijs
util/llmobj.ijs
)
