NB. ================================================================
NB. llmobj.ijs — object wrapper around a loaded LLM (OOP proof)
NB.
NB. Demonstrates J's modular-code OOP: an object locale holds the llm
NB. state; methods are found in the class locale (llmobj), so callers
NB. never juggle the box-of-boxes llm noun.
NB.
NB. Usage (after installing the addon):
NB.   require 'llm/inference/util/llmobj'
NB.   obj =. '/path/model.gguf' conew 'llmobj'
NB.   result =. infer__obj ('hello' ; <0 0 0.95 0.0)
NB.   text   =. generate__obj ('hello' ; 20 ; <0 0 0.95 0.0)
NB.   destroy__obj ''
NB.
NB. NOTE: wraps the active generic entry points (gemma3 by default).
NB. ================================================================
coclass 'llmobj'
require 'llm/inference'

NB. ---- create: load model into the object's state ----
create =: 3 : 0
  state =: load_gguf_to_llm_inference_ y
  'created'
)

NB. ---- methods: delegate to the generic core verbs ----
infer =: 3 : 0
  state infer_inference_ y
)

generate =: 3 : 0
  state generate_inference_ y
)

NB. ---- destroy: release the object locale ----
destroy =: 3 : 0
  codestroy ''
)
