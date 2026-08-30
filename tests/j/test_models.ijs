NB. ================================================================
NB. Model Catalog Test Suite — spec resolution, ~models registration,
NB. download-URL construction. No network: model_target/dl_url are pure.
NB. ================================================================
coclass 'inference'
load './inference.ijs'
load './tests/j/test_harness.ijs'
load './tests/j/pm_fixture.ijs'

test_models =: 3 : 0
  init_counters ''
  section_header 'Model catalog + spec resolution'

  NB. --- ~models registered to the per-user folder (NOT the install dir) ---
  assert_test ((jpath '~models/') -: jpath '~user/models/') ; '~models -> ~user/models (per-user)'
  assert_test (-. (jpath '~models/') -: jpath '~install/') ; '~models is NOT the install dir'

  NB. --- catalog lookup ---
  assert_test (0 = cat_idx 'gemma-3-270m-it') ; 'cat_idx gemma-3-270m-it = 0'
  assert_test (#catalog = cat_idx 'bogus-model') ; 'cat_idx unknown = #catalog'
  assert_test (21 = #catalog) ; 'catalog has 21 models'

  NB. --- target path convergence: id / HF path / URL -> same cache path ---
  exp =. jpath '~user/models/unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-F16.gguf'
  assert_test ((model_target 'gemma-3-270m-it') -: exp) ; 'model_target id -> ~user/models/...'
  assert_test ((model_target 'unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-F16.gguf') -: exp) ; 'model_target HF path -> same'
  assert_test ((model_target 'https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-F16.gguf') -: exp) ; 'model_target URL -> same'
  assert_test ((model_target '~models/unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-F16.gguf') -: exp) ; 'model_target ~models path -> same'

  NB. --- filesystem paths pass through unchanged ---
  assert_test ((model_target './models/x.gguf') -: './models/x.gguf') ; 'model_target ./ path unchanged'
  assert_test ((model_target '/abs/path.gguf') -: '/abs/path.gguf') ; 'model_target /abs path unchanged'

  NB. --- download URL construction ---
  assert_test ((dl_url 'gemma-3-270m-it') -: 'https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-F16.gguf') ; 'dl_url id'
  assert_test ((dl_url 'unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-BF16.gguf') -: 'https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-BF16.gguf') ; 'dl_url HF path'
  assert_test ((dl_url 'https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-BF16.gguf') -: 'https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-BF16.gguf') ; 'dl_url URL passthrough'

  NB. --- not-yet-supported: exactly one, granite-4.0-h-350m (dropped — 1M ctx + MoE) ---
  planned_idx =. I. ('' -: cat_arch)"0 i. #catalog
  assert_test (1 = # planned_idx) ; 'catalog has exactly 1 not-yet-supported model'
  assert_test ('granite-4.0-h-350m' -: > 0 { planned_idx { cat_ids) ; 'not-yet-supported model is granite-4.0-h-350m (dropped)'

  NB. --- roles: main file ---
  r0 =. > 0 { model_roles 'gemma-3-270m-it'
  assert_test ((1 = # model_roles 'gemma-3-270m-it') *. (r0 -: 'main';'gemma-3-270m-it-F16.gguf')) ; 'model_roles main file'

  show_summary 1
)
pm_start 1e8

test_models 0

pm_report ''
