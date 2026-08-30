NB. ================================================================
NB. util/models.ijs — model catalog, spec resolution, and downloader
NB.
NB. Model specs accepted by load_gguf_to_llm and model_path:
NB.   catalog id       'gemma-3-270m-it'
NB.   HF path          'unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-F16.gguf'
NB.   full URL         'https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-F16.gguf'
NB.   jpath path       '~models/...'  (registered as ~user/models)
NB.   filesystem path  './models/...' or '/abs/path.gguf'
NB.
NB. Downloads go to ~user/models (per-user, NOT the J install dir), cached by
NB. HF layout: ~user/models/<owner>/<repo>/[subdir/]<file>. Uses the builtin
NB. web/gethttp addon (wget/curl) — declared as DEPENDS in manifest.ijs.
NB. ================================================================

coclass 'inference'

require 'web/gethttp'

NB. --- register ~models as ~user/models (per-user, cross-platform) ---
NB. UserFolders_j_ is J's documented extension point for jpath folder tags.
NB. Models stay OUT of the install dir / addons.
reg_models =: 3 : 0
  if. _1 ~: 4!:0 <'UserFolders_j_' do.
    tags =. 0 {"1 UserFolders_j_
    if. (#tags) = tags i. <'models' do.
      UserFolders_j_ =: UserFolders_j_ , 'models';'~user/models'
    end.
  end.
)
reg_models ''
4!:55 <'reg_models'

NB. --- catalog: rows of <id; arch; repo; main_file; extras>
NB. arch '' = not-yet-supported (granite-4.0-h-350m is parser-supported +
NB. downloadable but DROPPED from the roadmap — 1M context + MoE out of scope;
NB. load_gguf_to_llm errors on it)
NB. extras = list of <role; file> for mmproj/mtp/dflash etc. (all empty now)
NB. One row per line; first line itemizes (,:), the rest append (catalog , row).
catalog =: ,: 'gemma-3-270m-it';'gemma3';'unsloth/gemma-3-270m-it-GGUF';'gemma-3-270m-it-F16.gguf';0$0
catalog =: catalog , 'gemma-3-1b-it';'gemma3';'unsloth/gemma-3-1b-it-GGUF';'gemma-3-1b-it-BF16.gguf';0$0
catalog =: catalog , 'qwen2.5-coder-0.5b';'qwen2';'unsloth/Qwen2.5-Coder-0.5B-Instruct-GGUF';'Qwen2.5-Coder-0.5B-Instruct-F16.gguf';0$0
catalog =: catalog , 'qwen2.5-coder-1.5b';'qwen2';'unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF';'Qwen2.5-Coder-1.5B-Instruct-F16.gguf';0$0
catalog =: catalog , 'qwen2.5-coder-3b';'qwen2';'unsloth/Qwen2.5-Coder-3B-Instruct-GGUF';'Qwen2.5-Coder-3B-Instruct-F16.gguf';0$0
catalog =: catalog , 'qwen3-0.6b';'qwen3';'unsloth/Qwen3-0.6B-GGUF';'Qwen3-0.6B-BF16.gguf';0$0
catalog =: catalog , 'qwen3-1.7b';'qwen3';'unsloth/Qwen3-1.7B-GGUF';'Qwen3-1.7B-BF16.gguf';0$0
catalog =: catalog , 'qwen3.5-0.8b';'qwen35';'unsloth/Qwen3.5-0.8B-MTP-GGUF';'Qwen3.5-0.8B-BF16.gguf';0$0
catalog =: catalog , 'qwen3.5-2b';'qwen35';'unsloth/Qwen3.5-2B-MTP-GGUF';'Qwen3.5-2B-BF16.gguf';0$0
catalog =: catalog , 'smollm2-135m';'llama';'unsloth/SmolLM2-135M-Instruct-GGUF';'SmolLM2-135M-Instruct-F16.gguf';0$0
catalog =: catalog , 'smollm2-360m';'llama';'unsloth/SmolLM2-360M-Instruct-GGUF';'SmolLM2-360M-Instruct-F16.gguf';0$0
catalog =: catalog , 'smollm2-1.7b';'llama';'unsloth/SmolLM2-1.7B-Instruct-GGUF';'SmolLM2-1.7B-Instruct-F16.gguf';0$0
catalog =: catalog , 'llama-3.2-1b';'llama';'unsloth/Llama-3.2-1B-Instruct-GGUF';'Llama-3.2-1B-Instruct-BF16.gguf';0$0
catalog =: catalog , 'granite-4.0-350m';'granite';'unsloth/granite-4.0-350m-GGUF';'granite-4.0-350m-BF16.gguf';0$0
catalog =: catalog , 'granite-4.0-h-350m';'';'unsloth/granite-4.0-h-350m-GGUF';'granite-4.0-h-350m-BF16.gguf';0$0
catalog =: catalog , 'ernie-4.5-0.3b';'ernie4_5';'unsloth/ERNIE-4.5-0.3B-PT-GGUF';'ERNIE-4.5-0.3B-PT-F16.gguf';0$0
catalog =: catalog , 'lfm2-350m';'lfm2';'unsloth/LFM2-350M-GGUF';'LFM2-350M-F16.gguf';0$0
catalog =: catalog , 'lfm2-700m';'lfm2';'unsloth/LFM2-700M-GGUF';'LFM2-700M-F16.gguf';0$0
catalog =: catalog , 'lfm2.5-230m';'lfm2';'unsloth/LFM2.5-230M-GGUF';'LFM2.5-230M-BF16.gguf';0$0
catalog =: catalog , 'lfm2.5-1.2b-instruct';'lfm2';'unsloth/LFM2.5-1.2B-Instruct-GGUF';'LFM2.5-1.2B-Instruct-BF16.gguf';0$0
catalog =: catalog , 'lfm2.5-1.2b-thinking';'lfm2';'unsloth/LFM2.5-1.2B-Thinking-GGUF';'LFM2.5-1.2B-Thinking-BF16.gguf';0$0
cat_ids =: 0 {"1 catalog

NB. --- catalog accessors (y = row index) ---
cat_arch   =: 3 : '> 1 { y { catalog'
cat_repo   =: 3 : '> 2 { y { catalog'
cat_main   =: 3 : '> 3 { y { catalog'
cat_extras =: 3 : '> 4 { y { catalog'
cat_idx    =: 3 : 'cat_ids i. <y'

NB. --- split 'owner/repo' -> <owner; repo> ---
split_owner =: 3 : 0
  i =. y i. '/'
  if. i = #y do. '' ; '' return. end.
  (i {. y) ; (i + 1) }. y
)

NB. --- parse a spec into <owner; repo; rest> (URL or HF path) ---
parse_hf =: 3 : 0
  s =. y
  if. 1 e. '://' E. s do.
    i =. ('//' E. s) i. 1
    s =. (i + 2) }. s
    i =. s i. '/'
    if. i = #s do. '' ; '' ; '' return. end.
    s =. (i + 1) }. s
  end.
  i1 =. s i. '/'
  if. i1 = #s do. '' ; '' ; '' return. end.
  owner =. i1 {. s
  r =. (i1 + 1) }. s
  i2 =. r i. '/'
  if. i2 = #r do.
    repo =. r
    rest =. ''
  else.
    repo =. i2 {. r
    rest =. (i2 + 1) }. r
  end.
  if. 1 e. 'resolve/main/' E. rest do.
    rest =. (13 + (('resolve/main/' E. rest) i. 1)) }. rest
  end.
  owner ; repo ; rest
)

NB. --- target cache path (no download): id / HF path / URL -> ~models/<owner>/<repo>/<rest> ---
model_target =: 3 : 0
  spec =. y
  if. '~' = {. spec do. jpath spec return. end.
  if. ('/' = {. spec) +. ('./' -: 2{.spec) +. ('../' -: 3{.spec) do. spec return. end.
  if. 1 e. '://' E. spec do.
    'owner repo rest' =. parse_hf spec
    if. 0 = #rest do. spec return. end.
    jpath ('~models/' , owner , '/' , repo , '/' , rest)
    return.
  end.
  if. ('.gguf' -: _5{.spec) *. (0 < +/ (spec = '/')) do.
    'owner repo rest' =. parse_hf spec
    jpath ('~models/' , owner , '/' , repo , '/' , rest)
    return.
  end.
  idx =. cat_idx spec
  if. idx = #catalog do. spec return. end.
  'owner repo' =. split_owner (cat_repo idx)
  file =. cat_main idx
  jpath ('~models/' , owner , '/' , repo , '/' , file)
)

NB. --- download URL for a spec (id / HF path / URL) ---
dl_url =: 3 : 0
  spec =. y
  if. 1 e. '://' E. spec do. spec return. end.
  idx =. cat_idx spec
  if. idx < #catalog do.
    'owner repo' =. split_owner (cat_repo idx)
    file =. cat_main idx
    'https://huggingface.co/' , owner , '/' , repo , '/resolve/main/' , file
  else.
    'owner repo rest' =. parse_hf spec
    'https://huggingface.co/' , owner , '/' , repo , '/resolve/main/' , rest
  end.
)

NB. --- download url to target via web/gethttp (wget -O / curl -o) ---
dl =: 3 : 0      NB. y = <url; target>
  'url target' =. y
  if. '/' e. target do.
    dir =. ({.~ i:&'/') target
    if. -. fexist dir do. mkdir_j_ dir end.
  end.
  if. IFWGET_wgethttp_ do.
    opts =. '-q -O ', target
  else.
    opts =. '-s -o ', target
  end.
  opts gethttp_wgethttp_ url
  target
)

NB. --- resolve a spec to a local file (download if needed) ---
NB. Feedback: direct paths echo the path as passed; catalog/HF/URL specs echo
NB. the cached target, or announce the download before fetching.
model_path =: 3 : 0
  spec =. y
  if. '~' = {. spec do. jpath spec return. end.
  if. ('/' = {. spec) +. ('./' -: 2{.spec) +. ('../' -: 3{.spec) do.
    echo 'loading: ' , spec
    spec return.
  end.
  target =. model_target spec
  if. -. fexist target do.
    url =. dl_url spec
    echo 'downloading model: ' , spec
    echo '  ' , url
    dl (url ; target)
    ('download failed: ' , url) assert fexist target
    echo '  -> ' , target
  else.
    echo 'loading (cached): ' , target
  end.
  target
)

model_download =: model_path

NB. --- list the catalog ---
model_list =: 3 : 0
  ids =. cat_ids
  archs =. 1 {"1 catalog
  echo 'llm/inference model catalog (id; arch; repo):'
  for_i. i. #catalog do.
    id =. > i { ids
    ar =. > i { archs
    if. 0 = #ar do. ar =. 'planned' end.
    echo ((": >: i) , '. ' , id , '  arch=' , ar)
  end.
  i. 0 0
)

NB. --- files (roles) for a model: list of <role; file> ---
model_roles =: 3 : 0
  idx =. cat_idx y
  if. idx = #catalog do. 0 $ 0 return. end.
  ((<'main'; cat_main idx)) , cat_extras idx
)

NB. --- path for a specific role (main / mmproj / mtp / dflash) ---
model_file =: 3 : 0      NB. y = <id; role>
  'id role' =. y
  idx =. cat_idx id
  if. idx = #catalog do. '' return. end.
  if. role -: 'main' do.
    file =. cat_main idx
  else.
    extras =. cat_extras idx
    if. 0 = #extras do.
      echo 'model_file: no ' , role , ' file for ' , id
      '' return.
    end.
    file =. ''
  end.
  'owner repo' =. split_owner (cat_repo idx)
  target =. jpath ('~models/' , owner , '/' , repo , '/' , file)
  if. -. fexist target do.
    url =. 'https://huggingface.co/' , owner , '/' , repo , '/resolve/main/' , file
    dl (url ; target)
    ('download failed: ' , url) assert fexist target
  end.
  target
)
