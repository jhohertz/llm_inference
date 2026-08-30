NB. ================================================================
NB. SentencePiece tokenizer (llama.cpp llm_tokenizer_spm)
NB. Used by ERNIE-4.5 / gemma3 (tokenizer.ggml.model='llama' -> SPM).
NB. Algorithm (llama.cpp src/llama-vocab.cpp ~110-239, 3301-3366):
NB.   - split text into UTF-8 codepoints (symbols)
NB.   - seed a max-heap work queue with every adjacent 2-codepoint pair
NB.     whose concatenation is a vocab token; priority = token score
NB.   - repeatedly pop the HIGHEST-scoring valid pair and merge it
NB.   - walk the merged symbol chain; each final symbol -> vocab token,
NB.     else byte fallback
NB. Escape: ' ' -> ▁ (U+2581). Leading space prepend (add_space_prefix)
NB. when the first raw-text fragment or after a special-token fragment
NB. (SPM default add_space_prefix=true; matches llama.cpp line 3336).
NB. Depends on: llm_core.ijs (template_markers/split_specials), gguf.ijs
NB. ================================================================
coclass 'inference'
require 'llm/inference/util/llm_core'
require 'llm/inference/gguf/gguf'

NB. ---- Input accessors (spm-prefixed) ----
spm_input_llm  =: >@(0&{)
spm_input_text =: >@(1&{)
spm_input_data =: >@(1&{)
spm_llm_tokenizer =: >@(3&{)

NB. ---- Build tokenizer from GGUF KV pairs ----
NB. y = kv_result = <kvs_flat; raw_bytes; count; kv_end_offset>
NB. Container (10 items):
NB.   <vocab; bos; eos; tk_len; sym; scores; sym_sorted; tok_sorted; specials; pre>
build_spm_tokenizer =: 3 : 0
  kvs =. > 0 { y
  raw =. > 1 { y
  vocab =. 'tokenizer.ggml.tokens' kv_string_array (<kvs) , (<raw)
  if. 0 = # vocab do.
    return. (<'' , <0 , <'' , <'' , <'' , <'' , <'' , <'')
  end.
  eos_id =. 'tokenizer.ggml.eos_token_id' kv_uint (<kvs) , (<raw)
  if. eos_id <: 0 do. eos_id =. 2 end.
  tk_len =. # vocab
  scores =. 'tokenizer.ggml.scores' kv_array (<kvs) , (<raw)
  if. 0 = # scores do. scores =. tk_len $ 0 end.
  sym =. s: vocab
  ord =. /: sym
  sym_sorted =. ord { sym
  tok_sorted =. ord { i. tk_len
  NB. Special tokens: markers used by the chat template (llama.cpp splits on
  NB. specials via tokenizer.ggml.token_type; we scope to the template markers).
  tmpl =. 'tokenizer.chat_template' kv_string (<kvs) , (<raw)
  specials =. template_markers tmpl
  specials =. specials #~ specials e. vocab
  pre =. 'tokenizer.ggml.pre' kv_string (<kvs) , (<raw)
  (<vocab) , (<eos_id) , (<sym) , (<scores) , (<sym_sorted) , (<tok_sorted) , (<specials) , (<pre)
)

NB. ---- Tokenizer field accessors ----
spm_vocab      =: >@(0&{)

spm_eos        =: >@(1&{)

spm_sym        =: >@(2&{)
spm_scores     =: >@(3&{)
spm_sym_sorted =: >@(4&{)
spm_tok_sorted =: >@(5&{)
spm_specials   =: >@(6&{)
spm_pre        =: >@(7&{)

NB. ---- Split string into UTF-8 codepoints (boxed) ----
spm_chars =: 3 : 0
  s =. y
  bytes =. a. i. s
  n =. # bytes
  res =. ''
  i =. 0
  while. i < n do.
    b =. i { bytes
    if. b < 128 do. ln =. 1
    elseif. b < 224 do. ln =. 2
    elseif. b < 240 do. ln =. 3
    else. ln =. 4
    end.
    res =. res , <(ln {. (i }. s))
    i =. i + ln
  end.
  res
)

NB. ---- Escape whitespace: ' ' -> ▁ (U+2581) ----
spm_escape =: 3 : 0
  s =. y
  res =. ''
  i =. 0
  while. i < # s do.
    c =. i { s
    if. c = ' ' do.
      res =. res , '▁'
    else.
      res =. res , c
    end.
    i =. i + 1
  end.
  res
)

NB. ---- Lookup token id for an exact piece (sorted-symbol binary search) ----
NB. x = <sym_sorted; tok_sorted>, y = piece string. Returns id or _1.
spm_lookup =: 4 : 0
  sym_sorted =. > 0 { x
  tok_sorted =. > 1 { x
  si =. s: <y
  lo =. 0
  hi =. # sym_sorted
  found =. 0
  idx =. 0
  while. lo < hi do.
    mid =. lo + <. (hi - lo) % 2
    msym =. mid { sym_sorted
    if. si > msym do.
      lo =. mid + 1
    elseif. si < msym do.
      hi =. mid
    else.
      found =. 1
      idx =. mid
      lo =. hi
    end.
  end.
  if. found do. idx { tok_sorted else. _1 end.
)

NB. ---- SPM bigram-merge encode ----
NB. x = tokenizer; y = escaped text (spaces already ▁; caller prepends the
NB. leading space). Returns boxed token-id list.
spm_encode =: 4 : 0
  tokenizer =. x
  scores =. spm_scores tokenizer
  lookup_tbl =. (<(spm_sym_sorted tokenizer)) , <(spm_tok_sorted tokenizer)
  if. 0 = # y do. return. '' end.
  cpts =. spm_chars y
  nsyms =. # cpts
  syms =. cpts
  nb =. ; (# each cpts)
  nxt =. (i. nsyms) + 1
  prv =. (i. nsyms) - 1
  q =. ''
  i =. 0
  while. i < nsyms - 1 do.
    l =. i
    r =. i + 1
    text =. (> l { syms) , > r { syms
    tok =. lookup_tbl spm_lookup text
    if. tok >: 0 do.
      q =. q , <l ; r ; (tok { scores) ; (# text)
    end.
    i =. i + 1
  end.
  while. # q do.
    scs =. ; (>@(2&{) each q)
    mi =. scs i. >./ scs
    bi =. > mi { q
    l =. > 0 { bi
    r =. > 1 { bi
    sz =. > 3 { bi
    q =. (mi {. q) , ((mi + 1) }. q)
    if. ((l { nb) > 0) *. ((r { nb) > 0) *. (((l { nb) + (r { nb)) = sz) do.
      syms =. (<((> l { syms) , > r { syms)) l} syms
      nb =. ((l { nb) + (r { nb)) l} nb
      nb =. 0 r} nb
      nr =. r { nxt
      nxt =. nr l} nxt
      if. nr < nsyms do. prv =. l nr} prv end.
      pl =. l { prv
      if. pl >: 0 do.
        text =. (> pl { syms) , > l { syms
        tok =. lookup_tbl spm_lookup text
        if. tok >: 0 do.
          q =. q , <pl ; l ; (tok { scores) ; (# text)
        end.
      end.
      nl =. l { nxt
      if. nl < nsyms do.
        text =. (> l { syms) , > nl { syms
        tok =. lookup_tbl spm_lookup text
        if. tok >: 0 do.
          q =. q , <l ; nl ; (tok { scores) ; (# text)
        end.
      end.
    end.
  end.
  res =. ''
  pos =. 0
  while. pos < nsyms do.
    if. 0 < pos { nb do.
      text =. > pos { syms
      tok =. lookup_tbl spm_lookup text
      if. tok >: 0 do.
        res =. res , <tok
      else.
        bytes =. a. i. text
        hex =. '0123456789ABCDEF'
        j =. 0
        while. j < # bytes do.
          b =. j { bytes
          bs =. '<0x' , (hex {~ <. b % 16) , (hex {~ 16 | b) , '>'
          bt =. lookup_tbl spm_lookup bs
          if. bt <: 0 do. bt =. lookup_tbl spm_lookup (1 $ b) end.
          res =. res , <bt
          j =. j + 1
        end.
      end.
    end.
    pos =. pos { nxt
  end.
  res
)

NB. ---- Tokenize text (SPM path with add_space_prefix) ----
NB. y = <llm; text>; llm has tokenizer at index 3.
NB. Splits special-token markers; each raw fragment gets a leading ▁ if it
NB. is the first fragment or follows a special token (llama.cpp is_prev_special).
spm_tokenize =: 3 : 0
  llm_data =. spm_input_llm y
  text =. spm_input_text y
  tokenizer =. spm_llm_tokenizer llm_data
  specials =. spm_specials tokenizer
  if. 0 < # specials do.
    segs =. specials split_specials text
  else.
    segs =. <text
  end.
  tokens =. ''
  is_prev_special =. 1
  si =. 0
  while. si < # segs do.
    seg =. > si { segs
    if. (specials i. <seg) < # specials do.
      tokens =. tokens , <(spm_sym tokenizer) i. s: <seg
      is_prev_special =. 1
    else.
      if. is_prev_special do.
        seg =. ' ' , seg
      end.
      esc =. spm_escape seg
      enc =. tokenizer spm_encode esc
      tokens =. tokens , enc
      is_prev_special =. 0
    end.
    si =. si + 1
  end.
  tokens
)

NB. ---- ▁ -> space for presentation ----
spm_sp_replace =: 3 : 0
  s =. y
  m =. '▁' E. s
  res =. ''
  i =. 0
  while. i < # s do.
    if. i { m do.
      res =. res , ' '
      i =. i + 3
    else.
      res =. res , (i { s)
      i =. i + 1
    end.
  end.
  res
)

NB. ---- Detokenize: ▁ -> space ----
NB. y = <llm; tokens>
spm_detokenize =: 3 : 0
  llm_data =. spm_input_llm y
  tokens_boxed =. spm_input_data y
  tokens =. > tokens_boxed
  if. 0 = # $ tokens do. tokens =. <tokens end.
  tokenizer =. spm_llm_tokenizer llm_data
  vocab =. spm_vocab tokenizer
  res =. ''
  i =. 0
  while. i < # tokens do.
    tid =. > i { tokens
    ts =. > tid { vocab
    res =. res , ts
    i =. i + 1
  end.
  spm_sp_replace res
)
