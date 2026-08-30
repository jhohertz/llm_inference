NB. ================================================================
NB. GPT-2 byte-level BPE tokenizer (SmolLM2, Qwen2/3, etc.)
NB. Uses PCRE regex for pre-tokenization (J regex library, system/main/regex.ijs)
NB. tokenizer.ggml.model = "gpt2" / "whitespace" / "hybriddna"
NB. ================================================================
coclass 'inference'
require 'regex'
require 'llm/inference/gguf/gguf'
require 'llm/inference/tokenizers/tokenizer_llama3'   NB. llama3_pre_tokenize (llama-bpe pre)

NB. ---- GPT-2 byte-level BPE regex (smollm main pattern, PCRE \p{L}/\p{N}/\s) ----
ap =: 39 { a.   NB. apostrophe char (avoids single-quote literal escaping fragility)
gpt2_regex =: ap , 's|' , ap , 't|' , ap , 're|' , ap , 've|' , ap , 'm|' , ap , 'll|' , ap , 'd| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)'

NB. ---- bytes<->unicode tables (GPT-2 byte encoding) ----
NB. byte b -> codepoint cpt:
NB.   b in 33..126 or 160..255 -> cpt = b
NB.   b in 0..32            -> cpt = 256 + b
NB.   b in 127..159         -> cpt = 256 + 33 + (b-127)
gpt2_build_tables =: 3 : 0
  cpt_tab =. 256 $ 0
  cpt_tab =. (33 + i. 94) (33 + i. 94)} cpt_tab
  cpt_tab =. (160 + i. 96) (160 + i. 96)} cpt_tab
  cpt_tab =. (256 + i. 33) (i. 33)} cpt_tab
  cpt_tab =. (289 + i. 33) (127 + i. 33)} cpt_tab
  byte_tab =. 322 $ _1
  byte_tab =. (33 + i. 94) (33 + i. 94)} byte_tab
  byte_tab =. (160 + i. 96) (160 + i. 96)} byte_tab
  byte_tab =. (i. 33) (256 + i. 33)} byte_tab
  byte_tab =. (127 + i. 33) (289 + i. 33)} byte_tab
  (<cpt_tab) , (<byte_tab)
)

NB. ---- Build tokenizer from GGUF KV pairs ----
NB. y = kv_result = <kvs_flat; raw_bytes; count; kv_end_offset>
NB. Returns 9-element boxed: <vocab; bos_id; eos_id; sym_vocab; sym_merges; cpt_tab; byte_tab; specials; pre>
build_gpt2_tokenizer =: 3 : 0
  kvs =. > 0 { y
  raw =. > 1 { y
  tk_tokens =. 'tokenizer.ggml.tokens' kv_string_array (<kvs) , (<raw)
  if. 0 = # tk_tokens do.
    return. <'' , <0 , <0 , <'' , <'' , <'' , <'' , <'' , <''
  end.
  vocab =. tk_tokens
  bos_id =. 'tokenizer.ggml.bos_token_id' kv_uint (<kvs) , (<raw)
  if. bos_id <: 0 do. bos_id =. 0 end.
  eos_id =. 'tokenizer.ggml.eos_token_id' kv_uint (<kvs) , (<raw)
  if. eos_id <: 0 do. eos_id =. 2 end.
  sym_vocab =. s: vocab
  merges =. 'tokenizer.ggml.merges' kv_string_array (<kvs) , (<raw)
  sym_merges =. s: merges
  tables =. gpt2_build_tables 0
  cpt_tab =. > 0 { tables
  byte_tab =. > 1 { tables
  NB. Special tokens: markers used by the model's chat template (llama.cpp
  NB. splits on specials via tokenizer.ggml.token_type; we scope to the
  NB. template's markers to keep tokenize fast). Store at index 8.
  tmpl =. 'tokenizer.chat_template' kv_string (<kvs) , (<raw)
  specials =. template_markers tmpl
  specials =. specials #~ specials e. vocab   NB. keep only real vocab entries
  vb =. <vocab
  bb =. <bos_id
  eb =. <eos_id
  tb =. <tk_len
  sv =. <sym_vocab
  sm =. <sym_merges
  ct =. <cpt_tab
  bt =. <byte_tab
  sp =. <specials
  NB. Pre-tokenizer name (llama-bpe = llama3 regex pre; else gpt2 regex pre).
  NB. Stored at index 9 so gpt2_tokenize can dispatch on it — the llama arch
  NB. module covers both SmolLM2 (pre 'smollm') and Llama-3.2 (pre 'llama-bpe').
  pre =. 'tokenizer.ggml.pre' kv_string (<kvs) , (<raw)
  pr =. <pre
  vb , bb , eb , sv , sm , ct , bt , sp , pr
)

NB. ---- Tokenizer field accessors ----
tokenizer_vocab_g  =: >@(0&{)
tokenizer_bos_g    =: >@(1&{)
tokenizer_eos_g    =: >@(2&{)
tokenizer_symv     =: >@(3&{)   NB. vocab symbols
tokenizer_symm     =: >@(4&{)   NB. merge symbols (rank = index)
tokenizer_cpt      =: >@(5&{)   NB. byte->codepoint table
tokenizer_byte     =: >@(6&{)   NB. codepoint->byte table
tokenizer_specials_g =: >@(7&{)  NB. special-token marker strings (boxed)
tokenizer_pre_g      =: >@(8&{)  NB. pre-tokenizer name ('llama-bpe' or gpt2-style)

NB. ---- Input accessors ----
input_llm  =: >@(0&{)
input_text =: >@(1&{)
input_data =: >@(1&{)

NB. ---- LLM field accessor ----
llm_tokenizer =: >@(3&{)

NB. ---- Split mapped string into single-char strings (utf-8 aware) ----
gpt2_chars =: 3 : 0
  s =. y
  if. 0 = # s do. return. '' end.
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
    res =. res , <(ln {. i }. s)
    i =. i + ln
  end.
  res
)

NB. ---- Map raw piece bytes -> GPT-2 unicode string ----
NB. x = cpt_tab, y = piece text
gpt2_map_bytes =: 4 : 0
  bytes =. a. i. y
  cpt =. x {~ bytes
  8 u: 4 u: cpt
)

NB. ---- BPE merge: greedily merge lowest-rank adjacent pairs ----
NB. x = sym_merges (s: of merges, rank = index), y = boxed list of char strings
gpt2_merge =: 4 : 0
  syms =. y
  n =. # syms
  while. 1 < n do.
    best =. _1
    bestr =. 1e9
    i =. 0
    while. i < n - 1 do.
      l =. > i { syms
      r =. > (i+1) { syms
      merge_str =. l , ' ' , r
      idx =. x i. s: <merge_str
      if. idx < # x do.
        if. idx < bestr do.
          bestr =. idx
          best =. i
        end.
      end.
      i =. i + 1
    end.
    if. best = _1 do. break. end.
    merged =. (> best { syms) , > (best+1) { syms
    syms =. (<merged) (best)} syms
    syms =. ((best+1) {. syms) , ((best+2) }. syms)
    n =. # syms
  end.
  syms
)

NB. ---- Pre-tokenizer: isolate digits individually, then regex-split with gaps ----
gpt2_pre_tokenize =: 3 : 0
  text =. y
  if. 0 = # text do. return. '' end.
  bytes =. a. i. text
  n =. # bytes
  pieces =. ''
  i =. 0
  while. i < n do.
    b =. i { bytes
    if. (48 <: b) *. (b <: 57) do.
      pieces =. pieces , <1 {. i }. text
      i =. i + 1
    else.
      start =. i
      while. i < n do.
        b2 =. i { bytes
        if. (48 <: b2) *. (b2 <: 57) do. break. end.
        i =. i + 1
      end.
      seg =. (i - start) {. start }. text
      ms =. gpt2_regex rxmatches seg
      if. 0 < {. $ ms do.
        n_ms =. {. $ ms
        ms2 =. (n_ms , 2) $ , ms   NB. rows of <start;len>
        last =. 0
        k =. 0
        while. k < n_ms do.
          row =. k { ms2
          st =. 0 { row
          ln =. 1 { row
          if. st > last do. pieces =. pieces , <(st - last) {. last }. seg end.
          pieces =. pieces , <(ln {. st }. seg)
          last =. st + ln
          k =. k + 1
        end.
        if. last < # seg do. pieces =. pieces , <((# seg) - last) {. last }. seg end.
      else.
        pieces =. pieces , <seg
      end.
    end.
  end.
  pieces
)

NB. ---- Tokenize text ----
NB. y = <llm; text>; returns boxed token-id list
gpt2_tokenize =: 3 : 0
  llm_data =. input_llm y
  text =. input_text y
  tokenizer =. llm_tokenizer llm_data
  sym_vocab =. tokenizer_symv tokenizer
  sym_merges =. tokenizer_symm tokenizer
  cpt_tab =. tokenizer_cpt tokenizer
  byte_tab =. tokenizer_byte tokenizer
  specials =. tokenizer_specials_g tokenizer
  
  NB. Split out special-token markers (chat template markers) first; tokenize
  NB. the text segments with the normal BPE path, splice the marker IDs.
  if. 0 < # specials do.
    segs =. specials split_specials text
  else.
    segs =. <text
  end.
  
  tokens =. ''
  si =. 0
  while. si < # segs do.
    seg =. > si { segs
    if. (specials i. <seg) < # specials do.
      tokens =. tokens , <(sym_vocab i. s: <seg)
    else.
      NB. Pre-tokenizer dispatch: llama-bpe (Llama-3.2), dbrx (Granite) and
      NB. lfm2 (LFM2) all use the llama3 regex pre; SmolLM2/Qwen use the gpt2
      NB. byte-level regex pre.
      pre =. tokenizer_pre_g tokenizer
      if. ('llama-bpe' -: pre) +. ('dbrx' -: pre) +. ('lfm2' -: pre) do.
        pieces =. llama3_pre_tokenize seg
      else.
        pieces =. gpt2_pre_tokenize seg
      end.
      i =. 0
      while. i < # pieces do.
        piece =. > i { pieces
        if. 0 < # piece do.
          word =. cpt_tab gpt2_map_bytes piece
          chars =. gpt2_chars word
          syms =. sym_merges gpt2_merge chars
          j =. 0
          while. j < # syms do.
            ss =. > j { syms
            idx =. sym_vocab i. s: <ss
            if. idx < # sym_vocab do.
              tokens =. tokens , <idx
            else.
              c2 =. gpt2_chars ss
              k =. 0
              while. k < # c2 do.
                cs =. > k { c2
                i2 =. sym_vocab i. s: <cs
                if. i2 < # sym_vocab do. tokens =. tokens , <i2
                else. tokens =. tokens , <0
                end.
                k =. k + 1
              end.
            end.
            j =. j + 1
          end.
        end.
        i =. i + 1
      end.
    end.
    si =. si + 1
  end.
  tokens
)

NB. ---- Detokenize: token ids -> text ----
NB. y = <llm; tokens>
gpt2_detokenize =: 3 : 0
  llm_data =. input_llm y
  tokens_boxed =. input_data y
  tokens =. > tokens_boxed
  if. 0 = # $ tokens do. tokens =. <tokens end.
  tokenizer =. llm_tokenizer llm_data
  vocab =. tokenizer_vocab_g tokenizer
  byte_tab =. tokenizer_byte tokenizer
  result =. ''
  i =. 0
  while. i < # tokens do.
    tid =. > i { tokens
    ts =. > tid { vocab
    if. 0 < # ts do.
      cpts =. 3 u: 7 u: ts   NB. decode mapped unicode -> integer codepoints
      bytes =. byte_tab {~ cpts
      result =. result , (a. {~ bytes)
    end.
    i =. i + 1
  end.
  result
)
