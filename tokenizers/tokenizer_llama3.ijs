NB. ================================================================
NB. Llama3 Tokenizer — regex pre-tokenizer + direct string lookup + byte fallback
NB. ================================================================

NB. ---- ASCII helpers ----
coclass 'inference'
is_letter =: 3 : 0
  b =. y
  ((b <: 90) *. b > 64) +. (b <: 122) *. b > 96
)

is_digit =: 3 : 0
  b =. y
  (-. b < 48) *. (-. b > 57)
)

is_ws =: 3 : 0
  b =. y
  (b = 32) + (b = 9) + (b = 10) + (b = 13)
)

tolower =: 3 : 0
  b =. y
  if. (65 <: b) *. (b <: 90) do. b + 32 else. b end.
)

byte =: 3 : 0
  a. i. y
)

NB. ---- Llama3 pre-tokenizer ----
llama3_pre_tokenize =: 3 : 0
  text =. y
  if. 0 = # text do. return. '' end.
  
  bytes =. byte text
  n =. # bytes
  pieces =. ''
  i =. 0
  
  while. i < n do.
    b =. i { bytes
    matched =. 0
    
    NB. 1. Contractions: 's, 't, 'm, 'd, 're, 've, 'll
    if. b = 27 do.
      if. (i + 1) < n do.
        c2l =. tolower i + 1 { bytes
        if. (c2l = 115) + (c2l = 116) + (c2l = 109) + (c2l = 100) do.
          pieces =. pieces , <2 {. i }. text
          i =. i + 2
          matched =. 1
        elseif. (i + 2) < n do.
          c3l =. tolower i + 2 { bytes
          if. ((c2l = 114) *. c3l = 101) + ((c2l = 118) *. c3l = 101) + ((c2l = 108) *. c3l = 108) do.
            pieces =. pieces , <3 {. i }. text
            i =. i + 3
            matched =. 1
          end.
        end.
      end.
    end.
    
    if. -. matched do.
      NB. 2. Letters
      NB. NOTE: `-. is_digit b` must be parenthesized — V V N makes a hook
      NB. `(-. is_digit) b` = `b -. is_digit b` (set difference), not NOT-of.
      NB. ALSO: the leading `-.` must be parenthesized with ITS argument —
      NB. `-. (b = 13) *. ...` applies `-.` to the WHOLE product (`-. 0` = 1).
      if. (-. (b = 13)) *. (-. (b = 10)) *. (-. (is_digit b)) do.
        if. is_letter b do.
          start =. i
          while. i < n do.
            if. -. is_letter (i { bytes) do. break. end.
            i =. i + 1
          end.
          pieces =. pieces , <(i - start) {. start }. text
          matched =. 1
        elseif. n > i + 1 *. is_letter (i + 1 { bytes) do.
          space_start =. i
          i =. i + 1
          start =. i
          while. i < n do.
            if. -. is_letter (i { bytes) do. break. end.
            i =. i + 1
          end.
          pieces =. pieces , <(i - space_start) {. space_start }. text
          matched =. 1
        end.
      end.
    end.
    
    if. -. matched do.
      NB. 3. Numbers
      if. is_digit b do.
        start =. i
        while. i < n do.
          if. -. is_digit (i { bytes) do. break. end.
          NB. `-. (i - start) < 3` — must parenthesize `i - start`; the bare
          NB. `-. i - start < 3` parses `i - (start < 3)` = `i - 1`, so `-.` = `2 - i`
          NB. which breaks at i=0 → empty piece → i never advances → infinite loop.
          if. -. (i - start) < 3 do. break. end.
          i =. i + 1
        end.
        pieces =. pieces , <(i - start) {. start }. text
        matched =. 1
      end.
    end.
    
    if. -. matched do.
      NB. 4. Non-word chars
      start =. i
      if. b = 32 do.
        if. (i + 1) < n do.
          nb =. i + 1 { bytes
          if. -. is_ws nb *. -. is_letter (nb) *. -. is_digit (nb) do.
            i =. i + 1
          else.
            i =. start
          end.
        end.
      end.
      while. i < n do.
        nb =. i { bytes
        if. is_ws (nb) + is_letter (nb) + is_digit (nb) do. break. end.
        i =. i + 1
      end.
      while. i < n do.
        nb =. i { bytes
        if. -. (nb = 13) + (nb = 10) do. break. end.
        i =. i + 1
      end.
      pieces =. pieces , <(i - start) {. start }. text
      matched =. 1
    end.
    
    if. -. matched do.
      pieces =. pieces , <1 {. i }. text
      i =. i + 1
    end.
  end.
  
  pieces
)

NB. ---- Strip trailing whitespace from a string ----
strip_ws =: 3 : 0
  space =. a. {~ 32
  i =. >./ I. y ~: space
  (i + 1) {. y
)

NB. ---- Build tokenizer from GGUF KV pairs ----
NB. Uses raw token strings for matching (avoids s: encoding issues)
NB. Returns: <vocab; bos_token_id; sym; eos; specials>
build_llama3_tokenizer =: 3 : 0
  kvs =. > 0 { y
  raw =. > 1 { y
  
  tk_tokens =. 'tokenizer.ggml.tokens' kv_string_array (<kvs) , (<raw)
  if. 0 = # tk_tokens do.
    return. <'' , <0 , <'' , <2 , <''
  end.
  
  NB. Strip trailing spaces from raw tokens
  vocab =. strip_ws each tk_tokens
  
  NB. Get BOS token ID from GGUF (default 2 for Llama3/Gemma3)
  bos_id =. 'tokenizer.ggml.bos_token_id' kv_uint (<kvs) , (<raw)
  if. bos_id <: 0 do. bos_id =. 2 end.
  
  NB. Get EOS token ID from GGUF (default 2 — llama3-style token 2 is EOS).
  NB. Gemma3-it sets it to 106; the arch generate verbs stop on this.
  eos_id =. 'tokenizer.ggml.eos_token_id' kv_uint (<kvs) , (<raw)
  if. eos_id <: 0 do. eos_id =. 2 end.
  
  
  NB. Build symbols for fast O(1)-ish hash lookup (s: uses a global symbol table;
  NB. no 64K limit — handles 262k+ vocab). Store in tokenizer index 3.
  sym =. s: vocab
  
  NB. Store as 6-element boxed container:
  NB. [vocab; bos_token_id; tk_len; sym; eos_token_id; specials]
  vb =. <vocab
  bb =. <bos_id
  tb =. <tk_len
  sb =. <sym
  eb =. <eos_id
  NB. Special tokens: markers used by the model's chat template (llama.cpp
  NB. splits on specials via tokenizer.ggml.token_type; we scope to the
  NB. template's markers to keep tokenize fast). Store at index 5.
  tmpl =. 'tokenizer.chat_template' kv_string (<kvs) , (<raw)
  specials =. template_markers tmpl
  specials =. specials #~ specials e. vocab   NB. keep only real vocab entries
  sp =. <specials
  vb , bb , sb , eb , sp
)

NB. ---- Tokenizer field accessors ----
NB. tokenizer = [vocab; bos_token_id; tk_len; sym; eos_token_id; specials]
tokenizer_vocab     =: >@(0&{)  NB. unbox vocab
tokenizer_bos_token =: >@(1&{)  NB. unbox BOS token ID
tokenizer_sym       =: >@(2&{)  NB. symbols for hash lookup
tokenizer_eos       =: >@(3&{)  NB. unbox EOS token ID
tokenizer_specials  =: >@(4&{)  NB. special-token marker strings (boxed)

NB. ---- Input accessors for tokenize/detokenize ----
NB. Both functions receive y = <llm; data> where data is either text or token list
input_llm   =: >@(0&{)  NB. unbox llm from input
input_text  =: >@(1&{)  NB. unbox text/tokens from input (for tokenize)
input_data  =: >@(1&{)  NB. unbox tokens from input (for detokenize)

NB. ---- LLM field accessors (shared with gemma3.ijs) ----
llm_tokenizer =: >@(3&{)  NB. extract tokenizer from llm noun

NB. ---- Tokenize text using llama3 pre-tokenizer + symbol lookup ----
NB. y = <llm; text> — receives llm (tokenizer at index 3) and text
llama3_tokenize =: 3 : 0
  llm_data =. input_llm y
  text =. input_text y
  tokenizer =. llm_tokenizer llm_data
  vocab =. tokenizer_vocab tokenizer
  bos_token =. tokenizer_bos_token tokenizer
  sym =. tokenizer_sym tokenizer
  specials =. tokenizer_specials tokenizer
  
  NB. Split out special-token markers (chat template markers) first; tokenize
  NB. the text segments with the normal piece path, splice the marker IDs.
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
      tokens =. tokens , <(sym i. s: <seg)
    else.
      pieces =. llama3_pre_tokenize seg
      i =. 0
      np =. # pieces
      while. i < np do.
        piece =. > i { pieces
        if. 0 = # piece do.
          i =. i + 1
        else.
          NB. Try the raw piece first — llama-bpe convention: tokens carry
          NB. literal leading spaces (" The" is one vocab entry), so a piece
          NB. like " The" must look up directly. Fall back to the ▁ form
          NB. (SentencePiece convention, e.g. gemma) only when the raw piece
          NB. is absent from the vocab. Byte fallback as a last resort.
          found_idx =. sym i. s: <piece
          if. found_idx < # sym do.
            tokens =. tokens , <found_idx
          elseif. ' ' = {. piece do.
            target =. '▁' , (1 }. piece)
            found_idx =. sym i. s: <target
            if. found_idx < # sym do.
              tokens =. tokens , <found_idx
            else.
              pbytes =. byte piece
              tokens =. tokens , (<"0) (65536 + pbytes)
            end.
          else.
            pbytes =. byte piece
            tokens =. tokens , (<"0) (65536 + pbytes)
          end.
          i =. i + 1
        end.
      end.
    end.
    si =. si + 1
  end.
  
  if. bos_token > 0 do.
    tokens =. (<bos_token) , tokens
  end.
  
  tokens
)

NB. ---- Detokenize: convert token IDs to text ----
NB. y = <llm; tokens> — receives llm (with tokenizer at index 3) and token IDs
NB. ---- Replace SentencePiece space-marker ▁ (UTF-8 E2 96 81, 3 chars in J) ----
NB. with a single space, for presentation and for round-trip persistence
NB. (spaces re-tokenize to the same ▁-pieces).
sp_replace =: 3 : 0
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
llama3_detokenize =: 3 : 0
  llm_data =. input_llm y
  tokens_boxed =. input_data y
  tokens =. > tokens_boxed
  if. 0 = #$ tokens do.   NB. scalar token — wrap in list
    tokens =. <tokens
  end.
  tokenizer =. llm_tokenizer llm_data
  vocab =. tokenizer_vocab tokenizer
  
  NB. Vectorized decode: byte tokens (65536..65791) -> raw byte char,
  NB. otherwise -> vocab string. Select per token via boolean mask.
  tid_list =. > tokens
  sv_byte =. (tid_list >: 65536) *. tid_list < 65792
  NB. Index vocab only for non-byte tokens (byte tids -> index 0, masked out)
  vb =. (tid_list * -. sv_byte) { vocab
  bc =. <"0 a. {~ (tid_list - 65536) * sv_byte
  s =. ; sv_byte {"_1 vb ,. bc
  NB. Presentation: SentencePiece marks leading spaces with ▁ (U+2581) — convert
  NB. to a real space so chat answers read naturally. This ALSO makes the answer
  NB. text re-tokenizable to the same piece stream (spaces -> ▁-pieces), so a
  NB. chat session's re-render prefix matches the stored tokens and the KV
  NB. cache can be resumed across turns.
  sp_replace s
)
