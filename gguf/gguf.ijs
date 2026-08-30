NB. ================================================================
NB. GGUF Parser — model-agnostic, sequential, spec-compliant
NB. ================================================================

NB. ---- Byte access: native J 3!:4 / 3!:5 dyad ----
NB. Type codes (from J Learning Ch 27 + Dictionary dx003):
NB.   _1 (3!:4) uint16 from 2 chars    _1 (3!:5) float32 from 4 chars
NB.   _2 (3!:4) uint32 from 4 chars    _2 (3!:5) float64 from 8 chars
NB.   _3 (3!:4) uint64 from 8 chars
NB. Direction: negative = char->number, positive = number->char
NB. NEVER DIY float32/float64 decode — J's representation verbs are
NB. the correct, native path. Bit-manipulation is wrong because J stores
NB. numbers with internal encoding that does not match raw IEEE 754 bits.
NB. F16/BF16 require DIY (no native support), but F32/F64 do not.

NB. Little-endian uint from n-byte group in char array
NB. J's 3!:4 interprets chars as little-endian (last char = MSB)
NB. Note: 3!:4 returns 1-element arrays, extract scalar with {.
NB. Tacit (Ch 42 13 : output; verified bit-exact + faster vs explicit):
NB.   13 : '0 { _1(3!:4) (x+i.2) { y'  ->  0 { (_1) 3!:4 ] { ~ 0 01 + [
NB.   13 : '0 { _2(3!:4) (x+i.4) { y'  ->  0 { (_2) 3!:4 ] { ~ 0 1 2 3 + [
NB.   13 : '0 { _3(3!:4) (x+i.8) { y'  ->  0 { (_3) 3!:4 ] { ~ (i.8) + [
NB. The converter right-bonds the fetch as a reflex (~): x+i.n is the
NB. index list, `] {~ idx` fetches (x+i.n){y. `(_k) 3!:4` decodes, `0 {`
NB. extracts the scalar.
coclass 'inference'
require 'jmf'
cocurrent 'inference'

NB. ---- Memory-mapped GGUF file (jmf addon) ----
NB. Map the file once as a JCHAR array; returns the mapped array.
NB. Avoids the full 1!: 1 read — pages are faulted in lazily on slice
NB. fetch. The mapping global is <name>_base_ in the base locale.
NB. CAVEAT: {. (take) on a mapped array is a view, but }. (drop) with a
NB. nonzero offset MATERIALIZES the suffix (tail copy) — so slice with
NB. the index-list (off + i. n) { raw, never n {. off }. raw.
mmap_gguf =: 3 : 0
  unmap_jmf_ 'gguf_raw'
  (2;'') map_jmf_ ('gguf_raw'; y)
  gguf_raw_base_
)
unmap_gguf =: 3 : 0
  unmap_jmf_ 'gguf_raw'
  ''
)

le32 =: 0 {  (_2) 3!:4 ] { ~ 0 1 2 3 + [
le64 =: 0 {  (_3) 3!:4 ] { ~ (i.8) + [

NB. Decode float32 values from char array y -> float64 array
NB. y = count; <chars> (2-element boxed list)
NB. Tacit (Ch 42, 13 : output): ((_1) 3!:5 ([: (i.) 4 * [: > 0 { ]) { [: > 1 { ])
f32_decode =: ((_1) 3!:5 ([: (i.) 4 * [: > 0 { ]) { [: > 1 { ])

NB. Decode float64 values from char array y -> float64 array
NB. y = count; <chars> (2-element boxed list)
NB. Tacit (Ch 42, 13 : output): ((_2) 3!:5 ([: (i.) 8 * [: > 0 { ]) { [: > 1 { ])
f64_decode =: ((_2) 3!:5 ([: (i.) 8 * [: > 0 { ]) { [: > 1 { ])

NB. ---- BF16 decode: zero-extend to F32 via 3!:5 ----
NB. BF16 is F32 with low 16 bits truncated. Zero-extend each LE uint16
NB. value with 2 leading zero chars, then _1(3!:5) decodes it (no reversal
NB. needed since J's 3!:5 is LE). Tacit (Ch 42, 13 : output):
bf16_to_f32 =: ((_1) 3!:5 [: , (0 00{a.) ,"1 (2 ,~ 2 %~ #) $ ])

NB. Decode BF16 values from char array y -> float64 array
NB. y = count; <chars> (2-element boxed list)
NB. Tacit (Ch 42, 13 : output) + f. (Ch 41): bf16_to_f32 is inlined at
NB. definition, so decode_bf16 is self-contained (no runtime name lookup).
decode_bf16 =: ([: bf16_to_f32 ([: (i.) 2 * [: > 0 { ]) { [: > 1 { ]) f.

NB. ---- 65536-entry F16->F64 lookup table (built once at load time) ----
NB. Replaces per-tensor bit extraction with simple table lookup
f16_build_table =: 3 : 0
  NB. All uint16 values 0..65535
  all =. i. 65536
  
  NB. Convert uint16 ints → chars → back to ints for F16 decode
  NB. 1(3!:4) converts J ints to 2-byte char sets (little-endian)
  chars =. 1 (3!:4) all
  
  NB. Convert chars back to uint16 ints
  ints =. _1 (3!:4) chars
  ints =. ints + 65536 * ints < 0
  
  NB. F16 bit extraction (run once over all 65536 values)
  sign =. <. ints % 32768
  exponent =. (<. ints % 1024) 17 b. 31
  mantissa =. ints 17 b. 1023
  smult =. 1 - 2 * sign
  
  NB. Precomputed powers of 2
  pow2 =. 2 ^ _15 + i. 32
  
  NB. Normal, subnormal, inf/NaN branches
  norm =. smult * (pow2 {~ exponent) * (1 + mantissa % 1024)
  sub =. smult * (2 ^ _14) * (mantissa % 1024)
  is_inf =. (31 = exponent) *. 0 = mantissa
  is_nan =. (31 = exponent) *. 0 < mantissa
  spec =. ((_ * smult) * is_inf) + (_. * is_nan)
  
  (norm * (0 < exponent) *. exponent < 31) + (sub * (0 = exponent) *. 0 < mantissa) + (spec * (is_inf +. is_nan))
)

NB. F16 lookup table — built once at module load time
f16_table =: f16_build_table ''

NB. Load F16 tensor data using precomputed lookup table
NB. y = <ne; <raw_bytes>> (same format as f16_decode)
NB. raw is a flat ne*2 byte slice; 0 (3!:4) reads 2-byte LE as UNSIGNED
NB. uint16 (0..65535), so it indexes f16_table directly — no signed
NB. conversion, no reshape/ravel (the old _1(3!:4) path returned signed
NB. int16 and needed ints + 65536 * ints < 0 to un-sign).
NB.
NB. Tacit form (Ch 42, explicit-to-tacit): ({&f16_table) right-bonds the
NB. table as the index operand, >&(1&{) is the second-box accessor, and
NB. 0&(3!:4) the unsigned 16-bit read. The 13: converter can't produce
NB. this — "{ f16_table" is a verb-noun RIGHT tine (V V N), so the noun
NB. must be right-bonded manually. Tacit drops runtime lexical handling
NB. (locals, name lookups): 270M values ~0.76s explicit -> ~0.37s tacit.
f16_load =: ({&f16_table) @ (0&(3!:4)) @ (>&(1&{))

NB. Decode F16 (half-float) values from char array y -> float64 array
NB. y = <count; <chars>>
NB. Uses bulk uint16 conversion and precomputed pow2 lookup for speed
f16_decode =: 3 : 0
  ne =. > 0 { y
  slice =. > 1 { y
  chars =. (ne, 2) $ slice
  
  NB. Bulk uint16 conversion — processes all values in one vectorized call
  ints =. _1 (3!:4) , chars
  ints =. ints + 65536 * ints < 0   NB. signed to unsigned
  
  NB. F16: sign(1bit) | exponent(5bit, bias=15) | mantissa(10bit)
  sign =. <. ints % 32768
  exponent =. (<. ints % 1024) 17 b. 31
  mantissa =. ints 17 b. 1023
  smult =. 1 - 2 * sign
  
  NB. Precomputed powers of 2 for all 32 exponent values
  pow2 =. 2 ^ _15 + i. 32
  
  NB. Normal: sign * 2^(exp-15) * (1 + mant/1024)
  norm =. smult * (pow2 {~ exponent) * (1 + mantissa % 1024)
  NB. Subnormal: sign * 2^-14 * (mant/1024)
  sub =. smult * (2 ^ _14) * (mantissa % 1024)
  NB. Inf/NaN
  is_inf =. (31 = exponent) *. 0 = mantissa
  is_nan =. (31 = exponent) *. 0 < mantissa
  spec =. ((_ * smult) * is_inf) + (_. * is_nan)
  
  (norm * (0 < exponent) *. exponent < 31) + (sub * (0 = exponent) *. 0 < mantissa) + (spec * (is_inf +. is_nan))
)

NB. ---- Parse header: <magic; version; tensor_count; kv_count> ----
NB. mmap-based: maps the file once, parses from the mapped bytes, unmaps
NB. (header scalars are already extracted). No full-file materialization.
parse_hdr =: 3 : 0
  raw =. mmap_gguf y
  hdr =. parse_hdr_raw raw
  unmap_gguf ''
  hdr
)

NB. Parse header from raw/mapped bytes (no file read)
parse_hdr_raw =: 3 : 0
  raw =. y
  (0 le32 raw), (4 le32 raw), (8 le64 raw), (16 le64 raw)
)

NB. ---- Read KV header at offset: <key; value_type; next_offset> ----
read_kv_hdr =: 4 : 0
  b =. y
  off =. x

  kl =. off le64 b
  if. kl > 65536 do. (<'_ERROR_'), (<_1), (<_1) return. end.
  key =. (off+8+i.kl) { b

  to =. off + 8 + kl
  vt =. to le32 b
  vo =. to + 4

  NB. Skip past value, compute next offset
  if. 8 = vt do.
    vl =. vo le64 b
    next =. vo + 8 + vl
  elseif. 9 = vt do.
    et =. vo le32 b
    ec =. (vo + 4) le64 b
    aoff =. vo + 12
    if. 8 = et do.
      count =. 0
      while. count < ec do.
        slen =. (aoff le64 b)
        aoff =. aoff + 8 + slen
        count =. count + 1
      end.
      next =. aoff
    elseif. (5 = et) +. (4 = et) +. (0 = et) +. (6 = et) do.
      next =. aoff + (ec * 4)
    elseif. (1 = et) +. (2 = et) +. (3 = et) do.
      next =. aoff + (ec * 2)
    elseif. 7 = et do.
      next =. aoff + ec
    elseif. (10 = et) +. (11 = et) +. (12 = et) do.
      next =. aoff + (ec * 8)
    else.
      next =. aoff + (ec * 8)
    end.
  elseif. (6 = vt) +. (5 = vt) +. (4 = vt) do.
    next =. vo + 4
  elseif. 7 = vt do.
    next =. vo + 1
  elseif. (0 = vt) +. (1 = vt) do.
    next =. vo + 1
  elseif. (2 = vt) +. (3 = vt) do.
    next =. vo + 2
  elseif. (10 = vt) +. (11 = vt) +. (12 = vt) do.
    next =. vo + 8
  else.
    next =. vo + 8
  end.

  NB. Return <key;vt;vo;next>
  (<key),(<vt),(<vo),(<next)
)

NB. ---- Parse all KV pairs sequentially ----
NB. Returns: <kvs_flat; raw_bytes; count; end_offset>
NB. kvs_flat: <key0; vt0; vo0; key1; vt1; vo1; ...>
NB. vo = value offset (where value data starts in raw bytes)
NB. NOTE: kept as a full 1!: 1 read. A mmap-based version that returns the
NB. live mapped array is a footgun — callers slice raw and then call other
NB. mapping verbs (gguf_dump/parse_hdr), and map_jmf_ refuses to remap
NB. 'gguf_raw' while the caller's boxed ref is alive. Loaders use the
NB. mmap'd parse_kv_pairs_raw instead (load_gguf_to_llm maps once).
parse_kv_pairs =: 3 : 0
  raw =. 1!: 1 < y
  parse_kv_pairs_raw raw
)

NB. Parse all KV pairs from raw/mapped bytes (no file read)
NB. Returns: <kvs_flat; raw_bytes; count; end_offset>
parse_kv_pairs_raw =: 3 : 0
  raw =. y
  header =. parse_hdr_raw raw
  n_kv =. > 3 { header

  NB. Build flat list using triplet concatenation
  kvs =. ''
  off =. 24
  i =. 0
  while. i < n_kv do.
    kvh =. off read_kv_hdr raw
    k =. > 0 { kvh
    if. '_' = {. k do. break. end.
    NB. kvh = <key;vt;vo;next> — concatenate as flat list
    kvs =. kvs , kvh
    off =. > 3 { kvh
    i =. i + 1
  end.

  (<kvs),(<raw),(<i),(<off)
)

NB. ---- Read one tensor info at offset ----
read_tensor_info =: 4 : 0
  b =. y
  off =. x

  name_len =. off le64 b
  name =. (off+8+i.name_len) { b

  doff =. off + 8 + name_len
  n_dims =. doff le32 b
  dims_off =. doff + 4
  dims =. n_dims $ 0
  j =. 0
  while. j < n_dims do.
    dims =. (j {. dims) , (((dims_off + j*8)) le64 b) , (1 + j) }. dims
    j =. j + 1
  end.

  type_off =. dims_off + (n_dims * 8)
  etype =. type_off le32 b
  data_off =. (type_off + 4) le64 b

  hdr_sz =. (8 + name_len + 4) + (n_dims * 8) + 4 + 8
  next =. off + hdr_sz

  (<name),(<dims),(<etype),(<data_off),(<hdr_sz),(<next)
)

NB. ---- Parse all tensor infos sequentially ----
NB. y = <raw_bytes; kv_end_offset; tensor_count>
parse_tensor_infos =: 3 : 0
  raw =. > 0 { y
  kv_end =. > 1 { y
  n_tensors =. > 2 { y

  infos =. ''
  off =. kv_end
  i =. 0
  while. i < n_tensors do.
    ti =. off read_tensor_info raw
    infos =. infos , ti
    off =. > 5 { ti
    i =. i + 1
  end.

  infos
)

NB. ---- Bytes per element for a tensor etype ----
NB. Quant types (2..29) use the block tables (qblk_elems/qblk_bytes).
etype_bpe =: 3 : 0
  et =. y
  if. (2 <: et) *. et <: 29 do.
    qi =. qblk_idx {~ et
    if. _1 = qi do. 4 return. end.
    (qblk_bytes {~ qi) % (qblk_elems {~ qi)
  elseif. 1 = et do. 2
  elseif. 30 = et do. 2
  else. 4 end.
)

NB. ---- Load tensor data ----
NB. y = <file_path; info_flat; tensor_data_start>
load_tensor_data =: 4 : 0
  path =. > 0 { y
  info =. > 1 { y
  tds =. > 2 { y

  etype =. > 2 { info
  data_rel_off =. > 3 { info
  dims =. > 1 { info
  ne =. */ dims

  file_off =. tds + data_rel_off
  raw =. 1!: 1 < path
  bpe =. etype_bpe etype
  NB. Hybrid slice — see load_tdata: take-of-drop when the tail copy is
  NB. cheaper than the index-list fetch (tail < nbytes * 18).
  nbytes =. <. ne * bpe
  tail =. (# raw) - file_off
  if. tail < nbytes * 18 do.
    slice =. nbytes {. file_off }. raw
  else.
    slice =. (file_off + i. nbytes) { raw
  end.

  flat =. decode_tensor_flat (<etype) , (<ne) , <slice
  dims tensor_reshape flat
)

NB. ---- Element type name (GGUF tensor etypes, ggml.h numbering) ----
elem_type_name =: 3 : 0
  et =. > y
  if. 0 = et do. 'F32' elseif. 1 = et do. 'F16' elseif. 2 = et do. 'Q4_0'
  elseif. 3 = et do. 'Q4_1' elseif. 6 = et do. 'Q5_0' elseif. 7 = et do. 'Q5_1'
  elseif. 8 = et do. 'Q8_0' elseif. 9 = et do. 'Q8_1' elseif. 10 = et do. 'Q2_K'
  elseif. 11 = et do. 'Q3_K' elseif. 12 = et do. 'Q4_K' elseif. 13 = et do. 'Q5_K'
  elseif. 14 = et do. 'Q6_K' elseif. 15 = et do. 'Q8_K' elseif. 16 = et do. 'IQ2_XXS'
  elseif. 17 = et do. 'IQ2_XS' elseif. 18 = et do. 'IQ3_XXS' elseif. 19 = et do. 'IQ1_S'
  elseif. 20 = et do. 'IQ4_NL' elseif. 21 = et do. 'IQ3_S' elseif. 22 = et do. 'IQ2_S'
  elseif. 23 = et do. 'IQ4_XS' elseif. 24 = et do. 'I8' elseif. 25 = et do. 'I16'
  elseif. 26 = et do. 'I32' elseif. 27 = et do. 'I64' elseif. 28 = et do. 'F64'
  elseif. 29 = et do. 'IQ1_M' elseif. 30 = et do. 'BF16'
  elseif. 34 = et do. 'TQ1_0' elseif. 35 = et do. 'TQ2_0'
  elseif. 39 = et do. 'MXFP4' elseif. 40 = et do. 'NVFP4'
  elseif. 41 = et do. 'Q1_0' elseif. 42 = et do. 'Q2_0'
  else. '<unknown>' end.
)

NB. ---- Value type name ----
val_type_name =: 3 : 0
  vt =. > y
  if. 0 = vt do. 'uint8' elseif. 1 = vt do. 'int8' elseif. 2 = vt do. 'uint16'
  elseif. 3 = vt do. 'int16' elseif. 4 = vt do. 'uint32' elseif. 5 = vt do. 'int32'
  elseif. 6 = vt do. 'float32' elseif. 7 = vt do. 'bool' elseif. 8 = vt do. 'string'
  elseif. 9 = vt do. 'array' elseif. 10 = vt do. 'uint64' elseif. 11 = vt do. 'int64'
  elseif. 12 = vt do. 'float64' else. '<unknown>' end.
)

NB. ---- KV pair value retrieval helpers ----

NB. Find KV pair index in flat list by key name
NB. flat list: <key0; vt0; vo0; next0; key1; vt1; vo1; next1; ...>
find_kv_idx =: 4 : 0
  target =. x
  kvs =. y
  count =. (#kvs) % 4
  result =. _1
  i =. 0
  while. i < count do.
    if. (> (i*4) { kvs) -: target do.
      result =. i
      i =. count
    else.
      i =. i + 1
    end.
  end.
  result
)

NB. Get uint value from KV pair by key
NB. x = key name, y = <kvs_flat; raw_bytes>
kv_uint =: 4 : 0
  key =. x
  data =. y
  kvs =. > 0 { data
  raw =. > 1 { data
  idx =. key find_kv_idx kvs
  if. 0 > idx do. _1 return. end.
  vt =. > ((idx*4)+1) { kvs
  if. (4 = vt) +. (10 = vt) +. (5 = vt) do.
    vo =. > ((idx*4)+2) { kvs
    if. (4 = vt) +. (5 = vt) do.
      vo le32 raw
    else.
      vo le64 raw
    end.
  else.
    _1
  end.
)

NB. Get float value from KV pair by key
NB. x = key name, y = <kvs_flat; raw_bytes>
kv_float =: 4 : 0
  key =. x
  data =. y
  kvs =. > 0 { data
  raw =. > 1 { data
  idx =. key find_kv_idx kvs
  if. 0 > idx do. _1 return. end.
  vt =. > ((idx*4)+1) { kvs
  if. (6 = vt) +. (12 = vt) do.
    vo =. > ((idx*4)+2) { kvs
    if. 6 = vt do.
      nb =. 4
    else.
      nb =. 8
    end.
    _1(3!:5) (vo + i.nb) { raw
  else.
    _1
  end.
)

NB. Get string value from KV pair by key
NB. x = key, y = <kvs_flat; raw_bytes>
kv_string =: 4 : 0
  key =. x
  data =. y
  kvs =. > 0 { data
  raw =. > 1 { data
  idx =. key find_kv_idx kvs
  if. 0 > idx do. '' return. end.
  vt =. > ((idx*4)+1) { kvs
  if. 8 = vt do.
    vo =. > ((idx*4)+2) { kvs
    slen =. vo le64 raw
    (vo+8+i.slen) { raw
  else.
    ''
  end.
)

NB. Get numeric array value from KV pair by key
NB. x = key, y = <kvs_flat; raw_bytes>
kv_array =: 4 : 0
  key =. x
  data =. y
  kvs =. > 0 { data
  raw =. > 1 { data
  idx =. key find_kv_idx kvs
  if. 0 > idx do. $0 return. end.
  vt =. > ((idx*4)+1) { kvs
  if. 9 = vt do.
    vo =. > ((idx*4)+2) { kvs
    et =. (vo le32 raw)
    ec =. ((vo + 4) le64 raw)
    aoff =. vo + 12
    if. (0 = et) +. (4 = et) +. (5 = et) +. (6 = et) do.
      if. 6 = et do.
        f32_decode (<ec) , <((aoff + i.ec*4) { raw)
      else.
         256 #. a. i. |."1 (ec, 4) $ (aoff + i.ec*4) { raw
      end.
    elseif. (1 = et) +. (2 = et) +. (3 = et) do.
      256 #. a. i. |."1 (ec, 2) $ (aoff + i.ec*2) { raw
    elseif. (10 = et) +. (11 = et) +. (12 = et) do.
      if. 12 = et do.
        f64_decode (<ec) , <((aoff + i.ec*8) { raw)
      else.
        i =. 0
        result =. ''
        while. i < ec do.
          result =. result , ((aoff + i * 8) le64 raw)
          i =. i + 1
        end.
        result
      end.
    elseif. 7 = et do.
      (aoff + i.ec) { raw
    else.
      $0
    end.
  else.
    $0
  end.
)

NB. Get string array from KV pair by key
kv_string_array =: 4 : 0
  key =. x
  data =. y
  kvs =. > 0 { data
  raw =. > 1 { data
  idx =. key find_kv_idx kvs
  if. 0 > idx do. $0 return. end.
  vt =. > ((idx*4)+1) { kvs
  if. 9 = vt do.
    vo =. > ((idx*4)+2) { kvs
    et =. (vo le32 raw)
    ec =. (((vo + 4)) le64 raw)
    if. 8 = et do.
      result =. ''
      aoff =. vo + 12
      count =. 0
      while. count < ec do.
        slen =. (aoff le64 raw)
        result =. result , (<((aoff+8+i.slen) { raw))
        aoff =. aoff + 8 + slen
        count =. count + 1
      end.
      result
    else.
      $0
    end.
  else.
    $0
  end.
)

NB. ---- Tensor info helpers ----

NB. Find tensor index in info flat list by name
NB. info_flat: <name0; n0; dims0; et0; off0; hdr0; next0; name1; ...>
NB. each tensor = 7 slots
NB. x = target name, y = info_flat
find_tensor_idx =: 4 : 0
  target =. x
  info =. y
  count =. (#info) % 6
  names =. (6 * i. count) { info
  idx =. names i.!.0 < target
  if. idx < count do. idx else. _1 end.
)

NB. Get full tensor info by name: <name; dims; etype; data_off; hdr_sz; next>
NB. x = name, y = info_flat
get_tensor_info =: 4 : 0
  name =. x
  info =. y
  idx =. name find_tensor_idx info
  if. 0 > idx do. $0 return. end.
  ((idx*6)+i.6) { info
)

NB. Get tensor shape (dims array) by name
NB. x = name, y = info_flat
get_tensor_shape =: 4 : 0
  name =. x
  info =. y
  ti =. name get_tensor_info info
  if. 0 = #ti do. $0 return. end.
  > 1 { ti
)

NB. Get element type code by name
NB. x = name, y = info_flat
get_tensor_type =: 4 : 0
  name =. x
  info =. y
  ti =. name get_tensor_info info
  if. 0 = #ti do. _1 return. end.
  > 2 { ti
)

NB. Load tensor data by name
NB. y = <file_path; info_flat; tensor_data_start; tensor_name>
load_tdata =: 3 : 0
  NB. y = <path; info; tds; name; raw>  (raw optional, cached file data)
  data =. y
  path =. > 0 { data
  info =. > 1 { data
  tds =. > 2 { data
  name =. > 3 { data
  ti =. name get_tensor_info info
  if. 0 = #ti do. $0 return. end.
  etype =. > 2 { ti
  data_rel_off =. > 3 { ti
  dims =. > 1 { ti
  ne =. */ dims

  file_off =. tds + data_rel_off
  raw =. > 4 { data
  if. 0 = # raw do.
    raw =. 1!: 1 < path
  end.
  bpe =. etype_bpe etype
  NB. Hybrid slice. The index-list (off + i. n) { raw allocates an n-int
  NB. index vector + element fetch (~4.6ns/elem measured — the big
  NB. embedding is 155M elems ≈ 0.72s). The take-of-drop n {. off }. raw
  NB. instead copies the FILE TAIL (fs - off) — cheap for tensors near the
  NB. END, catastrophic if used for every tensor (sums to ~130GB — the 20x
  NB. regression). Use take-of-drop only when the tail copy is cheaper:
  NB. tail < nbytes * 18 (bytes vs measured elem-cost) — the big early
  NB. tensors (token_embd) and late small-tail tensors; index-list elsewhere.
  nbytes =. <. ne * bpe
  tail =. (# raw) - file_off
  if. tail < nbytes * 18 do.
    slice =. nbytes {. file_off }. raw
  else.
    slice =. (file_off + i. nbytes) { raw
  end.

  flat =. decode_tensor_flat (<etype) , (<ne) , <slice
  dims tensor_reshape flat
)

NB. ---- Reshape decoded flat tensor data to match llama.cpp layout ----
NB. GGUF stores 2D weight data REVERSED vs the dims field: dims=[in,out],
NB. data is [out,in] (row = output neuron). Match llama.cpp by reshaping 2D
NB. tensors as reversed dims. 1D (norm) weights keep dims as-is.
tensor_reshape =: 4 : 0
  dims =. x
  dat =. y
  if. 2 = # dims do.
    (|. dims) $ dat
  else.
    dims $ dat
  end.
)


