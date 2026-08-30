NB. ================================================================
NB. GGUF Dump Utility — pretty-print info about any GGUF file
NB. Usage: load 'gguf_dump.ijs'; gguf_dump_inference_ 'path/to/model.gguf'
NB. Depends on: gguf/gguf.ijs
NB. ================================================================

coclass 'inference'
gguf_dump =: 3 : 0
  require 'llm/inference/gguf/gguf'
  
  path =. y
  
  NB. Map the file once; parse header/KV/tensor-infos from the mapped bytes
  NB. (no full-file materialization). Unmap at the end.
  raw =. mmap_gguf path
  
  NB. Header
  header =. parse_hdr_raw raw
  magic =. > 0 { header
  version =. > 1 { header
  n_tensors =. > 2 { header
  n_kv =. > 3 { header
  
  echo '========================================'
  echo 'GGUF File Dump'
  echo '========================================'
  s =. 'Path: ' , path
  echo s
  s =. 'Magic: ' , ": magic
  echo s
  s =. 'Version: ' , ": version
  echo s
  s =. 'Tensors: ' , ": n_tensors
  echo s
  s =. 'KV pairs: ' , ": n_kv
  echo ''
  
  NB. KV pairs
  echo '----------------------------------------'
  echo 'Metadata Key-Value Pairs'
  echo '----------------------------------------'
  
  result =. parse_kv_pairs_raw raw
  kvs =. > 0 { result
  raw =. > 1 { result
  count =. > 2 { result
  kv_end =. > 3 { result
  
  i =. 0
  while. i < count do.
    key =. > (i*4) { kvs
    vt =. > ((i*4)+1) { kvs
    vtname =. val_type_name (<vt)
    
    s =. key
    if. (8 = vt) +. (9 = vt) do.
      s =. s , ' [' , vtname , ']'
    elseif. (6 = vt) +. (12 = vt) do.
      val =. > ((i*4)+2) { kvs
      nb =. (6 = vt) * 4 + (12 = vt) * 8
      raw_bytes =. (val + i.nb) { raw
      if. 4 = nb do.
        fval =. _1(3!:5) raw_bytes
      else.
        fval =. _2(3!:5) raw_bytes
      end.
      s =. s , ' = ' , ": fval
    elseif. 7 = vt do.
      s =. s , ' = bool'
    elseif. (4 = vt) +. (5 = vt) do.
      val =. > ((i*4)+2) { kvs
      s =. s , ' = ' , ": (val le32 raw)
    elseif. (0 = vt) +. (1 = vt) +. (2 = vt) +. (3 = vt) do.
      val =. > ((i*4)+2) { kvs
      s =. s , ' = ' , ": (val le32 raw)
    elseif. ((10 = vt) +. (11 = vt)) do.
      val =. > ((i*4)+2) { kvs
      s =. s , ' = ' , ": (val le64 raw)
    else.
      val =. > ((i*4)+2) { kvs
      s =. s , ' = ' , ": val
    end.
    echo s
    
    i =. i + 1
  end.
  
  echo ''
  echo '----------------------------------------'
  s =. ('Tensor Infos (' , ": n_tensors) , ' total)'
  echo s
  echo '----------------------------------------'
  
  ti =. (<raw) , (<kv_end) , (<n_tensors)
  ti =. parse_tensor_infos ti
  
  NB. Header line
  echo 'Idx  Name                                      Shape            Type     Data Off'
  echo '---  ----------------------------------------  ---------------  -------  --------'
  
  i =. 0
  while. i < n_tensors do.
    nm =. > (i*6) { ti
    dm =. > ((i*6)+1) { ti
    et =. > ((i*6)+2) { ti
    do2 =. > ((i*6)+3) { ti
    etname =. elem_type_name (<et)
    
    NB. Format shape as comma-separated
    if. 0 = # dm do.
      shape_s =. '(scalar)'
    else.
      shape_s =. '{; ' , (": dm) , '; ]'
    end.
    
    NB. Pad shape to 15 chars
    pad_len =. 15 - # shape_s
    if. 0 > pad_len do. pad_len =. 0 end.
    shape_s =. shape_s , pad_len $ ' '
    
    NB. Pad name to 40 chars
    if. 40 < # nm do.
      nm_short =. 37 {. nm
      nm_short =. nm_short , '...'
    else.
      nm_short =. nm
    end.
    pad_len =. 40 - # nm_short
    if. 0 < pad_len do.
      nm_short =. nm_short , pad_len $ ' '
    end.
    
    s =. ": i
    if. 9 < # s do. s =. ' ' , s end.
    if. 8 < # s do. s =. ' ' , s end.
    s =. s , '  ' , nm_short , '  ' , shape_s , '  ' , etname , '  ' , ": do2
    echo s
    
    i =. i + 1
  end.
  
  NB. Tensor data section
  echo ''
  echo '----------------------------------------'
  echo 'Tensor Data Section'
  echo '----------------------------------------'
  
  file_size =. # raw
  NB. Last tensor info's "next" offset is the end of tensor info section
  n_tensors =. > 2 { header
  if. 0 < n_tensors do.
    ti_end =. > ((n_tensors*6)-1) { ti
  else.
    ti_end =. kv_end
  end.
  tds =. 32 * <. (ti_end + 31) % 32
  
  s =. 'KV end offset: ' , ": kv_end
  echo s
  s =. 'Tensor info end offset: ' , ": ti_end
  echo s
  s =. 'Alignment padding: ' , ": tds - ti_end
  echo s
  s =. 'Tensor data start: ' , ": tds
  echo s
  s =. 'File size: ' , ": file_size
  echo s
  s =. 'Tensor data size: ' , ": file_size - tds
  
  echo ''
  echo '========================================'
  echo 'Dump complete'
  echo '========================================'
  
  unmap_gguf ''
)

echo 'gguf_dump.ijs loaded — use: gguf_dump ''path/to/model.gguf'''
