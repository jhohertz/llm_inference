NB. ================================================================
NB. KV Cache — transformer KV cache management
NB. Representation: PRE-ALLOCATED flat arrays, one per kind, aligned with
NB. llama.cpp's kv-cache (allocate once per context, write in place; reset
NB. between generations). J's in-place amend fires only for a refcount-1
NB. noun with a SIMPLE selector, so the cache is ONE flat array per kind
NB. (no per-layer boxes — unboxing a boxed layer makes a second ref and the
NB. amend copies the whole layer):
NB.   k_cache_g = (n_layers * kv_batch_g * eff_seq, n_heads_kv * head_dim)
NB.              K, row-major; layer stride = kv_batch_g*eff_seq, sequence
NB.              stride = eff_seq (per-layer, B sequences contiguous),
NB.              position stride = n_kv*hd
NB.   v_cache_g = same for V.
NB.   kv_meta    = <n_layers; eff_seq; n_heads_kv; head_dim>  (4 items)
NB.   kv_batch_g = parallel sequences in the cache (default 1; batched decode
NB.              sets it before kv_create — seq base = (layer*kv_batch_g + seq))
NB.   kv_pos_g   = current used length (max across sequences)
NB.   kv_max_seq_g = context override for low-memory (default _1 = model max)
NB. Writes are IN-PLACE amends (measured: scalar row ~1us, 10-row list ~6us
NB. on a 102MB array): kv_write amends one row, kv_write_rows amends L rows.
NB. kv_write/kv_read/kv_write_rows take an OPTIONAL 5th seq item (default 0),
NB. so the single-sequence callers are unchanged.
NB. Reads gather the window (count rows) — a contiguous copy, unavoidable in
NB. J (no views). kv_reset only resets kv_pos_g (buffer reused, stale rows are
NB. never read — every position is written before it is read).
NB. (The old growing per-layer arrays appended via `,` — an O(used) full-array
NB. copy per write, O(n^2) total over a generation; at 128K that is unusable.
NB. The pre-alloc fixes the WRITE to O(cell); the window copy moves to the
NB. read. The K cyclic transpose (1 2 0 |:) remains — J's in-place amend
NB. requires positions-leading, so K cannot be stored transposed without
NB. per-layer column amends (blocked: refcount-2 boxes) or a batched-write
NB. refactor.)
NB. ================================================================

coclass 'inference'

kv_meta =: ''          NB. <n_layers; eff_seq; n_heads_kv; head_dim> or ''
k_cache_g =: ''        NB. (n_layers*kv_batch_g*eff_seq, n_kv*hd) K, or ''
v_cache_g =: ''        NB. (n_layers*kv_batch_g*eff_seq, n_kv*hd) V, or ''
kv_pos_g =: 0          NB. current used length (max across sequences)
kv_max_seq_g =: _1     NB. context override (default _1 = model max context)
kv_batch_g =: 1        NB. parallel sequences in the cache (batched decode)
kv_batch_alloc_g =: 1  NB. the kv_batch_g the buffer was allocated for
kv_seq_g =: 0          NB. default sequence for 4-item calls (batched prefill sets it)

NB. ---- Create/reset KV cache ----
NB. y = <n_layers; max_seq; n_heads_kv; head_dim>
NB. Allocates the flat buffers ONCE per session (eff_seq = min(max_seq,
NB. kv_max_seq_g), B = kv_batch_g); a repeat call with identical dims + batch
NB. just resets kv_pos_g (session-persistent — no realloc per generate).
NB. Returns ''.
kv_create =: 3 : 0
  n_layers =. > 0 { y
  max_seq =. > 1 { y
  n_heads_kv =. > 2 { y
  head_dim =. > 3 { y
  eff_seq =. max_seq
  if. 0 < kv_max_seq_g do. eff_seq =. max_seq <. kv_max_seq_g end.
  if. -. '' -: kv_meta do.
    if. (n_layers = > 0 { kv_meta) *. (eff_seq = > 1 { kv_meta) *. (n_heads_kv = > 2 { kv_meta) *. (head_dim = > 3 { kv_meta) *. (kv_batch_alloc_g = kv_batch_g) do.
      kv_pos_g =: 0
      '' return.
    end.
  end.
  kv_meta =: (<n_layers) , (<eff_seq) , (<n_heads_kv) , (<head_dim)
  k_cache_g =: ((n_layers * kv_batch_g * eff_seq) , (n_heads_kv * head_dim)) $ 0.0
  v_cache_g =: ((n_layers * kv_batch_g * eff_seq) , (n_heads_kv * head_dim)) $ 0.0
  kv_batch_alloc_g =: kv_batch_g
  kv_pos_g =: 0
  ''
)

NB. ---- Write one K/V row at pos for a layer ----
NB. y = <layer; pos; k_new; v_new; seq?>   seq default 0
NB. k_new/v_new shape: (n_heads_kv, head_dim). In-place row amend (scalar
NB. selector on the refcount-1 flat global) — O(cell), no array copy.
kv_write =: 3 : 0
  layer =. > 0 { y
  pos =. > 1 { y
  k_new =. > 2 { y
  v_new =. > 3 { y
  seq =. kv_seq_g
  if. 4 < # y do. seq =. > 4 { y end.
  base =. ((layer * kv_batch_g) + seq) * (> 1 { kv_meta)
  k_cache_g =: (, k_new) ((base + pos))} k_cache_g
  v_cache_g =: (, v_new) ((base + pos))} v_cache_g
  kv_pos_g =: kv_pos_g >. pos + 1
  ''
)

NB. ---- Bulk write L rows for a layer (prompt prefill) ----
NB. y = <kind; layer; start; rows; seq?>   rows = (L, n_heads_kv, head_dim)
NB. In-place list-selector amend (L contiguous rows) — O(L*cell), no copy.
kv_write_rows =: 3 : 0
  kind =. > 0 { y
  layer =. > 1 { y
  start =. > 2 { y
  rows =. > 3 { y
  seq =. kv_seq_g
  if. 4 < # y do. seq =. > 4 { y end.
  L =. {. $ rows
  rows =. (L , ((> 2 { kv_meta) * (> 3 { kv_meta))) $ , rows
  base =. ((layer * kv_batch_g) + seq) * (> 1 { kv_meta)
  idx =. base + start + i. L
  if. kind = 0 do.
    k_cache_g =: rows idx} k_cache_g
  else.
    v_cache_g =: rows idx} v_cache_g
  end.
  kv_pos_g =: kv_pos_g >. start + L
  ''
)

NB. ---- Read K/V for a layer ----
NB. y = <layer; pos; seq?>   seq default 0
NB. Returns: <k_rows; v_rows>  each (count, n_heads_kv, head_dim), count=pos+1.
NB. Gathers the contiguous window rows (a copy — unavoidable; the reshape to
NB. (count, n_kv, hd) is free on the contiguous data).
kv_read =: 3 : 0
  layer =. > 0 { y
  pos =. > 1 { y
  seq =. kv_seq_g
  if. 2 < # y do. seq =. > 2 { y end.
  count =. pos + 1
  base =. ((layer * kv_batch_g) + seq) * (> 1 { kv_meta)
  k_rows =. (count , (> 2 { kv_meta) , (> 3 { kv_meta)) $ , ((base + i. count) { k_cache_g)
  v_rows =. (count , (> 2 { kv_meta) , (> 3 { kv_meta)) $ , ((base + i. count) { v_cache_g)
  (<k_rows) , (<v_rows)
)

NB. ---- Reset KV cache to empty (keeps the buffer; no realloc) ----
NB. Stale rows beyond kv_pos_g are never read (every position is written
NB. before it is read in a fresh prefill + generation). Returns ''.
kv_reset =: 3 : 0
  if. 0 = # kv_meta do. '' return. end.
  kv_pos_g =: 0
  ''
)
