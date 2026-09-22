NB. ================================================================
NB.  util/session.ijs — per-session entity (foundation).
NB.
NB.  A session is a boxed noun holding per-session state.  The current
NB.  serial path uses ONE session (the global `session`); the future
NB.  concurrent/batched HTTP path will use one session per request.  This
NB.  lays the structure: the noun, accessors, and new/reset verbs.  The
NB.  globals migrate into it one category at a time (chat session first).
NB.
NB.  Layout (fixed, indexed accessors; callbacks are VERBS and cannot be
NB.  boxed into a noun — they stay as globals for now, staged later):
NB.    0 chat_session   (<arch ; messages ; output ; total ; max_steps ;
NB.                      params> or '')  — the persistent chat state
NB.    1 ct_tmpl        (real jinja template string or '')
NB.    2 ct_vars        (minja obj for extra template variables)
NB.    3 ct_now         (epoch override for pinned oracles; 0 = live)
NB.    4 ct_tools       (tools JSON string or '')
NB.    5 st_buf         (streaming detokenizer held bytes)
NB.    6 st_arch        (streaming arch)
NB.    7 sid            (response id)
NB.    8 created       (response created)
NB.    9 kv_seq        (KV cache slot; _1 = none)
NB.   10 kv_pos       (current used length for this session)
NB.   11 kv_batch    (parallel sequences B for this session; 1 = single)
NB.   12 kv_max_seq  (context override; _1 = model max)
NB.   13 kv_meta     (<n_layers; eff_seq; n_heads_kv; head_dim> or '')
NB.   14 llm         (shared model ref)
NB.   15 arch        (shared arch)
NB.   16 model       (shared model name)
NB.
NB.  J gotcha: a boxed noun is immutable — writing a field rebuilds the
NB.  noun (copy).  That is fine for the SMALL state here; the KV buffers
NB.  stay on the shared B-axis cache (in-place amend via globals), not in
NB.  the session.  The session holds only the KV META (seq/pos/batch/
NB.  max_seq/meta) + a slot reference into the shared buffers.
NB.  (All addon code lives in the `inference` locale — see the other util
NB.  scripts — so the session entity + accessors live there too.)
NB. ================================================================
coclass 'inference'

session =: 0 $ <''   NB. empty = no session

NB. ---- create a fresh (empty) session ----
NB.  (single line: a newline after a complete noun ends the statement in J)
session_new =: 3 : 0
  (<'') , (<'') , (<'') , (<0) , (<'') , (<'') , (<'') , (<'') , (<0) , (<_1) , (<0) , (<1) , (<_1) , (<'') , (<'') , (<'') , (<'')
)

NB. ---- clear the current session ----
session_reset =: 3 : 0
  session =: 0 $ <''
  ''
)

NB. ---- accessors (open the boxed field) ----
sess_chat   =: >@(0&{)    NB. the boxed chat-session list (2-cell)
sess_tmpl   =: >@(1&{)
sess_vars   =: >@(2&{)
sess_now    =: >@(3&{)
sess_tools  =: >@(4&{)
sess_stbuf  =: >@(5&{)
sess_starch =: >@(6&{)
sess_sid    =: >@(7&{)
sess_created=: >@(8&{)
sess_kvseq  =: >@(9&{)
sess_kvpos  =: >@(10&{)
sess_kvb    =: >@(11&{)
sess_kvmx   =: >@(12&{)
sess_kvmeta =: >@(13&{)
sess_llm    =: >@(14&{)
sess_arch   =: >@(15&{)
sess_model  =: >@(16&{)

NB. ---- replace field n of the session; returns the new session ----
NB.  x = field_idx; y = new_value.  The J amend on a boxed list is
NB.  (<NEW) (IDX) } LIST  — the new value is boxed (matching the cell) and
NB.  the index is parenthesized, attached to the }.
sess_put =: 4 : 0
  (<y) (x) } session
)

NB. ---- replace field n of the CURRENT session in place (rebinds global) ----
NB.  x = field_idx; y = new_value.  Rebinds the global `session`.
sess_set =: 4 : 0
  session =: sess_put x y
  ''
)
