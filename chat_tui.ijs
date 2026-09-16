NB. ================================================================
NB. Chat TUI — a minimal terminal chat UI driven by our llm_inference
NB. directly (j-kvm `vt` for raw-mode + key reads; no jpi).
NB. Usage:
NB.   jconsole chat_tui.ijs [MODEL]        (default mdl qwen3-0.6b)
NB.
NB. Stateless chat: each turn re-renders the full message history through
NB. the shared chat_completion verb (streaming: the reply appears live,
NB. token by token, via the chat_stream_cb per-token callback). The message
NB. list persists for the session (stateful KV-cache resume is the end
NB. goal, not yet).
NB.
NB. Controls: type a message + Enter to send; Backspace to edit;
NB. Ctrl-C or type `exit` to quit.
NB. ================================================================

NB. Run the TUI in the inference locale itself (not a separate chatu locale):
NB. the streaming consumer chat_cb_g is rebound to stream_delta, and J verb
NB. assignment aliases the NAME (resolved in the locale where it's CALLED, i.e.
NB. inference) — so the consumer + TUI state + drawing must share inference's
NB. locale, or chat_stream_cb's `chat_cb_g delta` can't resolve stream_delta.
load 'llm/inference'            NB. addon (must be installed)
cocurrent <'inference'

NB. Model: first ARGV arg or default catalog id.
get_model =: 3 : 0
  mdl =. 'qwen3-0.6b'
  if. 3 <: #ARGV do.
    m =. > 2 { ARGV
    if. 0 < # m do. mdl =. m end.
  end.
  mdl
)
mdl =: get_model ''

LLM =: load_gguf_to_llm_inference_ mdl
MAX_STEPS =: 100000
RUNNING =: 1
MSGS =: 0 $ <''                 NB. boxed list of <role ; content>
IN =: ''                        NB. current input line
STREAM =: ''                    NB. accumulating assistant text during streaming

NB. Terminal driver (fd-fixed j-kvm vt: read stdin fd 0, write stdout fd 1).
require 'llm/inference/util/vt'
coinsert 'vt'

NB. ---- Rendering ----
NB. Raw mode disables the terminal's LF->CRLF translation (ONLCR/OPOST), so a
NB. bare LF moves down without returning to column 0. Emit CRLF for line ends.
CRLF =: (13 { a.) , LF
puts_nl =: 3 : 0
  puts (y rplc LF;CRLF)
  ''
)

NB. wrap txt into rows of at most w bytes, never splitting a UTF-8 sequence
NB. (a byte >= 0xC0 at the cut is an incomplete lead -> drop; a 0x80..0xBF
NB. continuation at the cut -> drop). Returns boxed list of row strings.
NB. wrap txt into rows of at most w bytes, one row = one terminal line:
NB. never cross a LF, and never split a UTF-8 sequence. Returns boxed rows.
wrap =: 4 : 0
  w =. x
  txt =. y
  rows =. 0 $ <''
  while. 0 < # txt do.
    n =. w <. # txt
    li =. txt i. LF
    if. (li >: 0) *. (li < n) do. n =. li end.   NB. LF before width ends the row
    seg =. n {. txt
    NB. drop trailing bytes of an incomplete UTF-8 sequence
    while. 0 < # seg do.
      lb =. a. i. (seg {~ <: # seg)
      if. lb < 128 do.
        break.
      end.
      if. lb >: 224 do.
        seg =. }: seg
        break.
      end.
      seg =. }: seg
    end.
    if. 0 < # seg do.
      rows =. rows , < seg
      txt =. (# seg) }. txt
    else.
      txt =. 1 }. txt
    end.
    NB. consume a LF that ended the row
    if. 0 < # txt do.
      if. LF = {. txt do. txt =. }. txt end.
    end.
  end.
  rows
)

draw_conv =: monad define
  hw =. gethw ''
  h =. {. hw
  w =. {: hw
  in_row =. h - 1                     NB. input line occupies the last row
  cscr ''
  fgc 7
  puts 'llm_inference chat (stateless) - ' , mdl
  reset ''
  NB. parallel rows (boxed row strings) + cols (numeric colors per row)
  rows =. ''                          NB. boxed list of row strings
  cols =. 0 $ 0                       NB. color per row, parallel to rows
  for_m. MSGS do.
    m =. > m
    role =. > 0 { m
    content =. > 1 { m
    txt =. ('[' , role , '] ' , content) rplc CRLF;LF
    if. role -: 'user' do. c =. 6
    elseif. role -: 'assistant' do. c =. 2
    else. c =. 8 end.
    rs =. w wrap txt
    rows =. rows , rs
    cols =. cols , (c #~ # rs)
  end.
  NB. live streaming assistant row (grows as deltas arrive)
  if. 0 < # STREAM do.
    rs =. w wrap ('[assistant] ' , STREAM)
    rows =. rows , rs
    cols =. cols , (2 #~ # rs)
  end.
  NB. tail-window: only render rows that fit above the input line
  avail =. in_row - 1                 NB. row 0 is the header
  if. avail < # rows do.
    rows =. (- avail) {. rows
    cols =. (- avail) {. cols
  end.
  row =. 1
  for_i. i. # rows do.
    goxy 0 , row
    ceol ''
    fgc (i { cols)
    puts (> i { rows)
    row =. row + 1
  end.
  reset ''
  ''
)

draw_input =: monad define
  hw =. gethw ''
  row =. <: {. hw                 NB. last row (output area ends one row above)
  goxy 0 , row
  ceol ''
  fgc 3
  puts '> ' , IN
  reset ''
  ''
)

NB. ---- Streaming delta consumer ----
NB. Installed as chat_cb_g (the inference-locale delta verb); each text delta
NB. appends to STREAM and redraws the conversation live. chat_stream_cb calls
NB. this per generated token with only the complete UTF-8 text emitted.
stream_delta =: monad define
  STREAM =: STREAM , y
  draw_conv ''
  draw_input ''
  ''
)

NB. ---- Main loop ----
run =: monad define
  raw 1
  curs 0
  draw_conv ''
  draw_input ''
  while. RUNNING do.
    k =. rkey ''
    select. k
    case. 3 do. RUNNING =: 0            NB. Ctrl-C
    case. 10;13 do.                     NB. Enter
      if. (IN -: 'exit') +. (IN -: 'quit') do.
        RUNNING =: 0
      else.
        if. 0 < # IN do.
          MSGS =: MSGS , < ('user') ; IN
          IN =: ''
          STREAM =: ''
          draw_conv ''
          draw_input ''
          NB. streaming reply: install the per-token callback, generate live.
          NB. We run in the inference locale, so use simple names. chat_stream_start
          NB. arms gen_cb_on_g/gen_cb_g; chat_cb_g (rebound to our stream_delta,
          NB. also in inference) is the per-delta consumer.
          chat_stream_start ''
          chat_cb_g =: stream_delta
          res =. LLM chat_completion (MSGS ; '' ; MAX_STEPS ; 1 ; <(0 0 0.95 0.0))
          chat_stream_stop ''
          MSGS =: MSGS , < ('assistant') ; (> 0 { res)
          STREAM =: ''
          draw_conv ''
        end.
        draw_input ''
      end.
    case. 127;8 do.                     NB. Backspace
      if. 0 < # IN do. IN =: }: IN end.
      draw_input ''
    case. do.                           NB. printable chars
      if. (k >: 32) *. (k < 127) do.
        IN =: IN , (k { a.)
        draw_input ''
      end.
    end.
  end.
  raw 0
  curs 1
  reset ''
  puts_nl LF
  ''
)

run ''
2!:55 (0)   NB. exit jconsole cleanly after the TUI quits
