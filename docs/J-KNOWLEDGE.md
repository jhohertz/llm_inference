# J-KNOWLEDGE.md — J Language Knowledge & Idioms

Reusable J language knowledge: the chapter-by-chapter jforc idiom reviews
(Ch 22-43) distilled to general J lore, plus the hard-won J gotchas. This file
is **project-agnostic** — it is about J itself and can be shared with any J
project. It is **not** plan material (see **PLAN.md**), not architecture (see
**docs/ARCHITECTURE.md**), and not a project record (see
**docs/HISTORICAL.md**). Project-specific applications/decisions that were once
recorded here now live in those three files; only the J-language knowledge
stays.

Where this file notes a general idiom that a project *could* apply but has not,
the plan-relevant intent lives in **PLAN.md** under "Deferred J-idiom
applications"; the general knowledge lives here.

---

## J Idiom Review (jforc Loopless Code I: Verbs Have Rank)

Verb rank = cell size: `u"n` applies `u` to n-cells; `+/` is infinite-rank
(`+/"1` sums rows). Explicit `while.`/`for.` carry a penalty and are wise only
when the body is time-consuming (file I/O, stateful FSM) or the state is too
large to materialize looplessly.

Rank is the primary tool for loopless code: use `+/ "1`, `{."1`/`}."1`/`{ "1`,
`,"1` to operate on rows/cells without loops. `names i. <name` (boxed `i.`
matches exact boxes, no padding) replaces a scan-loop index search; `build
each i. n` replaces a fixed-count build loop.

## J Idiom Review (jforc More Verbs)

Arithmetic & boolean dyads are rank-0. Comparison verbs are **tolerant** by
default (`!.0` for exact); `i.!.0` is the intolerant, often-faster index-of.
`I.` finds the sorted-list insertion point. `\:` grade-down. `>`/`<` min-max
replace per-atom test loops. Constant verbs `n:`.

Use `i.!.0` for exact (boxed-string / integer) searches. Tolerant comparison
is relied on when exactness would split near-ties (e.g. argmax/softmax); verify
it never masks a real mismatch in exactness tests (`-:` on ints/strings is
exact).

## J Idiom Review (jforc Loopless Code II: Adverbs `/` and `~`)

`u/"n` applies at rank n. `~` reverses dyad operands (`x u~ y = y u x`) and
monad `u~ y = y u y` (reflex, e.g. `/:~ y` sorts ascending). Reductions
(`+/`, `>./`, `*/`, `+/"1`, `>./"1`), sort/grade reflex (`\:`, `/:~`, `\:~`),
revoke indexing (`{~`), AND-reduce `*./`.

Greedy-max selection: `y i. >./ y` (index-of first max) replaces
`{. I. y = >./ y`.

## J Idiom Review (jforc Compound Verbs)

`u@:v` (At, infinite rank — v on whole arg, u on whole result) vs `u@v` (Atop,
rank of v — cell-at-a-time). Prefer `@:` over `@` unless cell-at-a-time is
intended. Build composite verbs **tacitly** by composition rather than
`3 : 0`/`4 : 0` wrappers: `move_axes =: |:`, `matmul =: +/ .*`,
`rope_apply"1`, and accessors `3 : '> N { y'` → `>@(N&{)`.

**Gotcha**: `>@(N&{})` (errant `}` appended) makes `N&{}` a hook — `12&{}` ≠
`12&{` — causing domain errors. Use `>@(N&{)`. The `>@(N&{)` accessor form
relies on boxing rank semantics (`@` Atop + `>` open); confirm rank/cell
behavior on nested/2D boxes against a boxing chapter.

## J Idiom Review (jforc Empty Operands)

Empty **operand** ≠ empty **array**: an operand is empty iff its *frame* (w.r.t.
verb cell rank) contains a `0`. When empty, J runs the verb on a **cell of fills**
(`0`/`' '`/`a:`) and notes result shape `s`/type `t`; overall result = frame concat
`s`. If fill-cell execution errors, fallback scalar `0` (`5 + ''` → shape `0`
numeric). Empty *cells* are the verb's job, not fill-cell processing (`3 {. ''` →
char string, `3 {. 0$0` → numeric list; data vs control-info distinction).

**Main application = broadcasting**: leading-prefix broadcasts stay correct
over empty arrays because J executes on fill-cells rather than erroring.
Verified: `(0 4 $ 100) % (0 $ 0)` → `0 4` (no error), `0 % 0` → `0`.

## J Idiom Review (jforc Loopless Code III: Adverbs `\` and `\.`)

Scan adverbs: monad `u\` applies u to successive **prefixes**, monad `u\.` to
**suffixes**; dyad `x u\` applies u to **infixes** (rolling windows of length
`|x`, overlapping if x>0), `x u\.` to **outfixes** (y minus each infix). Most
used as `u/\` (running reductions): `+/\` running total, `>./\` running max,
`<:\` keeps leading 1s.

Cumulative-sum / crossing-point idiom: `cuts =. I. threshold < cumsum`,
`{. I. -. r > cum`. Note suffix `+/\\.` is faster than `+/\` per jforc, but
prefix scans need forward totals — keep `+/\`.

## J Idiom Review (jforc Loopless Code IV: Irregular Operations)

Control-information array + dyadic op per cell; Power/If/DoWhile conjunction
`u^:n` / `u^:v` / `u^:v^:_` (If: `u^:0`/`u^:1`, DoWhile: `u^:v^:_`); Tie ```` and
Agenda `m@.v` (switch — v selects verb from gerund m; **rank-0 agenda is slow**).

Default-param ifs → selection idiom: `threshold =. (0.95 , x) {~ 0 < x`
replaces `if. 0 < x do. ... end.` with control-info + selection. A byte-token
decode switch/loop can be vectorized as a boolean-mask selection instead of a
per-token while + if/elseif/else.

Caveats: `m@.v` agenda for scalar-per-token dispatch is rank-0 slow; `u^:v^:_`
DoWhile needs state threaded through a single verb; `u^:n` Power needs a
prebuilt gerund + fold when items differ per index.

## J Idiom Review (jforc More Verbs For Boxes)

Dyad `;` (Link — boxes x, boxes y only if unboxed), monad `;` (Raze — removes
one boxing level, concatenates contents), full `{` selection story (multi-axis
selectors, complementary `<<` exclusion, `a:` = all), `{::` (Fetch — multi-level
select+open), `L.` (Level), `;:` word-split, and **multiple assignment**
`'op1 op2 ...' =. y` (unboxes each item).

Multiple assignment replaces `> N { y` unpacking: `'a b c' =. y`. The `;` Link
"box the last operand" habit: `(x) , (y) , <z` boxes only trailing when needed.

## J Idiom Review (jforc Verb-Definition Revisited)

`m :n` (`3 :` monad, `4 :` dyad, `0 :0` multi-line noun/comment); compound
verbs assignable; `[:` suicide verb (fails on wrong valence, used as one side
of `u :v`); **dual-valence verbs `u :v`** — monadic uses u, dyadic uses v;
e.g. `pwr =: 2&pwr : (dyad : 'x^y')` assigns a default left operand.

## J Idiom Review (jforc Obverse / Apply-Under)

Obverse `u^:_1` (inverse of a verb, when it exists); apply-under `u&.:v` =
`v^:_1@u@v` (change representation via v, work with u, invert via `v^:_1`);
defined obverse `u :.v` (attach an explicit inverse verb to u).

`u&.:v` fits a nested transform→work→invert; it does NOT fit a sequential
pipeline (softmax/top-p/min-p) or boxed-arg unpacking (`u&.:>` adds nesting).
Natural obverse pairs (e.g. tokenize/detokenize) are only worth `u :.v`
definitions when a single "work in the other space, round-trip" verb is needed.

## J Idiom Review (jforc Loopless Code V: Partitions)

`~.`/`~:` (nub + nub-sieve), `u/.` (apply on keyed subsets), `u;.1`/`;.2`/
`;_1`/`;_2` (partition on frets), `E.` (find sequence), `u;.0` (extract
subarray), `u;.3`/`;._3` (sliding windows), `u^:a:` (fixpoint, return
intermediates). The `E.` + `;.1` string-splitting idiom maps to pattern-finding
(PCRE `rxmatches` is the regex equivalent); a unique-key lookup table (e.g. a
tokenizer's vocab) can use `s:` symbols for O(1)-ish hashing — no dedup needed
if the table is already unique.

## J Idiom Review (jforc When Programs Are Data)

Ways to pass verbs as data: published-name callback (`f_subfn =. g`), modifier
operands (`adverb define` / `conjunction define`, verb accessible as `u`/`v`),
**gerunds** (`u`v` = atomic verb rep; invoke via `m`:6`), `128!:2` Apply (string
→ verb, monadic only), and `".`/`5!:5` (execute/stringify sentences).

## J Idiom Review (jforc Loopless Code VI: Temporary Variables)

`LoopWithInitial` — fold state through items with a running carry, loopless:
box items, append initial, reverse (`/\.` faster), `u&.>/\.` suffix-scan, then
undo (reverse, drop initial, unbox). Each result carries the full state for the
next step. The chapter's caveat: it materializes every intermediate into a
rank-2 array — wasteful when state is large (a loop is then justified). It is
the tool for a small-state fold (e.g. piece accumulation) where space is not a
concern.

## J Idiom Review (jforc Loopless Code VII: Sequential Machines)

Dyad `;:` implements a sequential machine: `x ;: y` where `x = f;s;m;ijrd`. Map
each input item to a column via `m`; a row/action table `s` (row = state, column
= input class) yields `<new_row; action_code>`; actions (0-6) emit boxed words,
unboxed words, indices/lengths, coded row/col, or stop. Initial
`ijrd = i j r d` (input ptr, word ptr, row, final column).

A char-scanning state machine (classify bytes, transition state, emit boxed
word pieces) is exactly expressible as `;:` — a state table + column map
replacing a `while.` scanner with one tacit primitive call. Caveats: `;:`
action semantics (word pointer `j`, add-single vs add-multiple, end-of-input
`d`) are subtle; irregular rules (apostrophe contractions, run-merging,
max-digit limits) need extra states/columns and may lose the clarity gain. A
`;:` rewrite is high-risk vs a verified-correct loop; only pursue if tokenizer
clarity/speed demands it, validating piece-by-piece.

## J Idiom Review (jforc Modifying an Array: m})

Dyad `x m} y` — adverb `}` builds a derived verb that copies `y` and installs
`x` at positions `m` (same selection forms as `m{y`); result is the whole
modified array. Monad `I.` (rank 1) gives indexes of 1s in a Boolean vector;
interpreter special-cases compounds like `I.@:>`.

The two in-place forms the interpreter optimizes: `name =. x m} name` and
`name =. name , x`. `m}` builds tables; `{.`/`}.` row-drop after an amend is a
fresh copy (unavoidable reshape).

## J Idiom Review (jforc Modular Code — Ch 29)

Namespaces: every public name lives in one locale; a locative
`simplename_localename_` (or `simplename__var`) addresses a name in an explicit
locale. `=.` is private (lost on exit / lost under `load`), `=:` is public.
Simple-name lookup: private → current locale → its path (always ending `z`).
`cocurrent`/`coclass` set the current locale; `coinsert` adds inheritance.
`conew 'class'` creates an object (numbered locale + class in path) and calls
`create` (monadic — receives the left arg of `conew`). Methods are locatives
`method__obj`; `codestroy` erases the object locale.

Modern addon norm (cf. math/mt): all code in one locale (`coclass` in every
script), internal deps `require 'cat/name/...'` (no `.ijs` extension —
`getscripts` only addon-resolves extensionless names), entry file `cat/name.ijs`
so `load 'cat/name'` works (addon path → `~addons/cat/name/name.ijs`). No
z-locale publishing: call verbs from any locale with the `_locale_` suffix;
`load`/`require` restore the caller's locale on return.

## J Idiom Review (jforc Writing Your Own Modifiers — Ch 30)

User-written modifiers: `adverb define` / `conjunction define` (or `1 :` / `2 :`
one-liners). Operands arrive as `u` (and `v` for conjunctions); noun operands
also get `m`/`n`. Two kinds:

- **Not referring to x/y**: text is fully interpreted when `u`/`v` are supplied;
  result (usually a verb) replaces the modifier+operands in the sentence — zero
  interpretive overhead in the data path (`Ifany`, `Butifnull`, `playsound`,
  `LoopWithInitial`, `Undery`).
- **Referring to x/y**: text is interpreted only when the derived verb executes
  with its noun operands; result of the derived verb must be a noun (`InLocales`).
  Slower ("soporific" in low-rank verbs). Rule: refer to x/y only if necessary.

Operand passing: **verb arguments pass by name, noun arguments pass by value**
— a modifier can embed a verb operand in other modifiers but cannot *execute* it
directly in its own body. Introspection: `u b.0` → ranks; `v b._1` → string of
the inverse; `a: 1 :(v b._1)` converts the string to a verb (prefer the actual
inverse `#.` over `#:^:_1` — known compounds get special interpreter handling).

**Verified gotcha: anonymous-verb locale leak.** Named verbs restore the
current locale on completion; **anonymous derived verbs do not** — a `cocurrent`
inside an anonymous derived verb leaks the locale change (console test:
`(<'abc') t 0` leaves current locale `abc`; `tt =: (<'abc') t` then `tt 0`
restores `base`). A modifier that switches locales must save/restore
`18!:5 ''` itself (the `InLocales` pattern).

## J Idiom Review (jforc Odds and Ends — Ch 34)

Fast string searching `s:` (symbols) — O(1)-ish lookup; caveats: global symbol
table, no per-string free, `s:` encoding issues. Comparison tolerance `!.f`
(exact `i.!.0`). Bitwise Boolean `m b.` (16-31) — `(m-16) b.` = Boolean function
(used for F16-style masking). Generalized transpose `|:` — formal identity
`(<p{x) { p |: y` = `(<x) { y`. Amend `m}` in-place. Features not needed by most
numeric workloads: startup/IDE/config, form editor, CRC (`128!:3`), Unicode
(`u:`), Cartesian product monad `{`, `i:` (last-occurrence index), window
driver, tacit recursion `$:` / memoization `M.` (only memoizes small nonneg ints
— no fit for float tensors).

## J Idiom Review (jforc Performance: Measurement & Tips — Ch 35)

Measurement: `6!:2` (time) / `7!:0` + `7!:2` (space). The J Performance Monitor
(`load 'jpm'`; `$JINSTALL/system/util/pm.ijs`, locale `jpm`, NOT auto-loaded)
profiles named explicit verbs line-by-line. Console API: `start_jpm_ ''`
(SIZE=1e9 on 64-bit) → run → `showtotal_jpm_ ''` (summary) /
`showdetail_jpm_ ''` (per-line). `viewtotal_jpm_` (GUI) is gone in J9.7.
**NEVER call `stop_jpm_ ''` before `showtotal_jpm_ ''`** — it resets the tracing
buffer and `read` reports "no PM records"; `read` stops tracing itself
(start → run → showtotal). Trace overhead inflates `6!:2` numbers; heavy suites
overflow the buffer (lost records reported).

## J Idiom Review (jforc Tacit Programs — Ch 36)

Every verb is a tacit program (operands by position, not by name); simple
explicit verbs can be rewritten tacitly (`addrow =: +/"1`, `v =: 1.04&*@:+`,
`sortup =: /:~`). The grammar for "two operations on y combined" (forks/hooks)
is the next chapter.

## J Idiom Review (jforc Introduction to Forks — Ch 37)

Monadic fork: `(V0 V1 V2) Ny` = `(V0 Ny) V1 (V2 Ny)` — e.g. `mean =: +/ % #`.

## J Idiom Review (jforc Forks, Hooks, and Compound Adverbs — Ch 40)

The bident/trident rules:
- **`V V N` is NOT a fork** — it executes as `V (V N)` (rightmost verb monadic
  on the noun first). Noun tines only work on the LEFT (`N V V` noun fork).
  This explains: `1&{ , 'abc'` = `1&{ (, 'abc')` = `1&{ 'abc'` = 'b' (comma =
  monadic raze), and `(1&{ , <0 0 0.95 0)` = `1&{ (<0 0 0.95 0)` → index error.
- **Hooks and forks have infinite rank** (corrects the Ch 37 "fork rank = left
  tine's rank" misreading — `mean b.0` is `_ _ _` because all forks are `_`).
- **`[:` fork left tine** = "ignore the left branch": `([: u v)` ≡ `u@:v`,
  and `[:` is recognized by the parser, not executed — costs nothing.
- **`@`/`@:` semantics corrected**: `x (u@v) y` = `u (x v y)` — v dyadic, u
  MONADIC (both `@` and `@:`; the only difference is rank). So a dyadic-only
  verb can't be the `u` of `@`/`@:`.
- **Passing x through while building the right argument monadically**: the
  **dyadic hook** `x (u v) y` = `x u (v y)`, with a constant as a rank-`_` verb
  `(<0 0 0.95 0.0)"_` (ambivalent; `(<params>&[` is monadic-only). This solves
  default-param wrappers: `g_simple =: g (] ; (<params)"_)` /
  `g_simple =: g (0&{ , 1&{ , (<params)"_)`.

## J Idiom Review (jforc Readable Tacit Definitions — Ch 41)

Readability + `f.` (flatten adverb):
- Readability: group with spacing/parens, or split into named sub-verbs with
  comments. Splitting pollutes the namespace — use private assignment (`=.`)
  for helpers and `f.` on the public verb to inline them.
- **`f.`** replaces names in its operand with their definitions. Two perf
  effects: (1) removes per-rank name lookups (`abs1"0` 3.55s vs `abs1 f."0`
  3.01s — a named explicit verb is looked up for every atom); (2) restores
  interpreter special-code recognition (`#/.` 0.0158s vs `tally/.` 0.1547s —
  10x! `tally f./.` 0.0167s). Lesson: define verbs with proper rank; use J
  primitives directly so the interpreter can special-case compounds.
- `f.` has NO effect on explicit definitions (verified: `ex f.` still displays
  `3 : '| y'`; flattening a tacit verb displays the inlined body).

## J Idiom Review (jforc Explicit-To-Tacit Converter — Ch 42)

`13 :` auto-converts explicit definitions to tacit (monadic + dyadic; no
`14 :`; use `9!:3 (5)` for simplified display). Limits:
- **noun on the right of a dyadic verb** (`{ table`) is a `V V N` tine the
  converter can't emit; manual fix: right-bond `({&table)` (`({&n) y` = `y { n`).
- **conjunction-operand x/y** stays explicit (`13 : '+:^:x y'` → `4 :`; manual
  `+:@]^:[`).
- **assignments** via the `y [ s =. ...` pattern (`13 : '(s % t) [ s =. +/ y [ t =. #y'`
  → `+/ % #`).
- **modifier-operand routing** fails (`13 : '0 x} y'`) except exotic forms
  (`x 0:`[`]} y`, `x m&v y` = `m&v^:x y`, `x u&n y` = `u&n^:x y`).

**The fork-vs-At trap**: J's 3-verb trains are FORKS (`(f g h) y` =
`(f y) g (h y)`), NOT composition — so `+/"1 *: M` parses as `(+/"1 y) *: M`
(wrong); it needs At: `+/"1 @ (*: @ M)`. Same for `{: $ M` (must be
`{: @ $ @ M`) and `($ M) $ w` broadcast (use `($ @ M) $ w`; the bare hook
`($ M)` as a fork tine errors). Note: a `(f * )` (hook) vs `(f * g)` (fork) —
one extra paren flips the parse.

## J Idiom Review (jforc Parsing and Execution I — Ch 38)

Review-only (conceptual — execution internals):
- Conjunctions are executed before verbs, and only once: applying a modifier to
  operands produces an anonymous derived verb; the displayed sentence text is
  that verb's *pedigree* (how it was created), not a function list to re-run.
- Recognized compounds get customized interpreter routines (`-:&.^.` may run a
  different algorithm than `^ -: ^. y`) — a tacit rewrite can differ in
  *algorithm*, not just interpretive overhead. This validates verifying every
  tacit rewrite bit-exact vs the explicit form.
- Modifiers whose text refers to x/y are stored flagged; their text is
  interpreted only when the derived verb executes — the mechanism behind the
  Ch 30 "soporific" (referring-to-x/y) classification.

## J Idiom Review (jforc Parsing and Execution II — Ch 39)

Review-only (parsing mechanics). J parses/executes right-to-left with a
push-down stack: move the rightmost word onto the stack, check the top 4
against 9 executable patterns (Monad ×2, Dyad, Adverb, Conj, Fork, Hook/
Adverb, Is, Paren), execute the matching bident/trident fragment, repeat.
- **Conjunctions associate left-to-right**: the leftmost stack word is never a
  conjunction — a conjunction executes only in the third stack position. This
  is the ultimate source of the `-:&.^.` / `u&.>` chaining rule.
- Line 6 (Hook/Adverb): valid `CAVN CAVN` combos are `A A, C N, C V, N C,
  V C, V V`; `N A`/`V A` are adverbs (line 3); the rest are syntax errors.
- Bonded-fetch tines (`0&{`/`1&{`) are monadic-only (both `x (0&{) y` and
  `x (1&{) y` → valence error), and a `V V N` train with such a left tine does
  NOT form a fork/hook.

## J Idiom Review (jforc Common Mistakes — Ch 43, final)

Mechanics:
- **Adjacent numbers** form one list: `>: "0 1{y` fails (0 1 is one list) —
  use `>: "0 (1{y)` or `>: "0 ] 1{y` (gotcha 19).
- **Names in sequence do not form a list**: `0 y` errors even when `y`=1 —
  use `0 , y` (gotcha 20; use `,` everywhere).
- **Right-to-left desk calculator**: `5 - 2 + 1` = 2, not 4 (gotcha 0).
- **`{.`/`{:`/`}.`/`}:` mnemonic**: `{`=take, `}`=drop, single-dot=beginning,
  double-dot=end; `#.` atomic decode vs `#:` list encode.
- **Operand order `e.`/`i.`/`|`**: x control-like, y data-like; `x i. y` (x
  table), `x e. y` is backwards (y table), `x | y` = "x divides y" (gotcha 24).
- **Space before `:` alone**: `3:0`, `(+/%#):*`, `+/.*` are inflections.
- **`=:` in scripts**: keyboard `=.` gives a public assignment; scripts need `=:`.
- **No mixing `elseif.`/`else.`** in one structure.

Programming errors:
- **Dyad `;` asymmetry** — `x ; y` always boxes `x`, boxes `y` only if
  unboxed. `(<8) ; <chars` double-boxes the count and leaves chars unboxed.
  Build 2-box arg lists with `,`: `(<ne) , <slice` (gotcha 21). Note
  `<<0 ; 0 ; 0.95 ; 0.0` = `<< (0;0;0.95;0)` — the `;` chain binds the numbers
  first, then `<<` double-boxes.
- **Two operands to a monadic verb in a fork**: `x (* % +/) y` runs `x +/ y`
  (dyadic), not `+/ y` — use `x (* % +/@:]) y`.
- **`@:` unless `@` necessary** (rank) — `>@(n&{)` uses `@` but `n&{` is
  infinite-rank, so equivalent (gotcha 26).
- **Modifier greediness** — parenthesize compounds inside compounds:
  `+: @: 2&+ y` = `(+:@:2)&+ y` (illegal); `+: @: (2&+)` (gotcha 25).
- **A verb always executes on empty operands** (Ch 13) — broadcasts rely on
  the fill-cell fallback; still correct when the broadcast frame is 0.
- **`-:` doesn't check empty-operand type**: `(0 $ 0) -: (0 $ '')` → 1
  (shape-only); `{.` reveals. Verification vectors should be non-empty
  (gotcha 22).
- **`x u/ y` applies `u` to cells of `x`** (left rank of `u`) and whole `y`:
  `2 3 (+/) 10 20` → `12 22 13 23`; use `u"n/` to set x's cell rank (gotcha 23).
- **Modifier x/y has monadic+dyadic valences** — code the `:`-separated dyadic
  form explicitly (Ch 30 review).
- **Tacit code does not replace names by definitions** — `mean =: +/ % #` is a
  fork, not `+/ % # 1 2 3 4 5`; the paren-substitution mental model works.

---

# J Gotchas — distilled, project-agnostic

## Key J Gotchas — UNLEARN C THINKING

**J is NOT C. It is NOT Python. It is NOT any language you know.**
**Everything is right-to-left. There is NO operator precedence. There are NO loops at script level.**

0. **RIGHT-TO-LEFT, NO PRECEDENCE.** `*` does NOT bind tighter than `+`:
   `i*7+2` = `i * (7+2)`, not `(i*7)+2`. Always parenthesize compound
   expressions where you need specific grouping.

1. **`+.` IS NOT OR — IT'S A FORK.** `6 = vt +. 5 = vt` is a 3-fork; use
   `(6 = vt) +. (5 = vt) +. (4 = vt)`.

2. **ARITHMETIC WITH FUNCTION CALLS.** `ec =. (vo + 4 le64 b)` adds `vo` to the
   RESULT of `le64(4, b)`; `ec =. ((vo + 4) le64 b)` reads LE64 at offset `vo+4`.

3. **`":` + CONCATENATION — PUT FORMATTED VALUE LAST.** `echo ":i , ' [' , vtname , ']'`
   → domain error (`":` applies to whole concat). `s =. ": i; echo s , ' [' , vtname , ']'`.

4. **`> ((i*7) { ti)` — EXPLICIT PARENS FOR FETCH+UNBOX.** `dm =. > i*7+2 { ti`
   parses as `(> (i*7+2)) { ti`; use `dm =. > ((i*7+2) { ti)`.

5. **VERB APPLICATION — SEPARATE LIST CONSTRUCTION.** Build the argument list
   first, then apply the verb: `ti =. (<raw) , (<kv_end) , (<n_tensors)` then
   `ti =. parse_tensor_infos ti`.

6. **`;` IS ENCLOSEMENT, NOT STATEMENT SEPARATOR.** `a =. 5; b =. 10` makes `a` = `<5;10>`
   and leaves `b` undefined. Separate statements onto their own lines.

7. **`{` IS RIGHT-TO-LEFT.** `0 { 1 { AB` = `AB[1,0]`; `1 { 0 { AB` = `AB[0,1]`.

8. **`{"1` ON 2D ARRAYS EXTRACTS COLUMNS.** `h {"1 arr` = column `h`;
   `h { arr` = row `h`.

9. **CONTROL WORDS ONLY INSIDE EXPLICIT DEFINITIONS.** Script-level
   `while.`/`for.`/`try.`/`catch.`/`end.` → "invalid inflection". Must be inside
   `3 : 0` or `4 : 0`.

10. **STRING MATCHING — `i.` COMPARES ATOMS, NOT STRINGS.** `target i. names`
    finds each CHARACTER. Compare padded rows: `idx =. I. padded -: "1 arr`.

11. **`>=` AND `<=` PARSE AS FORKS.** Use `-. tid < 65536` (>=) and
    `-. r > cum` (<=) in `if.` conditions.

12. **`<"0` BOXES EACH ATOM; `<"1` BOXES WHOLE LISTS.** `dims =. <"0 1024 65536`;
    `<"1` boxes the whole list.

13. **NAN/INF CHECKING.** `echo +./ arr ~: arr` → true if any NaN;
    `echo +./ >./ arr > 1e300` → true if any Inf.

14. **FETCH `{::` — RIGHT ARG MUST BE VECTOR.** `0 {:: , <'hello'` works;
    `0 {:: <'hello'` → domain error (single box = scalar).

15. **`<a , b , c` DOES NOT BOX EACH.** `<` applies to the whole expression.
    `params =. <"0 flat` boxes each atom.

16. **`3 : 0` IS EXPLICIT — ARGUMENT IS `y`, NOT `x`.** In monadic `3 : 0`,
    `x` is UNDEFINED; `x` is the left arg only in dyadic `4 : 0`.

17. **AVOID SINGLE-CHAR VARIABLE NAMES.** `i` conflicts with builtin `i.`.
    Use `loop_idx` not `i`.

18. **ADJACENT NUMBERS FORM ONE LIST.** `>: "0 1{y` fails — `0 1` is one list.
    Use `>: "0 (1{y)` or `>: "0 ] 1{y`.

19. **NAMES IN SEQUENCE DO NOT FORM A LIST.** `0 y` is an error even if `y` is
    `1`. Use `0 , y`.

20. **DYAD `;` IS ASYMMETRIC.** `x ; y` always boxes `x`, but boxes `y` only if
    `y` is unboxed. `(<8) ; <chars` double-boxes the count and leaves `chars`
    unboxed. **To build a 2-box argument list, use `,` (comma):**
    `(<ne) , <slice`. (Ch 43.)
    Note `<<0 ; 0 ; 0.95 ; 0.0` = `<< (0 ; 0 ; 0.95 ; 0.0)` — the `;` chain
    binds the numbers first, then `<<` double-boxes.

21. **DYAD `-:` DOES NOT CHECK THE TYPE OF EMPTY OPERANDS.** `(0 $ 0) -: (0 $ '')`
    → `1` (shape-only match); `{.` would reveal the types. Verification vectors
    should be non-empty, so `-:` is safe there.

22. **`x u/ y` APPLIES `u` TO CELLS OF `x`** (cell rank = left rank of `u`) and
    the entire `y`. `2 3 (+/) 10 20` → `12 22 13 23`. Use `u"n/` to set `x`'s
    cell rank. Reductions (`+/`, `>./`) are monadic — no dyadic `u/` misuse.

23. **OPERAND ORDER `e.` / `i.` / `|`.** The dyad's `y` is data-like, `x` is
    control-like. `x i. y`: `x` is the table, `y` the looked-up values.
    `x e. y` is backwards (y is the table). `x | y` reads "x divides y" — the
    modulus is `x`.

24. **MODIFIER GREEDINESS — PARENTHESIZE COMPOUNDS INSIDE COMPOUNDS.** A modifier
    hoovers up everything until a left paren or unmodified verb. `+: @: 2&+ y`
    executes as `(+:@:2)&+ y` (illegal); write `+: @: (2&+)`. Parenthesize all
    compounds (`({&table) @ (0&(3!:4)) @ (>&(1&{))`).

25. **USE `@:` UNLESS `@` IS NECESSARY.** `u@:v` has infinite rank, `u@v` has
    `v`'s rank. Common `@` uses: `u@>` (open each box), `<@v` (box each cell).
    `>@(n&{)` uses `@` but `n&{` is infinite-rank, so equivalent.

26. **SCRIPT-LEVEL `=.` NAMES ARE TRANSIENT IN LOADED FILES.** In a `load`ed
    script, `name =. value` at script level creates a name that lasts only
    for the script's own execution: explicit definitions in the SAME script
    cannot see it (`value error` at call time — they resolve against the
    locale's permanent name table) and it is erased when the load finishes.
    Script-level `=:` creates the permanent global. Use `=:` for any script
    global that explicit defs reference. Interactively (not in a loaded script)
    `=.` at the console is fine.

27. **`x , y` ON DIFFERENT RANKS CONCATENATES, NOT BOXES.** `(3,6144) , (16,128,128)`
    gives a (17,128,6144) array, NOT a `<a ; b>` 2-box pair — to build a 2-box
    pair use `(<a) , <b`.

28. **`+/` REDUCES THE LEADING (FIRST) AXIS, NOT THE LAST.** Sum/Insert `+/`
    has rank `_ _ _` (applies to the whole array) and reduces the FIRST axis:
    `+/ t` where t is (2,3,4) → (3,4). To sum a trailing axis, use `+/"1`
    (rank 1 → reduce the last axis of each 1-cell) or `+/"2`. The C-mind
    assumes reduce-the-last; J Sum reduces the leading axis. (Useful for
    reducing a leading "tap" axis: `+/ (n, L, c)` over an (n,L,c) array sums
    the n taps directly, no transpose needed.)

29. **`$` ON A MATRIX RIGHT OPERAND IS RANK-DEPENDENT.** `(2 3) $ m` where m
    is (4,3) yields (2,3,3) — J applies Reshape at a per-cell rank, appending
    m's trailing axis, instead of reshaping the whole array. Flatten the
    right first: `(2 3) $ , m` gives (2,3).

## Numeric Representation Gotchas

- **NEVER DIY IEEE 754.** `3!:5` dyad: `_1(3!:5)` 4 chars → float32, `_2(3!:5)`
  8 chars → float64. `3!:4` dyad: `_1(3!:4)` 2 chars → **signed** int16
  (0xFFFF → `_1`), `0(3!:4)` 2 chars → **unsigned** uint16 (0xFFFF → 65535),
  `_2(3!:4)` 4 chars → uint32, `_3(3!:4)` 8 chars → uint64. All LE, no byte
  reversal.
- **CRITICAL: `3!:4/3!:5` return 1-element arrays, NOT scalars** — extract with
  `0 {`. `_3` as a bare number token next to `(3!:4)` is risky (`_ 3(3!:4)`
  negate-apply trap); parenthesize the type code. Tacit readers:
  `le16 =: 0 { (_1) 3!:4 ] { ~ 0 01 + [`, `le32 =: 0 { (_2) 3!:4 ] { ~ 0 1 2 3 + [`,
  `le64 =: 0 { (_3) 3!:4 ] { ~ (i.8) + [` (Ch 42 `13 :` output; ~1.5x faster
  than explicit `4 : 0` forms).
- **Scientific literals (Ch 19)**: `1e_5` is a string, not a number — use
  `1.0e_5` or `0.00001` (always a decimal point in scientific notation).

## Performance Idioms (Ch 35)

1. Built-in verbs (`+/` >> custom sum); avoid boxing; combine ops (`($,) y`);
   apply large verb-ranks to the biggest operands.
2. `/\.` faster than `/\`; `u/` and `u/\.` are compiled. `@.v` agenda avoids branches.
3. `3!:4/3!:5` for vectorized type conversion (seconds vs minutes).
4. `u&.>` (alias `each`) on boxed data — faster than `u"0`.
5. **In-place assignment** `name =. x m} name` — only fires for simple selectors
   on a refcount-1 noun; a **boxed selector on a local copies the whole array**
   (~151MB/call measured). It does NOT fire for rank-applied `x m}"r y` on a
   boxed item (unboxing a boxed global makes refcount 2 → the amend copies the
   whole layer at ~1.7GB/s). It DOES fire for `x m}"0 1 y` on a refcount-1 FLAT
   array (column amend ~18us/512 elems). Consequence: an in-place cache must be
   ONE flat array per kind with positions LEADING; per-layer boxes buy O(1)
   reads but force whole-layer copy writes.
6. `i.&1@:u` first-match; `m&i.` preinterpreted search; `s:` symbols for vocab
   (global table, no per-string free).
7. `x (<"1@[ { ]) y` unboxed x in fetch; `$, y` avoids ravel copy.

**Matrix product threading (hard-won)**: `+/ .*` parallelizes over N (output
columns), NOT M (rows) — and NOT at all for N=1. Measured on a 32-core box:
`(4096,1024) (+/ .*) (1024,1)` (N=1) ≈ 2 GFLOP/s (single-threaded,
memory-bound); N=64 ≈ 65 GFLOP/s; N=256 ≈ 145 GFLOP/s. Consequences:
- **M=1 generation matvecs are memory-bound** — the transposed orientation
  `(1,emb) (+/ .*) (emb,n)` parallelizes over N but reads the SAME weight bytes
  — no cold win. Batching (N≥4) is the lever.
- **Axis transposes**: `1 0 2 |:` (swap) is special-coded fast (~0.0002s/8MB);
  `1 2 0 |:` (cyclic permutation) is strided/slow (~0.0013s/8MB). Prefer the
  one-pass `1 2 0 |:` over the `1 0 2 |: + |:"2` two-pass (~4x slower), and
  `1 0 2 |:` over any two-swap route. Cyclic transpose `1 2 0 |:` measured
  ~1GB/s (strided) under load; the two-swap measured SLOWER — keep one-pass.
- **Cyclic boolean reshape is pathologically slow**: `(n_g,L,ctx) $ mask_2d`
  (cycling a 1MB boolean to 7.3MB) ≈ 5.4s at ctx=4096; the `(*/)` broadcast
  tile `2 0 1 |: (mask_2d (*/) (n_g $ 1))` ≈ 0.05s (~100x). Note `n_g # mask_2d`
  copies each ROW n_g times (t-major), NOT the r-major tile the group-major
  scores need — order matters.
- **`x ,"r y` appends along the FIRST axis of the rank-r cells**: for a 3D
  array `,"2` appends along axis 1, NOT the last axis. Appending a 2D slice to
  the last axis needs a 2D-reshape + `,` (a copy).
- **`x $ y` reshape with a rank-2 right does NOT ravel**: `(1 1 4) $ (1 4)`
  gives `1 1 4 4` (16 elems), not `1 1 4` — the reshape must see a 1D arg.
  Always ravel first: `(shape) $ , y` (the `$, y` idiom; free on contiguous).
- **List-gather `(base + i. n) { arr` is ~4-5GB/s, not memcpy**: the row-by-row
  indexed gather has per-row overhead (vs ~25GB/s for a contiguous `,`/`{.`
  copy).

**jpm profiling**: `load 'jpm'` (system `$JINSTALL/system/util/pm.ijs`, locale
`jpm`, NOT auto-loaded), `start_jpm_ ''` (SIZE=1e9), run, `showtotal_jpm_ ''`.
**NEVER call `stop_jpm_ ''` before `showtotal_jpm_ ''`** — it resets the
tracing buffer and `read` reports "no PM records"; `read` stops tracing itself
(start → run → showtotal).

## J Idiom Reference — where the deep material lives

This file holds the chapter-by-chapter idiom reviews (Ch 22-43) and the
distilled gotchas, kept project-agnostic. AGENTS.md points here. Deferred ideas
about applying an idiom to *this* codebase are surfaced in **PLAN.md**
("Deferred J-idiom applications").
- Ch 30 modifiers (locale-leak, soporific, playsound dispatch) → here
- Ch 35 performance & measurement → here (also "Performance Idioms" above)
- Ch 36 tacit accessors `>@(n&{)` → here
- Ch 37/39/40 forks, hooks, `@`/`@:`, `V V N`, dyadic-hook wrappers → here
- Ch 41 `f.` (flatten, special-code recognition) → here
- Ch 42 `13 :` explicit→tacit converter (limits: conjunction-operand x/y,
  assignments `y [ s =. ...`, modifier-operand routing, noun-on-right → manual
  right-bond `({&n)`) → here
- Dictionary verbs (`|:` transpose, `$` reshape, `m}` amend, `+/ .*`,
  `,:` itemize) → reviews here + ARCHITECTURE.md (RoPE, KV cache, attention)
