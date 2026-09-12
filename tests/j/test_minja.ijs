NB. ================================================================
NB. test_minja.ijs — Phase 5A foundation tests for util/minja.ijs
NB. (Value model + Context). NOT yet wired into run_all_tests.sh:
NB. the J boxing/rank hardening of eq / in / ctx_set is still in
NB. progress (see TODO at the end of util/minja.ijs). Only the
NB. verified-passing cases are asserted here so it can be wired later.
NB. Goldens: minja's Python-repr semantics (cross-checked via
NB. scripts/minja_goldens.py against Python jinja2).
NB. ================================================================
coclass 'inference'

load './tests/j/test_harness.ijs'
load './util/minja.ijs'

NB. dump is defined in the `minja` locale — call with the _minja_ suffix.
NB. `":` returns rank-1 char lists; single-char literals are rank-0 scalars, so
NB. `-:` against a literal fails. ravel-normalize expected with streq.
init_counters ''

streq =: 4 : '(x -: , y)'

section_header 'Value repr / dump (Python-style, single quotes)'
assert_test (((dump_minja_ (mkint_minja_ 5) streq '5')) ; 'dump int 5')
assert_test (((dump_minja_ (mkfloat_minja_ 1.2) streq '1.2')) ; 'dump float 1.2')
assert_test (((dump_minja_ (mkbool_minja_ 1) streq 'True')) ; 'dump true -> True')
assert_test (((dump_minja_ (mkbool_minja_ 0) streq 'False')) ; 'dump false -> False')
assert_test (((dump_minja_ (mknull_minja_ '') streq 'null')) ; 'dump null')
assert_test ((('''abc''') -: (dump_minja_ (mkstr_minja_ 'abc'))) ; 'dump str single-quoted')
assert_test ((('[1, 2]') -: (dump_minja_ (mkarr_minja_ ((<mkint_minja_ 1) , (<mkint_minja_ 2))))) ; 'dump arr')
assert_test (((dump_minja_ (mkobj_minja_ ((<'a') , (<(mkstr_minja_ 'b'))))) streq '{''a'': ''b''}') ; 'dump dict Python-style')

section_header 'Value repr / dumpj (to_json, double quotes)'
assert_test (((dumpj_minja_ (mkstr_minja_ 'abc') streq '"abc"')) ; 'tojson str')
assert_test (((dumpj_minja_ (mkbool_minja_ 1) streq 'true')) ; 'tojson true')
assert_test (((dumpj_minja_ (mknull_minja_ '') streq 'null')) ; 'tojson null')
assert_test (((dumpj_minja_ (mkarr_minja_ ((<mkstr_minja_ 'a') , (<(mkfloat_minja_ 1.2)))) streq '["a", 1.2]')) ; 'tojson arr')
assert_test ((('{"a": "b"}') -: (dumpj_minja_ (mkobj_minja_ ((<'a') , (<(mkstr_minja_ 'b')))))) ; 'tojson dict')

section_header 'Object accessors'
o =. mkobj_minja_ (('x') pair_minja_ (mkint_minja_ 10))
assert_test (((dump_minja_ (('x') obj_get_minja_ o) streq '10')) ; 'obj_get existing key')
assert_test (((dump_minja_ (('missing') obj_get_minja_ o) streq 'null')) ; 'obj_get missing -> null')

section_header 'Context (single-level, no set)'
c =. mkctx_minja_ (mkobj_minja_ (('x') pair_minja_ (mkint_minja_ 10)))
assert_test (((dump_minja_ (('x') streq '10' ctx_get_minja_ c))) ; 'ctx_get existing')
assert_test (((dump_minja_ (('missing') streq 'null' ctx_get_minja_ c))) ; 'ctx_get missing -> null')
assert_test ((1 -: ('x') ctx_contains_minja_ c) ; 'ctx_contains existing')
assert_test ((0 -: ('missing') ctx_contains_minja_ c) ; 'ctx_contains missing')

section_header 'TODO (J boxing/rank hardening in progress)'
NB. eq numeric-tolerant, in membership, ctx_set nested-scope recursion,
NB. ctx_get parent-chain to builtins, to_bool/to_int edge cases — these
NB. hit the J numeric-payload-as-1-list and `;`/`,<`/`>`-open gotchas and
NB. are NOT yet asserted. See the TODO list at the end of util/minja.ijs.

show_summary 0
