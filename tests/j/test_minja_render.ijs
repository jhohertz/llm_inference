NB. ================================================================
NB. test_minja_render.ijs — Phase 5B+ render cases for the minja port.
NB. Wired into run_all_tests.sh. These assert the full expression
NB. grammar / tokenizer / AST / renderer (util/minja.ijs Phase 5B-5E).
NB.
NB. Cases are drawn 1:1 from the upstream oracle
NB. reference/minja/tests/test-syntax.cpp `render("...", {}, {})`
NB. EXPECT_EQ cases (59 testable ones; the 8 newline-containing cases
NB. are skipped because literal newlines cannot live in J strings).
NB. render =: (context) render (template string) — full engine.
NB. ================================================================
coclass 'inference'

load './tests/j/test_harness.ijs'
load './util/minja.ijs'

init_counters ''

NB. render is in the `minja` locale — call with the _minja_ suffix.
NB. `":` returns rank-1 char lists; `-:` needs ravel-normalized strings.
streq =: 4 : '(x -: , y)'

NB. assert_render (template ; expected) -> assert_test ((render_minja_ ...) streq exp)
assert_render =: 3 : 0
  'tmpl exp' =. y
  assert_test (((('' render_minja_ tmpl) streq exp)) ; tmpl)
)

NB. assert_throw (template ; expected_substr) -> assert_test (err contains substr)
assert_throw =: 3 : 0
  'tmpl sub' =. y
  try.
    '' render_minja_ tmpl
    assert_test ((<0) ; (tmpl , ' -> expected throw: ' , sub))
  catcht.
    assert_test ((((err_msg_g_minja_ i. sub) < # err_msg_g_minja_)) ; (tmpl , ' -> err: ' , err_msg_g_minja_))
  end.
)

NB. assert_render_opt (template ; opts ; expected) — opts = 't'/'l'/'tl'/'k'
assert_render_opt =: 3 : 0
  'tmpl opts exp' =. y
  assert_test ((((,('' render_opt_minja_ (tmpl ; opts))) -: ,exp)) ; (tmpl , ' [' , opts , ']'))
)

section_header 'Literals / expressions (upstream test-syntax.cpp)'
assert_render ('{{ 1 }}' ; '1')
assert_render ('{{ 1.2 }}' ; '1.2')
assert_render ('{{ true }}' ; 'True')
assert_render ('{{ false }}' ; 'False')
assert_render ('{{ ''abc'' }}' ; 'abc')
assert_render ('{{ not [] }}' ; 'True')
assert_render ('{{ [1] + [2, 3] }}' ; '[1, 2, 3]')
assert_render ('{{ ''ab'' * 3 }}' ; 'ababab')
assert_render ('{{ [1, 2, 3][-1] }}' ; '3')

section_header 'Tests / type checks'
assert_render ('{{ [] is iterable }}' ; 'True')
assert_render ('{{ [] is not number }}' ; 'True')
assert_render ('{{ 1 is not string }}' ; 'True')
assert_render ('{{ [] == none }}' ; 'False')
assert_render ('{{ {} == none }}' ; 'False')
assert_render ('{{ [1, 2, 3] == none }}' ; 'False')
assert_render ('{{ none == [] }}' ; 'False')
assert_render ('{{ none == {} }}' ; 'False')
assert_render ('{{ none == [1, 2, 3] }}' ; 'False')
assert_render ('{{ none != [] }}' ; 'True')

section_header 'Filters / methods'
assert_render ('{{ ''ok''.capitalize() }}' ; 'Ok')
assert_render ('{{ ''aBc'' | capitalize }}' ; 'Abc')
assert_render ('{{ ''AbC'' | lower }}' ; 'abc')
assert_render ('{{ ''me'' | upper }}' ; 'ME')
assert_render ('{{ ''hello world''.upper() }}' ; 'HELLO WORLD')
assert_render ('{{ ''MiXeD''.upper() }}' ; 'MIXED')
assert_render ('{{ ''''.upper() }}' ; '')
assert_render ('{{ ''HELLO WORLD''.lower() }}' ; 'hello world')
assert_render ('{{ ''MiXeD''.lower() }}' ; 'mixed')
assert_render ('{{ ''''.lower() }}' ; '')
assert_render ('{{ '' a  '' | trim }}' ; 'a')
assert_render ('{{ '' a ''.strip() }}' ; 'a')
assert_render ('{{ '' a ''.lstrip() }}' ; 'a ')
assert_render ('{{ '' a ''.rstrip() }}' ; ' a')
assert_render ('{{ ''abcXYZabc''.strip(''ac'') }}' ; 'bcXYZab')
assert_render ('{{ ''abcXYZabcXYZabc''.replace(''bc'', ''oui'') }}' ; 'aouiXYZaouiXYZaoui')
assert_render ('{{ ''abcXYZabcXYZabc''.replace(''abc'', ''ok'', 2) }}' ; 'okXYZokXYZabc')
assert_render ('{{ ''abcXYZabcXYZabc''.replace(''def'', ''ok'') }}' ; 'abcXYZabcXYZabc')
assert_render ('{{ foo | default(''the default'') }}{{ 1 | default(''nope'') }}' ; 'the default1')
assert_render ('{{ '''' | default(''the default'', true) }}{{ 1 | default(''nope'', true) }}' ; 'the default1')

section_header 'List filters / joins'
assert_render ('{{ [1, 2, 3] | join('', '') }}' ; '1, 2, 3')
assert_render ('{{ [1, 2, 3] | join('', '') + ''...'' }}' ; '1, 2, 3...')
assert_render ('{{ ''Tools: '' + [1, 2, 3] | reject(''equalto'', 2) | join('', '') + ''...'' }}' ; 'Tools: 1, 3...')
assert_render ('{{ ''Tools: '' + [1, 2, 3] | select(''equalto'', 2) | join('', '') + ''...'' }}' ; 'Tools: 2...')
assert_render ('{{ [1, False, 2, ''3'', 1, ''3'', False] | unique | list }}' ; '[1, False, 2, ''3'']')
NB. test-syntax.cpp 399-407: e/escape over & < > " accumulated across a loop.
assert_render (('{%- set res = [] -%}' , LF , '{%- for c in ["<", ">", "&", ''"'' ] -%}' , LF , '    {%- set _ = res.append(c | e) -%}' , LF , '{%- endfor -%}' , LF , '{{- res | join(", ") -}}') ; '&lt;, &gt;, &amp;, &#34;')

section_header 'Filter with post-expression'
assert_render ('{{ range(5) | length % 2 }}' ; '1')
assert_render ('{{ range(5) | length % 2 == 1 }},{{ [] | length > 0 }}' ; 'True,False')

section_header 'Calls / expansion'
assert_render ('{{ range(*[2,4]) | list }}' ; '[2, 3]')
assert_render ('{{ range(3) | list }}{{ range(4, 7) | list }}{{ range(0, 10, 2) | list }}' ; '[0, 1, 2][4, 5, 6][0, 2, 4, 6, 8]')
assert_render ('{{ ''a'' + [] | length | string + ''b'' }}' ; 'a0b')

section_header 'Loops / control (for / if / break / continue / set)'
assert_render ('{% for i in range(3) %}{{i}},{% endfor %}' ; '0,1,2,')
assert_render ('{% for i in range(10) %}{{ i }},{% if i == 2 %}{% break %}{% endif %}{% endfor %}' ; '0,1,2,')
assert_render ('{% for i in range(10) %}{% if i % 2 %}{% continue %}{% endif %}{{ i }},{% endfor %}' ; '0,2,4,6,8,')
assert_render ('{% for i in [true, false, 10, -10, 10.1, -10.1, None, ''a'', ''2'', {}, [1]] %}{{ i | int }}, {% endfor %}' ; '1, 0, 10, -10, 10, -10, 0, 0, 2, 0, 0, ')
assert_render ('{% set foo %}Hello {{ ''there'' }}{% endset %}{{ 1 ~ foo ~ 2 }}' ; '1Hello there2')
assert_render ('{% if 1 %}{% elif 1 %}{% else %}{% endif %}' ; '')

section_header 'Whitespace control'
assert_render ('{%- for i in range(0) -%}NAH{% else %}OK{% endfor %}' ; 'OK')
assert_render (' a {{  ''b'' -}} c ' ; ' a bc ')
assert_render (' a {{- ''b''  }} c ' ; ' ab c ')
assert_render ('  {% set _ = 1 %}    ' ; '      ')

section_header 'Filter blocks'
assert_render ('{% filter trim %} abc {% endfilter %}' ; 'abc')

section_header 'Slicing / subscripting'
assert_render ('{% set x = [0, 1, 2, 3] %}{{ x[1:] }}{{ x[:2] }}{{ x[1:3] }}' ; '[1, 2, 3][0, 1][1, 2]')
assert_render ('{% set x = ''0123'' %}{{ x[1:] }};{{ x[:2] }};{{ x[1:3] }};{{ x[:] }};{{ x[::] }}' ; '123;01;12;0123;0123')
assert_render ('{% set x = [0, 1, 2, 3] %}{{ x[::-1] }}{{ x[:0:-1] }}{{ x[2::-1] }}{{ x[2:0:-1] }}{{ x[::2] }}{{ x[::-2] }}{{ x[-2::-2] }}' ; '[3, 2, 1, 0][3, 2, 1][2, 1, 0][2, 1][0, 2][3, 1][2, 0]')
assert_render ('{% set x = ''0123'' %}{{ x[::-1] }};{{ x[:0:-1] }};{{ x[2::-1] }};{{ x[2:0:-1] }};{{ x[::2] }};{{ x[::-2] }};{{ x[-2::-2] }}' ; '3210;321;210;21;02;31;20')

section_header 'Newline-containing templates (LF)'
NB. J has no '\n' literal; newlines are the LF noun (10{a.) — concatenate like CRLF/CR.
NB. Comment blocks spanning newlines.
assert_render (('{# Hey',LF,'Ho #}{#- Multiline...',LF,'Comments! -#}{{ ''ok'' }}{# yo #}') ; 'ok')
NB. Whitespace control across newlines: {{- strips the preceding LF, -}} strips the following.
assert_render (('a',LF,'{{- ''b''  }}',LF,'c') ; ('ab',LF,'c'))
assert_render (('a',LF,'{{  ''b'' -}}',LF,'c') ; ('a',LF,'bc'))
NB. {% generation %} block.
assert_render ('{% generation %}Foo{% endgeneration %}' ; 'Foo')
NB. Tuple unpacking in for with whitespace control.
assert_render (('{%- for x, y in [("a", "b"), ("c", "d")] -%}',LF,'{{- x }},{{ y -}};',LF,'{%- endfor -%}') ; 'a,b;c,d;')
NB. tojson over mixed literals (incl. True/False duplicates, int-key dict).
assert_render (('{%- for x in [1, 1.2, "a", true, True, false, False, None, [], [1], [1, 2], {}, {"a": 1}, {1: "b"}] -%}',LF,'{{- x | tojson -}},',LF,'{%- endfor -%}') ; '1,1.2,"a",true,true,false,false,null,[],[1],[1, 2],{},{"a": 1},{"1": "b"},')
NB. set + string concat, whitespace-controlled.
assert_render (('{%- set user = "Olivier" -%}',LF,'{%- set greeting = "Hello " ~ user -%}',LF,'{{- greeting -}}') ; 'Hello Olivier')

section_header 'Namespace mutation across a for-loop (ctx_set_found)'
NB. {% set ns.x = ... %} inside a {% for %} must update the shared namespace
NB. so the post-loop {{ ns.x }} sees the accumulated value.
assert_render ('{% set ns = namespace(content = "") %}{% for x in [1, 2] %}{% set ns.content = ns.content + "a" %}{% endfor %}{{ ns.content }}' ; 'aa')

section_header 'Recursive macro + namespace accumulation (test-syntax.cpp 465)'
assert_render (('' , LF , '            {%- macro recursive(obj) -%}' , LF , '            {%- set ns = namespace(content = caller()) -%}' , LF , '            {%- for key, value in obj.items() %}' , LF , '                {%- if value is mapping %}' , LF , '                    {%- call recursive(value) -%}' , LF , '                        {{ ''\n\nclass '' + key.title() + '':\n'' }}' , LF , '                    {%- endcall -%}' , LF , '                {%- else -%}' , LF , '                    {%- set ns.content = ns.content + ''  '' + key + '': '' + value + ''\n'' -%}' , LF , '                {%- endif -%}' , LF , '            {%- endfor -%}' , LF , '            {{ ns.content }}' , LF , '            {%- endmacro -%}' , LF , '' , LF , '            {%- call recursive({"a": {"b": "1", "c": "2"}}) -%}' , LF , '            {%- endcall -%}' , LF , '        ') ; (LF , LF , 'class A:' , LF , '  b: 1' , LF , '  c: 2' , LF))

section_header 'Additional test-syntax.cpp cases (LF)'
NB. mojitos loop: inline if-expr + loop.first/index/last + whitespace control.
assert_render (('' , LF , '            {%- for x in range(3) -%}' , LF , '                {%- if loop.first -%}' , LF , '                    but first, mojitos!' , LF , '                {%- endif -%}' , LF , '                {{ loop.index }}{{ "," if not loop.last -}}' , LF , '            {%- endfor -%}') ; 'but first, mojitos!1,2,3')
NB. trim_tmpl default render (keep_trailing_newline=false strips final LF).
assert_render (('' , LF , '  {% if true %}Hello{% endif %}  ' , LF , '...' , LF , LF) ; (LF , '  Hello  ' , LF , '...' , LF))
NB. trailing LF of a final text node is stripped.
assert_render (('a' , LF , 'b' , LF) ; ('a' , LF , 'b'))
NB. {{- strips preceding whitespace; trim_blocks coincidentally matches here.
assert_render (('  ' , '{{- '' a' , LF , '' , '}}') ; (' a' , LF))

section_header 'Error substrings (test-syntax.cpp ThrowsWithSubstr)'
assert_throw ('{{ "" | items }}' ; 'Can only get item pairs from a mapping')
assert_throw ('{{ [] | items }}' ; 'Can only get item pairs from a mapping')
assert_throw ('{{ None | items }}' ; 'Can only get item pairs from a mapping')
assert_throw ('{% break %}' ; 'break outside of a loop')
assert_throw ('{% continue %}' ; 'continue outside of a loop')
assert_throw ('{%- set _ = [].pop() -%}' ; 'pop from empty list')
assert_throw ('{%- set _ = {}.pop() -%}' ; 'pop')
assert_throw ('{%- set _ = {}.pop(''foooo'') -%}' ; 'foooo')
assert_throw ('{% else %}' ; 'Unexpected else')
assert_throw ('{% endif %}' ; 'Unexpected endif')
assert_throw ('{% elif 1 %}' ; 'Unexpected elif')
assert_throw ('{% endfor %}' ; 'Unexpected endfor')
assert_throw ('{% endfilter %}' ; 'Unexpected endfilter')
assert_throw ('{% endmacro %}' ; 'Unexpected endmacro')
assert_throw ('{% endcall %}' ; 'Unexpected endcall')
assert_throw ('{% if 1 %}' ; 'Unterminated if')
assert_throw ('{% for x in 1 %}' ; 'Unterminated for')
assert_throw ('{% generation %}' ; 'Unterminated generation')
assert_throw ('{% if 1 %}{% else %}' ; 'Unterminated if')
assert_throw ('{% if 1 %}{% else %}{% elif 1 %}{% endif %}' ; 'Unterminated if')
assert_throw ('{% filter trim %}' ; 'Unterminated filter')
assert_throw ('{# ' ; 'Missing end of comment tag')
assert_throw ('{% macro test() %}' ; 'Unterminated macro')
assert_throw ('{% call test %}' ; 'Unterminated call')
assert_throw (('{%- macro test() -%}content{%- endmacro -%}' , '{%- call test -%}caller_content{%- endcall -%}') ; 'Invalid call block syntax - expected function call')

section_header 'Options: trim_blocks / lstrip_blocks (test-syntax.cpp)'
NB. opts: 't' trim_blocks, 'l' lstrip_blocks, 'tl' both.
assert_render_opt ('  {% set _ = 1 %}    {% set _ = 2 %}b' ; 'tl' ; '    b')
assert_render_opt ('{%- if True %}        {% set _ = x %}{%- endif %}{{ 1 }}' ; 'tl' ; '        1')
assert_render_opt (('    {% if True %}' , LF , '    {% endif %}') ; 'l' ; LF)
assert_render_opt (('    {% if True %}' , LF , '    {% endif %}') ; 'tl' ; '')
assert_render_opt (('    {% if True %}' , LF , '    {% endif %}') ; 't' ; '        ')
assert_render_opt ('  {% set _ = 1 %}    ' ; 'l' ; '    ')
assert_render_opt ('  {% set _ = 1 %}    ' ; 't' ; '      ')
assert_render_opt ('  {% set _ = 1 %}    ' ; 'tl' ; '    ')
assert_render_opt (('  ' , LF , '    {% set _ = 1 %}        ' , LF , '                ') ; 'l' ; ('  ' , LF , '        ' , LF , '                '))
assert_render_opt (('  ' , LF , '    {% set _ = 1 %}        ' , LF , '                ') ; 't' ; ('  ' , LF , '            ' , LF , '                '))
assert_render_opt (('  ' , LF , '    {% set _ = 1 %}        ' , LF , '                ') ; 'tl' ; ('  ' , LF , '        ' , LF , '                '))
assert_render_opt (('{% set _ = 1 %}' , LF , '  ') ; 'l' ; (LF , '  '))
assert_render_opt (('{% set _ = 1 %}' , LF , '  ') ; 't' ; '  ')
assert_render_opt (('{% set _ = 1 %}' , LF , '  ') ; 'tl' ; '  ')
NB. trim_tmpl (keep_trailing_newline=false default).
assert_render_opt (('' , LF , '  {% if true %}Hello{% endif %}  ' , LF , '...' , LF , LF) ; 't' ; (LF , '  Hello  ' , LF , '...' , LF))
assert_render_opt (('' , LF , '  {% if true %}Hello{% endif %}  ' , LF , '...' , LF , LF) ; 'l' ; (LF , 'Hello  ' , LF , '...' , LF))
assert_render_opt (('' , LF , '  {% if true %}Hello{% endif %}  ' , LF , '...' , LF , LF) ; 'tl' ; (LF , 'Hello  ' , LF , '...' , LF))
NB. {{- ' a\n'}} under trim_blocks.
assert_render_opt (('  ' , '{{- '' a' , LF , '' , '}}') ; 't' ; (' a' , LF))

show_summary 0
