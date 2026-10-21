#!/usr/bin/env python3
"""Extract success cases from reference/minja/tests/test-syntax.cpp -> minja_cases.json + minja_cases.ijs.

Handles: normal strings w/ escapes, raw strings R"(...)" / R"delim(...)delim",
multi-line string concatenation, variable templates (trim_tmpl), C++ json
bindings literals ({{...}} objects, {...}/[...] arrays, json(), json::array(),
json::object(), json::parse(R"(...)")), and options mapping.
"""
import re, json, sys

SRC = open('reference/minja/tests/test-syntax.cpp').read()

# ---- option names -> J opts string ----
def opts_to_j(name):
    name = name.strip()
    if name == '{}': return ''
    if name == 'lstrip_blocks': return 'l'
    if name == 'trim_blocks': return 't'
    if name == 'lstrip_trim_blocks': return 'tl'
    return None

# ---- string literal parsing ----
def parse_raw(s, i):
    """s[i:i+2] == 'R"'. Find raw string (ends at first )delim"). Return (content, end_index)."""
    lp = s.find('(', i + 2)
    if lp < 0: return '', len(s)
    delim = s[i + 2: lp]
    close = s.find(')' + delim + '"', lp + 1)
    if close < 0: return s[lp + 1:], len(s)
    return s[lp + 1: close], close + len(delim) + 2

def parse_string(s, i):
    """Parse consecutive C++ string literals starting at s[i]. Return (value, i)."""
    val = []
    n = len(s)
    while True:
        while i < n and s[i].isspace(): i += 1
        if i >= n: break
        if s[i:i+2] == 'R"':
            content, i = parse_raw(s, i)
            val.append(content); continue
        if s[i] == '"':
            j = i + 1; buf = []
            while j < n:
                if s[j] == '\\':
                    nx = s[j+1]
                    if nx == 'n': buf.append('\n')
                    elif nx == 't': buf.append('\t')
                    elif nx == 'r': buf.append('\r')
                    elif nx == '\\': buf.append('\\')
                    elif nx == "'": buf.append("'")
                    elif nx == '"': buf.append('"')
                    elif nx == '0': buf.append('\0')
                    elif nx == 'x':
                        buf.append(chr(int(s[j+2:j+4], 16))); j += 4; continue
                    elif nx == 'u':
                        buf.append(chr(int(s[j+2:j+6], 16))); j += 6; continue
                    else: buf.append(nx)
                    j += 2; continue
                if s[j] == '"': break
                buf.append(s[j]); j += 1
            val.append(''.join(buf)); i = j + 1; continue
        break
    return ''.join(val), i

def skip_string(s, i):
    """Skip a single C++ string literal (normal or raw). Return index after."""
    if s[i:i+2] == 'R"':
        _, end = parse_raw(s, i)
        return end
    j = i + 1
    while j < len(s):
        if s[j] == '\\': j += 2; continue
        if s[j] == '"': return j + 1
        j += 1
    return len(s)

def find_matching(s, i):
    """s[i] is '(' -> find matching ')', string-aware, tracking (){}[]. Return index after close."""
    depth = 0; j = i
    while j < len(s):
        c = s[j]
        if c == '"' or s[j:j+2] == 'R"':
            j = skip_string(s, j); continue
        if c in '([{': depth += 1
        elif c in ')]}':
            depth -= 1
            if depth == 0: return j + 1
        j += 1
    return -1

def split_top(s, sep=','):
    """Split s on sep at depth 0, string-aware (track (){}[])."""
    parts = []; depth = 0; cur = []; i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == '"' or s[i:i+2] == 'R"':
            j = skip_string(s, i); cur.append(s[i:j]); i = j; continue
        if c in '([{': depth += 1
        elif c in ')]}': depth -= 1
        if c == sep and depth == 0:
            parts.append(''.join(cur)); cur = []; i += 1; continue
        cur.append(c); i += 1
    parts.append(''.join(cur))
    return parts

# ---- C++ json literal -> Python object (json.dumps ONCE at call site) ----
def parse_json_lit(s):
    s = s.strip()
    if s.startswith('json::parse('):
        inner = s[len('json::parse('):-1].strip()
        if inner.startswith('R"'):
            content, _ = parse_raw(inner, 0)
            return json.loads(content.strip())
        return json.loads(inner)
    if s.startswith('json::object()'):
        return {}
    if s.startswith('json::array('):
        inner = s[len('json::array('):-1].strip()
        return parse_json_lit(inner)
    if s.startswith('json('):
        inner = s[len('json('):-1].strip()
        return parse_json_lit(inner)
    # {{...}} object
    if s.startswith('{{'):
        j = find_matching_brace(s, 0)
        if j < 0: return {}
        body = s[2:j-2]
        items = split_top(body)
        obj = {}
        k = 0
        while k + 1 < len(items):
            key = parse_json_lit(items[k])
            val = parse_json_lit(items[k+1])
            obj[key] = val
            k += 2
        return obj
    # {...} array OR { {pair}, ... } object (nlohmann: list of pairs -> object)
    if s.startswith('{'):
        j = find_matching_brace(s, 0)
        if j < 0: return []
        body = s[1:j-1]
        items = split_top(body)
        is_obj = True
        for it in items:
            it2 = it.strip()
            if it2.startswith('{{'):       # nested object value -> not a pair
                is_obj = False; break
            if it2.startswith('{'):
                bj = find_matching_brace(it2, 0)
                if bj < 0: is_obj = False; break
                sub = split_top(it2[1:bj-1])
                if len(sub) >= 2 and sub[0].strip().startswith('"'):
                    continue               # {key, value} pair
            is_obj = False; break
        if is_obj:
            obj = {}
            for it in items:
                it2 = it.strip()
                bj = find_matching_brace(it2, 0)
                sub = split_top(it2[1:bj-1])
                obj[parse_json_lit(sub[0])] = parse_json_lit(sub[1])
            return obj
        return [parse_json_lit(x) for x in items]
    # [...] array
    if s.startswith('['):
        j = find_matching_bracket(s, 0)
        if j < 0: return []
        body = s[1:j-1]
        items = split_top(body)
        return [parse_json_lit(x) for x in items]
    # string
    if s.startswith('"'):
        m = re.match(r'"(?:\\.|[^"\\])*"', s, re.S)
        if m:
            raw = m.group(0)[1:-1]
            return re.sub(r'\\(.)', lambda mm: {'n':'\n','t':'\t','r':'\r','\\':'\\','"':'"','/':'/'}.get(mm.group(1), mm.group(1)), raw)
        return s.strip('"')
    # number / true / false / null
    t = s.strip()
    if t == 'true': return True
    if t == 'false': return False
    if t == 'null' or t == 'None': return None
    try:
        if re.fullmatch(r'-?\d+', t): return int(t)
        if re.fullmatch(r'-?\d+\.\d+', t): return float(t)
    except Exception: pass
    return t

def find_matching_brace(s, i):
    """s[i] starts '{' (or '{{') -> find matching close (string-aware, nested)."""
    depth = 0; j = i
    n = len(s)
    while j < n:
        c = s[j]
        if c == '"' or s[j:j+2] == 'R"':
            j = skip_string(s, j); continue
        if c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0: return j + 1
        j += 1
    return -1

def find_matching_bracket(s, i):
    depth = 0; j = i; n = len(s)
    while j < n:
        c = s[j]
        if c == '"' or s[j:j+2] == 'R"':
            j = skip_string(s, j); continue
        if c == '[': depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0: return j + 1
        j += 1
    return -1

# ---- variable template scan ----
var_tmpl = {}
def scan_vars():
    global SRC
    for m in re.finditer(r'std::string\s+(\w+)\s*=', SRC):
        name = m.group(1); i = m.end()
        val, i = parse_string(SRC, i)
        var_tmpl[name] = val

# ---- main extraction ----
def extract():
    scan_vars()
    cases = []
    # find EXPECT_EQ(...) blocks
    pos = 0
    n = len(SRC)
    while True:
        m = re.search(r'EXPECT_EQ\s*\(', SRC[pos:])
        if not m: break
        start = pos + m.start()
        # find matching ) for EXPECT_EQ
        lparen = pos + m.end() - 1
        j = find_matching(SRC, lparen)
        if j < 0:
            pos = start + 1; continue
        body = SRC[lparen + 1: j - 1]
        # split top-level -> expected, render(...)
        parts = split_top(body)
        if len(parts) < 2: pos = start + 1; continue
        expected_s = parts[0]
        render_s = parts[1]
        # expected value
        exp, _ = parse_string(expected_s, 0)
        # find render( in render_s
        rm = re.search(r'\brender\s*\(', render_s)
        if not rm: pos = start + 1; continue
        rlparen = rm.end() - 1
        rj = find_matching(render_s, rlparen)
        if rj < 0: pos = start + 1; continue
        rbody = render_s[rlparen + 1: rj - 1]
        rparts = split_top(rbody)
        if len(rparts) < 3: pos = start + 1; continue
        tmpl_s = rparts[0]
        bind_s = rparts[1].strip()
        opts_s = rparts[2].strip()
        # template value (resolve variable templates like trim_tmpl)
        tmpl, _ = parse_string(tmpl_s, 0)
        if not tmpl and tmpl_s.strip() in var_tmpl:
            tmpl = var_tmpl[tmpl_s.strip()]
        # bindings
        bindings = json.dumps(parse_json_lit(bind_s), ensure_ascii=False) if bind_s != '{}' else ''
        # opts
        opts = opts_to_j(opts_s)
        if opts is None: opts = ''
        cases.append({'template': tmpl, 'opts': opts, 'expected': exp, 'bindings': bindings})
        pos = start + 1
    return cases

def main():
    cases = extract()
    print('cases:', len(cases))
    with open('/tmp/minja_cases.json', 'w') as f:
        json.dump(cases, f, indent=1, ensure_ascii=False)
    # generate .ijs
    gen_ijs(cases)
    # report binding cases
    for c in cases:
        if c['bindings']:
            print('BIND:', repr(c['template']), '->', c['bindings'], '| exp=', repr(c['expected']))

def gen_ijs(cases):
    def js(s):
        # J string literal with '' for quote; newlines -> , LF , parts
        parts = s.split('\n')
        qparts = []
        for p in parts:
            qparts.append("'" + p.replace("'", "''") + "'")
        if len(qparts) == 1:
            return qparts[0]
        return '(' + ' , LF , '.join(qparts) + ')'
    arr = []
    for c in cases:
        arr.append(js(c['template']))
        arr.append(js(c['opts']))
        arr.append(js(c['expected']))
        arr.append(js(c['bindings']))
    lines = ['coclass \'inference\'', 'CASES =: ' + ' ; '.join(arr)]
    content = '\n'.join(lines) + '\n'
    with open('/tmp/minja_cases.ijs', 'w') as f:
        f.write(content)
    with open('/home/jhohertz/co/llm_inference/tests/j/minja_cases.ijs', 'w') as f:
        f.write(content)

if __name__ == '__main__':
    main()
