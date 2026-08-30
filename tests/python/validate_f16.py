"""
Validate J f16_decode against Python struct-based reference.
Does the comparison in J for full float64 precision.
"""
import struct
import subprocess
import sys
import os
import math
import tempfile
import glob

BASE = os.path.dirname(os.path.abspath(__file__))      # tests/python/
PROJ = os.path.abspath(os.path.join(BASE, '..', '..')) # checkout root

def find_jconsole():
    """Discover the J runtime in $HOME (~/j9.x); override with $JCONSOLE."""
    if os.environ.get('JCONSOLE'):
        return os.environ['JCONSOLE']
    for pattern in ('~/j9.*', '~/j9*'):
        dirs = sorted(glob.glob(os.path.expanduser(pattern)))
        for d in reversed(dirs):
            if not os.path.isdir(d):
                continue
            for rel in ('jconsole.sh', 'bin/jconsole'):
                p = os.path.join(d, rel)
                if os.path.exists(p):
                    return p
    raise SystemExit('validate_f16: no J runtime under ~ (looked for ~/j9.* and ~/j9*)')

REF_FILE = os.path.join(BASE, 'uint16-to-float16-list.txt')  # beside this test
J_CONSOLE = find_jconsole()
J_PROJ = PROJ

def read_reference(path):
    """Read the reference file: uint16 -> expected float64"""
    ref = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            hex_val = int(parts[0])
            val_str = ''
            for p in parts[1:]:
                if p.strip():
                    val_str = p.strip()
                    break
            if not val_str:
                continue
            if 'nan' in val_str.lower():
                ref[hex_val] = float('nan')
            elif 'inf' in val_str.lower():
                ref[hex_val] = float(val_str)
            else:
                ref[hex_val] = float(val_str)
    return ref

def format_j_float(v):
    """Format a float for J output with full double precision"""
    if math.isnan(v):
        return '_. '
    if math.isinf(v):
        return '_' if v > 0 else '__'
    # Use repr for full precision
    s = repr(v)
    # Python uses e- for negative exponent, J uses e_
    s = s.replace('e-', 'e_').replace('E-', 'E_')
    # Python uses - for negative sign, J uses _
    if s.startswith('-'):
        s = '_' + s[1:]
    return s

def run_j_script(script, expected_file=None):
    """Run J with the given script and return stdout"""
    cmd = [J_CONSOLE]
    if expected_file:
        script_with_file = f'expected_file =. {chr(39)}{expected_file}{chr(39)}\n' + script
        result = subprocess.run(
            cmd,
            input=script_with_file,
            capture_output=True,
            text=True,
            timeout=300
        )
    else:
        result = subprocess.run(
            cmd,
            input=script,
            capture_output=True,
            text=True,
            timeout=300
        )
    return result.stdout, result.stderr

def main():
    print("Reading reference file...")
    ref = read_reference(REF_FILE)
    print(f"  Loaded {len(ref)} reference entries")
    
    # Write expected values to a temp file
    # One float per line, in J-readable format
    expected_file = tempfile.mktemp(suffix='.txt')
    with open(expected_file, 'w') as f:
        for i in range(65536):
            if i in ref:
                f.write(format_j_float(ref[i]) + '\n')
            else:
                f.write('0\n')
    
    print("Generating J validation script...")
    script = "load '" + J_PROJ + "/lib/gguf.ijs'\n"
    script += "cocurrent <'inference'\n"
    script += f"expected_text =. 1!:1 <{chr(39)}{expected_file}{chr(39)}\n"
    script += "expected =. \". (' ' (I. (expected_text = LF)) } expected_text)\n"
    script += "expected -: f16_table\n"
    
    print("Running J...")
    stdout, stderr = run_j_script(script, expected_file)
    
    # Clean up temp file
    try:
        os.unlink(expected_file)
    except:
        pass
    
    # Check for real errors (ignore if. spelling errors which are expected at script level)
    has_real_error = any(e in stderr for e in ['domain error', 'value error', 'index error', 'length error', 'type error'])
    if has_real_error:
        print(f"J errors:\n{stderr}")
        sys.exit(1)
    
    print("J output:")
    print(stdout.strip())
    
    # Parse result: the last line should be 1 (match) or 0 (mismatch)
    lines = [l.strip() for l in stdout.strip().split('\n') if l.strip()]
    if lines:
        last = lines[-1]
        try:
            result = int(last)
            return result == 1
        except ValueError:
            pass
    
    # Fallback: check for keywords
    if 'PASS' in stdout:
        return True
    if 'FAIL' in stdout:
        return False
    
    print("Could not determine result")
    return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
