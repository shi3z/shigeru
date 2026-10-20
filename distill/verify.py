"""Verifiers for teacher outputs: math (numeric comparison with a gold answer), code (tests run in a subprocess
sandbox), JSON (parser + required keys). Each returns (status, detail)."""
import json, os, re, subprocess, sys, tempfile

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_final_number(text):
    m = re.findall(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if m:
        return m[-1].replace(",", "")
    m = re.findall(r"\\boxed\{\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if m:
        return m[-1].replace(",", "")
    nums = NUM.findall(text)
    return nums[-1].replace(",", "") if nums else None


def verify_math(response, gold):
    pred = extract_final_number(response)
    if pred is None:
        return "verified_wrong", "no number found"
    try:
        ok = abs(float(pred) - float(str(gold).replace(",", ""))) < 1e-6
    except ValueError:
        return "verified_wrong", f"unparsable {pred!r}"
    return ("verified_correct" if ok else "verified_wrong"), f"pred={pred} gold={gold}"


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len)
    return text


def run_sandbox(program, timeout=20):
    """Run a Python program in a temp dir with a clean environment and a time limit."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "HOME": d}
        try:
            p = subprocess.run([sys.executable, "-I", path], capture_output=True, timeout=timeout, text=True, cwd=d, env=env)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return p.returncode == 0, (p.stderr or p.stdout)[-800:]


def verify_code(response, tests, timeout=20):
    code = extract_code(response)
    if not code.strip():
        return "verified_fail", "no code"
    program = code + "\n\n" + "\n".join(tests) + "\nprint('ALL_TESTS_PASSED')\n"
    ok, out = run_sandbox(program, timeout)
    if ok and "ALL_TESTS_PASSED" in out or ok:
        return "verified_pass", "ok"
    return "verified_fail", out


def extract_json(text):
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    s = m.group(1) if m else text
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        i, j = s.find("{"), s.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(s[i:j + 1])
            except json.JSONDecodeError:
                return None
    return None


def verify_json(response, required_keys=()):
    obj = extract_json(response)
    if obj is None:
        return "verified_fail", "not JSON"
    if not isinstance(obj, dict):
        return "verified_fail", "not an object"
    missing = [k for k in required_keys if k not in obj]
    if missing:
        return "verified_fail", f"missing keys {missing}"
    return "verified_pass", "ok"
