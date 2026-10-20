#!/usr/bin/env python3
"""HumanEval / MBPP(sanitized) pass@1 (greedy) for any engine adapter in bench/engines.py, executing tests in a subprocess sandbox.
usage: run_code_eval.py --engine ours:0.9:graph --task humaneval [--n 164] --out results/humaneval_ours.json"""
import argparse, json, os, re, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_json
from engines import load

def extract_code(text, entry_point=None):
    m = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S)
    code = max(m, key=len) if m else text
    return code

def run_tests(program, timeout=15):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program); path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout, text=True)
        return p.returncode == 0, (p.stderr or "")[-300:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True); ap.add_argument("--task", choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--n", type=int, default=None); ap.add_argument("--gen-length", type=int, default=512); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if a.task == "humaneval":
        items = [json.loads(l) for l in open(os.path.join(root, "data", "HumanEval.jsonl"))]
        tasks = [{"id": it["task_id"], "prompt": f"Complete the following Python function. Reply with only the complete function in a ```python block.\n\n```python\n{it['prompt']}```",
                  "test": it["prompt"] + "{code}\n\n" + it["test"] + f"\n\ncheck({it['entry_point']})\n", "entry": it["entry_point"], "header": it["prompt"]} for it in items]
    else:
        items = json.load(open(os.path.join(root, "data", "mbpp_sanitized.json")))
        tasks = [{"id": f"mbpp/{it['task_id']}", "prompt": f"{it['prompt']}\nYour code should pass these tests:\n" + "\n".join(it["test_list"]) + "\nReply with only the code in a ```python block.",
                  "test": "\n".join(it.get("test_imports", [])) + "\n{code}\n\n" + "\n".join(it["test_list"]) + "\n", "entry": None, "header": ""} for it in items]
    if a.n: tasks = tasks[:a.n]
    gen, info = load(a.engine, gen_length=a.gen_length)
    results = []; t_all = time.time()
    for i, t in enumerate(tasks):
        t0 = time.time(); text = gen(t["prompt"]); dt = time.time() - t0
        code = extract_code(text)
        if a.task == "humaneval" and f"def {t['entry']}" not in code:
            code = t["header"] + code   # model returned only the body
        program = t["test"].replace("{code}", code if a.task == "mbpp" or f"def {t['entry']}" in code else t["header"] + code)
        if a.task == "humaneval" and f"def {t['entry']}" in code:
            program = code + "\n\n" + t["test"].split("{code}")[1]
        ok, err = run_tests(program)
        results.append({"id": t["id"], "pass": ok, "secs": round(dt, 1), "err": err if not ok else "", "answer": text[-400:]})
        print(f"[{i:3d}] {t['id']:14s} {'PASS' if ok else 'fail'} {dt:5.1f}s", flush=True)
    n_pass = sum(r["pass"] for r in results)
    res = {"task": a.task, **info, "n": len(results), "pass@1": n_pass / len(results), "seconds": time.time() - t_all, "results": results}
    save_json(a.out, res)
    print(f"=== {a.task} {a.engine}: pass@1 = {n_pass}/{len(results)} = {100*n_pass/len(results):.1f}% in {res['seconds']:.0f}s")

if __name__ == "__main__":
    main()
