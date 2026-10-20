"""Shared helpers for the Phase-1 benchmarks: prompt loading, GPU info, JSON output."""
import json, os, subprocess, time, torch

def load_prompts(path, ids=None):
    ps = json.load(open(path))
    return [p for p in ps if not ids or p["id"] in ids]

def gpu_info():
    i = torch.cuda.current_device()
    return {"name": torch.cuda.get_device_name(i), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__, "total_mem_gib": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}

def other_tenants():
    """memory used on this GPU by other processes (shared GPUs make numbers noisy): best effort via nvidia-smi"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout
        return out.strip().splitlines()
    except Exception:
        return None

def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print("wrote", path)

def summarize(runs, keys):
    agg = {}
    for k in keys:
        vals = [r[k] for r in runs if r.get(k) is not None]
        if vals:
            agg[k] = {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals), "n": len(vals)}
    return agg

import re
def extract_number(text):
    """last number in the text, preferring the one after '####'"""
    t = text.split('####')[-1] if '####' in text else text
    nums = re.findall(r'-?\d[\d,]*\.?\d*', t)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(',', '').rstrip('.'))
    except ValueError:
        return None

def score(prompt, answer_text):
    """returns True/False when the prompt carries a gold numeric answer, else None"""
    gold = prompt.get("answer")
    if gold is None:
        return None
    got = extract_number(answer_text)
    try:
        return got is not None and abs(got - float(gold)) < 1e-6
    except ValueError:
        return None

def accuracy(runs):
    sc = [r["correct"] for r in runs if r.get("correct") is not None]
    return {"n": len(sc), "accuracy": sum(sc) / len(sc)} if sc else None
