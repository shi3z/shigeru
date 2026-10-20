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
