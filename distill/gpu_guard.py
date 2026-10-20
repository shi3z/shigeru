"""GPU policy guard for SHIGERU training/inference processes (docs/gpu_policy.md, docs/distill_design.md §3).
Physical GPUs 0 and 1 are forbidden. Call check_env() before importing torch, and check_runtime() after the model is on
the GPU to verify from nvidia-smi that this process holds no memory on physical GPUs 0/1."""
import os, subprocess, sys

FORBIDDEN = {"0", "1"}


def check_env():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        sys.exit("CUDA_VISIBLE_DEVICES is unset: refusing to run (physical GPUs 0,1 are forbidden; set e.g. 4,5,6,7)")
    ids = [x.strip() for x in cvd.split(",") if x.strip()]
    bad = [x for x in ids if x in FORBIDDEN or x.startswith("GPU-")]
    if bad:
        sys.exit(f"CUDA_VISIBLE_DEVICES={cvd} includes forbidden or opaque ids {bad}; use physical indices among 2-7")
    return ids


def check_runtime(pid=None):
    """Return the list of physical GPU indices on which `pid` holds memory; raise if any is forbidden."""
    pid = pid or os.getpid()
    try:
        bus = subprocess.check_output(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"], text=True)
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id", "--format=csv,noheader"], text=True)
    except Exception as e:  # nvidia-smi unavailable: cannot verify, say so loudly
        print(f"[gpu_guard] cannot verify runtime GPU use: {e}", file=sys.stderr)
        return None
    idx = {}
    for line in bus.strip().splitlines():
        i, b = [x.strip() for x in line.split(",")]
        idx[b.upper()] = int(i)
    used = sorted({idx.get(b.strip().upper(), -1) for p, b in (l.split(",") for l in apps.strip().splitlines() if l.strip()) if int(p) == pid})
    if any(u in (0, 1) for u in used):
        raise SystemExit(f"[gpu_guard] process {pid} holds memory on forbidden physical GPU(s) {used}: aborting")
    return used
