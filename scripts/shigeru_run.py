#!/usr/bin/env python3
"""SHIGERU experiment manager: GPU policy enforcement + fleet pause/restore around a GPU job.

usage: shigeru_run.py --gpus 4,5,6,7 [--pause-fleet] [--drain-minutes 10] -- <command ...>
       shigeru_run.py --fleet status | pause | restore | recover
Rules: physical GPUs 0/1 are rejected; the fleet (H3 ComfyUI c4/c6/c7) is paused with the maintenance-file procedure from
docs/gpu_policy.md and restored on exit, SIGINT, SIGTERM or exception; state in results/fleet_pause_state.json; downtime logged
to results/fleet_downtime.log. Refuses --pause-fleet unless docs/fleet_procedure_confirmed exists (the fleet manager's OK)."""
import argparse, atexit, json, os, signal, subprocess, sys, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FLEET = "/mnt/ssdraid/project/h3-fleet"
MEMBERS = {"c4": (4, 8194), "c6": (6, 8189), "c7": (7, 8188)}
STATE = os.path.join(ROOT, "results", "fleet_pause_state.json")
DOWNTIME = os.path.join(ROOT, "results", "fleet_downtime.log")

def sh(cmd, timeout=60):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()

def gpu_mem():
    out = sh("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits")
    return {int(l.split(",")[0]): int(l.split(",")[1]) for l in out.splitlines() if l.strip()}

def backend_up(port):
    return subprocess.run(["curl", "-sf", "-m", "3", f"http://127.0.0.1:{port}/system_stats"], capture_output=True).returncode == 0

def worker_pids(port):
    pids = []
    for pid in sh("pgrep -f 'comfy_worker\\.py$'").split():
        try:
            env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
            if f"COMFY_PORT={port}".encode() in env:
                pids.append(int(pid))
        except OSError:
            pass
    return pids

def backend_pids(port):
    out = sh(f"ss -tlnp 2>/dev/null | grep '127.0.0.1:{port} '")
    import re
    return [int(p) for p in re.findall(r"pid=(\d+)", out)]

def fleet_status():
    st = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "members": {}, "queue": len(os.listdir(os.path.join(FLEET, "queue"))) if os.path.isdir(os.path.join(FLEET, "queue")) else None,
          "claimed": sorted(os.listdir(os.path.join(FLEET, "claimed"))) if os.path.isdir(os.path.join(FLEET, "claimed")) else None,
          "maintenance": sorted(os.listdir(os.path.join(FLEET, "maintenance"))), "gpu_mem_mib": gpu_mem()}
    for wid, (g, port) in MEMBERS.items():
        st["members"][wid] = {"gpu": g, "port": port, "backend_up": backend_up(port), "backend_pids": backend_pids(port), "worker_pids": worker_pids(port)}
    st["watchdog_pids"] = sh("ps -eo pid=,args= | awk '$2==\"bash\"&&$3~/comfy_watchdog\\.sh$/{print $1}'").split()
    st["running"] = any(m["backend_up"] or m["worker_pids"] for m in st["members"].values())
    return st

def log(msg):
    os.makedirs(os.path.dirname(DOWNTIME), exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True); open(DOWNTIME, "a").write(line + "\n")

def pause_fleet(drain_minutes):
    st = fleet_status()
    json.dump({"paused_at": time.time(), "was_running": st["running"], "status_before": st}, open(STATE, "w"), indent=1)
    if not st["running"]:
        log("fleet pause: fleet was not running; nothing to pause (no restore will be done)"); return
    log(f"fleet pause: begin (queue={st['queue']}, claimed={len(st['claimed'] or [])})")
    for wid in MEMBERS:
        open(os.path.join(FLEET, "maintenance", wid), "w").write(f"paused by shigeru_run.py {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    if drain_minutes > 0:
        # job-boundary wait (operator's recommendation): a worker finishing a job appends "done in" / "FAILED" / "REQUEUED" to
        # worker_p<port>.log; wait (bounded) until every member with a claim has logged a job end since the pause began
        marks = {}
        for wid, (g, port) in MEMBERS.items():
            lp = os.path.join(FLEET, f"worker_p{port}.log"); marks[wid] = os.path.getsize(lp) if os.path.exists(lp) else 0
        deadline = time.time() + 60 * drain_minutes
        while time.time() < deadline:
            pending = []
            for wid, (g, port) in MEMBERS.items():
                has_claim = any(c.startswith(wid + "__") for c in (fleet_status()["claimed"] or []))
                lp = os.path.join(FLEET, f"worker_p{port}.log")
                ended = False
                if os.path.exists(lp):
                    with open(lp, "rb") as f:
                        f.seek(marks[wid]); tail = f.read().decode("utf-8", "replace")
                    ended = any(k in tail for k in ("done in", "FAILED", "REQUEUED"))
                if has_claim and not ended: pending.append(wid)
            if not pending: break
            log(f"fleet pause: waiting for a job boundary on {pending}"); time.sleep(20)
    def stop(pids, what, grace):
        for pid in pids:
            try: os.kill(pid, signal.SIGCONT)   # thermal_guard may have SIGSTOPped it; a stopped process ignores SIGTERM
            except OSError: pass
            try: os.kill(pid, signal.SIGTERM); log(f"fleet pause: CONT+TERM {what} pid {pid}")
            except OSError: pass
        t_end = time.time() + grace
        while time.time() < t_end and any(os.path.exists(f"/proc/{pid}") for pid in pids):
            time.sleep(1)
        for pid in pids:
            if os.path.exists(f"/proc/{pid}"):
                try: os.kill(pid, signal.SIGKILL); log(f"fleet pause: KILL {what} pid {pid} (did not exit in {grace}s)")
                except OSError: pass
    for wid, (g, port) in MEMBERS.items():
        stop(worker_pids(port), f"worker {wid}", 5)
    # the watchdog skips members under maintenance, so its requeue branch never runs: put stranded claims back ourselves.
    # Fleet operator's rule (2026-10-20): move claimed/<wid>__X.json to queue/X unchanged (the file name is the job id that
    # story_editor tracks); never overwrite an existing queue/X (film_en and space_detective_v2 share cut numbers).
    for wid in MEMBERS:
        for c in sorted(os.listdir(os.path.join(FLEET, "claimed"))):
            if not c.startswith(wid + "__"):
                continue
            src, dst = os.path.join(FLEET, "claimed", c), os.path.join(FLEET, "queue", c[len(wid) + 2:])
            if os.path.exists(dst):
                log(f"fleet pause: NOT requeuing {c}: queue/{c[len(wid)+2:]} already exists (left in claimed/)"); continue
            try:
                os.rename(src, dst); log(f"fleet pause: requeued {c} -> queue/{c[len(wid)+2:]}")
            except OSError as e:
                log(f"fleet pause: could not requeue {c}: {e}")
    for wid, (g, port) in MEMBERS.items():
        stop(backend_pids(port), f"backend {wid} (gpu {g})", 10)
    for _ in range(40):
        mem = gpu_mem()
        if all(mem.get(g, 0) < 1024 for g, _ in MEMBERS.values()):
            break
        time.sleep(3)
    log(f"fleet pause: done, gpu mem now {gpu_mem()}")

def restore_fleet():
    if not os.path.exists(STATE):
        log("fleet restore: no pause state, nothing to do"); return
    s = json.load(open(STATE))
    if not s.get("was_running"):
        log("fleet restore: fleet was not running before the pause; leaving it stopped"); os.remove(STATE); return
    # operator's order: start_fleet.sh first (detached with setsid so a later Ctrl-C cannot kill the restored fleet),
    # verify, THEN remove the maintenance files (otherwise the 5-minute watchdog may race start_fleet.sh on the same ports)
    log("fleet restore: setsid start_fleet.sh")
    with open(os.path.join(ROOT, "results", "fleet_restore.log"), "a") as lf:
        subprocess.run(["setsid", "bash", os.path.join(FLEET, "start_fleet.sh")], cwd=FLEET, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, timeout=900)
    ok = False
    for _ in range(60):
        st = fleet_status()
        if all(m["backend_up"] and m["worker_pids"] for m in st["members"].values()):
            ok = True; break
        time.sleep(5)
    for wid in MEMBERS:   # never touch maintenance/c5 (GPU 5 was removed from the fleet on purpose)
        p = os.path.join(FLEET, "maintenance", wid)
        if os.path.exists(p): os.remove(p)
    down = time.time() - s["paused_at"]
    log(f"fleet restore: {'OK' if ok else 'INCOMPLETE'} after {down/60:.1f} min of downtime; status {json.dumps({w: (m['backend_up'], len(m['worker_pids'])) for w, m in fleet_status()['members'].items()})}")
    os.remove(STATE)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="4,5,6,7", help="physical GPU ids for CUDA_VISIBLE_DEVICES (0/1 rejected)")
    ap.add_argument("--pause-fleet", action="store_true"); ap.add_argument("--drain-minutes", type=float, default=10)
    ap.add_argument("--fleet", choices=["status", "pause", "restore", "recover"])
    ap.add_argument("cmd", nargs="*")
    a = ap.parse_args()
    if a.fleet:
        if a.fleet == "status": print(json.dumps(fleet_status(), indent=1)); return
        if a.fleet == "pause": pause_fleet(a.drain_minutes); return
        if a.fleet in ("restore", "recover"): restore_fleet(); return
    gpus = [int(g) for g in a.gpus.split(",")]
    if any(g in (0, 1) for g in gpus):
        sys.exit("refusing: physical GPUs 0/1 are not allowed for SHIGERU")
    if a.pause_fleet and not os.path.exists(os.path.join(ROOT, "docs", "fleet_procedure_confirmed")):
        sys.exit("refusing: fleet pause procedure not yet confirmed by the fleet manager (docs/fleet_procedure_confirmed missing)")
    if not a.cmd:
        sys.exit("no command given")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "gpus": gpus, "gpu_mem_before": gpu_mem(), "fleet": fleet_status()}, open(os.path.join(ROOT, "results", f"gpu_state_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)
    if a.pause_fleet:
        pause_fleet(a.drain_minutes)
        atexit.register(restore_fleet)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda s, f: sys.exit(128 + s))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)))
    log(f"run: GPUs {gpus}: {' '.join(a.cmd)}")
    proc = subprocess.Popen(a.cmd, env=env, cwd=ROOT)
    try:
        rc = proc.wait()
    except SystemExit:
        proc.terminate(); proc.wait(30); raise
    log(f"run: exit {rc}")
    sys.exit(rc)

if __name__ == "__main__":
    main()
