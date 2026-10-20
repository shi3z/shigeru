# GPU policy and fleet pause/restore (SHIGERU experiments)

## Hard rules (from the project brief)
* Physical GPUs **0 and 1 are never used** by SHIGERU: every GPU job runs with `CUDA_VISIBLE_DEVICES` restricted to a subset of
  physical 2–7 and `scripts/shigeru_run.py` refuses to start otherwise (it also re-checks `nvidia-smi` that no SHIGERU process
  is on 0/1 after start). After `CUDA_VISIBLE_DEVICES=2,3,4,5,6,7` the logical ids are 0–5 (`torchrun --nproc_per_node=6`).
* Services on GPUs 0/1, unrelated services, `killall`, indiscriminate `kill`, and GPU resets are forbidden.
* The H3 render fleet (ComfyUI members c4/c6/c7 on physical GPUs 4/6/7) is paused only for the duration of a SHIGERU GPU
  experiment and restored afterwards regardless of how the experiment ends (success, failure, SIGINT/SIGTERM, exception);
  only a fleet that was running when the experiment started is restored. Pause/restore times are logged.

## What actually holds the GPUs on this machine (2026-10-20)
| physical GPU | holder | notes |
|---|---|---|
| 0,1,2,3 | DeepSeek-V4.1 production server (`run_128k_4slots.sh`, `--devices 2,3,0,1`) + ASR 2.3 GB on 2 | **conflict**: the brief allows 2,3 but they are part of the 4-GPU production server, which also spans 0,1 (untouchable). Until the user decides, SHIGERU uses 4–7 only. |
| 4, 6, 7 | H3 fleet ComfyUI backends (41 GB each, ports 8194/8189/8188) + workers | paused per the procedure below |
| 5 | NEURAL RACER's live-commentary TTS/ASR (~10 GB, session donkeyhover-35: contact before any stop), hf_proxy (8.5 GB), Blender renders (varies; up to ~36 GB and 95% util seen) — removed from the fleet on 2026-10-06 | shared; never killed; usable VRAM varies 40-60 GB |

## Fleet anatomy (read from the scripts, not guessed)
* `/mnt/ssdraid/project/h3-fleet/start_fleet.sh` (idempotent): starts missing ComfyUI backends
  (`comfy-h3/.venv-exp/bin/python main.py --port 8194|8189|8188`, `CUDA_VISIBLE_DEVICES=4|6|7`), workers
  (`comfy_worker.py` with `H3_WORKER_ID=c4|c6|c7`, `COMFY_PORT`), and the watchdog `comfy_watchdog.sh`.
* `comfy_watchdog.sh` (5-minute loop) restarts a backend whose `/system_stats` fails and a worker that is not alive
  (requeueing its `claimed/<wid>__*.json` into `queue/` or `hold_norefs/`). **It skips a member while `maintenance/<wid>`
  exists** — the supported pause switch (`maintenance/c5` is how GPU 5 was removed).
* `thermal_guard.sh` (cron @reboot) SIGSTOPs/SIGCONTs workers and backends on heat; `remote_worker_guard.sh` (cron, 5 min)
  only manages remote members r1/r2. Neither starts local members on its own.
* The `h3-fleet` tmux session hosts a Claude session (fleet manager) — procedure confirmation requested from it on 2026-10-20.

## Review (2026-10-20): the procedure was cross-checked by the session living in the `h3-fleet` tmux window (donkeyhover-35, NEURAL RACER;
it reads the same scripts and is **not** the fleet's operator). Its corrections are applied below: (1) while `maintenance/<wid>`
exists the watchdog skips the member entirely, including its requeue branch, so claims of killed workers must be requeued by us
(same rule as the watchdog: no `refs` -> `hold_norefs/`, else `queue/`, written as `cut%04d.json`); (2) `comfy_worker.py` has no
SIGTERM handler, so a worker dies immediately and its claim stays in `claimed/`; (3) waiting for `claimed/` to drain does not
work (workers keep claiming), so pause = stop first, requeue second; (4) `maintenance/c5` must stay; (5) `thermal_guard.sh`
only SIGSTOP/SIGCONTs; `remote_worker_guard.sh` only manages remote r1/r2; the editor_server ControlNet may stop/restore the
c6 worker on its own. No one has yet named the fleet's operator; the user is asked to confirm or veto this procedure.

## Operator review (2026-10-20, session newh3-fd, the fleet's operator): four mandatory corrections, all applied
1. Requeue stranded claims by **moving** `claimed/<wid>__X.json` to `queue/X` unchanged (the file name is the job id that
   story_editor tracks; rewriting as `cut%04d.json` lost 132 jobs on 10-01); never overwrite an existing `queue/X`.
2. `thermal_guard.sh` may have SIGSTOPped a worker/backend (state T ignores SIGTERM): send SIGCONT, then SIGTERM, then
   SIGKILL after 5 s (workers) / 10 s (backends).
3. Restore order: run `start_fleet.sh` first, verify `/system_stats` and live workers, **then** remove `maintenance/c4,c6,c7`
   (removing first lets the 5-minute watchdog race `start_fleet.sh` on the same ports). `maintenance/c5` stays.
4. Run `start_fleet.sh` detached (`setsid … < /dev/null`) so a later Ctrl-C in the experiment's process group cannot kill the
   restored fleet.
Recommended and applied as an option: stop at a job boundary (`--drain-minutes N` waits until each claiming worker logs
`done in` / `FAILED` / `REQUEUED`). Caveats: the film_en editor_server ControlNet (`:8878`) can restart the c6 worker on GPU 6
by itself — the user should not use the webcam ControlNet during an experiment; remote members r1/r2 keep working.
Checks after pause: GPUs 4/6/7 ~0 MiB, no `claimed/c{4,6,7}__*`, queue count = before + requeued. After restore: 200 on
`:8194/:8189/:8188 /system_stats`, three live workers, claims reappearing within minutes.

## Pause procedure (`scripts/shigeru_run.py --fleet pause` implements it)
1. Record `nvidia-smi` and fleet state (backends up, workers alive, queue/claimed counts) to `results/fleet_state_<ts>.json`.
2. `touch maintenance/c4 maintenance/c6 maintenance/c7` so the watchdog does not restart anything.
3. (optional) wait for a job boundary; CONT+TERM(+KILL) the three workers; move their stranded claims to `queue/` (names
   unchanged, no overwrite); CONT+TERM(+KILL) the three backends; wait for the GPU memory on 4/6/7 to be released (< 1 GB).
4. Run the experiment with `CUDA_VISIBLE_DEVICES` restricted to the freed GPUs (+5).
5. Restore (always, via `atexit` + SIGINT/SIGTERM handlers): `setsid start_fleet.sh`, verify `/system_stats` on the three
   ports and three live workers, then remove the maintenance files; log the downtime.
6. Recovery after an OS crash: `scripts/shigeru_run.py --fleet recover` reads `results/fleet_pause_state.json` and
   restores the fleet if a pause was in progress.

### Operator notes added 2026-10-20 (after the confirmation)
* During `--drain-minutes`, if `thermal_guard.sh` pauses a worker (SIGSTOP) the job's "done in" line does not appear until
  the GPU cools below 73 °C; the drain then hits its time limit. A job cut off this way is not ingested and will be
  re-rendered later. This is accepted behaviour, not an error.
* A worker that claims a new job right after its "done in" line is normal; that claim is returned to `queue/` by the
  move-requeue step with its original name.
* Every pause/restore is logged in `results/fleet_downtime.log`; the operator reads that file when checking fleet health.
