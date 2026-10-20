#!/bin/bash
# Phase B pilot (Qwen3-4B LoRA) + its evaluation on physical GPUs 4,6,7 with the H3 fleet paused for the duration.
# Run ONLY after the user has confirmed the fleet pause in the SHIGERU session. scripts/shigeru_run.py pauses the fleet
# (operator-confirmed procedure, job-boundary wait up to 10 min), runs the command, and restores the fleet on exit
# (success, failure, SIGINT, SIGTERM). Downtime is logged to results/fleet_downtime.log.
#   (setsid bash distill/run_467.sh < /dev/null > results/run_467.log 2>&1 &)
cd "$(dirname "$0")/.."
exec .venv/bin/python scripts/shigeru_run.py --gpus 4,6,7 --pause-fleet --drain-minutes 10 -- bash -c '
  set -e; export PYTHONPATH=.
  echo "=== $(date +%T) GPUs: $CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  CUDA_VISIBLE_DEVICES=4 .venv-train/bin/python -m distill.train_sft distill/configs/phaseB_pilot_4b.yaml > results/train_phaseB_pilot_4b.log 2>&1
  GPU=4 bash bench/eval_ar_model.sh runs/phaseB_pilot_4b/merged phaseB_pilot_4b
  GPU=6 bash bench/eval_ar_model.sh /mnt/raid6/models/hf/Qwen3-4B qwen3_4b_idle   # clean speed numbers on an idle GPU
'
