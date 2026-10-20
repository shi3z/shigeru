#!/bin/bash
# Phase B pilot (Qwen3-4B LoRA) + evaluation + idle-GPU speed baseline on physical GPUs 4,6,7.
# The user stops the H3 fleet THEMSELVES (decision 2026-10-20: "停止は自分でやるので待て"); this script never pauses or
# restores the fleet. It refuses to start until GPUs 4, 6 and 7 are actually free (< 1 GiB used each), records the GPU
# state via scripts/shigeru_run.py (which also rejects GPUs 0/1), and logs to results/run_467.log.
#   (setsid bash distill/run_467.sh < /dev/null > results/run_467.log 2>&1 &)
cd "$(dirname "$0")/.."
for g in 4 6 7; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  if [ "$used" -gt 1024 ]; then echo "GPU $g has ${used} MiB in use (fleet still running?). Refusing to start; stop the fleet first."; exit 2; fi
done
echo "=== $(date +%T) GPUs 4,6,7 free; starting"
exec .venv/bin/python scripts/shigeru_run.py --gpus 4,6,7 -- bash -c '
  set -e; export PYTHONPATH=.
  echo "=== $(date +%T) GPUs: $CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  CUDA_VISIBLE_DEVICES=4 .venv-train/bin/python -m distill.train_sft distill/configs/phaseB_pilot_4b.yaml > results/train_phaseB_pilot_4b.log 2>&1
  GPU=4 bash bench/eval_ar_model.sh runs/phaseB_pilot_4b/merged phaseB_pilot_4b
  GPU=6 bash bench/eval_ar_model.sh /mnt/raid6/models/hf/Qwen3-4B qwen3_4b_idle   # clean speed numbers on an idle GPU
  echo "=== $(date +%T) run_467 done"
'
