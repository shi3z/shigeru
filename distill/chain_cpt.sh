#!/bin/bash
# Waits for the corpus shards and for the Phase F full evaluations (GPUs 4,6,7) to finish, then launches the 3-GPU
# continued-pretraining conversion (distill/configs/phaseC_cpt_4b_3gpu.yaml). Never touches the fleet or GPUs 0/1/5.
cd "$(dirname "$0")/.."; LOG=results/phaseB_runs.log
until [ -z "$(ls /mnt/raid6/data/*/*.aria2 2>/dev/null)" ]; do sleep 60; done
echo "=== $(date +%T) corpus shards complete" >> $LOG
until grep -q "phaseF_full done" results/phaseF_runs.log 2>/dev/null; do sleep 60; done
for g in 4 6 7; do until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)" -lt 1024 ]; do sleep 30; done; done
echo "=== $(date +%T) start phaseC_cpt_4b (torchrun 3 ranks on GPUs 4,6,7) commit=$(git rev-parse --short HEAD)" >> $LOG
CUDA_VISIBLE_DEVICES=4,6,7 PYTHONPATH=. .venv-train/bin/torchrun --nproc_per_node=3 -m distill.train_cdlm distill/configs/phaseC_cpt_4b_3gpu.yaml > results/train_phaseC_cpt_4b.log 2>&1
echo "=== $(date +%T) phaseC_cpt_4b rc=$?" >> $LOG
