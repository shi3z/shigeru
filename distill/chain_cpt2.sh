#!/bin/bash
# CPT continuation on GPUs 4,6 after the medium-set Phase B chain has finished with them and the new shards are down.
cd "$(dirname "$0")/.."; LOG=results/phaseB_runs.log
until grep -q "chain_medium done" $LOG 2>/dev/null; do sleep 120; done
until [ -z "$(ls /mnt/raid6/data/*/*.aria2 2>/dev/null)" ]; do sleep 60; done
for g in 4 6; do until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)" -lt 1024 ]; do sleep 30; done; done
echo "=== $(date +%T) start phaseC_cpt2_4b (torchrun 2 ranks on GPUs 4,6) commit=$(git rev-parse --short HEAD)" >> $LOG
CUDA_VISIBLE_DEVICES=4,6 PYTHONPATH=. .venv-train/bin/torchrun --nproc_per_node=2 -m distill.train_cdlm distill/configs/phaseC_cpt2_4b_2gpu.yaml > results/train_phaseC_cpt2_4b.log 2>&1
echo "=== $(date +%T) phaseC_cpt2_4b rc=$?" >> $LOG
