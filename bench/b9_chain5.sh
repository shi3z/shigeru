#!/bin/bash
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b9_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; R=results/pure/dev_refs_e5.json; E="PYTHONPATH=. .venv-train/bin/python"
CUDA_VISIBLE_DEVICES=6 env $E bench/eval_global.py --model $M --adapter $A --refs $R --n 50 --modes remask --steps 4,8,16 --out results/pure/b9_global_dev50_remask_v2.json > results/pure/b9_global_dev50_remask_v2.log 2>&1
echo "=== $(date +%T) [gpu6] b9 global remask v2 rc=$?" >> $L
