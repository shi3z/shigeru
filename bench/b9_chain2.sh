#!/bin/bash
# B9-C global refinement samplers without training, dev-50 (GPU 7 after the B8 top-k diagnostic)
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b9_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; R=results/pure/dev_refs_e5.json; E="PYTHONPATH=. .venv-train/bin/python"
until grep -q "diag_pairwise topk gsm8k left rc=" results/pure/b8_chain.log; do sleep 30; done
CUDA_VISIBLE_DEVICES=7 env $E bench/eval_global.py --model $M --adapter $A --refs $R --n 50 --modes monotonic,sync,remask,soft,anneal --steps 4,8,16 --out results/pure/b9_global_dev50.json > results/pure/b9_global_dev50.log 2>&1
echo "=== $(date +%T) [gpu7] b9 global dev50 rc=$?" >> $L
