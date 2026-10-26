#!/bin/bash
# baseline (undistilled adapter) with the non-monotonic sampler on dev2 S=8 and S=4/16 — the missing comparison row (GPU 7)
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b11_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; E="PYTHONPATH=. .venv-train/bin/python"
CUDA_VISIBLE_DEVICES=7 env $E bench/eval_global.py --model $M --adapter $A --prompts bench/prompts_gsm8k_dev2.json --n 150 --modes remask --steps 4,8,16 --skip-tf --out results/pure/b11_eval_base_dev2.json > results/pure/b11_eval_base_dev2.log 2>&1
echo "=== $(date +%T) [gpu7] b11 eval baseline remask dev2 rc=$? $(grep '^=== b9_gen' results/pure/b11_eval_base_dev2.log | sed 's/.*=== b9_gen //' | cut -c1-60 | tr '\n' ' ')" >> $L
