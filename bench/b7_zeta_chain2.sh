#!/bin/bash
# B7 dev2 (150) extra rows on zeta: S=16 baselines, confidence+lowprob, adaptive+dependency, left+dependency b1
cd ~/shigeru/open-mercury
L=results/pure/b7_zeta_chain.log; E="SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-train/bin/python"; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; P=bench/prompts_gsm8k_dev2.json
env $E bench/eval_trust.py --model $M --adapter $A --no-time --prompts $P --n 150 --steps 16 --unmask confidence,left --remask none --skip-tf --out results/pure/b7_gen_dev2_lr1e4_s16.json > results/pure/b7_gen_dev2_lr1e4_s16.log 2>&1
echo "=== $(date +%T) [zeta] dev2 S16 baselines rc=$?" >> $L
env $E bench/eval_trust.py --model $M --adapter $A --no-time --prompts $P --n 150 --steps 8 --unmask confidence --remask lowprob --skip-tf --out results/pure/b7_gen_dev2_lr1e4_conf_lowprob.json > results/pure/b7_gen_dev2_lr1e4_conf_lowprob.log 2>&1
echo "=== $(date +%T) [zeta] dev2 conf lowprob rc=$?" >> $L
env $E bench/eval_trust.py --model $M --adapter $A --no-time --prompts $P --n 150 --steps 8 --unmask adaptive --remask none,dependency --skip-tf --out results/pure/b7_gen_dev2_lr1e4_adaptive.json > results/pure/b7_gen_dev2_lr1e4_adaptive.log 2>&1
echo "=== $(date +%T) [zeta] dev2 adaptive rc=$?" >> $L
env $E bench/eval_trust.py --model $M --adapter $A --no-time --prompts $P --n 150 --steps 8 --unmask left --remask dependency --remask-budget 1 --skip-tf --out results/pure/b7_gen_dev2_lr1e4_dep_b1.json > results/pure/b7_gen_dev2_lr1e4_dep_b1.log 2>&1
echo "=== $(date +%T) [zeta] dev2 left dependency b1 rc=$?" >> $L
