#!/bin/bash
# B7 dev2 follow-ups on GPU 7 after sweep a: lowprob (dev-50 88 %), stability budgets, lowprob threshold
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b7_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; E="PYTHONPATH=. .venv-train/bin/python"
P=bench/prompts_gsm8k_dev2.json
until grep -q "b7 dev2 sweep a rc=" $L; do sleep 30; done
CUDA_VISIBLE_DEVICES=7 env $E bench/eval_trust.py --model $M --adapter $A --no-time --prompts $P --n 150 --steps 8 --unmask left --remask lowprob --tau-remask 0.5 --skip-tf --out results/pure/b7_gen_dev2_lr1e4_lowprob.json > results/pure/b7_gen_dev2_lr1e4_lowprob.log 2>&1
echo "=== $(date +%T) [gpu7] b7 dev2 lowprob rc=$?" >> $L
for B in 1 4; do CUDA_VISIBLE_DEVICES=7 env $E bench/eval_trust.py --model $M --adapter $A --no-time --prompts $P --n 150 --steps 8 --unmask left --remask stability --remask-budget $B --skip-tf --out results/pure/b7_gen_dev2_lr1e4_stab_b$B.json > results/pure/b7_gen_dev2_lr1e4_stab_b$B.log 2>&1
echo "=== $(date +%T) [gpu7] b7 dev2 stability b$B rc=$?" >> $L; done
CUDA_VISIBLE_DEVICES=7 env $E bench/eval_trust.py --model $M --adapter $A --no-time --prompts $P --n 150 --steps 8 --unmask left --remask lowprob --tau-remask 0.3 --skip-tf --out results/pure/b7_gen_dev2_lr1e4_lowprob03.json > results/pure/b7_gen_dev2_lr1e4_lowprob03.log 2>&1
echo "=== $(date +%T) [gpu7] b7 dev2 lowprob t0.3 rc=$?" >> $L
