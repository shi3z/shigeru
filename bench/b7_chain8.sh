#!/bin/bash
# B7 final: GSM8K test (1319) of the one rule with a positive held-out effect — S=8 left-to-right + stability re-mask, budget 2 (GPU 4 after the TF S16 run)
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b7_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; E="PYTHONPATH=. .venv-train/bin/python"
until grep -q "b7 TF S16 rc=" $L; do sleep 30; done
CUDA_VISIBLE_DEVICES=4 env $E bench/eval_trust.py --model $M --adapter $A --no-time --prompts bench/prompts_gsm8k_full.json --n 1319 --steps 8 --unmask left --remask stability --remask-budget 2 --skip-tf --out results/pure/b7_test_s8_left_stab2.json > results/pure/b7_test_s8_left_stab2.log 2>&1
echo "=== $(date +%T) [gpu4] TEST S8 left+stability b2 rc=$? $(grep '^=== b7_gen' results/pure/b7_test_s8_left_stab2.log | cut -c1-300)" >> $L
