#!/bin/bash
# GSM8K test (1319) of student E at S=8 left (GPU 6)
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b10_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; E="PYTHONPATH=. .venv-train/bin/python"
CUDA_VISIBLE_DEVICES=6 env $E bench/eval_trust.py --model $M --adapter $A --adapter2 runs/consist_e/adapter --no-time --prompts bench/prompts_gsm8k_full.json --n 1319 --steps 8 --unmask left --remask none --skip-tf --out results/pure/b10_test_e_s8.json > results/pure/b10_test_e_s8.log 2>&1
echo "=== $(date +%T) [gpu6] TEST student E S8 rc=$? $(grep '^=== b7_gen' results/pure/b10_test_e_s8.log | cut -c1-300)" >> $L
