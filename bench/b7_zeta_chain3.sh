#!/bin/bash
# B7: GSM8K test (1319) of S=16 left-to-right (dev2 90 % vs 82 % confidence) on zeta, after chain2
cd ~/shigeru/open-mercury
L=results/pure/b7_zeta_chain.log; E="SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-train/bin/python"; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter
until grep -q "dev2 left dependency b1 rc=" $L && ! pgrep -f "b7_zeta_chain[2]" > /dev/null; do sleep 30; done
env $E bench/eval_trust.py --model $M --adapter $A --no-time --prompts bench/prompts_gsm8k_full.json --n 1319 --steps 16 --unmask left --remask none --skip-tf --out results/pure/b7_test_s16_left.json > results/pure/b7_test_s16_left.log 2>&1
echo "=== $(date +%T) [zeta] TEST S16 left rc=$? $(grep '^=== b7_gen' results/pure/b7_test_s16_left.log | cut -c1-300)" >> $L
