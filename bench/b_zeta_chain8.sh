#!/bin/bash
# zeta: final GSM8K test for the lr 2e-4 no-time adapter at S=8 (confidence) and S=16, fixed settings.
cd ~/shigeru/open-mercury
M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_notime/adapter; P=bench/prompts_gsm8k_full.json; R=results/pure/dev_refs_sft3.json; L=results/pure/bdiff_chain.log
E="SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=."
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter $A --no-time --refs $R --prompts $P --n 1319 --steps 8 --skip-tf --out results/pure/final_test_b_e5_notime_S8.json > results/pure/final_test_b_e5_notime_S8.log 2>&1; echo "=== $(date +%T) [zeta] TEST B-series e5 no-time S=8 $(grep -E '^=== bdiff_gen' results/pure/final_test_b_e5_notime_S8.log | cut -c1-400)" >> $L
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter $A --no-time --refs $R --prompts $P --n 1319 --steps 16 --skip-tf --out results/pure/final_test_b_e5_notime_S16.json > results/pure/final_test_b_e5_notime_S16.log 2>&1; echo "=== $(date +%T) [zeta] TEST B-series e5 no-time S=16 $(grep -E '^=== bdiff_gen' results/pure/final_test_b_e5_notime_S16.log | cut -c1-400)" >> $L
echo "=== $(date +%T) [zeta] b chain8 done" >> $L
