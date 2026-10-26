#!/bin/bash
# B11 final GSM8K test rows (settings fixed on dev2): arm B with left-to-right S=8 (best left model), arm E with the global remask sampler S=8 (best global) — GPU 6
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b11_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; E="PYTHONPATH=. .venv-train/bin/python"
CUDA_VISIBLE_DEVICES=6 env $E bench/eval_global.py --model $M --adapter $A --adapter2 runs/sa_B_left_plain/adapter --prompts bench/prompts_gsm8k_full.json --n 1319 --modes monotonic --steps 8 --skip-tf --out results/pure/b11_test_B_left_s8.json > results/pure/b11_test_B_left_s8.log 2>&1
echo "=== $(date +%T) [gpu6] TEST arm B left S8 rc=$? $(grep '^=== b9_gen' results/pure/b11_test_B_left_s8.log | sed 's/.*=== b9_gen //' | cut -c1-200)" >> $L
CUDA_VISIBLE_DEVICES=6 env $E bench/eval_global.py --model $M --adapter $A --adapter2 runs/sa_E_remask_corr/adapter --prompts bench/prompts_gsm8k_full.json --n 1319 --modes remask --steps 8 --skip-tf --out results/pure/b11_test_E_remask_s8.json > results/pure/b11_test_E_remask_s8.log 2>&1
echo "=== $(date +%T) [gpu6] TEST arm E remask S8 rc=$? $(grep '^=== b9_gen' results/pure/b11_test_E_remask_s8.log | sed 's/.*=== b9_gen //' | cut -c1-200)" >> $L
