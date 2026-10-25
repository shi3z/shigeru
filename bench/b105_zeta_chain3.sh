#!/bin/bash
# GSM8K test (1319) of ablation arm B (masked CE only on 12k teacher-trajectory states) at S=8 left — zeta
cd ~/shigeru/open-mercury
L=results/pure/b105_zeta_chain.log; E="SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-train/bin/python"; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter
env $E bench/eval_trust.py --model $M --adapter $A --adapter2 runs/abl_maskce/adapter --no-time --prompts bench/prompts_gsm8k_full.json --n 1319 --steps 8 --unmask left --remask none --skip-tf --out results/pure/b105_test_maskce_s8.json > results/pure/b105_test_maskce_s8.log 2>&1
echo "=== $(date +%T) [zeta] TEST arm B mask_only S8 rc=$? $(grep '^=== b7_gen' results/pure/b105_test_maskce_s8.log | cut -c1-300)" >> $L
