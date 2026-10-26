#!/bin/bash
# B11-C: on-policy states of the non-monotonic remask sampler (prompts 800-1099) and of the left sampler (1100-1399) with teacher targets — zeta
cd ~/shigeru/open-mercury
L=results/pure/b11_zeta_chain.log; E="SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-train/bin/python"; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter
mkdir -p data/distill
env $E distill/gen_global_states.py --model $M --adapter $A --prompts bench/prompts_gsm8k_pairtrain.json --skip 800 --n 300 --sampler remask --out data/distill/gstates_remask > results/pure/gen_gstates_remask.log 2>&1
echo "=== $(date +%T) [zeta] gstates remask rc=$? $(tail -n1 results/pure/gen_gstates_remask.log | cut -c1-200)" >> $L
env $E distill/gen_global_states.py --model $M --adapter $A --prompts bench/prompts_gsm8k_pairtrain.json --skip 1100 --n 300 --sampler left --out data/distill/gstates_left > results/pure/gen_gstates_left.log 2>&1
echo "=== $(date +%T) [zeta] gstates left rc=$? $(tail -n1 results/pure/gen_gstates_left.log | cut -c1-200)" >> $L
