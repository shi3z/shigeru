#!/bin/bash
# B10-A joint-teacher data on the student's own S=8 trajectories (zeta): GSM8K pair-train prompts then MBPP prompts; k=4 candidates
cd ~/shigeru/open-mercury
L=results/pure/b10_zeta_chain.log; E="SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-train/bin/python"; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter
mkdir -p data/distill
env $E distill/gen_pairs.py --model $M --adapter $A --prompts bench/prompts_gsm8k_pairtrain.json --n 800 --k 4 --out data/distill/pairs_gsm8k > results/pure/gen_pairs_gsm8k.log 2>&1
echo "=== $(date +%T) [zeta] gen_pairs gsm8k rc=$? $(tail -n1 results/pure/gen_pairs_gsm8k.log | cut -c1-200)" >> $L
env $E distill/gen_pairs.py --model $M --adapter $A --prompts bench/prompts_mbpp_pairtrain.json --n 300 --k 4 --max-new-tokens 512 --out data/distill/pairs_mbpp > results/pure/gen_pairs_mbpp.log 2>&1
echo "=== $(date +%T) [zeta] gen_pairs mbpp rc=$? $(tail -n1 results/pure/gen_pairs_mbpp.log | cut -c1-200)" >> $L
