#!/bin/bash
# B10 variant B (GPU 4, after the pilot S16 eval): pair-KL + look-ahead transition distillation, hard x4, same 12k states
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b10_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; E="PYTHONPATH=. .venv-train/bin/python"
until grep -q "consist pilot dev50 S16" $L; do sleep 30; done
CUDA_VISIBLE_DEVICES=4 env $E distill/train_consist.py --model $M --adapter $A --data data/distill/pairs_pilot/pilot --steps 1000 --accum 8 --lr 1e-4 --hard-weight 4 --lookahead 1.0 --eval-every 250 --save-every 500 --out runs/consist_b > results/pure/train_consist_b.log 2>&1
echo "=== $(date +%T) [gpu4] consist B (pair-KL + lookahead, hard x4) train rc=$? $(grep 'held-out' results/pure/train_consist_b.log | tail -n 1 | cut -c1-250)" >> $L
CUDA_VISIBLE_DEVICES=4 env $E bench/eval_trust.py --model $M --adapter $A --adapter2 runs/consist_b/adapter --no-time --n 50 --steps 8 --unmask left --remask none --skip-tf --out results/pure/b10_b_dev50_s8.json > results/pure/b10_b_dev50_s8.log 2>&1
echo "=== $(date +%T) [gpu4] consist B dev50 S8 rc=$? $(grep '^=== b7_gen' results/pure/b10_b_dev50_s8.log | cut -c1-250)" >> $L
CUDA_VISIBLE_DEVICES=4 env $E bench/eval_trust.py --model $M --adapter $A --adapter2 runs/consist_b/adapter --no-time --prompts bench/prompts_gsm8k_dev2.json --n 150 --steps 8 --unmask left --remask none --skip-tf --out results/pure/b10_b_dev2_s8.json > results/pure/b10_b_dev2_s8.log 2>&1
echo "=== $(date +%T) [gpu4] consist B dev2 S8 rc=$? $(grep '^=== b7_gen' results/pure/b10_b_dev2_s8.log | cut -c1-250)" >> $L
