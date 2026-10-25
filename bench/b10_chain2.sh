#!/bin/bash
# B10 Stage 2 on the first 12k teacher states (GPU 6): pair-KL distillation with hard-state upweighting, then S=8 dev-50 / dev2 generation
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b10_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; E="PYTHONPATH=. .venv-train/bin/python"
CUDA_VISIBLE_DEVICES=6 env $E distill/train_consist.py --model $M --adapter $A --data data/distill/pairs_pilot/pilot --steps 1000 --accum 8 --lr 1e-4 --hard-weight 4 --eval-every 250 --save-every 500 --out runs/consist_a > results/pure/train_consist_a.log 2>&1
echo "=== $(date +%T) [gpu6] consist A (pair-KL, hard x4, 12k states) train rc=$? $(grep 'held-out' results/pure/train_consist_a.log | tail -n 1 | cut -c1-250)" >> $L
CUDA_VISIBLE_DEVICES=6 env $E bench/eval_trust.py --model $M --adapter $A --adapter2 runs/consist_a/adapter --no-time --n 50 --steps 8 --unmask left --remask none --skip-tf --out results/pure/b10_a_dev50_s8.json > results/pure/b10_a_dev50_s8.log 2>&1
echo "=== $(date +%T) [gpu6] consist A dev50 S8 rc=$? $(grep '^=== b7_gen' results/pure/b10_a_dev50_s8.log | cut -c1-250)" >> $L
CUDA_VISIBLE_DEVICES=6 env $E bench/eval_trust.py --model $M --adapter $A --adapter2 runs/consist_a/adapter --no-time --prompts bench/prompts_gsm8k_dev2.json --n 150 --steps 8 --unmask left --remask none --skip-tf --out results/pure/b10_a_dev2_s8.json > results/pure/b10_a_dev2_s8.log 2>&1
echo "=== $(date +%T) [gpu6] consist A dev2 S8 rc=$? $(grep '^=== b7_gen' results/pure/b10_a_dev2_s8.log | cut -c1-250)" >> $L
