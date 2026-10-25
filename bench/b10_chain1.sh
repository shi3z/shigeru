#!/bin/bash
# B10 Stage 2 pilot: conditional-distribution distillation on the first 8k teacher states (GPU 4), then S=8/S=16 dev-50 generation with the student
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b10_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; E="PYTHONPATH=. .venv-train/bin/python"
CUDA_VISIBLE_DEVICES=4 env $E distill/train_consist.py --model $M --adapter $A --data data/distill/pairs_pilot/pilot --steps 400 --accum 8 --lr 1e-4 --eval-every 100 --save-every 200 --out runs/consist_pilot > results/pure/train_consist_pilot.log 2>&1
echo "=== $(date +%T) [gpu4] consist pilot train rc=$? $(grep 'held-out' results/pure/train_consist_pilot.log | tail -n 1 | cut -c1-200)" >> $L
CUDA_VISIBLE_DEVICES=4 env $E bench/eval_trust.py --model $M --adapter $A --adapter2 runs/consist_pilot/adapter --no-time --n 50 --steps 8 --unmask left --remask none --skip-tf --out results/pure/b10_pilot_dev50_s8.json > results/pure/b10_pilot_dev50_s8.log 2>&1
echo "=== $(date +%T) [gpu4] consist pilot dev50 S8 rc=$? $(grep '^=== b7_gen' results/pure/b10_pilot_dev50_s8.log | cut -c1-250)" >> $L
CUDA_VISIBLE_DEVICES=4 env $E bench/eval_trust.py --model $M --adapter $A --adapter2 runs/consist_pilot/adapter --no-time --n 50 --steps 16 --unmask left --remask none --skip-tf --out results/pure/b10_pilot_dev50_s16.json > results/pure/b10_pilot_dev50_s16.log 2>&1
echo "=== $(date +%T) [gpu4] consist pilot dev50 S16 rc=$? $(grep '^=== b7_gen' results/pure/b10_pilot_dev50_s16.log | cut -c1-250)" >> $L
