#!/bin/bash
# student A' (GPU 4): student A's recipe with end-of-text pairs excluded from the distillation states; dev2 S=8
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b10_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; E="PYTHONPATH=. .venv-train/bin/python"
CUDA_VISIBLE_DEVICES=4 env $E distill/train_consist.py --model $M --adapter $A --data data/distill/pairs_pilot/pilot --steps 1000 --accum 8 --lr 1e-4 --hard-weight 4 --exclude-eos --eval-every 500 --save-every 1000 --out runs/consist_a2 > results/pure/train_consist_a2.log 2>&1
echo "=== $(date +%T) [gpu4] consist A' (exclude-eos) train rc=$? $(grep 'exclude-eos\|held-out' results/pure/train_consist_a2.log | tail -n 2 | tr '\n' ' ' | cut -c1-300)" >> $L
CUDA_VISIBLE_DEVICES=4 env $E bench/eval_trust.py --model $M --adapter $A --adapter2 runs/consist_a2/adapter --no-time --prompts bench/prompts_gsm8k_dev2.json --n 150 --steps 8 --unmask left --remask none --skip-tf --out results/pure/b10_a2_dev2_s8.json > results/pure/b10_a2_dev2_s8.log 2>&1
echo "=== $(date +%T) [gpu4] consist A' dev2 S8 rc=$? $(grep '^=== b7_gen' results/pure/b10_a2_dev2_s8.log | cut -c1-250)" >> $L
