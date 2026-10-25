#!/bin/bash
# B10 ablation D (GPU 4): same as A but the second token gets a hard CE to b* instead of the KL to the teacher conditional
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b10_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; E="PYTHONPATH=. .venv-train/bin/python"
CUDA_VISIBLE_DEVICES=4 env $E distill/train_consist.py --model $M --adapter $A --data data/distill/pairs_pilot/pilot --steps 1000 --accum 8 --lr 1e-4 --hard-weight 4 --j-ce --eval-every 250 --save-every 500 --out runs/consist_d > results/pure/train_consist_d.log 2>&1
echo "=== $(date +%T) [gpu4] consist D (pair CE labels only, hard x4) train rc=$? $(grep 'held-out' results/pure/train_consist_d.log | tail -n 1 | cut -c1-250)" >> $L
CUDA_VISIBLE_DEVICES=4 env $E bench/eval_trust.py --model $M --adapter $A --adapter2 runs/consist_d/adapter --no-time --prompts bench/prompts_gsm8k_dev2.json --n 150 --steps 8 --unmask left --remask none --skip-tf --out results/pure/b10_d_dev2_s8.json > results/pure/b10_d_dev2_s8.log 2>&1
echo "=== $(date +%T) [gpu4] consist D dev2 S8 rc=$? $(grep '^=== b7_gen' results/pure/b10_d_dev2_s8.log | cut -c1-250)" >> $L
