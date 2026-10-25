#!/bin/bash
# B10 variant C (GPU 7): pair-KL distillation on ALL GSM8K teacher states collected so far (~47k), 2000 steps x 8, hard x4; then S=8 dev-50 / dev2 and S=16 dev2
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b10_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; E="PYTHONPATH=. .venv-train/bin/python"
CUDA_VISIBLE_DEVICES=7 env $E distill/train_consist.py --model $M --adapter $A --data data/distill/pairs_gsm8k/pairs_gsm8k --steps 2000 --accum 8 --lr 1e-4 --hard-weight 4 --eval-every 250 --save-every 500 --out runs/consist_c > results/pure/train_consist_c.log 2>&1
echo "=== $(date +%T) [gpu7] consist C (pair-KL, hard x4, all gsm8k states, 2000 steps) train rc=$? $(grep 'held-out' results/pure/train_consist_c.log | tail -n 1 | cut -c1-250)" >> $L
for cfg in "bench/prompts_gsm8k_dev.json 50 8 dev50_s8" "bench/prompts_gsm8k_dev2.json 150 8 dev2_s8" "bench/prompts_gsm8k_dev2.json 150 16 dev2_s16"; do set -- $cfg
  CUDA_VISIBLE_DEVICES=7 env $E bench/eval_trust.py --model $M --adapter $A --adapter2 runs/consist_c/adapter --no-time --prompts $1 --n $2 --steps $3 --unmask left --remask none --skip-tf --out results/pure/b10_c_$4.json > results/pure/b10_c_$4.log 2>&1
  echo "=== $(date +%T) [gpu7] consist C $4 rc=$? $(grep '^=== b7_gen' results/pure/b10_c_$4.log | cut -c1-250)" >> $L; done
