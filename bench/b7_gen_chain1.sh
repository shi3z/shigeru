#!/bin/bash
# B7 §3/§4 free-generation sweep, dev items 0-49, fixed 8 forwards, lr1e-4 adapter (GEN only; TF part waits for the e5 refs)
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b7_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter
RM=none,disagree,lowprob,stability,dependency
( CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv-train/bin/python bench/eval_trust.py --model $M --adapter $A --no-time --n 50 --steps 8 --unmask confidence,left --remask $RM --skip-tf --out results/pure/b7_gen_dev50_lr1e4_a.json > results/pure/b7_gen_dev50_lr1e4_a.log 2>&1; echo "=== $(date +%T) [gpu4] b7 gen sweep a rc=$?" >> $L ) &
( CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. .venv-train/bin/python bench/eval_trust.py --model $M --adapter $A --no-time --n 50 --steps 8 --unmask nonadjacent,dependency,adaptive --remask $RM --skip-tf --out results/pure/b7_gen_dev50_lr1e4_b.json > results/pure/b7_gen_dev50_lr1e4_b.log 2>&1; echo "=== $(date +%T) [gpu6] b7 gen sweep b rc=$?" >> $L ) &
wait
