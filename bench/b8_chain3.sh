#!/bin/bash
# B8 §2 extended top-k oracle diagnostic (k = 1..32, recalls, distances, candidate dump) on GSM8K dev (GPU 7)
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b8_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; E="PYTHONPATH=. .venv-train/bin/python"
CUDA_VISIBLE_DEVICES=7 env $E bench/diag_pairwise.py --model $M --adapter $A --no-time --refs results/pure/dev_refs_e5.json --n 50 --unmask left --rerank-ks 2,4 --save-candidates results/pure/b8_candidates_gsm8k_left.jsonl --out results/pure/diag_pairwise_lr1e4_left_topk.json > results/pure/diag_pairwise_lr1e4_left_topk.log 2>&1
echo "=== $(date +%T) [gpu7] diag_pairwise topk gsm8k left rc=$?" >> $L
