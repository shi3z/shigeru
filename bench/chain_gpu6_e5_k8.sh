#!/bin/bash
# GPU 6: B-series with K = 8 blocks (no time, e5, 2000 x 8); S = 8 / 4 / 2 -> 1 / 2 / 4 tokens per network call.
cd "$(dirname "$0")/.."
M=runs/zeta_sft_final_e5/final; D=data/distill/phaseA_stage2c_train.jsonl; R=results/pure/dev_refs_sft3.json; L=results/pure/b_chain.log; E="CUDA_VISIBLE_DEVICES=6 PYTHONPATH=."
env $E .venv-train/bin/python distill/train_bdiff.py --model $M --data $D --out runs/bdiff_e5_k8 --K 8 --no-time --steps 2000 --accum 8 --lr 2e-4 --save-every 500 > results/pure/train_bdiff_e5_k8.log 2>&1; echo "=== $(date +%T) [gpu6] train e5_k8 rc=$? $(tail -n 2 results/pure/train_bdiff_e5_k8.log | head -n1 | cut -c1-300)" >> $L
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter runs/bdiff_e5_k8/adapter --no-time --K 8 --refs $R --n 50 --steps 8,4,2 --skip-tf --out results/pure/bdiff_e5_k8_eval.json > results/pure/bdiff_e5_k8_eval.log 2>&1; echo "=== $(date +%T) [gpu6] eval e5_k8 dev $(grep -E '^=== bdiff_gen' results/pure/bdiff_e5_k8_eval.log | cut -c1-600 | sed ':a;N;$!ba;s/\n/ /g')" >> $L
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter runs/bdiff_e5_k8/adapter --no-time --K 8 --refs $R --skip 50 --n 150 --steps 8,4 --skip-tf --out results/pure/bdiff_e5_k8_heldout.json > results/pure/bdiff_e5_k8_heldout.log 2>&1; echo "=== $(date +%T) [gpu6] e5_k8 held-out 50-199 $(grep -E '^=== bdiff_gen' results/pure/bdiff_e5_k8_heldout.log | cut -c1-500 | sed ':a;N;$!ba;s/\n/ /g')" >> $L
echo "=== $(date +%T) [gpu6] e5_k8 chain done" >> $L
