#!/bin/bash
# GPU 7: learning-rate ablation — no-time recipe on e5 at lr 1e-4 (2000 x 8); dev + held-out.
cd "$(dirname "$0")/.."
M=runs/zeta_sft_final_e5/final; D=data/distill/phaseA_stage2c_train.jsonl; R=results/pure/dev_refs_sft3.json; L=results/pure/b_chain.log; E="CUDA_VISIBLE_DEVICES=7 PYTHONPATH=."
env $E .venv-train/bin/python distill/train_bdiff.py --model $M --data $D --out runs/bdiff_e5_lr1e4 --no-time --steps 2000 --accum 8 --lr 1e-4 --save-every 500 > results/pure/train_bdiff_e5_lr1e4.log 2>&1; echo "=== $(date +%T) [gpu7] train e5_lr1e4 rc=$? $(tail -n 2 results/pure/train_bdiff_e5_lr1e4.log | head -n1 | cut -c1-300)" >> $L
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter runs/bdiff_e5_lr1e4/adapter --no-time --refs $R --n 50 --steps 16,8,4 --skip-tf --out results/pure/bdiff_e5_lr1e4_eval.json > results/pure/bdiff_e5_lr1e4_eval.log 2>&1; echo "=== $(date +%T) [gpu7] eval e5_lr1e4 dev $(grep -E '^=== bdiff_gen' results/pure/bdiff_e5_lr1e4_eval.log | cut -c1-600 | sed ':a;N;$!ba;s/\n/ /g')" >> $L
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter runs/bdiff_e5_lr1e4/adapter --no-time --refs $R --skip 50 --n 150 --steps 16,8 --skip-tf --out results/pure/bdiff_e5_lr1e4_heldout.json > results/pure/bdiff_e5_lr1e4_heldout.log 2>&1; echo "=== $(date +%T) [gpu7] e5_lr1e4 held-out 50-199 $(grep -E '^=== bdiff_gen' results/pure/bdiff_e5_lr1e4_heldout.log | cut -c1-500 | sed ':a;N;$!ba;s/\n/ /g')" >> $L
echo "=== $(date +%T) [gpu7] e5_lr1e4 chain done" >> $L
