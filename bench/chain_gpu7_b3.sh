#!/bin/bash
# GPU 7: B3 — B-sampler on-policy states from the B1-sft3 adapter, training from B1 (lambda_vis 0.3, 30 % states), evaluation.
cd "$(dirname "$0")/.."
M=runs/phaseC_sft3_4b/final; D=data/distill/phaseA_stage2c_train.jsonl; R=results/pure/dev_refs_sft3.json; L=results/pure/b_chain.log; E="CUDA_VISIBLE_DEVICES=7 PYTHONPATH=."
A=runs/bdiff_sft3_b1/adapter; T=runs/bdiff_sft3_b1/time_embed.pt
env $E .venv-train/bin/python distill/gen_bstates.py --mode states --model $M --adapter $A --time-embed $T --data $D --limit 3000 --cuts 2 --states-per-block 2 --steps 16 --out data/distill/bstates_sft3_b1.jsonl > results/pure/gen_bstates_sft3_b1.log 2>&1; echo "=== $(date +%T) [gpu7] b3 states rc=$? $(tail -n 1 results/pure/gen_bstates_sft3_b1.log | cut -c1-200)" >> $L
env $E .venv-train/bin/python distill/train_bdiff.py --model $M --data $D --init-adapter $A --states data/distill/bstates_sft3_b1.jsonl --p-state 0.3 --lambda-vis 0.3 --out runs/bdiff_sft3_b3 --steps 1500 --accum 8 --lr 1e-4 --save-every 500 > results/pure/train_bdiff_sft3_b3.log 2>&1; echo "=== $(date +%T) [gpu7] b3 train rc=$? $(tail -n 2 results/pure/train_bdiff_sft3_b3.log | head -n1 | cut -c1-300)" >> $L
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter runs/bdiff_sft3_b3/adapter --time-embed runs/bdiff_sft3_b3/time_embed.pt --refs $R --states data/distill/bstates_sft3_b1.jsonl --n 50 --steps 16,8,4 --out results/pure/bdiff_sft3_b3_eval.json > results/pure/bdiff_sft3_b3_eval.log 2>&1; echo "=== $(date +%T) [gpu7] b3 eval rc=$? $(grep -E '^=== bdiff' results/pure/bdiff_sft3_b3_eval.log | cut -c1-700 | sed ':a;N;$!ba;s/\n/ /g')" >> $L
echo "=== $(date +%T) [gpu7] b3 chain done" >> $L
