#!/bin/bash
# GPU 4 (after the B5 chain): the recipe update from §10.4 — no time conditioning — at 3x steps on the sft3 base, dev + held-out.
cd "$(dirname "$0")/.."
until grep -q "\[gpu4\] b chain done" results/pure/b_chain.log; do sleep 60; done
M=runs/phaseC_sft3_4b/final; D=data/distill/phaseA_stage2c_train.jsonl; R=results/pure/dev_refs_sft3.json; L=results/pure/b_chain.log; E="CUDA_VISIBLE_DEVICES=4 PYTHONPATH=."
env $E .venv-train/bin/python distill/train_bdiff.py --model $M --data $D --out runs/bdiff_sft3_notime_x3 --no-time --steps 6000 --accum 8 --lr 2e-4 --save-every 1000 > results/pure/train_bdiff_sft3_notime_x3.log 2>&1; echo "=== $(date +%T) [gpu4] train notime_x3 rc=$? $(tail -n 2 results/pure/train_bdiff_sft3_notime_x3.log | head -n1 | cut -c1-300)" >> $L
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter runs/bdiff_sft3_notime_x3/adapter --no-time --refs $R --n 50 --steps 16,8,4 --skip-tf --out results/pure/bdiff_sft3_notime_x3_eval.json > results/pure/bdiff_sft3_notime_x3_eval.log 2>&1; echo "=== $(date +%T) [gpu4] eval notime_x3 dev $(grep -E '^=== bdiff_gen' results/pure/bdiff_sft3_notime_x3_eval.log | cut -c1-600 | sed ':a;N;$!ba;s/\n/ /g')" >> $L
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter runs/bdiff_sft3_notime_x3/adapter --no-time --refs $R --skip 50 --n 150 --steps 16,8 --skip-tf --out results/pure/bdiff_sft3_notime_x3_heldout.json > results/pure/bdiff_sft3_notime_x3_heldout.log 2>&1; echo "=== $(date +%T) [gpu4] notime_x3 held-out 50-199 $(grep -E '^=== bdiff_gen' results/pure/bdiff_sft3_notime_x3_heldout.log | cut -c1-500 | sed ':a;N;$!ba;s/\n/ /g')" >> $L
echo "=== $(date +%T) [gpu4] b chain2 done" >> $L
