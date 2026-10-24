#!/bin/bash
# zeta: batch ablation — no-time recipe on e5 with accum 16 (2000 steps = 2x blocks); dev + held-out.
cd ~/shigeru/open-mercury
M=runs/zeta_sft_final_e5/final; D=data/distill/phaseA_stage2c_train.jsonl; R=results/pure/dev_refs_sft3.json; L=results/pure/bdiff_chain.log
E="SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=."
env $E .venv-train/bin/python distill/train_bdiff.py --model $M --data $D --no-time --out runs/bdiff_e5_acc16 --steps 2000 --accum 16 --lr 2e-4 --save-every 500 > results/pure/train_bdiff_e5_acc16.log 2>&1; echo "=== $(date +%T) [zeta] train e5_acc16 rc=$? $(tail -n 2 results/pure/train_bdiff_e5_acc16.log | head -n1 | cut -c1-300)" >> $L
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter runs/bdiff_e5_acc16/adapter --no-time --refs $R --n 50 --steps 16,8,4 --skip-tf --out results/pure/bdiff_e5_acc16_eval.json > results/pure/bdiff_e5_acc16_eval.log 2>&1; echo "=== $(date +%T) [zeta] eval e5_acc16 dev $(grep -E '^=== bdiff_gen' results/pure/bdiff_e5_acc16_eval.log | cut -c1-600 | sed ':a;N;$!ba;s/\n/ /g')" >> $L
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter runs/bdiff_e5_acc16/adapter --no-time --refs $R --skip 50 --n 150 --steps 16,8 --skip-tf --out results/pure/bdiff_e5_acc16_heldout.json > results/pure/bdiff_e5_acc16_heldout.log 2>&1; echo "=== $(date +%T) [zeta] e5_acc16 held-out 50-199 $(grep -E '^=== bdiff_gen' results/pure/bdiff_e5_acc16_heldout.log | cut -c1-500 | sed ':a;N;$!ba;s/\n/ /g')" >> $L
echo "=== $(date +%T) [zeta] b chain7 done" >> $L
