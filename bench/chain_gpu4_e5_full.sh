#!/bin/bash
# GPU 4: full fine-tune (no LoRA) of the no-time recipe on e5, 2000 x 8, fp32 master + bf16 autocast, gradient checkpointing.
cd "$(dirname "$0")/.."
M=runs/zeta_sft_final_e5/final; D=data/distill/phaseA_stage2c_train.jsonl; R=results/pure/dev_refs_sft3.json; L=results/pure/b_chain.log; E="CUDA_VISIBLE_DEVICES=4 PYTHONPATH=."
env $E .venv-train/bin/python distill/train_bdiff.py --model $M --data $D --out runs/bdiff_e5_full --no-time --full --grad-ckpt --steps 2000 --accum 8 --lr 1e-5 --save-every 2000 > results/pure/train_bdiff_e5_full.log 2>&1; echo "=== $(date +%T) [gpu4] train e5_full rc=$? $(tail -n 2 results/pure/train_bdiff_e5_full.log | head -n1 | cut -c1-300)" >> $L
env $E .venv-train/bin/python bench/eval_bdiff.py --model runs/bdiff_e5_full/final --no-time --refs $R --n 50 --steps 16,8,4 --skip-tf --out results/pure/bdiff_e5_full_eval.json > results/pure/bdiff_e5_full_eval.log 2>&1; echo "=== $(date +%T) [gpu4] eval e5_full dev $(grep -E '^=== bdiff_gen' results/pure/bdiff_e5_full_eval.log | cut -c1-600 | sed ':a;N;$!ba;s/\n/ /g')" >> $L
env $E .venv-train/bin/python bench/eval_bdiff.py --model runs/bdiff_e5_full/final --no-time --refs $R --skip 50 --n 150 --steps 16,8 --skip-tf --out results/pure/bdiff_e5_full_heldout.json > results/pure/bdiff_e5_full_heldout.log 2>&1; echo "=== $(date +%T) [gpu4] e5_full held-out 50-199 $(grep -E '^=== bdiff_gen' results/pure/bdiff_e5_full_heldout.log | cut -c1-500 | sed ':a;N;$!ba;s/\n/ /g')" >> $L
echo "=== $(date +%T) [gpu4] e5_full chain done" >> $L
