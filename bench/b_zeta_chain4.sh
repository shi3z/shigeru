#!/bin/bash
# zeta: B1 x3 steps on the e5 base (the base-comparison partner of GPU 6's sft3 run), then dev + held-out evaluation.
cd ~/shigeru/open-mercury
M=runs/zeta_sft_final_e5/final; D=data/distill/phaseA_stage2c_train.jsonl; R=results/pure/dev_refs_sft3.json; L=results/pure/bdiff_chain.log
E="SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=."
env $E .venv-train/bin/python distill/train_bdiff.py --model $M --data $D --out runs/bdiff_b1x3 --steps 6000 --accum 8 --lr 2e-4 --save-every 1000 > results/pure/train_bdiff_b1x3.log 2>&1; echo "=== $(date +%T) [zeta] train b1x3 rc=$? $(tail -n 2 results/pure/train_bdiff_b1x3.log | head -n1 | cut -c1-300)" >> $L
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter runs/bdiff_b1x3/adapter --time-embed runs/bdiff_b1x3/time_embed.pt --refs $R --n 50 --steps 16,8,4 --skip-tf --out results/pure/bdiff_b1x3_eval.json > results/pure/bdiff_b1x3_eval.log 2>&1; echo "=== $(date +%T) [zeta] eval b1x3 dev $(grep -E '^=== bdiff_gen' results/pure/bdiff_b1x3_eval.log | cut -c1-600 | tr '\n' ' ')" >> $L
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter runs/bdiff_b1x3/adapter --time-embed runs/bdiff_b1x3/time_embed.pt --refs $R --skip 50 --n 150 --steps 16,8 --skip-tf --out results/pure/bdiff_b1x3_heldout.json > results/pure/bdiff_b1x3_heldout.log 2>&1; echo "=== $(date +%T) [zeta] b1x3 held-out 50-199 $(grep -E '^=== bdiff_gen' results/pure/bdiff_b1x3_heldout.log | cut -c1-500 | tr '\n' ' ')" >> $L
echo "=== $(date +%T) [zeta] b chain4 done" >> $L
