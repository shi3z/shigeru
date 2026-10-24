#!/bin/bash
# zeta: capacity ablation — LoRA rank 256 (alpha 512), no time, e5 base, B1 budget; dev + held-out.
cd ~/shigeru/open-mercury
M=runs/zeta_sft_final_e5/final; D=data/distill/phaseA_stage2c_train.jsonl; R=results/pure/dev_refs_sft3.json; L=results/pure/bdiff_chain.log
E="SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=."
env $E .venv-train/bin/python distill/train_bdiff.py --model $M --data $D --no-time --lora-r 256 --lora-alpha 512 --out runs/bdiff_e5_r256 --steps 2000 --accum 8 --lr 1e-4 --save-every 500 > results/pure/train_bdiff_e5_r256.log 2>&1; echo "=== $(date +%T) [zeta] train e5_r256 rc=$? $(tail -n 2 results/pure/train_bdiff_e5_r256.log | head -n1 | cut -c1-300)" >> $L
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter runs/bdiff_e5_r256/adapter --no-time --refs $R --n 50 --steps 16,8,4 --skip-tf --out results/pure/bdiff_e5_r256_eval.json > results/pure/bdiff_e5_r256_eval.log 2>&1; echo "=== $(date +%T) [zeta] eval e5_r256 dev $(grep -E '^=== bdiff_gen' results/pure/bdiff_e5_r256_eval.log | cut -c1-600 | sed ':a;N;$!ba;s/\n/ /g')" >> $L
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter runs/bdiff_e5_r256/adapter --no-time --refs $R --skip 50 --n 150 --steps 16,8 --skip-tf --out results/pure/bdiff_e5_r256_heldout.json > results/pure/bdiff_e5_r256_heldout.log 2>&1; echo "=== $(date +%T) [zeta] e5_r256 held-out 50-199 $(grep -E '^=== bdiff_gen' results/pure/bdiff_e5_r256_heldout.log | cut -c1-500 | sed ':a;N;$!ba;s/\n/ /g')" >> $L
echo "=== $(date +%T) [zeta] b chain6 done" >> $L
