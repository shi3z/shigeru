#!/bin/bash
# zeta: no-time B1 with 50 % raw FineWeb blocks (CPT token stream) + 50 % conversations, 4000 steps; dev + held-out.
cd ~/shigeru/open-mercury
M=runs/zeta_sft_final_e5/final; D=data/distill/phaseA_stage2c_train.jsonl; R=results/pure/dev_refs_sft3.json; L=results/pure/bdiff_chain.log; TS=data/cpt/fineweb_1b.bin
E="SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=."
env $E .venv-train/bin/python distill/train_bdiff.py --model $M --data $D --data-tokens $TS --p-tokens 0.5 --no-time --out runs/bdiff_b7_mix --steps 4000 --accum 8 --lr 2e-4 --save-every 1000 > results/pure/train_bdiff_b7_mix.log 2>&1; echo "=== $(date +%T) [zeta] train b7_mix rc=$? $(tail -n 2 results/pure/train_bdiff_b7_mix.log | head -n1 | cut -c1-300)" >> $L
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter runs/bdiff_b7_mix/adapter --no-time --refs $R --n 50 --steps 16,8,4 --skip-tf --out results/pure/bdiff_b7_mix_eval.json > results/pure/bdiff_b7_mix_eval.log 2>&1; echo "=== $(date +%T) [zeta] eval b7_mix dev $(grep -E '^=== bdiff_gen' results/pure/bdiff_b7_mix_eval.log | cut -c1-600 | sed ':a;N;$!ba;s/\n/ /g')" >> $L
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter runs/bdiff_b7_mix/adapter --no-time --refs $R --skip 50 --n 150 --steps 16,8 --skip-tf --out results/pure/bdiff_b7_mix_heldout.json > results/pure/bdiff_b7_mix_heldout.log 2>&1; echo "=== $(date +%T) [zeta] b7_mix held-out 50-199 $(grep -E '^=== bdiff_gen' results/pure/bdiff_b7_mix_heldout.log | cut -c1-500 | sed ':a;N;$!ba;s/\n/ /g')" >> $L
echo "=== $(date +%T) [zeta] b chain5 done" >> $L
