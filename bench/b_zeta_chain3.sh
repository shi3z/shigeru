#!/bin/bash
# zeta: redo the B4 sweep + held-out on the B1 adapter after chain2 (its first attempt hit a division-by-zero in the evaluator).
cd ~/shigeru/open-mercury
until grep -q "b chain2 done" results/pure/bdiff_chain.log; do sleep 60; done
M=runs/zeta_sft_final_e5/final; R=results/pure/dev_refs_sft3.json; L=results/pure/bdiff_chain.log
E="SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=."
A=runs/bdiff_b1/adapter; T=runs/bdiff_b1/time_embed.pt
for b in 1 2 4; do for S in 16 8; do env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter $A --time-embed $T --refs $R --n 50 --steps $S --remask --remask-budget $b --skip-tf --out results/pure/bdiff_b1_remask${b}_S$S.json > results/pure/bdiff_b1_remask${b}_S$S.log 2>&1; echo "=== $(date +%T) [zeta] b4(b1) remask$b S$S $(grep -E '^=== bdiff_gen' results/pure/bdiff_b1_remask${b}_S$S.log | cut -c1-400) $(grep -E 'Error' results/pure/bdiff_b1_remask${b}_S$S.log | tail -n1 | cut -c1-120)" >> $L; done; done
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter $A --time-embed $T --refs $R --n 50 --steps 16,8 --schedule cosine --skip-tf --out results/pure/bdiff_b1_cosine_sampling.json > results/pure/bdiff_b1_cosine_sampling.log 2>&1; echo "=== $(date +%T) [zeta] b4(b1) cosine-sampling $(grep -E '^=== bdiff_gen' results/pure/bdiff_b1_cosine_sampling.log | cut -c1-400 | tr '\n' ' ')" >> $L
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter $A --time-embed $T --refs $R --n 50 --steps 16,8 --orders random,left --skip-tf --out results/pure/bdiff_b1_orders.json > results/pure/bdiff_b1_orders.log 2>&1; echo "=== $(date +%T) [zeta] b4(b1) orders $(grep -E '^=== bdiff_gen' results/pure/bdiff_b1_orders.log | cut -c1-600 | tr '\n' ' ')" >> $L
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter $A --time-embed $T --refs $R --skip 50 --n 150 --steps 16,8 --skip-tf --out results/pure/bdiff_b1_heldout.json > results/pure/bdiff_b1_heldout.log 2>&1; echo "=== $(date +%T) [zeta] b1 held-out 50-199 $(grep -E '^=== bdiff_gen' results/pure/bdiff_b1_heldout.log | cut -c1-500 | tr '\n' ' ')" >> $L
echo "=== $(date +%T) [zeta] b chain3 done" >> $L
