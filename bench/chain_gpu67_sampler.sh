#!/bin/bash
# GPUs 6/7 (eval only): S = 12 / 6 points and left-to-right order at S = 8 on the two best adapters, held-out items 50-199.
cd "$(dirname "$0")/.."
M=runs/zeta_sft_final_e5/final; R=results/pure/dev_refs_sft3.json; L=results/pure/b_chain.log
( E="CUDA_VISIBLE_DEVICES=6 PYTHONPATH=."; A=runs/bdiff_e5_notime/adapter
  env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter $A --no-time --refs $R --skip 50 --n 150 --steps 12,6 --skip-tf --out results/pure/bdiff_e5_notime_heldout_s12_6.json > results/pure/bdiff_e5_notime_heldout_s12_6.log 2>&1; echo "=== $(date +%T) [gpu6] e5_notime held-out S=12/6 $(grep -E '^=== bdiff_gen' results/pure/bdiff_e5_notime_heldout_s12_6.log | cut -c1-500 | sed ':a;N;$!ba;s/\n/ /g')" >> $L
  env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter $A --no-time --refs $R --skip 50 --n 150 --steps 8 --orders left --skip-tf --out results/pure/bdiff_e5_notime_heldout_left8.json > results/pure/bdiff_e5_notime_heldout_left8.log 2>&1; echo "=== $(date +%T) [gpu6] e5_notime held-out S=8 left $(grep -E '^=== bdiff_gen' results/pure/bdiff_e5_notime_heldout_left8.log | cut -c1-500 | sed ':a;N;$!ba;s/\n/ /g')" >> $L ) &
( E="CUDA_VISIBLE_DEVICES=7 PYTHONPATH=."; A=runs/bdiff_e5_lr1e4/adapter
  env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter $A --no-time --refs $R --skip 50 --n 150 --steps 12,6 --skip-tf --out results/pure/bdiff_e5_lr1e4_heldout_s12_6.json > results/pure/bdiff_e5_lr1e4_heldout_s12_6.log 2>&1; echo "=== $(date +%T) [gpu7] e5_lr1e4 held-out S=12/6 $(grep -E '^=== bdiff_gen' results/pure/bdiff_e5_lr1e4_heldout_s12_6.log | cut -c1-500 | sed ':a;N;$!ba;s/\n/ /g')" >> $L
  env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter $A --no-time --refs $R --skip 50 --n 150 --steps 8 --orders left --skip-tf --out results/pure/bdiff_e5_lr1e4_heldout_left8.json > results/pure/bdiff_e5_lr1e4_heldout_left8.log 2>&1; echo "=== $(date +%T) [gpu7] e5_lr1e4 held-out S=8 left $(grep -E '^=== bdiff_gen' results/pure/bdiff_e5_lr1e4_heldout_left8.log | cut -c1-500 | sed ':a;N;$!ba;s/\n/ /g')" >> $L ) &
wait; echo "=== $(date +%T) [gpu6/7] sampler chain done" >> $L
