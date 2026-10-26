#!/bin/bash
# B11 §2/§5: mismatch taxonomy + global-sampler dynamics on dev-50 (baseline adapter): remask on GPU 4 (after the lc diagnostic), left on GPU 6 (shared with the collection)
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b11_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; E="PYTHONPATH=. .venv-train/bin/python"
( until grep -q "b11 states lc rc=" $L; do sleep 30; done
  CUDA_VISIBLE_DEVICES=4 env $E bench/diag_mismatch.py --model $M --adapter $A --n 50 --sampler remask --out results/pure/b11_mismatch_remask.json > results/pure/b11_mismatch_remask.log 2>&1
  echo "=== $(date +%T) [gpu4] b11 mismatch remask rc=$? $(grep '^=== mismatch ' results/pure/b11_mismatch_remask.log | cut -c1-600)" >> $L ) &
( CUDA_VISIBLE_DEVICES=6 env $E bench/diag_mismatch.py --model $M --adapter $A --n 50 --sampler left --out results/pure/b11_mismatch_left.json > results/pure/b11_mismatch_left.log 2>&1
  echo "=== $(date +%T) [gpu6] b11 mismatch left rc=$? $(grep '^=== mismatch ' results/pure/b11_mismatch_left.log | cut -c1-600)" >> $L ) &
wait
