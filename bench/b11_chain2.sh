#!/bin/bash
# B11-C in parallel with zeta (which does remask 800-1099): GPU 6 = left states (1100-1399) after its diagnostic; GPU 7 = extra remask states (1400-1499) after its diagnostic
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b11_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; E="PYTHONPATH=. .venv-train/bin/python"
mkdir -p data/distill
( until grep -q "b11 states remask rc=" $L; do sleep 30; done
  CUDA_VISIBLE_DEVICES=6 env $E distill/gen_global_states.py --model $M --adapter $A --prompts bench/prompts_gsm8k_pairtrain.json --skip 1100 --n 300 --sampler left --out data/distill/gstates_left > results/pure/gen_gstates_left.log 2>&1
  echo "=== $(date +%T) [gpu6] gstates left rc=$? $(tail -n1 results/pure/gen_gstates_left.log | cut -c1-200)" >> $L ) &
( until grep -q "b11 states sync rc=" $L; do sleep 30; done
  CUDA_VISIBLE_DEVICES=7 env $E distill/gen_global_states.py --model $M --adapter $A --prompts bench/prompts_gsm8k_pairtrain.json --skip 1400 --n 100 --sampler remask --out data/distill/gstates_remask_b > results/pure/gen_gstates_remask_b.log 2>&1
  echo "=== $(date +%T) [gpu7] gstates remask_b rc=$? $(tail -n1 results/pure/gen_gstates_remask_b.log | cut -c1-200)" >> $L ) &
wait
