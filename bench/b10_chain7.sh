#!/bin/bash
# student A at other budgets on dev2 (GPU 6): S=4 and S=12, plus baseline S=4/S=12 for the quality-per-forward curve
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b10_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; S=runs/consist_a/adapter; E="PYTHONPATH=. .venv-train/bin/python"
for st in 4 12; do
  CUDA_VISIBLE_DEVICES=6 env $E bench/eval_trust.py --model $M --adapter $A --adapter2 $S --no-time --prompts bench/prompts_gsm8k_dev2.json --n 150 --steps $st --unmask left --remask none --skip-tf --out results/pure/b10_a_dev2_s$st.json > results/pure/b10_a_dev2_s$st.log 2>&1
  echo "=== $(date +%T) [gpu6] consist A dev2 S$st rc=$? $(grep '^=== b7_gen' results/pure/b10_a_dev2_s$st.log | cut -c1-250)" >> $L
  CUDA_VISIBLE_DEVICES=6 env $E bench/eval_trust.py --model $M --adapter $A --no-time --prompts bench/prompts_gsm8k_dev2.json --n 150 --steps $st --unmask left --remask none --skip-tf --out results/pure/b10_base_dev2_s$st.json > results/pure/b10_base_dev2_s$st.log 2>&1
  echo "=== $(date +%T) [gpu6] baseline dev2 S$st rc=$? $(grep '^=== b7_gen' results/pure/b10_base_dev2_s$st.log | cut -c1-250)" >> $L
done
