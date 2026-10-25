#!/bin/bash
# B10 variant A follow-ups (GPU 6): dev2 S=16 (no-damage check), HumanEval S=8 and S=16 with the student
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b10_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; S=runs/consist_a/adapter; E="PYTHONPATH=. .venv-train/bin/python"
CUDA_VISIBLE_DEVICES=6 env $E bench/eval_trust.py --model $M --adapter $A --adapter2 $S --no-time --prompts bench/prompts_gsm8k_dev2.json --n 150 --steps 16 --unmask left --remask none --skip-tf --out results/pure/b10_a_dev2_s16.json > results/pure/b10_a_dev2_s16.log 2>&1
echo "=== $(date +%T) [gpu6] consist A dev2 S16 rc=$? $(grep '^=== b7_gen' results/pure/b10_a_dev2_s16.log | cut -c1-250)" >> $L
for cfg in "8:left|a_s8" "16:left|a_s16"; do eng=${cfg%|*}; tag=${cfg#*|}
  CUDA_VISIBLE_DEVICES=6 env $E bench/run_code_eval.py --engine bdiff:$A+$S:$eng --task humaneval --out results/pure/b10_humaneval_$tag.json > results/pure/b10_humaneval_$tag.log 2>&1
  echo "=== $(date +%T) [gpu6] consist A humaneval $tag rc=$? $(grep '^=== humaneval' results/pure/b10_humaneval_$tag.log | cut -c1-200)" >> $L; done
