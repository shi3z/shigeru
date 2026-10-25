#!/bin/bash
# B10.5 §3-4 code-structure diagnostics (GPU 4, after student A'): GEN S=8/16 for student E, student A, baseline; TF by token category on HumanEval and GSM8K refs
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b10_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; E="PYTHONPATH=. .venv-train/bin/python"
until grep -q "consist A' dev2 S8" $L; do sleep 30; done
for cfg in "runs/consist_e/adapter|e" "runs/consist_a/adapter|a" "|base"; do ad=${cfg%|*}; tag=${cfg#*|}
  X=""; [ -n "$ad" ] && X="--adapter2 $ad"
  CUDA_VISIBLE_DEVICES=4 env $E bench/diag_code_structure.py --model $M --adapter $A $X --steps 8,16 --n 164 --out results/pure/code_struct_gen_$tag.json > results/pure/code_struct_gen_$tag.log 2>&1
  echo "=== $(date +%T) [gpu4] code_struct gen $tag rc=$? $(grep '^=== gen 8' results/pure/code_struct_gen_$tag.log | cut -c1-300)" >> $L
done
for cfg in "|base" "runs/consist_e/adapter|e"; do ad=${cfg%|*}; tag=${cfg#*|}
  X=""; [ -n "$ad" ] && X="--adapter2 $ad"
  CUDA_VISIBLE_DEVICES=4 env $E bench/diag_code_structure.py --model $M --adapter $A $X --steps 8,16 --n 64 --prompts bench/prompts_humaneval.json --refs results/pure/humaneval_refs_e5.json --out results/pure/code_struct_tf_he_$tag.json > results/pure/code_struct_tf_he_$tag.log 2>&1
  echo "=== $(date +%T) [gpu4] code_struct tf humaneval $tag rc=$?" >> $L
  CUDA_VISIBLE_DEVICES=4 env $E bench/diag_code_structure.py --model $M --adapter $A $X --steps 8,16 --n 50 --prompts bench/prompts_gsm8k_dev.json --refs results/pure/dev_refs_e5.json --max-blocks 6 --out results/pure/code_struct_tf_gsm_$tag.json > results/pure/code_struct_tf_gsm_$tag.log 2>&1
  echo "=== $(date +%T) [gpu4] code_struct tf gsm8k $tag rc=$?" >> $L
done
