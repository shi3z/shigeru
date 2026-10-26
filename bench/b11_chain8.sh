#!/bin/bash
# B11 §6 (protect/measure abilities) with arm E on GPU 4: fill accuracy on the global sampler's own states (states diag with adapter2), HumanEval S=8/16
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b11_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; S=runs/sa_E_remask_corr/adapter; R=results/pure/dev_refs_e5.json; E="PYTHONPATH=. .venv-train/bin/python"
CUDA_VISIBLE_DEVICES=4 env $E bench/diag_sampler_states.py --model $M --adapter $A --adapter2 $S --refs $R --n 50 --samplers remask --out results/pure/b11_states_remask_E.json > results/pure/b11_states_remask_E.log 2>&1
echo "=== $(date +%T) [gpu4] b11 states remask with arm E rc=$?" >> $L
for cfg in "8:left|E_s8" "16:left|E_s16"; do eng=${cfg%|*}; tag=${cfg#*|}
  CUDA_VISIBLE_DEVICES=4 env $E bench/run_code_eval.py --engine "bdiff:$A+$S:$eng" --task humaneval --out results/pure/b11_humaneval_$tag.json > results/pure/b11_humaneval_$tag.log 2>&1
  echo "=== $(date +%T) [gpu4] b11 humaneval $tag rc=$? $(grep '^=== humaneval' results/pure/b11_humaneval_$tag.log | cut -c1-200)" >> $L; done
