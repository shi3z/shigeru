#!/bin/bash
# B7 HumanEval baselines on zeta: AR greedy (e5 base), B-series S=16 confidence, S=8 left (lr1e-4 adapter); 512-token budget
cd ~/shigeru/open-mercury
L=results/pure/b7_zeta_chain.log; E="PYTHONPATH=. .venv-train/bin/python"
A=runs/bdiff_e5_lr1e4/adapter
true
true
for cfg in "bdiff:$A:16:confidence|b_s16_conf" "bdiff:$A:8:left|b_s8_left" "bdiff:$A:8:confidence|b_s8_conf"; do
  eng=${cfg%|*}; tag=${cfg#*|}
  env $E bench/run_code_eval.py --engine $eng --task humaneval --out results/pure/b7_humaneval_$tag.json > results/pure/b7_humaneval_$tag.log 2>&1
  echo "=== $(date +%T) [zeta] humaneval $tag rc=$? $(grep '^=== humaneval' results/pure/b7_humaneval_$tag.log | cut -c1-200)" >> $L
done
