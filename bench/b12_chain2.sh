#!/bin/bash
# B12: zero-cost forced visible re-masking inside the global sampler (arm E), dev-50: k=1/2/3 lowest-confidence and random, vs plain remask — GPU 4
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b12_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; S=runs/sa_E_remask_corr/adapter; R=results/pure/dev_refs_e5.json; E="PYTHONPATH=. .venv-train/bin/python"
for cfg in "1 lowconf" "2 lowconf" "3 lowconf" "2 random" "3 random"; do set -- $cfg
  CUDA_VISIBLE_DEVICES=4 env $E bench/eval_global.py --model $M --adapter $A --adapter2 $S --refs $R --n 50 --modes remask_vis --steps 8 --vis-k $1 --vis-rule $2 --skip-tf --out results/pure/b12_rv_dev50_k$1_$2.json > results/pure/b12_rv_dev50_k$1_$2.log 2>&1
  echo "=== $(date +%T) [gpu4] b12 remask_vis k$1 $2 dev50 rc=$? $(grep '^=== b9_gen' results/pure/b12_rv_dev50_k$1_$2.log | sed 's/.*=== b9_gen //' | cut -c1-120)" >> $L
done
