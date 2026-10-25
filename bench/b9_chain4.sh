#!/bin/bash
# B9-C variants in parallel: GPU 4 = soft/anneal with milder annealing (t_end 0.5, alpha_end 0.5); GPU 6 = remask with cosine schedule
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b9_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; R=results/pure/dev_refs_e5.json; E="PYTHONPATH=. .venv-train/bin/python"
( CUDA_VISIBLE_DEVICES=4 env $E bench/eval_global.py --model $M --adapter $A --refs $R --n 50 --modes anneal --steps 4,8,16 --t-end 0.5 --alpha-end 0.5 --out results/pure/b9_global_dev50_anneal_mild.json > results/pure/b9_global_dev50_anneal_mild.log 2>&1
  echo "=== $(date +%T) [gpu4] b9 global anneal mild rc=$?" >> $L ) &
( CUDA_VISIBLE_DEVICES=6 env $E bench/eval_global.py --model $M --adapter $A --refs $R --n 50 --modes remask --steps 4,8,16 --schedule cosine --out results/pure/b9_global_dev50_remask_cosine.json > results/pure/b9_global_dev50_remask_cosine.log 2>&1
  echo "=== $(date +%T) [gpu6] b9 global remask cosine rc=$?" >> $L ) &
wait
