#!/bin/bash
# dev2 (150) confirmation of the forced visible re-mask variants (arm E, S=8): GPU 4 after its dev-50 sweep, GPU 7 after counterfactual part b
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b12_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; S=runs/sa_E_remask_corr/adapter; E="PYTHONPATH=. .venv-train/bin/python"
run() { CUDA_VISIBLE_DEVICES=$1 env $E bench/eval_global.py --model $M --adapter $A --adapter2 $S --prompts bench/prompts_gsm8k_dev2.json --n 150 --modes remask_vis --steps 8 --vis-k $2 --vis-rule $3 --skip-tf --out results/pure/b12_rv_dev2_k$2_$3.json > results/pure/b12_rv_dev2_k$2_$3.log 2>&1
  echo "=== $(date +%T) [gpu$1] b12 remask_vis k$2 $3 dev2 rc=$? $(grep '^=== b9_gen' results/pure/b12_rv_dev2_k$2_$3.log | sed 's/.*=== b9_gen //' | cut -c1-120)" >> $L; }
( until grep -q "remask_vis k3 random dev50" $L; do sleep 30; done; run 4 1 lowconf; run 4 2 lowconf; run 4 1 random ) &
( until grep -q "counterfactual E part b" $L; do sleep 30; done; run 7 3 lowconf; run 7 2 random ) &
wait
