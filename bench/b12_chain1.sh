#!/bin/bash
# B12-A/B/C counterfactual interventions on the B11 best global sampler (arm E), dev-50, blocks 0-2, intervention after step 4: GPU 4 prompts 0-24, GPU 7 prompts 25-49
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b12_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; S=runs/sa_E_remask_corr/adapter; E="PYTHONPATH=. .venv-train/bin/python"
for cfg in "4 0 a" "7 25 b"; do set -- $cfg
  ( CUDA_VISIBLE_DEVICES=$1 env $E bench/diag_counterfactual.py --model $M --adapter $A --adapter2 $S --skip $2 --n 25 --blocks 0,1,2 --at-step 4 --out results/pure/b12_cf_E_$3.json > results/pure/b12_cf_E_$3.log 2>&1
    echo "=== $(date +%T) [gpu$1] b12 counterfactual E part $3 rc=$? $(grep '^=== counterfactual' results/pure/b12_cf_E_$3.log | cut -c1-600)" >> $L ) &
done
wait
