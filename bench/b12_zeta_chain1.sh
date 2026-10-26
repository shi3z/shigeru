#!/bin/bash
# B12 counterfactual interventions on the UNDISTILLED adapter (comparison: does the frozen model have the same repair room?) — zeta, dev-50, blocks 0-2
cd ~/shigeru/open-mercury
L=results/pure/b12_zeta_chain.log; E="SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-train/bin/python"; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter
env $E bench/diag_counterfactual.py --model $M --adapter $A --n 50 --blocks 0,1,2 --at-step 4 --out results/pure/b12_cf_base.json > results/pure/b12_cf_base.log 2>&1
echo "=== $(date +%T) [zeta] b12 counterfactual baseline rc=$? $(grep '^=== counterfactual' results/pure/b12_cf_base.log | cut -c1-600)" >> $L
