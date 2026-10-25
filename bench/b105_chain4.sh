#!/bin/bash
# B10.5 §6: does the arm-B student help a GLOBAL (non-monotonic re-masking) sampler? dev-50, remask vs monotonic, S=4/8/16 (GPU 4)
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b10_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; R=results/pure/dev_refs_e5.json; E="PYTHONPATH=. .venv-train/bin/python"
CUDA_VISIBLE_DEVICES=4 env $E bench/eval_global.py --model $M --adapter $A --adapter2 runs/abl_maskce/adapter --refs $R --n 50 --modes monotonic,remask --steps 4,8,16 --out results/pure/b105_global_maskce_dev50.json > results/pure/b105_global_maskce_dev50.log 2>&1
echo "=== $(date +%T) [gpu4] global samplers with arm-B student rc=$? $(grep '^=== b9_gen' results/pure/b105_global_maskce_dev50.log | sed 's/.*=== b9_gen //' | cut -c1-60 | tr '\n' ' ')" >> $L
