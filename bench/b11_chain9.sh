#!/bin/bash
# arm E on GPU 7: dev2 remask S=4/16, and the dynamics/mismatch diagnostic with E under the global sampler (dev-50)
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b11_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; S=runs/sa_E_remask_corr/adapter; E="PYTHONPATH=. .venv-train/bin/python"
CUDA_VISIBLE_DEVICES=7 env $E bench/eval_global.py --model $M --adapter $A --adapter2 $S --prompts bench/prompts_gsm8k_dev2.json --n 150 --modes remask --steps 4,16 --skip-tf --out results/pure/b11_eval_E_dev2_s416.json > results/pure/b11_eval_E_dev2_s416.log 2>&1
echo "=== $(date +%T) [gpu7] b11 eval E dev2 remask S4/S16 rc=$? $(grep '^=== b9_gen' results/pure/b11_eval_E_dev2_s416.log | sed 's/.*=== b9_gen //' | cut -c1-60 | tr '\n' ' ')" >> $L
CUDA_VISIBLE_DEVICES=7 env $E bench/diag_mismatch.py --model $M --adapter $A --adapter2 $S --n 50 --sampler remask --out results/pure/b11_mismatch_remask_E.json > results/pure/b11_mismatch_remask_E.log 2>&1
echo "=== $(date +%T) [gpu7] b11 mismatch remask with arm E rc=$? $(grep '^=== mismatch ' results/pure/b11_mismatch_remask_E.log | cut -c1-500)" >> $L
