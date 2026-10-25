#!/bin/bash
# B8: research-ceiling generation on the fresh dev2 set — chain-rule joint re-ranking (2 extra forwards per step) at S=8 left (GPU 4)
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b8_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; E="PYTHONPATH=. .venv-train/bin/python"
CUDA_VISIBLE_DEVICES=4 env $E bench/eval_trust.py --model $M --adapter $A --no-time --prompts bench/prompts_gsm8k_dev2.json --n 150 --steps 8 --unmask left --remask none --pair joint_fwd --pair-k 2 --skip-tf --out results/pure/b8_gen_dev2_lr1e4_joint_fwd_k2.json > results/pure/b8_gen_dev2_lr1e4_joint_fwd_k2.log 2>&1
echo "=== $(date +%T) [gpu4] b8 dev2 joint_fwd k2 rc=$? $(grep '^=== b7_gen' results/pure/b8_gen_dev2_lr1e4_joint_fwd_k2.log | cut -c1-300)" >> $L
