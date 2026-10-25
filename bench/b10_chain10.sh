#!/bin/bash
# B8 joint teacher reference rows (GPU 7): HumanEval at S=8 (22 forwards/block) and GSM8K test at S=8
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b10_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; E="PYTHONPATH=. .venv-train/bin/python"
CUDA_VISIBLE_DEVICES=7 env $E bench/run_code_eval.py --engine "bdiff:$A:8:left:none:2::joint_fwd:2" --task humaneval --out results/pure/b10_humaneval_teacher_s8.json > results/pure/b10_humaneval_teacher_s8.log 2>&1
echo "=== $(date +%T) [gpu7] TEACHER humaneval S8 joint rc=$? $(grep '^=== humaneval' results/pure/b10_humaneval_teacher_s8.log | cut -c1-200)" >> $L
CUDA_VISIBLE_DEVICES=7 env $E bench/eval_trust.py --model $M --adapter $A --no-time --prompts bench/prompts_gsm8k_full.json --n 1319 --steps 8 --unmask left --remask none --pair joint_fwd --pair-k 2 --skip-tf --out results/pure/b10_test_teacher_s8.json > results/pure/b10_test_teacher_s8.log 2>&1
echo "=== $(date +%T) [gpu7] TEST teacher S8 joint rc=$? $(grep '^=== b7_gen' results/pure/b10_test_teacher_s8.log | cut -c1-300)" >> $L
