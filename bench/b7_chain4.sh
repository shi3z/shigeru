#!/bin/bash
# B7 round 3 (post tie-break fix): GPU 4 = teacher-forced stats; GPU 6 = dev-50 generation sweep; GPU 7 = dev2 (150) candidates
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b7_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; R=results/pure/dev_refs_e5.json
T=results/pure/trust_lr_e5refs_lr1e4_left_half.json; TM=results/pure/trust_lr_e5refs_lr1e4_left_masked_half.json
E="PYTHONPATH=. .venv-train/bin/python"; RM=none,disagree,lowprob,stability,dependency,trust
( CUDA_VISIBLE_DEVICES=4 env $E bench/eval_trust.py --model $M --adapter $A --no-time --refs $R --n 50 --steps 8 --unmask confidence,left,trust --remask $RM --trust $T --trust-mask $TM --skip-gen --out results/pure/b7_tf_dev50_lr1e4.json > results/pure/b7_tf_dev50_lr1e4.log 2>&1
  echo "=== $(date +%T) [gpu4] b7 TF dev50 rc=$?" >> $L
  for B in 1 4; do CUDA_VISIBLE_DEVICES=4 env $E bench/eval_trust.py --model $M --adapter $A --no-time --refs $R --n 50 --steps 8 --unmask left --remask disagree,trust --trust $T --remask-budget $B --skip-gen --out results/pure/b7_tf_dev50_lr1e4_b$B.json > results/pure/b7_tf_dev50_lr1e4_b$B.log 2>&1
  echo "=== $(date +%T) [gpu4] b7 TF dev50 budget $B rc=$?" >> $L; done
  CUDA_VISIBLE_DEVICES=4 env $E bench/eval_trust.py --model $M --adapter $A --no-time --refs $R --n 50 --steps 8 --unmask nonadjacent,dependency,adaptive --remask none,disagree,dependency --skip-tf --out results/pure/b7_gen_dev50_lr1e4_b.json > results/pure/b7_gen_dev50_lr1e4_b.log 2>&1
  echo "=== $(date +%T) [gpu4] b7 gen dev50 sweep b rc=$?" >> $L ) &
( CUDA_VISIBLE_DEVICES=6 env $E bench/eval_trust.py --model $M --adapter $A --no-time --n 50 --steps 8 --unmask confidence,left,trust --remask $RM --trust $T --trust-mask $TM --skip-tf --out results/pure/b7_gen_dev50_lr1e4_a.json > results/pure/b7_gen_dev50_lr1e4_a.log 2>&1
  echo "=== $(date +%T) [gpu6] b7 gen dev50 sweep a rc=$?" >> $L
  for B in 1 4; do CUDA_VISIBLE_DEVICES=6 env $E bench/eval_trust.py --model $M --adapter $A --no-time --n 50 --steps 8 --unmask left,confidence --remask disagree,stability,trust --trust $T --remask-budget $B --skip-tf --out results/pure/b7_gen_dev50_lr1e4_b$B.json > results/pure/b7_gen_dev50_lr1e4_b$B.log 2>&1
  echo "=== $(date +%T) [gpu6] b7 gen dev50 budget $B rc=$?" >> $L; done
  CUDA_VISIBLE_DEVICES=6 env $E bench/eval_trust.py --model $M --adapter $A --no-time --n 50 --steps 8 --unmask left --remask trust --trust $T --tau-remask 0.3 --skip-tf --out results/pure/b7_gen_dev50_lr1e4_trust03.json > results/pure/b7_gen_dev50_lr1e4_trust03.log 2>&1
  echo "=== $(date +%T) [gpu6] b7 gen dev50 trust t0.3 rc=$?" >> $L ) &
( CUDA_VISIBLE_DEVICES=7 env $E bench/eval_trust.py --model $M --adapter $A --no-time --prompts bench/prompts_gsm8k_dev2.json --n 150 --steps 8 --unmask left,confidence --remask none,disagree,stability,dependency,trust --trust $T --skip-tf --out results/pure/b7_gen_dev2_lr1e4_a.json > results/pure/b7_gen_dev2_lr1e4_a.log 2>&1
  echo "=== $(date +%T) [gpu7] b7 dev2 sweep a rc=$?" >> $L ) &
wait
