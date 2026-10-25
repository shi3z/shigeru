#!/bin/bash
# B7 round 2: GPU 4 = old-sampler consistency check + teacher-forced correction stats (e5 refs) for every rule;
# GPU 6 = trust-based remask/unmask + budget variants on dev 50; GPU 7 = dev2 held-out (150) for the first candidates.
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b7_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; R=results/pure/dev_refs_e5.json
T=results/pure/trust_lr_e5refs_lr1e4_left_half.json; TM=results/pure/trust_lr_e5refs_lr1e4_left_masked_half.json
E="PYTHONPATH=. .venv-train/bin/python"
( CUDA_VISIBLE_DEVICES=4 env $E bench/eval_bdiff.py --model $M --adapter $A --no-time --refs $R --n 50 --steps 8 --orders confidence,left --skip-tf --out results/pure/b7_oldsampler_dev50_lr1e4.json > results/pure/b7_oldsampler_dev50_lr1e4.log 2>&1
  echo "=== $(date +%T) [gpu4] old sampler S8 dev50 rc=$? $(grep '^=== bdiff_gen' results/pure/b7_oldsampler_dev50_lr1e4.log | cut -c1-200 | tr '\n' ' ')" >> $L
  CUDA_VISIBLE_DEVICES=4 env $E bench/eval_trust.py --model $M --adapter $A --no-time --refs $R --n 50 --steps 8 --unmask confidence,left --remask none,disagree,lowprob,stability,dependency,trust --trust $T --skip-gen --out results/pure/b7_tf_dev50_lr1e4.json > results/pure/b7_tf_dev50_lr1e4.log 2>&1
  echo "=== $(date +%T) [gpu4] b7 TF dev50 rc=$?" >> $L
  CUDA_VISIBLE_DEVICES=4 env $E bench/eval_trust.py --model $M --adapter $A --no-time --refs $R --n 50 --steps 8 --unmask left --remask disagree --remask-budget 4 --skip-gen --out results/pure/b7_tf_dev50_lr1e4_b4.json > results/pure/b7_tf_dev50_lr1e4_b4.log 2>&1
  echo "=== $(date +%T) [gpu4] b7 TF dev50 b4 rc=$?" >> $L ) &
( CUDA_VISIBLE_DEVICES=6 env $E bench/eval_trust.py --model $M --adapter $A --no-time --n 50 --steps 8 --unmask confidence,left,trust --remask none,trust --trust $T --trust-mask $TM --tau-remask 0.5 --skip-tf --out results/pure/b7_gen_dev50_lr1e4_trust.json > results/pure/b7_gen_dev50_lr1e4_trust.log 2>&1
  echo "=== $(date +%T) [gpu6] b7 gen trust rc=$?" >> $L
  CUDA_VISIBLE_DEVICES=6 env $E bench/eval_trust.py --model $M --adapter $A --no-time --n 50 --steps 8 --unmask left --remask trust --trust $T --tau-remask 0.3 --skip-tf --out results/pure/b7_gen_dev50_lr1e4_trust03.json > results/pure/b7_gen_dev50_lr1e4_trust03.log 2>&1
  echo "=== $(date +%T) [gpu6] b7 gen trust t0.3 rc=$?" >> $L
  for B in 1 4; do
    CUDA_VISIBLE_DEVICES=6 env $E bench/eval_trust.py --model $M --adapter $A --no-time --n 50 --steps 8 --unmask left,confidence --remask disagree,stability --remask-budget $B --skip-tf --out results/pure/b7_gen_dev50_lr1e4_b$B.json > results/pure/b7_gen_dev50_lr1e4_b$B.log 2>&1
    echo "=== $(date +%T) [gpu6] b7 gen budget $B rc=$?" >> $L
  done ) &
( CUDA_VISIBLE_DEVICES=7 env $E bench/eval_trust.py --model $M --adapter $A --no-time --prompts bench/prompts_gsm8k_dev2.json --n 150 --steps 8 --unmask left,confidence --remask none,disagree,stability,dependency --skip-tf --out results/pure/b7_gen_dev2_lr1e4_a.json > results/pure/b7_gen_dev2_lr1e4_a.log 2>&1
  echo "=== $(date +%T) [gpu7] b7 dev2 sweep a rc=$?" >> $L ) &
wait
