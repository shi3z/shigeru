#!/bin/bash
# B7: oracle-remask ceiling (GPU 4, teacher-forced); cosine S=8 schedule + HumanEval of the candidates (GPU 6)
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b7_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; R=results/pure/dev_refs_e5.json; E="PYTHONPATH=. .venv-train/bin/python"; P2=bench/prompts_gsm8k_dev2.json
( for B in 2 4 16; do CUDA_VISIBLE_DEVICES=4 env $E bench/eval_trust.py --model $M --adapter $A --no-time --refs $R --n 50 --steps 8 --unmask left,confidence --remask oracle --remask-budget $B --skip-gen --out results/pure/b7_tf_dev50_lr1e4_oracle_b$B.json > results/pure/b7_tf_dev50_lr1e4_oracle_b$B.log 2>&1
  echo "=== $(date +%T) [gpu4] b7 TF oracle b$B rc=$? $(grep '^=== b7_tf' results/pure/b7_tf_dev50_lr1e4_oracle_b$B.log | cut -c1-300 | tr '\n' ' ')" >> $L; done
  CUDA_VISIBLE_DEVICES=4 env $E bench/eval_trust.py --model $M --adapter $A --no-time --refs $R --n 50 --steps 16 --unmask left,confidence --remask none,stability --skip-gen --out results/pure/b7_tf_dev50_lr1e4_s16.json > results/pure/b7_tf_dev50_lr1e4_s16.log 2>&1
  echo "=== $(date +%T) [gpu4] b7 TF S16 rc=$?" >> $L ) &
( CUDA_VISIBLE_DEVICES=6 env $E bench/eval_trust.py --model $M --adapter $A --no-time --prompts $P2 --n 150 --steps 8 --unmask left --remask none,stability --schedule cosine --skip-tf --out results/pure/b7_gen_dev2_lr1e4_cosine.json > results/pure/b7_gen_dev2_lr1e4_cosine.log 2>&1
  echo "=== $(date +%T) [gpu6] b7 dev2 cosine rc=$?" >> $L
  for cfg in "bdiff:$A:8:left:stability:2|b_s8_left_stab2" "bdiff:$A:16:left|b_s16_left"; do eng=${cfg%|*}; tag=${cfg#*|}
    CUDA_VISIBLE_DEVICES=6 env $E bench/run_code_eval.py --engine $eng --task humaneval --out results/pure/b7_humaneval_$tag.json > results/pure/b7_humaneval_$tag.log 2>&1
    echo "=== $(date +%T) [gpu6] humaneval $tag rc=$? $(grep '^=== humaneval' results/pure/b7_humaneval_$tag.log | cut -c1-200)" >> $L; done ) &
wait
