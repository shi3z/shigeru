#!/bin/bash
# B8 §2: pairwise dependency diagnostic (GPU 4: left order; GPU 6: generation with joint re-ranking, then confidence-order diagnostic)
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b8_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; R=results/pure/dev_refs_e5.json; E="PYTHONPATH=. .venv-train/bin/python"
( CUDA_VISIBLE_DEVICES=4 env $E bench/diag_pairwise.py --model $M --adapter $A --no-time --refs $R --n 50 --unmask left --out results/pure/diag_pairwise_lr1e4_left.json > results/pure/diag_pairwise_lr1e4_left.log 2>&1
  echo "=== $(date +%T) [gpu4] diag_pairwise left rc=$? $(grep '^=== pairwise' results/pure/diag_pairwise_lr1e4_left.log | cut -c1-1500)" >> $L ) &
( for cfg in "joint_fwd 2" "joint_both 2" "joint_fwd 4"; do set -- $cfg
    CUDA_VISIBLE_DEVICES=6 env $E bench/eval_trust.py --model $M --adapter $A --no-time --n 50 --steps 8 --unmask left --remask none --pair $1 --pair-k $2 --skip-tf --out results/pure/b8_gen_dev50_lr1e4_$1_k$2.json > results/pure/b8_gen_dev50_lr1e4_$1_k$2.log 2>&1
    echo "=== $(date +%T) [gpu6] b8 gen dev50 $1 k$2 rc=$? $(grep '^=== b7_gen' results/pure/b8_gen_dev50_lr1e4_$1_k$2.log | cut -c1-300)" >> $L; done
  CUDA_VISIBLE_DEVICES=6 env $E bench/diag_pairwise.py --model $M --adapter $A --no-time --refs $R --n 50 --unmask confidence --out results/pure/diag_pairwise_lr1e4_conf.json > results/pure/diag_pairwise_lr1e4_conf.log 2>&1
  echo "=== $(date +%T) [gpu6] diag_pairwise confidence rc=$? $(grep '^=== pairwise' results/pure/diag_pairwise_lr1e4_conf.log | cut -c1-1500)" >> $L ) &
wait
