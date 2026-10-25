#!/bin/bash
# B9-A trajectory analysis: GPU 4 = S=16 left teacher-forced (REF + FINAL targets, hidden states saved); GPU 6 = free generation S=16/S=8 left, then S=16 confidence tf
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b9_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; R=results/pure/dev_refs_e5.json; E="PYTHONPATH=. .venv-train/bin/python"
( CUDA_VISIBLE_DEVICES=4 env $E bench/diag_trajectory.py --model $M --adapter $A --n 50 --steps 16 --unmask left --mode tf --refs $R --save-hidden results/pure/traj_s16_left_tf_hidden.pt --out results/pure/traj_s16_left_tf.json > results/pure/traj_s16_left_tf.log 2>&1
  echo "=== $(date +%T) [gpu4] traj S16 left tf rc=$?" >> $L
  CUDA_VISIBLE_DEVICES=4 env $E bench/diag_trajectory.py --model $M --adapter $A --n 50 --steps 8 --unmask left --mode tf --refs $R --out results/pure/traj_s8_left_tf.json > results/pure/traj_s8_left_tf.log 2>&1
  echo "=== $(date +%T) [gpu4] traj S8 left tf rc=$?" >> $L ) &
( CUDA_VISIBLE_DEVICES=6 env $E bench/diag_trajectory.py --model $M --adapter $A --n 50 --steps 16 --unmask left --mode free --out results/pure/traj_s16_left_free.json > results/pure/traj_s16_left_free.log 2>&1
  echo "=== $(date +%T) [gpu6] traj S16 left free rc=$?" >> $L
  CUDA_VISIBLE_DEVICES=6 env $E bench/diag_trajectory.py --model $M --adapter $A --n 50 --steps 8 --unmask left --mode free --out results/pure/traj_s8_left_free.json > results/pure/traj_s8_left_free.log 2>&1
  echo "=== $(date +%T) [gpu6] traj S8 left free rc=$?" >> $L
  CUDA_VISIBLE_DEVICES=6 env $E bench/diag_trajectory.py --model $M --adapter $A --n 50 --steps 16 --unmask confidence --mode tf --refs $R --out results/pure/traj_s16_conf_tf.json > results/pure/traj_s16_conf_tf.log 2>&1
  echo "=== $(date +%T) [gpu6] traj S16 conf tf rc=$?" >> $L ) &
wait
