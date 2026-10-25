#!/bin/bash
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b9_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; R=results/pure/dev_refs_e5.json; E="PYTHONPATH=. .venv-train/bin/python"
until grep -q "traj S8 left tf rc=" $L; do sleep 20; done
CUDA_VISIBLE_DEVICES=4 env $E bench/diag_trajectory.py --model $M --adapter $A --n 50 --steps 16 --unmask left --mode tf --refs $R --out results/pure/traj_s16_left_tf.json > results/pure/traj_s16_left_tf.log 2>&1
echo "=== $(date +%T) [gpu4] traj S16 left tf (with distance JSD) rc=$?" >> $L
CUDA_VISIBLE_DEVICES=4 env $E bench/diag_trajectory.py --model $M --adapter $A --n 50 --steps 4 --unmask left --mode tf --refs $R --out results/pure/traj_s4_left_tf.json > results/pure/traj_s4_left_tf.log 2>&1
echo "=== $(date +%T) [gpu4] traj S4 left tf rc=$?" >> $L
