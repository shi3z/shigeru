#!/bin/bash
# B11 §7: training-vs-inference state distributions once the remask states are fetched (GPU 4, shared), and again with the left states
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b11_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; E="PYTHONPATH=. .venv-train/bin/python"
until grep -q "zeta remask states fetched" $L; do sleep 60; done
CUDA_VISIBLE_DEVICES=4 env $E bench/diag_state_dist.py --model $M --adapter $A --train remask=data/distill/gstates_remask --sampler remask --n 50 --out results/pure/b11_state_dist_remask.json > results/pure/b11_state_dist_remask.log 2>&1
echo "=== $(date +%T) [gpu4] b11 state_dist (remask training vs remask inference) rc=$? $(grep '^=== state_dist overlap' results/pure/b11_state_dist_remask.log | cut -c1-400)" >> $L
until grep -q "gstates left rc=" $L; do sleep 60; done
CUDA_VISIBLE_DEVICES=4 env $E bench/diag_state_dist.py --model $M --adapter $A --train remask=data/distill/gstates_remask left=data/distill/gstates_left --sampler remask --n 50 --out results/pure/b11_state_dist_all.json > results/pure/b11_state_dist_all.log 2>&1
echo "=== $(date +%T) [gpu4] b11 state_dist (remask+left training vs remask inference) rc=$? $(grep '^=== state_dist overlap' results/pure/b11_state_dist_all.log | cut -c1-400)" >> $L
CUDA_VISIBLE_DEVICES=4 env $E bench/diag_state_dist.py --model $M --adapter $A --train remask=data/distill/gstates_remask left=data/distill/gstates_left --sampler left --n 50 --out results/pure/b11_state_dist_leftinf.json > results/pure/b11_state_dist_leftinf.log 2>&1
echo "=== $(date +%T) [gpu4] b11 state_dist (vs LEFT inference) rc=$? $(grep '^=== state_dist overlap' results/pure/b11_state_dist_leftinf.log | cut -c1-400)" >> $L
