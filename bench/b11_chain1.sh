#!/bin/bash
# B11-A/B: sampler state distributions + teacher-correction feasibility on dev-50 reference blocks (GPU 4: left,confidence; GPU 6: remask; GPU 7: sync)
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b11_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; R=results/pure/dev_refs_e5.json; E="PYTHONPATH=. .venv-train/bin/python"
for cfg in "4 left,confidence lc" "6 remask remask" "7 sync sync"; do set -- $cfg
  ( CUDA_VISIBLE_DEVICES=$1 env $E bench/diag_sampler_states.py --model $M --adapter $A --refs $R --n 50 --samplers $2 --out results/pure/b11_states_$3.json > results/pure/b11_states_$3.log 2>&1
    echo "=== $(date +%T) [gpu$1] b11 states $3 rc=$?" >> $L ) &
done
wait
