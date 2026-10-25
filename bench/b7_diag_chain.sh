#!/bin/bash
# Phase B7 §2 trust diagnostics (GPU 4: lr1e-4 adapter, GPU 6: no-time lr2e-4 adapter)
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b7_chain.log
M=runs/zeta_sft_final_e5/final; R=results/pure/dev_refs_sft3.json
( CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv-train/bin/python bench/diag_trust.py --model $M --adapter runs/bdiff_e5_lr1e4/adapter --no-time --refs $R --n 50 --steps 8 --unmask confidence,left --oracle --oracle-blocks 40 --out results/pure/diag_trust_lr1e4.json > results/pure/diag_trust_lr1e4.log 2>&1; echo "=== $(date +%T) [gpu4] diag_trust lr1e4 rc=$? $(grep '^=== trust ' results/pure/diag_trust_lr1e4.log | cut -c1-600)" >> $L ) &
( CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. .venv-train/bin/python bench/diag_trust.py --model $M --adapter runs/bdiff_e5_notime/adapter --no-time --refs $R --n 50 --steps 8 --unmask confidence,left --oracle --oracle-blocks 40 --out results/pure/diag_trust_notime.json > results/pure/diag_trust_notime.log 2>&1; echo "=== $(date +%T) [gpu6] diag_trust notime rc=$? $(grep '^=== trust ' results/pure/diag_trust_notime.log | cut -c1-600)" >> $L ) &
wait
