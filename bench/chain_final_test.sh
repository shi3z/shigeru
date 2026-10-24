#!/bin/bash
# Final GSM8K test (1 319) rows with FIXED settings (chosen on dev/held-out, never tuned on test):
#   GPU 6: A-series K=4 refineN Jacobi <=6, CUDA graphs (sft3 checkpoint)
#   GPU 7: B-series e5 no-time lr 1e-4 adapter, S=16 confidence and S=8 left-to-right (K=16)
cd /mnt/ssdraid/project/open-mercury
P=bench/prompts_gsm8k_full.json; L=results/pure/final_test.log; R=results/pure/dev_refs_sft3.json
( CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. .venv-train/bin/python bench/run_block.py --model runs/phaseC_sft3_4b/final --prompts $P --n 1319 --block 4 --engine graph --decoder k4 --k4-strategy refineN --anchor-leftmost --max-rounds 6 --timing --out results/pure/final_test_k4_refineN_r6_graph.json > results/pure/final_test_k4_refineN_r6_graph.log 2>&1; echo "=== $(date +%T) [gpu6] TEST A-series K=4 refineN r6 graph $(grep '^=== block' results/pure/final_test_k4_refineN_r6_graph.log | cut -c11-700)" >> $L ) &
( E="CUDA_VISIBLE_DEVICES=7 PYTHONPATH=."; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter
  env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter $A --no-time --refs $R --prompts $P --n 1319 --steps 16 --skip-tf --out results/pure/final_test_b_e5_lr1e4_S16.json > results/pure/final_test_b_e5_lr1e4_S16.log 2>&1; echo "=== $(date +%T) [gpu7] TEST B-series e5 lr1e4 S=16 $(grep -E '^=== bdiff_gen' results/pure/final_test_b_e5_lr1e4_S16.log | cut -c1-400)" >> $L
  env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter $A --no-time --refs $R --prompts $P --n 1319 --steps 8 --orders left --skip-tf --out results/pure/final_test_b_e5_lr1e4_S8left.json > results/pure/final_test_b_e5_lr1e4_S8left.log 2>&1; echo "=== $(date +%T) [gpu7] TEST B-series e5 lr1e4 S=8 left $(grep -E '^=== bdiff_gen' results/pure/final_test_b_e5_lr1e4_S8left.log | cut -c1-400)" >> $L ) &
wait; echo "=== $(date +%T) final test chain done" >> $L
