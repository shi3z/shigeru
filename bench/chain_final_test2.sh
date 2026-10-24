#!/bin/bash
# GPU 6: second A-series test row — K=4 refineN Jacobi <=3 rounds (the faster point), CUDA graphs, fixed beforehand.
cd /mnt/ssdraid/project/open-mercury
P=bench/prompts_gsm8k_full.json; L=results/pure/final_test.log
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. .venv-train/bin/python bench/run_block.py --model runs/phaseC_sft3_4b/final --prompts $P --n 1319 --block 4 --engine graph --decoder k4 --k4-strategy refineN --anchor-leftmost --max-rounds 3 --timing --out results/pure/final_test_k4_refineN_r3_graph.json > results/pure/final_test_k4_refineN_r3_graph.log 2>&1; echo "=== $(date +%T) [gpu6] TEST A-series K=4 refineN r3 graph $(grep '^=== block' results/pure/final_test_k4_refineN_r3_graph.log | cut -c11-700)" >> $L
