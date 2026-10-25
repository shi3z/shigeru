#!/bin/bash
# B7 §7: AR greedy (HF generate) of the B-series base model on the full GSM8K test — accuracy reference for the comparison table (GPU 7)
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b7_chain.log; E="PYTHONPATH=. .venv-train/bin/python"
CUDA_VISIBLE_DEVICES=7 env $E bench/make_refs.py --model runs/zeta_sft_final_e5/final --prompts bench/prompts_gsm8k_full.json --n 1319 --out results/pure/test_refs_e5.json > results/pure/test_refs_e5.log 2>&1
echo "=== $(date +%T) [gpu7] AR e5 GSM8K test rc=$? $(tail -n1 results/pure/test_refs_e5.log | cut -c1-200)" >> $L
