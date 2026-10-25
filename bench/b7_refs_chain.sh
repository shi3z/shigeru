#!/bin/bash
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b7_chain.log; M=runs/zeta_sft_final_e5/final
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. .venv-train/bin/python bench/make_refs.py --model $M --prompts bench/prompts_gsm8k_dev.json --n 50 --out results/pure/dev_refs_e5.json > results/pure/make_refs_e5.log 2>&1
echo "=== $(date +%T) [gpu7] refs e5 dev50 rc=$? $(tail -n1 results/pure/make_refs_e5.log)" >> $L
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. .venv-train/bin/python bench/make_refs.py --model $M --prompts bench/prompts_gsm8k_dev2.json --n 150 --out results/pure/dev2_refs_e5.json > results/pure/make_refs_e5_dev2.log 2>&1
echo "=== $(date +%T) [gpu7] refs e5 dev2 rc=$? $(tail -n1 results/pure/make_refs_e5_dev2.log)" >> $L
