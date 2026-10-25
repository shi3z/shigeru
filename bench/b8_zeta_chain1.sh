#!/bin/bash
# B8 §2: code side — AR references on HumanEval prompts, then the top-k oracle diagnostic on code blocks (zeta)
cd ~/shigeru/open-mercury
L=results/pure/b8_zeta_chain.log; E="SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-train/bin/python"; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter
env $E bench/make_refs.py --model $M --prompts bench/prompts_humaneval.json --n 164 --max-new-tokens 512 --out results/pure/humaneval_refs_e5.json > results/pure/humaneval_refs_e5.log 2>&1
echo "=== $(date +%T) [zeta] humaneval refs rc=$? $(tail -n1 results/pure/humaneval_refs_e5.log | cut -c1-100)" >> $L
env $E bench/diag_pairwise.py --model $M --adapter $A --no-time --prompts bench/prompts_humaneval.json --refs results/pure/humaneval_refs_e5.json --n 64 --unmask left --rerank-ks 2 --max-blocks 8 --save-candidates results/pure/b8_candidates_humaneval_left.jsonl --out results/pure/diag_pairwise_lr1e4_left_humaneval.json > results/pure/diag_pairwise_lr1e4_left_humaneval.log 2>&1
echo "=== $(date +%T) [zeta] diag_pairwise humaneval rc=$?" >> $L
