#!/bin/bash
# After stage 2b (CPT x2 + no-code/codecomp SFT): full GSM8K with the CUDA-graph verified decoder on GPU 7.
cd "$(dirname "$0")/.."; until grep -q "chain_after_cpt2 done" results/phaseB_runs.log 2>/dev/null; do sleep 120; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7)" -lt 1024 ]; do sleep 60; done
out=results/cdlm_sft2_4b_full_fast_graph_verify.json; echo "=== $(date +%T) start $out (gpu 7, 1319 items, CUDA graphs)" >> results/phaseE_runs.log
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. .venv-train/bin/python bench/run_cdlm.py --model runs/phaseC_sft2_4b/final --engine fast_graph --verify --tau 1.0 --n 1319 --prompts bench/prompts_gsm8k_full.json --out $out > results/cdlm_sft2_4b_full_graph.log 2>&1
grep -E "^=== cdlm|Traceback|Error" results/cdlm_sft2_4b_full_graph.log | sed "s/^=== cdlm/=== cdlm sft2_full_fast_graph_verify/" | cut -c1-600 >> results/phaseE_runs.log; echo "=== $(date +%T) end $out" >> results/phaseE_runs.log
echo "=== $(date +%T) chain_after_sft2 done" >> results/phaseB_runs.log
