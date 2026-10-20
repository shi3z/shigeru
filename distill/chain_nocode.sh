#!/bin/bash
# Ablation 2 after the full verified run has released GPU 7: 4B LoRA on the medium set minus code, then GSM8K-50 + HumanEval.
cd "$(dirname "$0")/.."; until grep -q "end results/cdlm_sft_4b_full_fast_verify.json" results/phaseE_runs.log 2>/dev/null; do sleep 120; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7)" -lt 1024 ]; do sleep 30; done
GPU=7 MIN_FREE_GIB=60 bash distill/run_phaseB.sh distill/configs/phaseB_medium_4b_nocode.yaml phaseB_medium_4b_nocode
out=results/humaneval_phaseB_medium_4b_nocode.json; echo "=== $(date +%T) start $out (gpu 7)" >> results/phaseF_runs.log
CUDA_VISIBLE_DEVICES=7 .venv/bin/python bench/run_code_eval.py --engine hf:runs/phaseB_medium_4b_nocode/merged --task humaneval --out $out 2>&1 | grep -E "^===|rror" | cut -c1-200 >> results/phaseF_runs.log
.venv/bin/python -c "import json; r=json.load(open('$out')); print('=== accuracy', '$out', r.get('pass@1'))" >> results/phaseF_runs.log 2>&1
echo "=== $(date +%T) chain_nocode done" >> results/phaseB_runs.log
