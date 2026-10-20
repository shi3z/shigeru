#!/bin/bash
# Phase F rows R0/R1: released Qwen3-8B and Qwen3-4B, AR greedy, on the shared GPU (default physical 5): speed on
# bench/prompts.json and GSM8K-50 accuracy. Detached; log in results/phaseF_runs.log. Skips outputs that exist.
cd "$(dirname "$0")/.."; export CUDA_VISIBLE_DEVICES=${GPU:-5}; PY=.venv/bin/python; LOG=results/phaseF_runs.log
run() { out=$1; shift; if [ -s "$out" ]; then echo "skip $out (exists)" >> $LOG; return; fi; echo "=== $(date +%T) start $out" >> $LOG; "$@" --out "$out" 2>&1 | grep -vE "Warning|warn|^\s*$" | grep -E "^===|wrote|rror" | cut -c1-220 >> $LOG; $PY -c "import json; r=json.load(open('$out')); print('=== accuracy', '$out', r.get('accuracy'))" >> $LOG 2>&1; echo "=== $(date +%T) end $out" >> $LOG; }
M4=/mnt/raid6/models/hf/Qwen3-4B
run results/qwen3_4b_hf_256.json      $PY bench/run_ar.py --model $M4 --max-new-tokens 256
run results/gsm8k50_qwen3_4b_hf.json  $PY bench/run_ar.py --model $M4 --max-new-tokens 384 --prompts bench/prompts_gsm8k50.json
echo "=== $(date +%T) phaseF_baselines done" >> $LOG
