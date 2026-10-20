#!/bin/bash
# eval_ar_model.sh <model_dir> <tag> : AR greedy speed (bench/prompts.json, 256 new tokens) + GSM8K-50 accuracy for any
# HF causal LM (Phase B/F rows). Shared GPU (default physical 5), detached-friendly, log results/phaseF_runs.log.
cd "$(dirname "$0")/.."; export CUDA_VISIBLE_DEVICES=${GPU:-5}; PY=.venv/bin/python; LOG=results/phaseF_runs.log
M=$1; T=$2; [ -d "$M" ] && [ -n "$T" ] || { echo "usage: $0 <model_dir> <tag>"; exit 1; }
run() { out=$1; shift; if [ -s "$out" ]; then echo "skip $out (exists)" >> $LOG; return; fi; echo "=== $(date +%T) start $out" >> $LOG; "$@" --out "$out" 2>&1 | grep -vE "Warning|warn|^\s*$" | grep -E "^===|wrote|rror" | cut -c1-220 >> $LOG; $PY -c "import json; r=json.load(open('$out')); print('=== accuracy', '$out', r.get('accuracy'))" >> $LOG 2>&1; echo "=== $(date +%T) end $out" >> $LOG; }
run results/${T}_256.json      $PY bench/run_ar.py --model "$M" --max-new-tokens 256
run results/gsm8k50_${T}.json  $PY bench/run_ar.py --model "$M" --max-new-tokens 384 --prompts bench/prompts_gsm8k50.json
echo "=== $(date +%T) eval_ar_model $T done" >> $LOG
