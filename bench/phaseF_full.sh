#!/bin/bash
# Phase F: full GSM8K test (1319) + HumanEval for the AR rows, on physical GPUs 4,6,7 (fleet stopped by the user).
# GPU 4: Qwen3-4B released, then Phase B 4B pilot; GPU 6: Qwen3-8B released, then Phase B 8B pilot; GPU 7: HumanEval of both pilots.
# Log results/phaseF_runs.log. Launch: (setsid bash bench/phaseF_full.sh < /dev/null > /dev/null 2>&1 &)
cd "$(dirname "$0")/.."; PY=.venv/bin/python; LOG=results/phaseF_runs.log; P=bench/prompts_gsm8k_full.json
run() { gpu=$1; out=$2; shift 2; if [ -s "$out" ]; then echo "skip $out (exists)" >> $LOG; return; fi; echo "=== $(date +%T) start $out (gpu $gpu)" >> $LOG; CUDA_VISIBLE_DEVICES=$gpu "$@" --out "$out" 2>&1 | grep -vE "Warning|warn|^\s*$" | grep -E "^===|wrote|rror" | cut -c1-220 >> $LOG; $PY -c "import json; r=json.load(open('$out')); print('=== accuracy', '$out', r.get('accuracy') or r.get('pass@1'))" >> $LOG 2>&1; echo "=== $(date +%T) end $out" >> $LOG; }
( run 4 results/gsm8k_full_qwen3_4b.json        $PY bench/run_ar.py --model /mnt/raid6/models/hf/Qwen3-4B --max-new-tokens 384 --prompts $P
  run 4 results/gsm8k_full_phaseB_pilot_4b.json $PY bench/run_ar.py --model runs/phaseB_pilot_4b/merged --max-new-tokens 384 --prompts $P ) &
( run 6 results/gsm8k_full_qwen3_8b.json        $PY bench/run_ar.py --model /mnt/ssd/models/hf/Qwen3-8B --max-new-tokens 384 --prompts $P
  run 6 results/gsm8k_full_phaseB_pilot_8b.json $PY bench/run_ar.py --model runs/phaseB_pilot_8b/merged --max-new-tokens 384 --prompts $P ) &
( run 7 results/humaneval_phaseB_pilot_4b.json  $PY bench/run_code_eval.py --engine hf:runs/phaseB_pilot_4b/merged --task humaneval
  run 7 results/humaneval_phaseB_pilot_8b.json  $PY bench/run_code_eval.py --engine hf:runs/phaseB_pilot_8b/merged --task humaneval
  run 7 results/humaneval_qwen3_4b.json         $PY bench/run_code_eval.py --engine hf:/mnt/raid6/models/hf/Qwen3-4B --task humaneval ) &
wait; echo "=== $(date +%T) phaseF_full done" >> $LOG
