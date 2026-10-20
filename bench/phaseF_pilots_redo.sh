#!/bin/bash
# Re-evaluate the Phase B pilots after the config fix (distill/fix_config.py): speed, GSM8K-50 and HumanEval on GPU 7,
# after the phaseF_full GPU-7 branch (humaneval_qwen3_4b) has finished. Log results/phaseF_runs.log.
cd "$(dirname "$0")/.."; PY=.venv/bin/python; LOG=results/phaseF_runs.log
until grep -q "end results/humaneval_qwen3_4b.json" $LOG 2>/dev/null; do sleep 30; done
run() { out=$1; shift; if [ -s "$out" ]; then echo "skip $out (exists)" >> $LOG; return; fi; echo "=== $(date +%T) start $out (gpu 7, config fixed)" >> $LOG; CUDA_VISIBLE_DEVICES=7 "$@" --out "$out" 2>&1 | grep -vE "Warning|warn|^\s*$" | grep -E "^===|wrote|rror" | cut -c1-220 >> $LOG; $PY -c "import json; r=json.load(open('$out')); print('=== accuracy', '$out', r.get('accuracy') or r.get('pass@1'))" >> $LOG 2>&1; echo "=== $(date +%T) end $out" >> $LOG; }
run results/phaseB_pilot_4b_256.json      $PY bench/run_ar.py --model runs/phaseB_pilot_4b/merged --max-new-tokens 256
run results/gsm8k50_phaseB_pilot_4b.json  $PY bench/run_ar.py --model runs/phaseB_pilot_4b/merged --max-new-tokens 384 --prompts bench/prompts_gsm8k50.json
run results/phaseB_pilot_8b_256.json      $PY bench/run_ar.py --model runs/phaseB_pilot_8b/merged --max-new-tokens 256
run results/gsm8k50_phaseB_pilot_8b.json  $PY bench/run_ar.py --model runs/phaseB_pilot_8b/merged --max-new-tokens 384 --prompts bench/prompts_gsm8k50.json
run results/humaneval_phaseB_pilot_4b.json $PY bench/run_code_eval.py --engine hf:runs/phaseB_pilot_4b/merged --task humaneval
run results/humaneval_phaseB_pilot_8b.json $PY bench/run_code_eval.py --engine hf:runs/phaseB_pilot_8b/merged --task humaneval
echo "=== $(date +%T) phaseF_pilots_redo done" >> $LOG
