#!/bin/bash
# LLaDA2.1-mini (MoE) through its HF remote generate, after phase1b_v2.sh.
cd "$(dirname "$0")/.."; export CUDA_VISIBLE_DEVICES=${GPU:-5}; PY=.venv/bin/python; LOG=results/phase1_runs.log
until grep -q "phase1b_v2 done" $LOG 2>/dev/null; do sleep 30; done
run() { out=$1; shift; if [ -s "$out" ]; then echo "skip $out" >> $LOG; return; fi; echo "=== $(date +%T) start $out" >> $LOG; "$@" --out "$out" 2>&1 | grep -vE "Warning|warn|^\s*$" | grep -E "^===|wrote|rror|Traceback" | cut -c1-240 >> $LOG; echo "=== $(date +%T) end $out" >> $LOG; }
run results/llada2_mini_t0.5_e0.json   $PY bench/run_llada2.py --gen-length 256 --threshold 0.5 --editing-threshold 0
run results/llada2_mini_t0.7_e0.5.json $PY bench/run_llada2.py --gen-length 256 --threshold 0.7 --editing-threshold 0.5
run results/gsm8k50_llada2_mini_t0.7_e0.5.json $PY bench/run_llada2.py --gen-length 256 --threshold 0.7 --editing-threshold 0.5 --prompts bench/prompts_gsm8k50.json
echo "=== $(date +%T) phase1c_llada2 done" >> $LOG
