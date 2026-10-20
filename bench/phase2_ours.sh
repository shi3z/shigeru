#!/bin/bash
# Our engine: parity run (fixed 256 steps, no cache) vs the official sampler, then cached threshold decoding, then GSM8K-50.
cd "$(dirname "$0")/.."; export CUDA_VISIBLE_DEVICES=${GPU:-5}; PY=.venv/bin/python; LOG=results/phase1_runs.log
until grep -q "phase1_quality done" $LOG 2>/dev/null; do sleep 30; done
run() { out=$1; shift; if [ -s "$out" ]; then echo "skip $out" >> $LOG; return; fi; echo "=== $(date +%T) start $out" >> $LOG; "$@" --out "$out" 2>&1 | grep -vE "Warning|warn|^\s*$" | grep -E "^===|wrote|rror|Traceback" | cut -c1-240 >> $LOG; echo "=== $(date +%T) end $out" >> $LOG; }
run results/ours_fixed_s256_b32_nocache.json $PY bench/run_ours.py --mode fixed --steps 256 --gen-length 256 --block-length 32 --no-cache
run results/ours_fixed_s256_b32_cache.json   $PY bench/run_ours.py --mode fixed --steps 256 --gen-length 256 --block-length 32
run results/ours_t0.9_b32_cache.json         $PY bench/run_ours.py --mode threshold --threshold 0.9 --gen-length 256 --block-length 32
run results/ours_t0.8_b32_cache.json         $PY bench/run_ours.py --mode threshold --threshold 0.8 --gen-length 256 --block-length 32
run results/gsm8k50_ours_t0.9_b32_cache.json $PY bench/run_ours.py --mode threshold --threshold 0.9 --gen-length 256 --block-length 32 --prompts bench/prompts_gsm8k50.json
echo "=== $(date +%T) phase2_ours done" >> $LOG
