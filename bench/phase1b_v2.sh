#!/bin/bash
# Fast-dLLM v2 speed runs (rerun after installing einops), after phase2_graph.sh.
cd "$(dirname "$0")/.."; export CUDA_VISIBLE_DEVICES=${GPU:-5}; PY=.venv/bin/python; LOG=results/phase1_runs.log
until grep -q "phase2_graph done" $LOG 2>/dev/null; do sleep 30; done
run() { out=$1; shift; if [ -s "$out" ]; then echo "skip $out" >> $LOG; return; fi; echo "=== $(date +%T) start $out" >> $LOG; "$@" --out "$out" 2>&1 | grep -vE "Warning|warn|^\s*$" | grep -E "^===|wrote|rror|Traceback" | cut -c1-240 >> $LOG; echo "=== $(date +%T) end $out" >> $LOG; }
run results/fastdllm_v2_t0.9_sb8_nocache.json $PY bench/run_fastdllm_v2.py --max-new-tokens 256 --threshold 0.9 --small-block-size 8
run results/fastdllm_v2_t0.9_sb8_cache.json   $PY bench/run_fastdllm_v2.py --max-new-tokens 256 --threshold 0.9 --small-block-size 8 --block-cache
run results/fastdllm_v2_t1.0_sb8_nocache.json $PY bench/run_fastdllm_v2.py --max-new-tokens 256 --threshold 1.0 --small-block-size 8
echo "=== $(date +%T) phase1b_v2 done" >> $LOG
