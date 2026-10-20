#!/bin/bash
# Rerun the CUDA-graph configurations with the fixed runner (the first graph results were produced by a stale-mask bug).
cd "$(dirname "$0")/.."; export CUDA_VISIBLE_DEVICES=${GPU:-5}; PY=.venv/bin/python; LOG=results/phase1_runs.log
until grep -q "phase5_code done" $LOG 2>/dev/null; do sleep 60; done
mkdir -p results/invalid; mv results/ours_t0.9_b32_graph.json results/ours_t0.9_b32_graph_eos.json results/ours_fixed_s256_b32_graph.json results/gsm8k50_ours_t0.9_b32_graph_eos.json results/invalid/ 2>/dev/null
echo "=== $(date +%T) graph rerun: stale-mask results moved to results/invalid/" >> $LOG
run() { out=$1; shift; echo "=== $(date +%T) start $out" >> $LOG; "$@" --out "$out" 2>&1 | grep -vE "Warning|warn|^\s*$" | grep -E "^===|wrote|rror|Traceback" | cut -c1-240 >> $LOG; echo "=== $(date +%T) end $out" >> $LOG; }
run results/ours_t0.9_b32_graph.json        $PY bench/run_ours.py --mode threshold --threshold 0.9 --gen-length 256 --block-length 32 --graph
run results/ours_t0.9_b32_graph_eos.json    $PY bench/run_ours.py --mode threshold --threshold 0.9 --gen-length 256 --block-length 32 --graph --stop-on-eos
run results/gsm8k50_ours_t0.9_b32_graph_eos.json $PY bench/run_ours.py --mode threshold --threshold 0.9 --gen-length 256 --block-length 32 --graph --stop-on-eos --prompts bench/prompts_gsm8k50.json
echo "=== $(date +%T) phase2_graph_rerun done" >> $LOG
