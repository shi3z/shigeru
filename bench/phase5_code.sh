#!/bin/bash
# HumanEval pass@1 for the main engines (greedy, 512 tokens), after the Phase-1/2 chain.
cd "$(dirname "$0")/.."; export CUDA_VISIBLE_DEVICES=${GPU:-5}; PY=.venv/bin/python; LOG=results/phase1_runs.log
until grep -q "phase1c_llada2 done" $LOG 2>/dev/null; do sleep 30; done
run() { out=$1; shift; if [ -s "$out" ]; then echo "skip $out" >> $LOG; return; fi; echo "=== $(date +%T) start $out" >> $LOG; "$@" --out "$out" 2>&1 | grep -vE "Warning|warn|^\s*$" | grep -E "^===|wrote|rror|Traceback" | cut -c1-240 >> $LOG; echo "=== $(date +%T) end $out" >> $LOG; }
run results/humaneval_ours_t0.9_graph.json $PY bench/run_code_eval.py --engine ours:0.9:graph --task humaneval
run results/humaneval_qwen3_8b_hf.json     $PY bench/run_code_eval.py --engine hf:/mnt/ssd/models/hf/Qwen3-8B --task humaneval
run results/humaneval_llada_official.json  $PY bench/run_code_eval.py --engine llada:512 --task humaneval
run results/humaneval_fastdllm_v1_t0.9.json $PY bench/run_code_eval.py --engine fastdllm_v1:0.9 --task humaneval
run results/humaneval_fastdllm_v2_t0.9.json $PY bench/run_code_eval.py --engine fastdllm_v2:0.9 --task humaneval
echo "=== $(date +%T) phase5_code done" >> $LOG
