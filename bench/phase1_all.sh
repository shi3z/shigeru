#!/bin/bash
# Phase-1 sequence on one GPU (default GPU 5): AR baseline, LLaDA step sweep, Fast-dLLM v1 variants, Fast-dLLM v2 variants.
# Detached-friendly: logs to results/phase1_runs.log, one JSON per run. Skips runs whose JSON already exists.
cd "$(dirname "$0")/.."; export CUDA_VISIBLE_DEVICES=${GPU:-5}; PY=.venv/bin/python; LOG=results/phase1_runs.log
run() { out=$1; shift; if [ -s "$out" ]; then echo "skip $out (exists)" >> $LOG; return; fi; echo "=== $(date +%T) start $out" >> $LOG; "$@" --out "$out" 2>&1 | grep -vE "Warning|warn|^\s*$" | cut -c1-220 >> $LOG; echo "=== $(date +%T) end $out" >> $LOG; }
run results/qwen3_8b_hf_256.json $PY bench/run_ar.py --model /mnt/ssd/models/hf/Qwen3-8B --max-new-tokens 256
for S in 128 64 32; do run results/llada8b_g256_s${S}_b32.json $PY bench/run_llada.py --gen-length 256 --steps $S --block-length 32; done
run results/fastdllm_v1_dual_cache_t0.9.json   $PY bench/run_fastdllm_v1.py --mode dual_cache   --threshold 0.9 --gen-length 256 --steps 256 --block-length 32
run results/fastdllm_v1_prefix_cache_t0.9.json $PY bench/run_fastdllm_v1.py --mode prefix_cache --threshold 0.9 --gen-length 256 --steps 256 --block-length 32
run results/fastdllm_v1_dual_cache_tnone.json  $PY bench/run_fastdllm_v1.py --mode dual_cache                   --gen-length 256 --steps 256 --block-length 32
run results/fastdllm_v1_vanilla_tnone.json     $PY bench/run_fastdllm_v1.py --mode vanilla                      --gen-length 256 --steps 256 --block-length 32
run results/fastdllm_v2_t0.9_sb8_nocache.json  $PY bench/run_fastdllm_v2.py --max-new-tokens 256 --threshold 0.9 --small-block-size 8
run results/fastdllm_v2_t0.9_sb8_cache.json    $PY bench/run_fastdllm_v2.py --max-new-tokens 256 --threshold 0.9 --small-block-size 8 --block-cache
run results/fastdllm_v2_t1.0_sb8_nocache.json  $PY bench/run_fastdllm_v2.py --max-new-tokens 256 --threshold 1.0 --small-block-size 8
echo "=== $(date +%T) phase1_all done" >> $LOG
