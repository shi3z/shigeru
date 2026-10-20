#!/bin/bash
# Clean speed numbers for the key diffusion configurations on an IDLE GPU (physical 7, fleet stopped by the user):
# GPU 5 is shared with TTS/hf_proxy at 99 % utilization, so today's GPU-5 speed runs are not comparable. Waits for the
# GPU-5 graph rerun and for the Phase E timing to finish, then writes *_idle.json. Log results/phase1_runs.log.
cd "$(dirname "$0")/.."; export CUDA_VISIBLE_DEVICES=${GPU:-7}; PY=.venv/bin/python; LOG=results/phase1_runs.log
until grep -q "phaseC_cpt_4b rc=" results/phaseB_runs.log 2>/dev/null; do sleep 120; done; until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CUDA_VISIBLE_DEVICES)" -lt 1024 ]; do sleep 30; done
run() { out=$1; shift; [ -s "$out" ] && return; echo "=== $(date +%T) start $out (idle GPU $CUDA_VISIBLE_DEVICES)" >> $LOG; "$@" --out "$out" 2>&1 | grep -vE "Warning|warn|^\s*$" | grep -E "^===|wrote|rror|Traceback" | cut -c1-240 >> $LOG; echo "=== $(date +%T) end $out" >> $LOG; }
run results/ours_t0.9_b32_cache_idle.json      $PY bench/run_ours.py --mode threshold --threshold 0.9 --gen-length 256 --block-length 32
run results/ours_t0.9_b32_graph_idle.json      $PY bench/run_ours.py --mode threshold --threshold 0.9 --gen-length 256 --block-length 32 --graph
run results/ours_t0.9_b32_graph_eos_idle.json  $PY bench/run_ours.py --mode threshold --threshold 0.9 --gen-length 256 --block-length 32 --graph --stop-on-eos
run results/ours_fixed_s256_b32_nocache_idle.json $PY bench/run_ours.py --mode fixed --steps 256 --gen-length 256 --block-length 32 --no-cache
run results/llada8b_s256_idle.json             $PY bench/run_llada.py --gen-length 256 --steps 256 --block-length 32
run results/fastdllm_v1_dual_t0.9_idle.json    $PY bench/run_fastdllm_v1.py --mode dual_cache --threshold 0.9 --gen-length 256 --steps 256 --block-length 32
run results/fastdllm_v2_t0.9_sb8_idle.json     $PY bench/run_fastdllm_v2.py --max-new-tokens 256 --threshold 0.9 --small-block-size 8
run results/qwen3_8b_hf_256_idle.json          $PY bench/run_ar.py --model /mnt/ssd/models/hf/Qwen3-8B --max-new-tokens 256
echo "=== $(date +%T) phase2_idle done" >> $LOG
