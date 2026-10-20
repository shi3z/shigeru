#!/bin/bash
# GSM8K-50 accuracy for the key Phase-1 configurations (same GPU, greedy). Waits for phase1_all.sh to finish.
cd "$(dirname "$0")/.."; export CUDA_VISIBLE_DEVICES=${GPU:-5}; PY=.venv/bin/python; LOG=results/phase1_runs.log; P=bench/prompts_gsm8k50.json
until grep -q "phase1_all done" $LOG 2>/dev/null; do sleep 30; done
run() { out=$1; shift; if [ -s "$out" ]; then echo "skip $out (exists)" >> $LOG; return; fi; echo "=== $(date +%T) start $out" >> $LOG; "$@" --prompts $P --out "$out" 2>&1 | grep -vE "Warning|warn|^\s*$" | grep -E "^===|wrote|rror" | cut -c1-220 >> $LOG; $PY -c "import json; r=json.load(open('$out')); print('=== accuracy', '$out', r.get('accuracy'))" >> $LOG 2>&1; echo "=== $(date +%T) end $out" >> $LOG; }
run results/gsm8k50_qwen3_8b_hf.json          $PY bench/run_ar.py --model /mnt/ssd/models/hf/Qwen3-8B --max-new-tokens 384
run results/gsm8k50_llada8b_s256.json         $PY bench/run_llada.py --gen-length 256 --steps 256 --block-length 32
run results/gsm8k50_llada8b_s128.json         $PY bench/run_llada.py --gen-length 256 --steps 128 --block-length 32
run results/gsm8k50_llada8b_s64.json          $PY bench/run_llada.py --gen-length 256 --steps 64  --block-length 32
run results/gsm8k50_fastdllm_v1_dual_t0.9.json $PY bench/run_fastdllm_v1.py --mode dual_cache --threshold 0.9 --gen-length 256 --steps 256 --block-length 32
run results/gsm8k50_fastdllm_v2_t0.9.json     $PY bench/run_fastdllm_v2.py --max-new-tokens 384 --threshold 0.9 --small-block-size 8
echo "=== $(date +%T) phase1_quality done" >> $LOG
