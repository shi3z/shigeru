#!/bin/bash
# Phase E timing on one idle GPU (default physical 7): the converted 4B checkpoint decoded by the HF-cache decoder vs the
# own engine (plain / static shapes / CUDA graphs), τ=None and τ=0.4, 10 GSM8K prompts. Also the τ=None exactness gate
# in bf16. Runs in .venv-train. Log results/phaseE_runs.log. Launch: (setsid bash bench/phaseE_timing.sh <ckpt> <tag> < /dev/null > /dev/null 2>&1 &)
cd "$(dirname "$0")/.."; CK=${1:-runs/phaseC_pilot_4b/final}; TAG=${2:-pilot}; GPU=${GPU:-7}; LOG=results/phaseE_runs.log
for e in hf fast fast_static fast_graph; do
  out=results/cdlm_${TAG}_${e}.json; [ -s "$out" ] && { echo "skip $out" >> $LOG; continue; }
  echo "=== $(date +%T) start $out" >> $LOG
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. .venv-train/bin/python bench/run_cdlm.py --model "$CK" --engine $e --n 10 --tau none,0.4 $( [ $e = hf ] && echo --ar ) --out "$out" 2>&1 | grep -E "^=== cdlm|Traceback|Error" | cut -c1-400 >> $LOG
  echo "=== $(date +%T) end $out" >> $LOG
done
echo "=== $(date +%T) phaseE_timing $TAG done" >> $LOG
