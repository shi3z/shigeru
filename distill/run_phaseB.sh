#!/bin/bash
# run_phaseB.sh <config.yaml> <tag> : Phase B LoRA SFT on the shared GPU (physical 5 by default), then AR evaluation of the
# merged model. Waits until the GPU has >= MIN_FREE_GIB free (other tenants: hf_proxy, TTS, benches) and never touches them.
# Detached use: (setsid bash distill/run_phaseB.sh distill/configs/phaseB_pilot_4b.yaml phaseB_pilot_4b < /dev/null > /dev/null 2>&1 &)
cd "$(dirname "$0")/.."; CFG=$1; TAG=$2; GPU=${GPU:-5}; MIN_FREE_GIB=${MIN_FREE_GIB:-30}; LOG=results/phaseB_runs.log
[ -f "$CFG" ] && [ -n "$TAG" ] || { echo "usage: $0 <config.yaml> <tag>"; exit 1; }
free_gib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU | awk '{printf "%d", $1/1024}'; }
while [ "$(free_gib)" -lt "$MIN_FREE_GIB" ]; do echo "$(date +%T) GPU$GPU free $(free_gib) GiB < $MIN_FREE_GIB, waiting" >> $LOG; sleep 60; done
echo "=== $(date +%T) start train $TAG cfg=$CFG gpu=$GPU free=$(free_gib)GiB commit=$(git rev-parse --short HEAD)" >> $LOG
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. .venv-train/bin/python -m distill.train_sft "$CFG" > results/train_${TAG}.log 2>&1; rc=$?
echo "=== $(date +%T) end train $TAG rc=$rc" >> $LOG
OUT=$(python3 -c "import yaml,sys; print(yaml.safe_load(open('$CFG'))['out_dir'])" 2>/dev/null || .venv/bin/python -c "import yaml; print(yaml.safe_load(open('$CFG'))['out_dir'])")
if [ $rc -eq 0 ] && [ -d "$OUT/merged" ]; then GPU=$GPU bash bench/eval_ar_model.sh "$OUT/merged" "$TAG"; echo "=== $(date +%T) eval $TAG done" >> $LOG; fi
