#!/bin/bash
# After the CPT conversion and the idle-GPU speed rerun (both on GPU 7): benchmark the CPT checkpoint with the diffusion
# decoder (τ=None gate vs AR, parallel decoding), then stage-2 fine-tune on the medium teacher set, then benchmark again.
# GPU 7 only; never touches the fleet or GPUs 0/1/5. Log results/phaseB_runs.log, results/phaseE_runs.log.
cd "$(dirname "$0")/.."; LOG=results/phaseB_runs.log; PY=.venv-train/bin/python
until grep -q "phase2_idle done" results/phase1_runs.log 2>/dev/null; do sleep 120; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7)" -lt 1024 ]; do sleep 30; done
bench() { ck=$1; tag=$2; out=results/cdlm_${tag}_hf.json; [ -s "$out" ] && return; echo "=== $(date +%T) start $out" >> results/phaseE_runs.log; CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. $PY bench/run_cdlm.py --model "$ck" --engine hf --n 20 --tau none,0.4,0.7,1.0 --ar --out "$out" 2>&1 | grep -E "^=== cdlm|Traceback|Error" | cut -c1-600 >> results/phaseE_runs.log; echo "=== $(date +%T) end $out" >> results/phaseE_runs.log; }
bench runs/phaseC_cpt_4b/final cpt_4b
until [ -s data/distill/phaseA_medium_train.jsonl ]; do sleep 120; done
echo "=== $(date +%T) start phaseC_sft_4b (GPU 7) commit=$(git rev-parse --short HEAD)" >> $LOG
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. $PY -m distill.train_cdlm distill/configs/phaseC_sft_4b.yaml > results/train_phaseC_sft_4b.log 2>&1
echo "=== $(date +%T) phaseC_sft_4b rc=$?" >> $LOG
bench runs/phaseC_sft_4b/final sft_4b
echo "=== $(date +%T) chain_after_cpt done" >> $LOG
