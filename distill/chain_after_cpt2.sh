#!/bin/bash
# After the CPT continuation: stage-2 teacher fine-tune of its output (GPU 4), then verified-decoding benchmarks
# (20 prompts quick, then the full 1319 with the own engine) to compare with runs/phaseC_sft_4b (60M CPT tokens).
cd "$(dirname "$0")/.."; LOG=results/phaseB_runs.log; PY=.venv-train/bin/python
until grep -q "phaseC_cpt2_4b rc=" $LOG 2>/dev/null; do sleep 120; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4)" -lt 1024 ]; do sleep 30; done
echo "=== $(date +%T) start phaseC_sft2_4b (GPU 4) commit=$(git rev-parse --short HEAD)" >> $LOG
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. $PY -m distill.train_cdlm distill/configs/phaseC_sft2_4b.yaml > results/train_phaseC_sft2_4b.log 2>&1
echo "=== $(date +%T) phaseC_sft2_4b rc=$?" >> $LOG
rm -rf runs/phaseC_cpt2_4b/checkpoint-* runs/phaseC_sft2_4b/checkpoint-*
bench() { tag=$1; shift; out=results/cdlm_sft2_4b_$tag.json; echo "=== $(date +%T) start $out (gpu 4)" >> results/phaseE_runs.log; CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. $PY bench/run_cdlm.py --model runs/phaseC_sft2_4b/final "$@" --out $out 2>&1 | grep -E "^=== cdlm|Traceback|Error" | sed "s/^=== cdlm/=== cdlm sft2_$tag/" | cut -c1-700 >> results/phaseE_runs.log; echo "=== $(date +%T) end $out" >> results/phaseE_runs.log; }
bench quick --engine fast --n 20 --verify --tau 1.0 --ar
bench plain --engine hf --n 20 --tau none,0.4
bench full_fast_verify --engine fast --n 1319 --prompts bench/prompts_gsm8k_full.json --verify --tau 1.0 --ar
echo "=== $(date +%T) chain_after_cpt2 done" >> $LOG
