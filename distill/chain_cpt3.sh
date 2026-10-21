#!/bin/bash
# Long CPT: waits for the pre-tokenized stream, launches 3-rank training on GPUs 4,6,7 (auto-resume on restart),
# and runs a side evaluation of each new HF snapshot on GPU 4 (fits beside the rank: ~9 GiB). Log results/phaseB_runs.log.
cd "$(dirname "$0")/.."; LOG=results/phaseB_runs.log
until grep -q "pretokenize rc=0" results/pretokenize_1b.log 2>/dev/null; do sleep 60; done
for g in 4 6 7; do until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)" -lt 1024 ]; do sleep 30; done; done
echo "=== $(date +%T) start phaseC_cpt3_4b (torchrun 3 ranks on GPUs 4,6,7, 1B tokens) commit=$(git rev-parse --short HEAD)" >> $LOG
( last=""; while sleep 600; do s=$(ls -d runs/phaseC_cpt3_4b/snapshot-* 2>/dev/null | sort -t- -k2 -n | tail -1); [ -n "$s" ] && [ "$s" != "$last" ] && [ -f "$s/model.safetensors" ] || continue; sleep 120; last=$s; tag=cpt3_$(basename $s); out=results/cdlm_${tag}_hf.json; echo "=== $(date +%T) start $out (gpu 4, beside training)" >> results/phaseE_runs.log; CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv-train/bin/python bench/run_cdlm.py --model $s --engine hf --n 20 --tau none,0.4 --out $out 2>&1 | grep -E "^=== cdlm|Traceback|Error" | sed "s/^=== cdlm/=== cdlm $tag/" | cut -c1-400 >> results/phaseE_runs.log; CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv-train/bin/python bench/run_cdlm.py --model $s --engine fast --n 20 --verify --tau 1.0 --out results/cdlm_${tag}_verify.json 2>&1 | grep -E "^=== cdlm|Traceback|Error" | sed "s/^=== cdlm/=== cdlm $tag verify/" | cut -c1-400 >> results/phaseE_runs.log; grep -q "phaseC_cpt3_4b rc=" $LOG && break; done ) &
EVAL_PID=$!
CUDA_VISIBLE_DEVICES=4,6,7 PYTHONPATH=. .venv-train/bin/torchrun --nproc_per_node=3 -m distill.train_cdlm distill/configs/phaseC_cpt3_4b_3gpu.yaml > results/train_phaseC_cpt3_4b.log 2>&1
echo "=== $(date +%T) phaseC_cpt3_4b rc=$?" >> $LOG
kill $EVAL_PID 2>/dev/null
