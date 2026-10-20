#!/bin/bash
# After the medium teacher set finishes: curate it; then, once the 3-GPU CPT run has released the GPUs, train the
# medium Phase B LoRA students (4B on GPU 4, 8B on GPU 6) and evaluate them (speed, GSM8K-50, full GSM8K, HumanEval).
# Never touches the fleet, GPUs 0/1/5. Log results/phaseB_runs.log.
cd "$(dirname "$0")/.."; LOG=results/phaseB_runs.log; PY=.venv/bin/python
while pgrep -f "distill.gen_teacher distill/configs/phaseA_medium.yaml" > /dev/null; do sleep 60; done
echo "=== $(date +%T) medium generator exited; $(grep -c . data/distill/phaseA_medium.jsonl) records" >> $LOG
PYTHONPATH=. $PY -m distill.curate data/distill/phaseA_medium.jsonl --out data/distill/phaseA_medium >> results/curate_medium.log 2>&1
echo "=== $(date +%T) medium curated: $(grep -c . data/distill/phaseA_medium_train.jsonl) train / $(grep -c . data/distill/phaseA_medium_val.jsonl) val / $(grep -c . data/distill/phaseA_medium_test.jsonl) test" >> $LOG
until grep -q "phaseC_cpt_4b rc=" $LOG 2>/dev/null; do sleep 120; done
for g in 4 6; do until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)" -lt 1024 ]; do sleep 30; done; done
evalfull() { gpu=$1; m=$2; tag=$3; GPU=$gpu bash bench/eval_ar_model.sh "$m" "$tag"; for spec in "gsm8k_full_${tag}.json:$PY bench/run_ar.py --model $m --max-new-tokens 384 --prompts bench/prompts_gsm8k_full.json" "humaneval_${tag}.json:$PY bench/run_code_eval.py --engine hf:$m --task humaneval"; do out=results/${spec%%:*}; cmd=${spec#*:}; [ -s "$out" ] && continue; echo "=== $(date +%T) start $out (gpu $gpu)" >> results/phaseF_runs.log; CUDA_VISIBLE_DEVICES=$gpu $cmd --out "$out" 2>&1 | grep -E "^===|rror" | cut -c1-200 >> results/phaseF_runs.log; $PY -c "import json; r=json.load(open('$out')); print('=== accuracy', '$out', r.get('accuracy') or r.get('pass@1'))" >> results/phaseF_runs.log 2>&1; done; }
( GPU=4 MIN_FREE_GIB=60 bash distill/run_phaseB.sh distill/configs/phaseB_medium_4b.yaml phaseB_medium_4b; evalfull 4 runs/phaseB_medium_4b/merged phaseB_medium_4b ) &
( GPU=6 MIN_FREE_GIB=60 bash distill/run_phaseB.sh distill/configs/phaseB_medium_8b.yaml phaseB_medium_8b; evalfull 6 runs/phaseB_medium_8b/merged phaseB_medium_8b ) &
wait; echo "=== $(date +%T) chain_medium done" >> $LOG
