#!/bin/bash
# After the 1B-token CPT: stage 2 with the enlarged mix (GPU 4), then benchmarks (quick verified, plain, full GSM8K with CUDA graphs),
# HumanEval of the AR rows (bench venv, base tokenizer copy), and the Phase F table. Log results/phaseB_runs.log.
cd "$(dirname "$0")/.."; LOG=results/phaseB_runs.log; PY=.venv-train/bin/python
until grep -q "phaseC_cpt3_4b rc=" $LOG 2>/dev/null; do sleep 300; done
for g in 4 6 7; do until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)" -lt 1024 ]; do sleep 30; done; done
rm -rf runs/phaseC_cpt3_4b/checkpoint-*
echo "=== $(date +%T) start phaseC_sft3_4b (GPU 4) commit=$(git rev-parse --short HEAD)" >> $LOG
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. $PY -m distill.train_cdlm distill/configs/phaseC_sft3_4b.yaml > results/train_phaseC_sft3_4b.log 2>&1
echo "=== $(date +%T) phaseC_sft3_4b rc=$?" >> $LOG
rm -rf runs/phaseC_sft3_4b/checkpoint-*
M=runs/phaseC_sft3_4b/final
bench() { tag=$1; gpu=$2; shift 2; out=results/cdlm_sft3_4b_$tag.json; echo "=== $(date +%T) start $out (gpu $gpu)" >> results/phaseE_runs.log; CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. $PY bench/run_cdlm.py --model $M "$@" --out $out 2>&1 | grep -E "^=== cdlm|Traceback|Error" | sed "s/^=== cdlm/=== cdlm sft3_$tag/" | cut -c1-700 >> results/phaseE_runs.log; echo "=== $(date +%T) end $out" >> results/phaseE_runs.log; }
bench quick 4 --engine fast --n 20 --verify --tau 1.0 --ar &
bench plain 6 --engine hf --n 20 --tau none,0.4,0.7 &
wait
bench full_fast_graph_verify 7 --engine fast_graph --n 1319 --prompts bench/prompts_gsm8k_full.json --verify --tau 1.0 &
( D=$M/../final_eval; mkdir -p $D; ln -sf ../final/model.safetensors $D/model.safetensors; cp $M/config.json $M/generation_config.json $D/ 2>/dev/null; for fn in tokenizer.json tokenizer_config.json vocab.json merges.txt; do cp /mnt/raid6/models/hf/Qwen3-4B/$fn $D/; done
  for spec in "humaneval_phaseC_sft3_4b_ar.json:.venv/bin/python bench/run_code_eval.py --engine hf:$D --task humaneval" "gsm8k_full_phaseC_sft3_4b_ar.json:.venv/bin/python bench/run_ar.py --model $D --max-new-tokens 384 --prompts bench/prompts_gsm8k_full.json"; do out=results/${spec%%:*}; cmd=${spec#*:}; echo "=== $(date +%T) start $out (gpu 4)" >> results/phaseF_runs.log; CUDA_VISIBLE_DEVICES=4 $cmd --out $out 2>&1 | grep -E "^===|rror" | cut -c1-200 >> results/phaseF_runs.log; .venv/bin/python -c "import json; r=json.load(open('$out')); print('=== accuracy', '$out', r.get('accuracy') or r.get('pass@1'))" >> results/phaseF_runs.log 2>&1; done ) &
wait
.venv/bin/python bench/summarize_distill.py > /dev/null 2>&1
echo "=== $(date +%T) chain_after_cpt3 done" >> $LOG
