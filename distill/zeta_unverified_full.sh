#!/bin/bash
# On zeta, after the suffix/mixed experiment: unverified parallel decoding (tau=0.4, HF decoder) on the full GSM8K for the
# random-, suffix- and mixed-mode stage-2 models of snapshot-6000. Runs beside the v4 cycle (fits in 96 GB).
cd "$(dirname "$0")/.."; LOG=results/zeta_runs.log; Z="ssh -o LogLevel=ERROR zeta-gpu"
until grep -q "zeta suffix-exp finished" $LOG 2>/dev/null; do sleep 120; done
for m in zeta_sft_snapshot-6000 zeta_sft_snapshot-6000_suffix zeta_sft_snapshot-6000_mixed; do
  echo "=== $(date +%T) zeta unverified full: $m" >> $LOG
  $Z "cd ~/shigeru/open-mercury && SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-train/bin/python bench/run_cdlm.py --model runs/$m/final --engine hf --n 1319 --prompts bench/prompts_gsm8k_full.json --tau 0.4 --out results/zeta_${m}_unverified_full.json 2>&1 | grep -E '^=== cdlm|Traceback' | sed 's/^=== cdlm/=== cdlm ${m}_unverified_full/' | cut -c1-400" >> $LOG 2>&1
  rsync -a -e "ssh -o LogLevel=ERROR" "zeta-gpu:~/shigeru/open-mercury/results/zeta_${m}_unverified_full.json" results/ >> $LOG 2>&1
done
echo "=== $(date +%T) zeta unverified full evals done" >> $LOG
