#!/bin/bash
# One-off on zeta: stage 2 of snapshot-6000 with mask_mode suffix and mixed (1 epoch each), then quick + full verified
# benchmarks, compared with the random-mode cycle (89.5 %, 1.43 tok/step). Log results/zeta_runs.log.
cd "$(dirname "$0")/.."; LOG=results/zeta_runs.log; Z="ssh -o LogLevel=ERROR zeta-gpu"
for mode in suffix mixed; do
  echo "=== $(date +%T) zeta suffix-exp: stage 2 on snapshot-6000, mask_mode=$mode" >> $LOG
  $Z "cd ~/shigeru/open-mercury && git pull -q 2>/dev/null; sed 's#^model: .*#model: runs/phaseC_cpt3_4b/snapshot-6000#; s#^out_dir: .*#out_dir: runs/zeta_sft_snapshot-6000_$mode#' distill/configs/phaseC_sft3_4b.yaml | sed -e s/^epochs:\\ 2/epochs:\\ 1/ > distill/configs/zeta_sft_snapshot-6000_$mode.yaml && echo 'mask_mode: $mode' >> distill/configs/zeta_sft_snapshot-6000_$mode.yaml && grep -E '^model|^epochs|^mask_mode' distill/configs/zeta_sft_snapshot-6000_$mode.yaml" >> $LOG 2>&1
  $Z "cd ~/shigeru/open-mercury && SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-train/bin/python -m distill.train_cdlm distill/configs/zeta_sft_snapshot-6000_$mode.yaml > results/train_zeta_sft_snapshot-6000_$mode.log 2>&1; echo \"train rc=\$?\"; rm -rf runs/zeta_sft_snapshot-6000_$mode/checkpoint-*" >> $LOG 2>&1
  M=runs/zeta_sft_snapshot-6000_$mode/final; tag=cpt3_snapshot-6000_$mode
  $Z "cd ~/shigeru/open-mercury && export SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.; \
      .venv-train/bin/python bench/run_cdlm.py --model $M --engine fast_graph --n 20 --verify --tau 1.0 --ar --out results/zeta_${tag}_quick.json 2>&1 | grep -E '^=== cdlm|Traceback' | sed 's/^=== cdlm/=== cdlm zeta_${tag}_quick/' | cut -c1-600; \
      .venv-train/bin/python bench/run_cdlm.py --model $M --engine hf --n 20 --tau none,0.4 --out results/zeta_${tag}_plain.json 2>&1 | grep -E '^=== cdlm|Traceback' | sed 's/^=== cdlm/=== cdlm zeta_${tag}_plain/' | cut -c1-400; \
      .venv-train/bin/python bench/run_cdlm.py --model $M --engine fast_graph --n 1319 --prompts bench/prompts_gsm8k_full.json --verify --tau 1.0 --out results/zeta_${tag}_full_verify.json 2>&1 | grep -E '^=== cdlm|Traceback' | sed 's/^=== cdlm/=== cdlm zeta_${tag}_full/' | cut -c1-400" >> $LOG 2>&1
  rsync -a -e "ssh -o LogLevel=ERROR" "zeta-gpu:~/shigeru/open-mercury/results/zeta_*" results/ >> $LOG 2>&1 && $Z "cd ~/shigeru/open-mercury && rm -f results/zeta_*" >> $LOG 2>&1
  echo "=== $(date +%T) zeta suffix-exp $mode done" >> $LOG
done
echo "=== $(date +%T) zeta suffix-exp finished; relaunching the v4 cycle" >> $LOG
setsid bash distill/chain_zeta_stage2_v4.sh < /dev/null > /dev/null 2>&1 &
