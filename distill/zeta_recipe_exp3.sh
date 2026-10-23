#!/bin/bash
# One-off on zeta: stage-2 recipe variants on snapshot-28000, compared with the cycle result (88.5 %, 1.71 tok/step at tau=inf):
#   a) suffix masks + ar_weight 0.3 (MLM-heavier), 1 epoch
#   b) suffix masks + lr 2e-5 + 2 epochs
# Then quick (tau=inf, --ar), plain, full (tau=inf) benchmarks; results copied back; then the v5 cycle resumes.
cd "$(dirname "$0")/.."; LOG=results/zeta_runs.log; Z="ssh -o LogLevel=ERROR zeta-gpu"
run_variant() { tag=$1; extra=$2; epochs=$3
  echo "=== $(date +%T) zeta recipe-exp $tag: stage 2 on snapshot-28000 ($extra, epochs $epochs)" >> $LOG
  $Z "cd ~/shigeru/open-mercury && git pull -q 2>/dev/null; sed 's#^model: .*#model: runs/phaseC_cpt3_4b/snapshot-28000#; s#^out_dir: .*#out_dir: runs/zeta_sft_snapshot-28000_$tag#' distill/configs/phaseC_sft3_4b.yaml | sed -e s/^epochs:\\ 2/epochs:\\ $epochs/ > distill/configs/zeta_sft_snapshot-28000_$tag.yaml && printf 'mask_mode: suffix\n$extra\n' >> distill/configs/zeta_sft_snapshot-28000_$tag.yaml && grep -E '^model|^epochs|^mask_mode|^ar_weight|^lr' distill/configs/zeta_sft_snapshot-28000_$tag.yaml" >> $LOG 2>&1
  $Z "cd ~/shigeru/open-mercury && SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-train/bin/python -m distill.train_cdlm distill/configs/zeta_sft_snapshot-28000_$tag.yaml > results/train_zeta_sft_snapshot-28000_$tag.log 2>&1; echo \"train rc=\$?\"; rm -rf runs/zeta_sft_snapshot-28000_$tag/checkpoint-*" >> $LOG 2>&1
  M=runs/zeta_sft_snapshot-28000_$tag/final; t=cpt3_snapshot-28000_$tag
  $Z "cd ~/shigeru/open-mercury && export SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.; \
      .venv-train/bin/python bench/run_cdlm.py --model $M --engine fast_graph --n 20 --verify --tau 1e9 --ar --out results/zeta_${t}_quick.json 2>&1 | grep -E '^=== cdlm|Traceback' | sed 's/^=== cdlm/=== cdlm zeta_${t}_quick/' | cut -c1-600; \
      .venv-train/bin/python bench/run_cdlm.py --model $M --engine hf --n 20 --tau none,0.4 --out results/zeta_${t}_plain.json 2>&1 | grep -E '^=== cdlm|Traceback' | sed 's/^=== cdlm/=== cdlm zeta_${t}_plain/' | cut -c1-400; \
      .venv-train/bin/python bench/run_cdlm.py --model $M --engine fast_graph --n 1319 --prompts bench/prompts_gsm8k_full.json --verify --tau 1e9 --out results/zeta_${t}_full_verify.json 2>&1 | grep -E '^=== cdlm|Traceback' | sed 's/^=== cdlm/=== cdlm zeta_${t}_full/' | cut -c1-400" >> $LOG 2>&1
  rsync -a -e "ssh -o LogLevel=ERROR" "zeta-gpu:~/shigeru/open-mercury/results/zeta_*" results/ >> $LOG 2>&1 && $Z "cd ~/shigeru/open-mercury && rm -f results/zeta_*" >> $LOG 2>&1
  echo "=== $(date +%T) zeta recipe-exp $tag done" >> $LOG
}
run_variant lr3e5_e5 "lr: 3e-5" 5
echo "=== $(date +%T) zeta recipe-exp finished; no cycle relaunch" >> $LOG
