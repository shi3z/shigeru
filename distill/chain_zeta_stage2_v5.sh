#!/bin/bash
# While the 1B-token CPT runs on the A100s: for every new HF snapshot (every 2000 steps), copy it to zeta (RTX PRO 6000),
# run stage 2 there (enlarged mix, phaseC_sft3-style config), evaluate (verified 20 + full GSM8K with CUDA graphs, AR-row
# HumanEval), and copy the results back. One cycle ~3 h, matching the snapshot cadence. Log results/zeta_runs.log.
cd "$(dirname "$0")/.."; LOG=results/zeta_runs.log; Z="ssh -o LogLevel=ERROR zeta-gpu"; last="runs/phaseC_cpt3_4b/snapshot-18000"
while true; do
  s=$(ls -d runs/phaseC_cpt3_4b/snapshot-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  if [ -n "$s" ] && [ "$s" != "$last" ] && [ -f "$s/model.safetensors" ]; then
    sleep 120; last=$s; name=$(basename $s); tag=cpt3_${name}
    echo "=== $(date +%T) zeta cycle $name: rsync" >> $LOG
    rsync -a -e "ssh -o LogLevel=ERROR" $s zeta-gpu:~/shigeru/open-mercury/runs/phaseC_cpt3_4b/ >> $LOG 2>&1
    $Z "cd ~/shigeru/open-mercury && git pull -q 2>/dev/null; sed 's#^model: .*#model: runs/phaseC_cpt3_4b/$name#; s#^out_dir: .*#out_dir: runs/zeta_sft_$name#' distill/configs/phaseC_sft3_4b.yaml | sed -e s/^epochs:\\ 2/epochs:\\ 1/ > distill/configs/zeta_sft_$name.yaml && echo cfg ok" >> $LOG 2>&1
    echo "=== $(date +%T) zeta stage 2 on $name" >> $LOG
    $Z "cd ~/shigeru/open-mercury && SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-train/bin/python -m distill.train_cdlm distill/configs/zeta_sft_$name.yaml > results/train_zeta_sft_$name.log 2>&1; echo \"train rc=\$?\"; rm -rf runs/zeta_sft_$name/checkpoint-*" >> $LOG 2>&1
    M=runs/zeta_sft_$name/final
    $Z "cd ~/shigeru/open-mercury && export SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.; \
        .venv-train/bin/python bench/run_cdlm.py --model $M --engine fast_graph --n 20 --verify --tau 1e9 --ar --out results/zeta_${tag}_quick.json 2>&1 | grep -E '^=== cdlm|Traceback' | sed 's/^=== cdlm/=== cdlm zeta_${tag}_quick/' | cut -c1-600; \
        .venv-train/bin/python bench/run_cdlm.py --model $M --engine hf --n 20 --tau none,0.4 --out results/zeta_${tag}_plain.json 2>&1 | grep -E '^=== cdlm|Traceback' | sed 's/^=== cdlm/=== cdlm zeta_${tag}_plain/' | cut -c1-400; \
        .venv-train/bin/python bench/run_cdlm.py --model $M --engine fast_graph --n 1319 --prompts bench/prompts_gsm8k_full.json --verify --tau 1e9 --out results/zeta_${tag}_full_verify.json 2>&1 | grep -E '^=== cdlm|Traceback' | sed 's/^=== cdlm/=== cdlm zeta_${tag}_full/' | cut -c1-400; \
        D=$M/../final_eval; mkdir -p \$D; ln -sf ../final/model.safetensors \$D/model.safetensors; cp $M/config.json $M/generation_config.json \$D/ 2>/dev/null; for fn in tokenizer.json tokenizer_config.json vocab.json merges.txt; do cp ~/models/hf/Qwen3-4B/\$fn \$D/; done; \
        .venv/bin/python bench/run_code_eval.py --engine hf:\$D --task humaneval --out results/zeta_humaneval_${tag}.json 2>&1 | grep -E '^===|Traceback' | cut -c1-200" >> $LOG 2>&1
    rsync -a -e "ssh -o LogLevel=ERROR" "zeta-gpu:~/shigeru/open-mercury/results/zeta_*" results/ && $Z "cd ~/shigeru/open-mercury && rm -f results/zeta_*" >> $LOG 2>&1
    echo "=== $(date +%T) zeta cycle $name done" >> $LOG
  fi
  grep -q "phaseC_cpt3_4b rc=" results/phaseB_runs.log 2>/dev/null && [ "$last" = "$s" ] && [ -n "$s" ] && break
  sleep 300
done
echo "=== $(date +%T) chain_zeta_stage2 finished" >> $LOG
