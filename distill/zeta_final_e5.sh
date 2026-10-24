#!/bin/bash
# zeta: the 5-epoch recipe (best tokens/step so far) on the FINAL 1B-token CPT checkpoint, with quick/plain/full (tau=inf) + HumanEval.
cd "$(dirname "$0")/.."; LOG=results/zeta_runs.log; Z="ssh -o LogLevel=ERROR zeta-gpu"; name=final; tag=cpt3_final_e5
mkdir -p /dev/null 2>/dev/null; rsync -a -e "ssh -o LogLevel=ERROR" runs/phaseC_cpt3_4b/final zeta-gpu:~/shigeru/open-mercury/runs/phaseC_cpt3_4b/ >> $LOG 2>&1
echo "=== $(date +%T) zeta final-e5: stage 2 (suffix, lr 3e-5, 5 epochs) on the final CPT checkpoint" >> $LOG
$Z "cd ~/shigeru/open-mercury && git pull -q 2>/dev/null; sed 's#^model: .*#model: runs/phaseC_cpt3_4b/final#; s#^out_dir: .*#out_dir: runs/zeta_sft_final_e5#' distill/configs/phaseC_sft3_4b.yaml | sed -e s/^epochs:.*/epochs:\\ 5/ > distill/configs/zeta_sft_final_e5.yaml && grep -E '^model|^epochs|^lr|^mask_mode' distill/configs/zeta_sft_final_e5.yaml" >> $LOG 2>&1
$Z "cd ~/shigeru/open-mercury && SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-train/bin/python -m distill.train_cdlm distill/configs/zeta_sft_final_e5.yaml > results/train_zeta_sft_final_e5.log 2>&1; echo \"train rc=\$?\"; rm -rf runs/zeta_sft_final_e5/checkpoint-*" >> $LOG 2>&1
M=runs/zeta_sft_final_e5/final
$Z "cd ~/shigeru/open-mercury && export SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.; \
    .venv-train/bin/python bench/run_cdlm.py --model $M --engine fast_graph --n 20 --verify --tau 1e9 --ar --out results/zeta_${tag}_quick.json 2>&1 | grep -E '^=== cdlm|Traceback' | sed 's/^=== cdlm/=== cdlm zeta_${tag}_quick/' | cut -c1-600; \
    .venv-train/bin/python bench/run_cdlm.py --model $M --engine hf --n 20 --tau none,0.4 --out results/zeta_${tag}_plain.json 2>&1 | grep -E '^=== cdlm|Traceback' | sed 's/^=== cdlm/=== cdlm zeta_${tag}_plain/' | cut -c1-400; \
    .venv-train/bin/python bench/run_cdlm.py --model $M --engine fast_graph --n 1319 --prompts bench/prompts_gsm8k_full.json --verify --tau 1e9 --out results/zeta_${tag}_full_verify.json 2>&1 | grep -E '^=== cdlm|Traceback' | sed 's/^=== cdlm/=== cdlm zeta_${tag}_full/' | cut -c1-400; \
    D=$M/../final_eval; mkdir -p \$D; ln -sf ../final/model.safetensors \$D/model.safetensors; cp $M/config.json $M/generation_config.json \$D/ 2>/dev/null; for fn in tokenizer.json tokenizer_config.json vocab.json merges.txt; do cp ~/models/hf/Qwen3-4B/\$fn \$D/; done; \
    .venv/bin/python bench/run_code_eval.py --engine hf:\$D --task humaneval --out results/zeta_humaneval_${tag}.json 2>&1 | grep -E '^===|Traceback' | cut -c1-200" >> $LOG 2>&1
rsync -a -e "ssh -o LogLevel=ERROR" "zeta-gpu:~/shigeru/open-mercury/results/zeta_*" results/ >> $LOG 2>&1 && $Z "cd ~/shigeru/open-mercury && rm -f results/zeta_*" >> $LOG 2>&1
echo "=== $(date +%T) zeta final-e5 done" >> $LOG
