#!/bin/bash
# After the recipe sweep: best recipe (suffix, lr 3e-5, 3 epochs) on the latest snapshot (28000) to see whether a later
# CPT checkpoint compounds with the stronger stage 2. Then quick/plain/full (tau=inf) benchmarks + HumanEval.
cd "$(dirname "$0")/.."; LOG=results/zeta_runs.log; Z="ssh -o LogLevel=ERROR zeta-gpu"
until grep -q "zeta recipe-exp finished" $LOG 2>/dev/null; do sleep 120; done
name=snapshot-28000; tag=cpt3_${name}_best
rsync -a -e "ssh -o LogLevel=ERROR" runs/phaseC_cpt3_4b/$name zeta-gpu:~/shigeru/open-mercury/runs/phaseC_cpt3_4b/ >> $LOG 2>&1
echo "=== $(date +%T) zeta best-recipe on $name" >> $LOG
$Z "cd ~/shigeru/open-mercury && git pull -q 2>/dev/null; sed 's#^model: .*#model: runs/phaseC_cpt3_4b/$name#; s#^out_dir: .*#out_dir: runs/zeta_sft_${name}_best#' distill/configs/phaseC_sft3_4b.yaml > distill/configs/zeta_sft_${name}_best.yaml && grep -E '^model|^epochs|^lr|^mask_mode' distill/configs/zeta_sft_${name}_best.yaml" >> $LOG 2>&1
$Z "cd ~/shigeru/open-mercury && SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-train/bin/python -m distill.train_cdlm distill/configs/zeta_sft_${name}_best.yaml > results/train_zeta_sft_${name}_best.log 2>&1; echo \"train rc=\$?\"; rm -rf runs/zeta_sft_${name}_best/checkpoint-*" >> $LOG 2>&1
M=runs/zeta_sft_${name}_best/final
$Z "cd ~/shigeru/open-mercury && export SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.; \
    .venv-train/bin/python bench/run_cdlm.py --model $M --engine fast_graph --n 20 --verify --tau 1e9 --ar --out results/zeta_${tag}_quick.json 2>&1 | grep -E '^=== cdlm|Traceback' | sed 's/^=== cdlm/=== cdlm zeta_${tag}_quick/' | cut -c1-600; \
    .venv-train/bin/python bench/run_cdlm.py --model $M --engine hf --n 20 --tau none,0.4 --out results/zeta_${tag}_plain.json 2>&1 | grep -E '^=== cdlm|Traceback' | sed 's/^=== cdlm/=== cdlm zeta_${tag}_plain/' | cut -c1-400; \
    .venv-train/bin/python bench/run_cdlm.py --model $M --engine fast_graph --n 1319 --prompts bench/prompts_gsm8k_full.json --verify --tau 1e9 --out results/zeta_${tag}_full_verify.json 2>&1 | grep -E '^=== cdlm|Traceback' | sed 's/^=== cdlm/=== cdlm zeta_${tag}_full/' | cut -c1-400; \
    D=$M/../final_eval; mkdir -p \$D; ln -sf ../final/model.safetensors \$D/model.safetensors; cp $M/config.json $M/generation_config.json \$D/ 2>/dev/null; for fn in tokenizer.json tokenizer_config.json vocab.json merges.txt; do cp ~/models/hf/Qwen3-4B/\$fn \$D/; done; \
    .venv/bin/python bench/run_code_eval.py --engine hf:\$D --task humaneval --out results/zeta_humaneval_${tag}.json 2>&1 | grep -E '^===|Traceback' | cut -c1-200" >> $LOG 2>&1
rsync -a -e "ssh -o LogLevel=ERROR" "zeta-gpu:~/shigeru/open-mercury/results/zeta_*" results/ >> $LOG 2>&1 && $Z "cd ~/shigeru/open-mercury && rm -f results/zeta_*" >> $LOG 2>&1
echo "=== $(date +%T) zeta best-recipe on $name done" >> $LOG
