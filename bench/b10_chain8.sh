#!/bin/bash
# B10 variant E (GPU 4, after ablation D): pair-KL on ALL GSM8K (65k) + MBPP (~11k) teacher states, 3000 steps x 8, hard x4; then dev2 S=8/S=16, HumanEval S=8/S=16
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b10_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; E="PYTHONPATH=. .venv-train/bin/python"
until ssh -o BatchMode=yes -o ConnectTimeout=20 zeta-gpu 'grep -q "gen_pairs mbpp rc=0" ~/shigeru/open-mercury/results/pure/b10_zeta_chain.log' 2>/dev/null; do sleep 60; done
mkdir -p data/distill/pairs_mbpp
scp -q zeta-gpu:~/shigeru/open-mercury/data/distill/pairs_mbpp_*.pt zeta-gpu:~/shigeru/open-mercury/data/distill/pairs_mbpp_prompts.jsonl zeta-gpu:~/shigeru/open-mercury/data/distill/pairs_mbpp_meta.json data/distill/pairs_mbpp/
echo "=== $(date +%T) [local] mbpp teacher data fetched: $(ls data/distill/pairs_mbpp | wc -l) files" >> $L
until grep -q "consist D dev2 S8" $L; do sleep 30; done
CUDA_VISIBLE_DEVICES=4 env $E distill/train_consist.py --model $M --adapter $A --data data/distill/pairs_gsm8k/pairs_gsm8k data/distill/pairs_mbpp/pairs_mbpp --steps 3000 --accum 8 --lr 1e-4 --hard-weight 4 --eval-every 500 --save-every 1000 --out runs/consist_e > results/pure/train_consist_e.log 2>&1
echo "=== $(date +%T) [gpu4] consist E (pair-KL, gsm8k+mbpp, 3000 steps) train rc=$? $(grep 'held-out' results/pure/train_consist_e.log | tail -n 1 | cut -c1-250)" >> $L
for cfg in "bench/prompts_gsm8k_dev2.json 150 8 dev2_s8" "bench/prompts_gsm8k_dev2.json 150 16 dev2_s16"; do set -- $cfg
  CUDA_VISIBLE_DEVICES=4 env $E bench/eval_trust.py --model $M --adapter $A --adapter2 runs/consist_e/adapter --no-time --prompts $1 --n $2 --steps $3 --unmask left --remask none --skip-tf --out results/pure/b10_e_$4.json > results/pure/b10_e_$4.log 2>&1
  echo "=== $(date +%T) [gpu4] consist E $4 rc=$? $(grep '^=== b7_gen' results/pure/b10_e_$4.log | cut -c1-250)" >> $L; done
for cfg in "8:left|e_s8" "16:left|e_s16"; do eng=${cfg%|*}; tag=${cfg#*|}
  CUDA_VISIBLE_DEVICES=4 env $E bench/run_code_eval.py --engine bdiff:$A+runs/consist_e/adapter:$eng --task humaneval --out results/pure/b10_humaneval_$tag.json > results/pure/b10_humaneval_$tag.log 2>&1
  echo "=== $(date +%T) [gpu4] consist E humaneval $tag rc=$? $(grep '^=== humaneval' results/pure/b10_humaneval_$tag.log | cut -c1-200)" >> $L; done
