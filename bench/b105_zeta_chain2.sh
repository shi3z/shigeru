#!/bin/bash
# B10.5 arm B-large (zeta): masked CE only on ALL 65k GSM8K teacher-trajectory states, 2000 steps x 8; dev2 S=8/S=16, then HumanEval S=8
cd ~/shigeru/open-mercury
L=results/pure/b105_zeta_chain.log; E="SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-train/bin/python"; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter
env $E distill/train_consist.py --model $M --adapter $A --data data/distill/pairs_gsm8k --steps 2000 --accum 8 --lr 1e-4 --hard-weight 4 --loss-mode mask_only --eval-every 500 --save-every 1000 --out runs/abl_maskce_large > results/pure/train_abl_maskce_large.log 2>&1
echo "=== $(date +%T) [zeta] abl B-large mask_only 65k train rc=$? $(grep 'held-out' results/pure/train_abl_maskce_large.log | tail -n 1 | cut -c1-250)" >> $L
for st in 8 16; do
  env $E bench/eval_trust.py --model $M --adapter $A --adapter2 runs/abl_maskce_large/adapter --no-time --prompts bench/prompts_gsm8k_dev2.json --n 150 --steps $st --unmask left --remask none --skip-tf --out results/pure/b105_maskce_large_dev2_s$st.json > results/pure/b105_maskce_large_dev2_s$st.log 2>&1
  echo "=== $(date +%T) [zeta] abl B-large dev2 S$st rc=$? $(grep '^=== b7_gen' results/pure/b105_maskce_large_dev2_s$st.log | cut -c1-250)" >> $L; done
env $E bench/run_code_eval.py --engine "bdiff:$A+runs/abl_maskce_large/adapter:8:left" --task humaneval --out results/pure/b105_humaneval_maskce_large_s8.json > results/pure/b105_humaneval_maskce_large_s8.log 2>&1
echo "=== $(date +%T) [zeta] abl B-large humaneval S8 rc=$? $(grep '^=== humaneval' results/pure/b105_humaneval_maskce_large_s8.log | cut -c1-200)" >> $L
