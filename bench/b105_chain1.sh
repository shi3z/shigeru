#!/bin/bash
# B10.5 ablation arms on tsuginosuke: GPU 6 (after the E test) = arm D kl_only; GPU 7 (after the teacher test) = arm F ordinary-data continuation of the lr1e-4 adapter (same steps/lr/rank)
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b10_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; E="PYTHONPATH=. .venv-train/bin/python"
( until grep -q "TEST student E S8" $L; do sleep 30; done
  CUDA_VISIBLE_DEVICES=6 env $E distill/train_consist.py --model $M --adapter $A --data data/distill/pairs_pilot/pilot --steps 1000 --accum 8 --lr 1e-4 --hard-weight 4 --loss-mode kl_only --eval-every 500 --save-every 1000 --out runs/abl_klonly > results/pure/train_abl_klonly.log 2>&1
  echo "=== $(date +%T) [gpu6] abl D kl_only train rc=$? $(grep 'held-out' results/pure/train_abl_klonly.log | tail -n 1 | cut -c1-250)" >> $L
  for st in 8 16; do CUDA_VISIBLE_DEVICES=6 env $E bench/eval_trust.py --model $M --adapter $A --adapter2 runs/abl_klonly/adapter --no-time --prompts bench/prompts_gsm8k_dev2.json --n 150 --steps $st --unmask left --remask none --skip-tf --out results/pure/b105_klonly_dev2_s$st.json > results/pure/b105_klonly_dev2_s$st.log 2>&1
    echo "=== $(date +%T) [gpu6] abl D kl_only dev2 S$st rc=$? $(grep '^=== b7_gen' results/pure/b105_klonly_dev2_s$st.log | cut -c1-250)" >> $L; done ) &
( until grep -q "TEST teacher S8" $L; do sleep 30; done
  CUDA_VISIBLE_DEVICES=7 env $E distill/train_bdiff.py --model $M --data data/distill/phaseA_stage2c_train.jsonl --init-adapter $A --no-time --schedule linear --lr 1e-4 --steps 1000 --accum 8 --out runs/abl_ordinary > results/pure/train_abl_ordinary.log 2>&1
  echo "=== $(date +%T) [gpu7] abl F ordinary-data continuation train rc=$? $(tail -n 2 results/pure/train_abl_ordinary.log | head -n1 | cut -c1-200)" >> $L
  for st in 8 16; do CUDA_VISIBLE_DEVICES=7 env $E bench/eval_trust.py --model $M --adapter runs/abl_ordinary/adapter --no-time --prompts bench/prompts_gsm8k_dev2.json --n 150 --steps $st --unmask left --remask none --skip-tf --out results/pure/b105_ordinary_dev2_s$st.json > results/pure/b105_ordinary_dev2_s$st.log 2>&1
    echo "=== $(date +%T) [gpu7] abl F ordinary dev2 S$st rc=$? $(grep '^=== b7_gen' results/pure/b105_ordinary_dev2_s$st.log | cut -c1-250)" >> $L; done ) &
wait
