#!/bin/bash
# B10.5 §5 minimal structural-consistency training (data side): teacher states on MBPP problems rendered in the function-completion format (GPU 7 after arm F),
# then arm-B recipe (masked CE only) on 12k GSM8K states + these code states; HumanEval S=8/16 and dev2 S=8
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b10_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; E="PYTHONPATH=. .venv-train/bin/python"
until grep -q "abl F ordinary dev2 S16" $L; do sleep 30; done
mkdir -p data/distill/pairs_mbppfn
CUDA_VISIBLE_DEVICES=7 env $E distill/gen_pairs.py --model $M --adapter $A --prompts bench/prompts_mbpp_fnformat.json --n 400 --k 4 --max-new-tokens 512 --out data/distill/pairs_mbppfn/pairs_mbppfn > results/pure/gen_pairs_mbppfn.log 2>&1
echo "=== $(date +%T) [gpu7] gen_pairs mbpp fn-format rc=$? $(tail -n1 results/pure/gen_pairs_mbppfn.log | cut -c1-200)" >> $L
CUDA_VISIBLE_DEVICES=7 env $E distill/train_consist.py --model $M --adapter $A --data data/distill/pairs_pilot/pilot data/distill/pairs_mbppfn/pairs_mbppfn --steps 1500 --accum 8 --lr 1e-4 --hard-weight 4 --loss-mode mask_only --eval-every 500 --save-every 1500 --out runs/abl_maskce_code > results/pure/train_abl_maskce_code.log 2>&1
echo "=== $(date +%T) [gpu7] abl B+code mask_only train rc=$? $(grep 'held-out' results/pure/train_abl_maskce_code.log | tail -n 1 | cut -c1-250)" >> $L
for cfg in "8:left|bcode_s8" "16:left|bcode_s16"; do eng=${cfg%|*}; tag=${cfg#*|}
  CUDA_VISIBLE_DEVICES=7 env $E bench/run_code_eval.py --engine "bdiff:$A+runs/abl_maskce_code/adapter:$eng" --task humaneval --out results/pure/b105_humaneval_$tag.json > results/pure/b105_humaneval_$tag.log 2>&1
  echo "=== $(date +%T) [gpu7] abl B+code humaneval $tag rc=$? $(grep '^=== humaneval' results/pure/b105_humaneval_$tag.log | cut -c1-200)" >> $L; done
CUDA_VISIBLE_DEVICES=7 env $E bench/eval_trust.py --model $M --adapter $A --adapter2 runs/abl_maskce_code/adapter --no-time --prompts bench/prompts_gsm8k_dev2.json --n 150 --steps 8 --unmask left --remask none --skip-tf --out results/pure/b105_bcode_dev2_s8.json > results/pure/b105_bcode_dev2_s8.log 2>&1
echo "=== $(date +%T) [gpu7] abl B+code dev2 S8 rc=$? $(grep '^=== b7_gen' results/pure/b105_bcode_dev2_s8.log | cut -c1-250)" >> $L
