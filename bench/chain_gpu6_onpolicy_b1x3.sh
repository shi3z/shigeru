#!/bin/bash
# GPU 6: evaluate the on-policy (config D) checkpoint when its training ends, then B1 x3 (6000 steps, mask-only) on the sft3 base.
cd "$(dirname "$0")/.."
until grep -q "pure_sft3_onpolicy rc=0" results/pure/train_chain.log; do sleep 120; done
P=bench/prompts_gsm8k_dev.json; m=onpolicy; M=runs/pure_sft3_onpolicy/final; R=results/pure/dev_refs_pure_$m.json; L=results/pure/eval_pure_chain.log; E="CUDA_VISIBLE_DEVICES=6 PYTHONPATH=."
env $E .venv-train/bin/python bench/diag_mask_recovery.py --model $M --prompts $P --n 100 --refs $R --out results/pure/diag_recovery_pure_$m.json > results/pure/diag_recovery_pure_$m.log 2>&1; echo "=== $(date +%T) $m recovery $(grep '^=== recovery' results/pure/diag_recovery_pure_$m.log | cut -c14-700)" >> $L
run() { tag=$1; shift; out=results/pure/eval_pure_${m}_$tag; env $E .venv-train/bin/python bench/run_block.py --model $M --prompts $P --n 50 --engine plain --ar-ref $R "$@" --out $out.json > $out.log 2>&1; echo "=== $(date +%T) $m $tag $(grep '^=== block' $out.log | cut -c11-600)" >> $L; }
run k4_refineN_r3 --block 4 --decoder k4 --k4-strategy refineN --anchor-leftmost --max-rounds 3
run k4_refineN_r6 --block 4 --decoder k4 --k4-strategy refineN --anchor-leftmost --max-rounds 6
run k8_color1_r6 --block 8 --decoder k4 --k4-strategy refineN --anchor-leftmost --adopt color1 --max-rounds 6
run anchor_tau99 --block 32 --steps adaptive --tau 0.99 --anchor-leftmost
run anchor_tau90 --block 32 --steps adaptive --tau 0.9 --anchor-leftmost
run S16_conf --block 32 --steps 16 --order confidence
env $E .venv-train/bin/python bench/diag_k4.py --model $M --refs $R --n 50 --block 4 --out results/pure/diag_k4_pure_$m.json > results/pure/diag_k4_pure_$m.log 2>&1; echo "=== $(date +%T) $m diag_k4 $(grep '^=== k4diag' results/pure/diag_k4_pure_$m.log | cut -c12-1200)" >> $L
echo "=== $(date +%T) eval $m done" >> $L
# B1 x3 on the A-series base
M=runs/phaseC_sft3_4b/final; D=data/distill/phaseA_stage2c_train.jsonl; R=results/pure/dev_refs_sft3.json; ST=data/distill/pure_states_random.jsonl; L=results/pure/b_chain.log
env $E .venv-train/bin/python distill/train_bdiff.py --model $M --data $D --out runs/bdiff_sft3_b1x3 --steps 6000 --accum 8 --lr 2e-4 --save-every 1000 > results/pure/train_bdiff_sft3_b1x3.log 2>&1; echo "=== $(date +%T) [gpu6] train b1x3 rc=$? $(tail -n 2 results/pure/train_bdiff_sft3_b1x3.log | head -n1 | cut -c1-300)" >> $L
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter runs/bdiff_sft3_b1x3/adapter --time-embed runs/bdiff_sft3_b1x3/time_embed.pt --refs $R --states $ST --n 50 --steps 16,8,4 --out results/pure/bdiff_sft3_b1x3_eval.json > results/pure/bdiff_sft3_b1x3_eval.log 2>&1; echo "=== $(date +%T) [gpu6] eval b1x3 rc=$? $(grep -E '^=== bdiff' results/pure/bdiff_sft3_b1x3_eval.log | cut -c1-700 | tr '\n' ' ')" >> $L
echo "=== $(date +%T) [gpu6] b1x3 chain done" >> $L
