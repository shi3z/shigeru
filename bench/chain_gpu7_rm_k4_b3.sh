#!/bin/bash
# GPU 7: K=4/6/8 evaluations of the random and mixed checkpoints after their current chain, then B3 (B-sampler on-policy states).
cd "$(dirname "$0")/.."
until grep -q "gpu7 eval of pure_sft3_random/mixed done" results/pure/train_chain.log; do sleep 60; done
P=bench/prompts_gsm8k_dev.json; L=results/pure/eval_pure_chain.log; E="CUDA_VISIBLE_DEVICES=7 PYTHONPATH=."
for m in random mixed; do M=runs/pure_sft3_$m/final; R=results/pure/dev_refs_pure_$m.json
run() { tag=$1; shift; out=results/pure/eval_pure_${m}_$tag; env $E .venv-train/bin/python bench/run_block.py --model $M --prompts $P --n 50 --engine plain --ar-ref $R --decoder k4 "$@" --out $out.json > $out.log 2>&1; echo "=== $(date +%T) $m $tag $(grep '^=== block' $out.log | cut -c11-600)" >> $L; }
run k4_refineN_r3 --block 4 --k4-strategy refineN --anchor-leftmost --max-rounds 3
run k4_refineN_r6 --block 4 --k4-strategy refineN --anchor-leftmost --max-rounds 6
run k8_color1_r6 --block 8 --k4-strategy refineN --anchor-leftmost --adopt color1 --max-rounds 6
env $E .venv-train/bin/python bench/diag_k4.py --model $M --refs $R --n 50 --block 4 --out results/pure/diag_k4_pure_$m.json > results/pure/diag_k4_pure_$m.log 2>&1; echo "=== $(date +%T) $m diag_k4 $(grep '^=== k4diag' results/pure/diag_k4_pure_$m.log | cut -c12-1200)" >> $L
done; echo "=== $(date +%T) eval random/mixed k4 done" >> $L
# B3: wait for the GPU 4 B1 control adapter, generate the B-sampler's own states, train from B1 with those states
until [ -f runs/bdiff_sft3_b1/adapter/adapter_config.json ] && grep -q "\[gpu4\] eval b1" results/pure/b_chain.log; do sleep 120; done
M=runs/phaseC_sft3_4b/final; D=data/distill/phaseA_stage2c_train.jsonl; R=results/pure/dev_refs_sft3.json; L=results/pure/b_chain.log
A=runs/bdiff_sft3_b1/adapter; T=runs/bdiff_sft3_b1/time_embed.pt
env $E .venv-train/bin/python distill/gen_bstates.py --mode states --model $M --adapter $A --time-embed $T --data $D --limit 3000 --cuts 2 --states-per-block 2 --steps 16 --out data/distill/bstates_sft3_b1.jsonl > results/pure/gen_bstates_sft3_b1.log 2>&1; echo "=== $(date +%T) [gpu7] b3 states rc=$? $(tail -n 1 results/pure/gen_bstates_sft3_b1.log | cut -c1-200)" >> $L
env $E .venv-train/bin/python distill/train_bdiff.py --model $M --data $D --init-adapter $A --states data/distill/bstates_sft3_b1.jsonl --p-state 0.3 --lambda-vis 0.3 --out runs/bdiff_sft3_b3 --steps 1500 --accum 8 --lr 1e-4 --save-every 500 > results/pure/train_bdiff_sft3_b3.log 2>&1; echo "=== $(date +%T) [gpu7] b3 train rc=$? $(tail -n 2 results/pure/train_bdiff_sft3_b3.log | head -n1 | cut -c1-300)" >> $L
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter runs/bdiff_sft3_b3/adapter --time-embed runs/bdiff_sft3_b3/time_embed.pt --refs $R --states data/distill/bstates_sft3_b1.jsonl --n 50 --steps 16,8,4 --out results/pure/bdiff_sft3_b3_eval.json > results/pure/bdiff_sft3_b3_eval.log 2>&1; echo "=== $(date +%T) [gpu7] b3 eval rc=$? $(grep -E '^=== bdiff' results/pure/bdiff_sft3_b3_eval.log | cut -c1-700 | tr '\n' ' ')" >> $L
echo "=== $(date +%T) [gpu7] b3 chain done" >> $L
