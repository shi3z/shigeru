#!/bin/bash
cd "$(dirname "$0")/.."
M=runs/phaseC_sft3_4b/final; R=results/pure/dev_refs_sft3.json; L=results/pure/b_chain.log; E="CUDA_VISIBLE_DEVICES=7 PYTHONPATH=."
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter runs/bdiff_sft3_b3/adapter --time-embed runs/bdiff_sft3_b3/time_embed.pt --refs $R --states data/distill/bstates_sft3_b1.jsonl --n 50 --steps 16,8,4 --out results/pure/bdiff_sft3_b3_eval.json > results/pure/bdiff_sft3_b3_eval.log 2>&1; echo "=== $(date +%T) [gpu7] b3 eval(2) rc=$? $(grep -E '^=== bdiff' results/pure/bdiff_sft3_b3_eval.log | cut -c1-700 | sed ':a;N;$!ba;s/\n/ /g')" >> $L
# B1 (sft3) on-policy correction test with the B-sampler states, for the B3 comparison
env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter runs/bdiff_sft3_b1/adapter --time-embed runs/bdiff_sft3_b1/time_embed.pt --refs $R --states data/distill/bstates_sft3_b1.jsonl --n 50 --skip-gen --out results/pure/bdiff_sft3_b1_bstates_eval.json > results/pure/bdiff_sft3_b1_bstates_eval.log 2>&1; echo "=== $(date +%T) [gpu7] b1 B3-test with B-states rc=$? $(grep -E '^=== bdiff_B3' results/pure/bdiff_sft3_b1_bstates_eval.log | cut -c1-900)" >> $L
echo "=== $(date +%T) [gpu7] b3 eval chain done" >> $L
