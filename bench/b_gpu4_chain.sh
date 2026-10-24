#!/bin/bash
# B-series on tsuginosuke GPU 4 (A100): same base as the A-series (runs/phaseC_sft3_4b/final).  B1 control, then B5 ablations.
cd "$(dirname "$0")/.."
until grep -q "e3 heldout done" results/pure/e3_chain.log; do sleep 60; done
M=runs/phaseC_sft3_4b/final; D=data/distill/phaseA_stage2c_train.jsonl; R=results/pure/dev_refs_sft3.json; ST=data/distill/pure_states_random.jsonl; L=results/pure/b_chain.log
E="CUDA_VISIBLE_DEVICES=4 PYTHONPATH=."
tr() { name=$1; shift; env $E .venv-train/bin/python distill/train_bdiff.py --model $M --data $D --out runs/bdiff_sft3_$name --steps 2000 --accum 8 --lr 2e-4 --save-every 500 "$@" > results/pure/train_bdiff_sft3_$name.log 2>&1; echo "=== $(date +%T) [gpu4] train $name rc=$? $(tail -n 2 results/pure/train_bdiff_sft3_$name.log | head -n1 | cut -c1-300)" >> $L; }
ev() { name=$1; shift; env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter runs/bdiff_sft3_$name/adapter --time-embed runs/bdiff_sft3_$name/time_embed.pt --refs $R --states $ST --n 50 --steps 16,8,4 "$@" --out results/pure/bdiff_sft3_${name}_eval.json > results/pure/bdiff_sft3_${name}_eval.log 2>&1; echo "=== $(date +%T) [gpu4] eval $name rc=$? $(grep -E '^=== bdiff' results/pure/bdiff_sft3_${name}_eval.log | cut -c1-700 | tr '\n' ' ')" >> $L; }
tr b1; ev b1
tr b5_notime --no-time; env $E .venv-train/bin/python bench/eval_bdiff.py --model $M --adapter runs/bdiff_sft3_b5_notime/adapter --no-time --refs $R --states $ST --n 50 --steps 16,8,4 --out results/pure/bdiff_sft3_b5_notime_eval.json > results/pure/bdiff_sft3_b5_notime_eval.log 2>&1; echo "=== $(date +%T) [gpu4] eval b5_notime rc=$? $(grep -E '^=== bdiff' results/pure/bdiff_sft3_b5_notime_eval.log | cut -c1-700 | tr '\n' ' ')" >> $L
tr b5_cosine --schedule cosine; ev b5_cosine --schedule cosine
echo "=== $(date +%T) [gpu4] b chain done" >> $L
