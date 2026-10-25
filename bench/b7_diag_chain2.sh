#!/bin/bash
# B7 §2 re-run of the trust diagnostic with the e5 base model's OWN greedy answers as reference (GPU 7, after make_refs)
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b7_chain.log; M=runs/zeta_sft_final_e5/final; R=results/pure/dev_refs_e5.json
until grep -q "refs e5 dev50 rc=0 === refs" $L; do sleep 30; done
for A in lr1e4 notime; do
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. .venv-train/bin/python bench/diag_trust.py --model $M --adapter runs/bdiff_e5_$A/adapter --no-time --refs $R --n 50 --steps 8 --unmask confidence,left --oracle --oracle-blocks 40 --tag e5refs_$A --out results/pure/diag_trust_e5refs_$A.json > results/pure/diag_trust_e5refs_$A.log 2>&1
  echo "=== $(date +%T) [gpu7] diag_trust e5refs $A rc=$? $(grep '^=== trust ' results/pure/diag_trust_e5refs_$A.log | cut -c1-500)" >> $L
done
