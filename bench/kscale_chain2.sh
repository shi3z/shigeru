#!/bin/bash
# Phase E2 additions (docs/k_scaling_failure_analysis.md): runs after bench/kscale_chain.sh on GPU 7.
cd "$(dirname "$0")/.."
until grep -q "kscale chain done" results/pure/kscale_chain.log; do sleep 60; done
P=bench/prompts_gsm8k_dev.json; L=results/pure/kscale_chain.log; M=runs/phaseC_sft3_4b/final; R=results/pure/gsm8k_dev_sft3_ar.json
run() { tag=$1; K=$2; shift 2; out=results/pure/kscale_K${K}_$tag; CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. .venv-train/bin/python bench/run_block.py --model $M --prompts $P --n 50 --block $K --engine plain --ar-ref $R --decoder k4 --timing "$@" --out $out.json > $out.log 2>&1; echo "=== $(date +%T) kscale K=$K $tag $(grep "^=== block" $out.log | cut -c11-700)" >> $L; }
# §6 D: 4+4 initialisation without refinement; §9 D: top-2 adoption; K=6 pairwise init with other rules / more rounds
run init4+4_norefine 8 --k4-strategy s1 --anchor-leftmost --init 4+4
run two_r3 6 --k4-strategy refineN --anchor-leftmost --max-rounds 3 --adopt two
run two_r3 8 --k4-strategy refineN --anchor-leftmost --max-rounds 3 --adopt two
run two_r6 8 --k4-strategy refineN --anchor-leftmost --max-rounds 6 --adopt two
run init2+2+2_r6 6 --k4-strategy refineN --anchor-leftmost --max-rounds 6 --init 2+2+2
run init2+2+2_nonadj_r3 6 --k4-strategy refineN --anchor-leftmost --max-rounds 3 --init 2+2+2 --adopt nonadjacent
run init2x4_r6 8 --k4-strategy refineN --anchor-leftmost --max-rounds 6 --init 2+2+2+2
run init4+4_one_r12 8 --k4-strategy refineN --anchor-leftmost --max-rounds 12 --adopt one --init 4+4
# §8 per-K teacher-forced direction statistics (right->wrong, wrong->right) via diag_k4 at K=6 and K=8
for K in 6 8; do CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. .venv-train/bin/python bench/diag_k4.py --model $M --refs results/pure/dev_refs_sft3.json --n 50 --block $K --strategies s1 refine refineN seq --out results/pure/diag_k${K}_sft3.json > results/pure/diag_k${K}_sft3.log 2>&1; echo "=== $(date +%T) diag_k$K sft3 $(grep "^=== k4diag" results/pure/diag_k${K}_sft3.log | cut -c12-1500)" >> $L; done
# §13 held-out re-evaluation of the best K=6 setting (items 50-199)
out=results/pure/kscale_K6_init2+2+2_r3_heldout; CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. .venv-train/bin/python bench/run_block.py --model $M --prompts $P --skip 50 --n 150 --block 6 --engine plain --ar-ref $R --decoder k4 --k4-strategy refineN --anchor-leftmost --max-rounds 3 --init 2+2+2 --out $out.json > $out.log 2>&1; echo "=== $(date +%T) kscale K=6 init2+2+2_r3 (held-out 50-199) $(grep "^=== block" $out.log | cut -c11-600)" >> $L
echo "=== $(date +%T) kscale chain2 done" >> $L
