#!/bin/bash
# Phase E3 matrix on the idle A100 (GPU 4). docs/dependency_aware_refinement.md
cd "$(dirname "$0")/.."
P=bench/prompts_gsm8k_dev.json; L=results/pure/e3_chain.log; M=runs/phaseC_sft3_4b/final; R=results/pure/gsm8k_dev_sft3_ar.json; G=4
run() { tag=$1; K=$2; shift 2; out=results/pure/e3_K${K}_$tag; CUDA_VISIBLE_DEVICES=$G PYTHONPATH=. .venv-train/bin/python bench/run_block.py --model $M --prompts $P --n 50 --block $K --decoder k4 --timing "$@" --out $out.json > $out.log 2>&1; echo "=== $(date +%T) e3 K=$K $tag $(grep "^=== block" $out.log | cut -c11-800) $(grep -E "Traceback|Error" $out.log | head -n1)" >> $L; }
# Step 1: reproduction on an IDLE A100 (plain and graph), AR speed on the same GPU
CUDA_VISIBLE_DEVICES=$G PYTHONPATH=. .venv-train/bin/python bench/run_cdlm.py --model $M --prompts $P --n 50 --engine fast_graph --tau none --ar --window 16 --out results/pure/e3_ar_graph_a100.json > results/pure/e3_ar_graph_a100.log 2>&1; echo "=== $(date +%T) e3 AR/graph A100 $(grep '^=== cdlm' results/pure/e3_ar_graph_a100.log | cut -c9-400)" >> $L
run repro_r3_plain 4 --engine plain --k4-strategy refineN --anchor-leftmost --max-rounds 3
run repro_r3_graph 4 --engine graph --k4-strategy refineN --anchor-leftmost --max-rounds 3
run repro_r6_graph 4 --engine graph --k4-strategy refineN --anchor-leftmost --max-rounds 6
run repro_one_r12_plain 8 --engine plain --k4-strategy refineN --anchor-leftmost --adopt one --max-rounds 12
# Step 2-3: adaptive Top-M at K=8 (r3 and the equal-budget r6)
for m in 1 2 3 4; do run top${m}_r6 8 --engine plain --k4-strategy refineN --anchor-leftmost --adopt top$m --max-rounds 6; done
run top2_r12 8 --engine plain --k4-strategy refineN --anchor-leftmost --adopt top2 --max-rounds 12
run top2_r3 8 --engine plain --k4-strategy refineN --anchor-leftmost --adopt top2 --max-rounds 3
# Step 5-6: dependency-aware (distance-graph) adoption at K=8
for d in 1 2; do run wmis${d}_r6 8 --engine plain --k4-strategy refineN --anchor-leftmost --adopt wmis$d --max-rounds 6; run color${d}_r6 8 --engine plain --k4-strategy refineN --anchor-leftmost --adopt color$d --max-rounds 6; done
run wmis1_r12 8 --engine plain --k4-strategy refineN --anchor-leftmost --adopt wmis1 --max-rounds 12
run always_maxch2_r6 8 --engine plain --k4-strategy refineN --anchor-leftmost --max-rounds 6 --max-changes 2
# Step 7: K=4 early stopping (graph engine for tok/s)
run stop_two_quiet_r6_graph 4 --engine graph --k4-strategy refineN --anchor-leftmost --max-rounds 6 --stop-rule two_quiet
run stop_conf_stable_r6_graph 4 --engine graph --k4-strategy refineN --anchor-leftmost --max-rounds 6 --stop-rule conf_stable --stop-tau 0.9
run stop_conf_stable99_r6_graph 4 --engine graph --k4-strategy refineN --anchor-leftmost --max-rounds 6 --stop-rule conf_stable --stop-tau 0.99
# Step 8: adaptive block size 8 -> 4
run fb4_always_r3 8 --engine plain --k4-strategy refineN --anchor-leftmost --max-rounds 3 --fallback-k 4
run fb4_top2_r6 8 --engine plain --k4-strategy refineN --anchor-leftmost --adopt top2 --max-rounds 6 --fallback-k 4
# Step 4: dependency measurement
for K in 8 4; do CUDA_VISIBLE_DEVICES=$G PYTHONPATH=. .venv-train/bin/python bench/diag_dependency.py --model $M --refs results/pure/dev_refs_sft3.json --n 50 --K $K --out results/pure/diag_dependency_K${K}_sft3.json > results/pure/diag_dependency_K${K}_sft3.log 2>&1; echo "=== $(date +%T) e3 dependency K=$K $(grep '^=== dependency' results/pure/diag_dependency_K${K}_sft3.log | cut -c15-1500)" >> $L; done
echo "=== $(date +%T) e3 chain done" >> $L
