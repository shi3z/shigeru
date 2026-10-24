#!/bin/bash
# K-scaling failure analysis matrix (docs/k_scaling_failure_analysis.md). GPU 7, shared; waits for the Phase E confirmation runs.
cd "$(dirname "$0")/.."
until grep -q "k4 confirm done" results/pure/k4_chain.log; do sleep 60; done
P=bench/prompts_gsm8k_dev.json; L=results/pure/kscale_chain.log; M=runs/phaseC_sft3_4b/final; R=results/pure/gsm8k_dev_sft3_ar.json
run() { tag=$1; K=$2; shift 2; out=results/pure/kscale_K${K}_$tag; CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. .venv-train/bin/python bench/run_block.py --model $M --prompts $P --n 50 --block $K --engine plain --ar-ref $R --decoder k4 --timing "$@" --out $out.json > $out.log 2>&1; echo "=== $(date +%T) kscale K=$K $tag $(grep "^=== block" $out.log | cut -c11-700)" >> $L; }
# Step 1: reproduction
run repro_refineN_r3 4 --k4-strategy refineN --anchor-leftmost --max-rounds 3
run repro_refineN_r3 8 --k4-strategy refineN --anchor-leftmost --max-rounds 3
# Experiment A: K scaling (K=6 first)
for K in 6 5 7 3; do run refineN_r3 $K --k4-strategy refineN --anchor-leftmost --max-rounds 3; done
# Experiment D: split initialisation
run init3+3_r3 6 --k4-strategy refineN --anchor-leftmost --max-rounds 3 --init 3+3
run init2+2+2_r3 6 --k4-strategy refineN --anchor-leftmost --max-rounds 3 --init 2+2+2
run init4+4_r3 8 --k4-strategy refineN --anchor-leftmost --max-rounds 3 --init 4+4
run init2x4_r3 8 --k4-strategy refineN --anchor-leftmost --max-rounds 3 --init 2+2+2+2
# Experiment E: update policies at K=6 and K=8 (r3 budget; oddeven uses 2 probes/round so r3 == 6 probes; always_r6 is the equal-forward control)
for K in 6 8; do
  for ad in nonadjacent conf one damped; do run ${ad}_r3 $K --k4-strategy refineN --anchor-leftmost --max-rounds 3 --adopt $ad; done
  run oddeven_r3 $K --k4-strategy refineN --anchor-leftmost --max-rounds 3 --adopt oddeven
  run always_r6 $K --k4-strategy refineN --anchor-leftmost --max-rounds 6
  run one_r12 $K --k4-strategy refineN --anchor-leftmost --max-rounds 12 --adopt one
done
# Experiment G: EOS policies at K=8 (and K=6)
for K in 8 6; do for pol in suppress_first verify; do run eos_${pol}_r3 $K --k4-strategy refineN --anchor-leftmost --max-rounds 3 --eos-policy $pol; done; done
# Experiments B / C / 10 (teacher-forced) and the KV/attention audit
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. .venv-train/bin/python bench/diag_kscale.py --model $M --refs results/pure/dev_refs_sft3.json --n 50 --K 3 4 5 6 7 8 --inject-K 6 8 --out results/pure/diag_kscale_sft3.json > results/pure/diag_kscale_sft3.log 2>&1
echo "=== $(date +%T) diag_kscale rc=$? $(grep -E '^=== kscale_B' results/pure/diag_kscale_sft3.log | cut -c1-1500)" >> $L
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. .venv-train/bin/python bench/audit_k_cache.py --model $M --refs results/pure/dev_refs_sft3.json --n 8 --out results/pure/audit_k_cache_sft3.json > results/pure/audit_k_cache_sft3.log 2>&1
echo "=== $(date +%T) audit rc=$? $(grep '^=== audit' results/pure/audit_k_cache_sft3.log | cut -c1-900)" >> $L
echo "=== $(date +%T) kscale chain done" >> $L
