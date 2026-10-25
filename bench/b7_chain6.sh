#!/bin/bash
# B7 §7 HumanEval for the A-series K=4 decoder (GPU 4, after the dev-50 sweep b)
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b7_chain.log; E="PYTHONPATH=. .venv-train/bin/python"
until grep -q "b7 gen dev50 sweep b rc=" $L; do sleep 30; done
CUDA_VISIBLE_DEVICES=4 env $E bench/run_code_eval.py --engine k4:runs/phaseC_sft3_4b/final:6 --task humaneval --n 3 --out results/pure/b7_he_k4_smoke.json > results/pure/b7_he_k4_smoke.log 2>&1
echo "=== $(date +%T) [gpu4] he k4 smoke rc=$? $(tail -n1 results/pure/b7_he_k4_smoke.log | cut -c1-200)" >> $L
CUDA_VISIBLE_DEVICES=4 env $E bench/run_code_eval.py --engine k4:runs/phaseC_sft3_4b/final:6 --task humaneval --out results/pure/b7_humaneval_k4_r6.json > results/pure/b7_humaneval_k4_r6.log 2>&1
echo "=== $(date +%T) [gpu4] humaneval k4 r6 rc=$? $(grep '^=== humaneval' results/pure/b7_humaneval_k4_r6.log | cut -c1-200)" >> $L
