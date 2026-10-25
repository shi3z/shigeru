#!/bin/bash
# MBPP pass@1 (sanitized) for the undistilled adapter and student A (both never trained on MBPP) at S=8/16 — GPU 6
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b10_chain.log; A=runs/bdiff_e5_lr1e4/adapter; E="PYTHONPATH=. .venv-train/bin/python"
for cfg in "$A:8:left|base_s8" "$A:16:left|base_s16" "$A+runs/consist_a/adapter:8:left|a_s8" "$A+runs/consist_a/adapter:16:left|a_s16"; do eng=${cfg%|*}; tag=${cfg#*|}
  CUDA_VISIBLE_DEVICES=6 env $E bench/run_code_eval.py --engine "bdiff:$eng" --task mbpp --out results/pure/b105_mbpp_$tag.json > results/pure/b105_mbpp_$tag.log 2>&1
  echo "=== $(date +%T) [gpu6] mbpp $tag rc=$? $(grep '^=== mbpp' results/pure/b105_mbpp_$tag.log | cut -c1-200)" >> $L; done
