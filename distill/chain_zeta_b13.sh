#!/bin/bash
# B13 chain on zeta (RTX PRO 6000, GPU 0): train the auxiliary-conditioned drafter (+ alignment), evaluate on dev2
# (verified decoding W=16 tau=1e9: tokens/step and tok/s with the rank-0 auxiliary input vs the [MASK] input, and AR),
# then the auxiliary-only ablation, then GSM8K test (1319) for both.  Control: runs/phaseC_sft3_4b (results/clover_dev2_exact.json
# = 2.22 tok/step, 186 tok/s, 93.3 % on the same GPU).  Log: results/b13_chain.log.  Runs detached (nohup) from the repo root.
cd "$(dirname "$0")/.."; export SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.
PY=.venv-train/bin/python; LOG=results/b13_chain.log; DEV2=bench/prompts_gsm8k_dev2.json; FULL=bench/prompts_gsm8k_full.json
log() { echo "=== $(date +%F_%T) $*" >> $LOG; }
run_eval() {  # $1 model dir, $2 tag, $3 prompts, $4 n, extra args...
  local M=$1 tag=$2 P=$3 n=$4; shift 4
  $PY bench/run_cdlm.py --model $M --prompts $P --n $n --engine fast_graph --verify --tau 1e9 --window 16 "$@" --out results/b13_${tag}.json \
      > results/b13_${tag}.log 2>&1; grep -E '^=== cdlm|Traceback|Error' results/b13_${tag}.log | cut -c1-700 | sed "s/^=== cdlm/=== cdlm $tag/" >> $LOG
}
for run in b13_auxalign_4b b13_aux_4b; do
  if [ ! -f runs/$run/final/model.safetensors ]; then
    log "train $run"
    $PY -m distill.train_cdlm distill/configs/$run.yaml > results/train_$run.log 2>&1; log "train $run rc=$?"
    grep -E '"step": (1|500|1000|1500|2000|2097),' results/train_$run.log | cut -c1-300 >> $LOG
    rm -rf runs/$run/checkpoint-*
  fi
  M=runs/$run/final
  log "eval dev2 $run"
  run_eval $M ${run}_dev2_aux0 $DEV2 150 --draft-input aux0 --ar
  run_eval $M ${run}_dev2_mask $DEV2 150 --draft-input mask_excl
done
for run in b13_auxalign_4b b13_aux_4b; do
  M=runs/$run/final
  log "eval test $run"
  run_eval $M ${run}_test_aux0 $FULL 1319 --draft-input aux0
done
log "chain done"
