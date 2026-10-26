#!/bin/bash
# B11-D/E: sampler-aligned masked CE arms (1000 x 8, lr 1e-4, r64) once the state collections are in, then global + monotonic evaluation
cd /mnt/ssdraid/project/open-mercury
L=results/pure/b11_chain.log; M=runs/zeta_sft_final_e5/final; A=runs/bdiff_e5_lr1e4/adapter; R=results/pure/dev_refs_e5.json; E="PYTHONPATH=. .venv-train/bin/python"
# wait for zeta's remask collection and fetch it
until ssh -o BatchMode=yes -o ConnectTimeout=20 zeta-gpu 'grep -q "gstates remask rc=0" ~/shigeru/open-mercury/results/pure/b11_zeta_chain.log' 2>/dev/null; do sleep 60; done
scp -q zeta-gpu:~/shigeru/open-mercury/data/distill/gstates_remask_\* data/distill/ 2>/dev/null
echo "=== $(date +%T) [local] zeta remask states fetched: $(ls data/distill | grep -c gstates_remask_)" >> $L
until grep -q "gstates remask_b rc=" $L && grep -q "gstates left rc=" $L; do sleep 30; done
evalg() {  # $1 gpu, $2 adapter2, $3 tag
  CUDA_VISIBLE_DEVICES=$1 env $E bench/eval_global.py --model $M --adapter $A --adapter2 $2 --refs $R --n 50 --modes monotonic,remask --steps 4,8,16 --out results/pure/b11_eval_$3_dev50.json > results/pure/b11_eval_$3_dev50.log 2>&1
  echo "=== $(date +%T) [gpu$1] b11 eval $3 dev50 rc=$? $(grep '^=== b9_gen' results/pure/b11_eval_$3_dev50.log | sed 's/.*=== b9_gen //' | cut -c1-50 | tr '\n' ' ')" >> $L
  CUDA_VISIBLE_DEVICES=$1 env $E bench/eval_global.py --model $M --adapter $A --adapter2 $2 --prompts bench/prompts_gsm8k_dev2.json --n 150 --modes remask,monotonic --steps 8 --skip-tf --out results/pure/b11_eval_$3_dev2.json > results/pure/b11_eval_$3_dev2.log 2>&1
  echo "=== $(date +%T) [gpu$1] b11 eval $3 dev2 rc=$? $(grep '^=== b9_gen' results/pure/b11_eval_$3_dev2.log | sed 's/.*=== b9_gen //' | cut -c1-60 | tr '\n' ' ')" >> $L
}
( CUDA_VISIBLE_DEVICES=4 env $E distill/train_states.py --model $M --adapter $A --data data/distill/gstates_remask data/distill/gstates_remask_b --target plain --steps 1000 --out runs/sa_C_remask_plain > results/pure/train_sa_C.log 2>&1
  echo "=== $(date +%T) [gpu4] arm C (remask states, plain targets) train rc=$? $(grep 'held-out' results/pure/train_sa_C.log | tail -n1 | cut -c1-200)" >> $L
  evalg 4 runs/sa_C_remask_plain/adapter C
  CUDA_VISIBLE_DEVICES=4 env $E distill/train_states.py --model $M --adapter $A --data data/distill/gstates_left --target plain --steps 1000 --out runs/sa_B_left_plain > results/pure/train_sa_B.log 2>&1
  echo "=== $(date +%T) [gpu4] arm B (left states, plain targets) train rc=$? $(grep 'held-out' results/pure/train_sa_B.log | tail -n1 | cut -c1-200)" >> $L
  evalg 4 runs/sa_B_left_plain/adapter B ) &
( CUDA_VISIBLE_DEVICES=6 env $E distill/train_states.py --model $M --adapter $A --data data/distill/gstates_remask data/distill/gstates_remask_b --target mix --steps 1000 --out runs/sa_E_remask_corr > results/pure/train_sa_E.log 2>&1
  echo "=== $(date +%T) [gpu6] arm E (remask states, plain+corrected targets) train rc=$? $(grep 'held-out' results/pure/train_sa_E.log | tail -n1 | cut -c1-200)" >> $L
  evalg 6 runs/sa_E_remask_corr/adapter E ) &
( CUDA_VISIBLE_DEVICES=7 env $E distill/train_states.py --model $M --adapter $A --data data/distill/gstates_remask data/distill/gstates_remask_b data/distill/gstates_left --target plain --steps 1000 --out runs/sa_D_mixed_plain > results/pure/train_sa_D.log 2>&1
  echo "=== $(date +%T) [gpu7] arm D (remask + left states, plain targets) train rc=$? $(grep 'held-out' results/pure/train_sa_D.log | tail -n1 | cut -c1-200)" >> $L
  evalg 7 runs/sa_D_mixed_plain/adapter D ) &
wait
