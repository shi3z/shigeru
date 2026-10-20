#!/bin/bash
# After the Phase A small pilot finishes: curate it and start the medium Phase A generation (teacher only, no GPU).
# Training launches are NOT chained here: the user directed experiments to GPUs 4,6,7 (fleet pause needed, see
# docs/distill_design.md §3), so they start via distill/run_467.sh after the pause is confirmed.
cd "$(dirname "$0")/.."; LOG=results/distill_chain.log; PY=.venv/bin/python
while pgrep -f "distill.gen_teacher distill/configs/phaseA_small.yaml" > /dev/null; do sleep 30; done
echo "=== $(date +%T) pilot generator exited; $(grep -c . data/distill/phaseA_small.jsonl) records" >> $LOG
PYTHONPATH=. $PY -m distill.curate data/distill/phaseA_small.jsonl --out data/distill/phaseA_small >> $LOG 2>&1
echo "=== $(date +%T) curated: $(grep -c . data/distill/phaseA_small_train.jsonl) train / $(grep -c . data/distill/phaseA_small_val.jsonl) val / $(grep -c . data/distill/phaseA_small_test.jsonl) test" >> $LOG
(setsid bash -c "PYTHONPATH=. $PY -m distill.gen_teacher distill/configs/phaseA_medium.yaml > results/phaseA_medium.log 2>&1" < /dev/null > /dev/null 2>&1 &)
echo "=== $(date +%T) phaseA_medium generation launched (results/phaseA_medium.log)" >> $LOG
