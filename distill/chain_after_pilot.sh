#!/bin/bash
# After the Phase A small pilot finishes: curate it, launch the Phase B 4B LoRA pilot (GPU 5, waits for free memory),
# and start the medium Phase A generation (teacher, concurrency 2). Detached; log results/distill_chain.log.
cd "$(dirname "$0")/.."; LOG=results/distill_chain.log; PY=.venv/bin/python
while pgrep -f "distill.gen_teacher distill/configs/phaseA_small.yaml" > /dev/null; do sleep 30; done
echo "=== $(date +%T) pilot generator exited; $(grep -c . data/distill/phaseA_small.jsonl) records" >> $LOG
PYTHONPATH=. $PY -m distill.curate data/distill/phaseA_small.jsonl --out data/distill/phaseA_small >> $LOG 2>&1
echo "=== $(date +%T) curated: $(grep -c . data/distill/phaseA_small_train.jsonl) train / $(grep -c . data/distill/phaseA_small_val.jsonl) val / $(grep -c . data/distill/phaseA_small_test.jsonl) test" >> $LOG
(setsid bash distill/run_phaseB.sh distill/configs/phaseB_pilot_4b.yaml phaseB_pilot_4b < /dev/null > /dev/null 2>&1 &)
echo "=== $(date +%T) phaseB_pilot_4b launched (results/phaseB_runs.log)" >> $LOG
(setsid bash -c "PYTHONPATH=. $PY -m distill.gen_teacher distill/configs/phaseA_medium.yaml > results/phaseA_medium.log 2>&1" < /dev/null > /dev/null 2>&1 &)
echo "=== $(date +%T) phaseA_medium generation launched (results/phaseA_medium.log)" >> $LOG
