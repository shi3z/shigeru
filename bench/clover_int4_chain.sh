#!/bin/bash
# Clover-35B-A3B-Preview Q4_K_M on llama.cpp (PR 31306 branch), GPU 5: GSM8K dev2 with self-speculative decoding
# (draft-self, the PR's example flags) and plain decoding; then GSM8K full with the faster mode.
cd /mnt/ssdraid/project/open-mercury || exit 1
LOG=results/clover_int4.log; mv -f results/clover_int4.log results/clover_int4_attempt3.log 2>/dev/null; GGUF=/mnt/raid6/models/gguf/clover/Diffuspace-Clover-35B-A3B-Preview-Q4_K_M.gguf
BIN=/mnt/ssdraid/git/llama.cpp-clover/build/bin/llama-server; PORT=8081
until [ ! -e "$GGUF.aria2" ] && grep -qx REBUILD_OK /mnt/ssdraid/git/llama.cpp-clover.log 2>/dev/null; do sleep 20; done
echo "=== ready $(date +%H:%M:%S)" >> $LOG
run_server() {  # $1 = mode name, rest = extra server flags
  mode=$1; shift
  CUDA_VISIBLE_DEVICES=5 setsid nohup $BIN -m $GGUF -ngl 999 -c 8192 -b 512 -np 1 --host 127.0.0.1 --port $PORT -ctk bf16 -ctv bf16 --jinja --chat-template-file /mnt/raid6/models/gguf/clover/chat_template.jinja "$@" > results/clover_int4_server_$mode.log 2>&1 < /dev/null &
  SPID=$!
  for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:$PORT/health | grep -q '"ok"' && break; sleep 5; done
  curl -s -m 3 http://127.0.0.1:$PORT/health >> $LOG; echo " (server $mode pid $SPID)" >> $LOG
}
stop_server() { kill $SPID 2>/dev/null; for i in $(seq 1 30); do kill -0 $SPID 2>/dev/null || break; sleep 2; done; }
X='{"chat_template_kwargs": {"enable_thinking": false}}'
for mode in spec_thr spec_exact plain; do
  case $mode in
    spec_thr)   run_server $mode --spec-type draft-self --spec-draft-n-max 15 --spec-accept-ratio 0.9 --spec-accept-delta 0.6 --temp 0 --repeat-penalty 1.0 --typical 1.0 ;;
    spec_exact) run_server $mode --spec-type draft-self --spec-draft-n-max 15 ;;
    plain)      run_server $mode ;;
  esac
  echo "=== dev2 $mode $(date +%H:%M:%S)" >> $LOG
  .venv/bin/python bench/run_api.py --base-url http://127.0.0.1:$PORT/v1 --model clover --prompts bench/prompts_gsm8k_dev2.json --n 150 --max-tokens 384 --extra "$X" --tag $mode --out results/clover_int4_dev2_$mode.json >> $LOG 2>&1
  stop_server
done
echo "=== dev2 done $(date +%H:%M:%S)" >> $LOG
