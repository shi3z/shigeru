#!/bin/bash
# follow-ups after the 384-token dev2 trio: (1) plain, thinking off, 1024-token budget (29/150 answers were truncated at
# 384); (2) plain, thinking ON (Clover's intended mode), 2048-token budget.
cd /mnt/ssdraid/project/open-mercury || exit 1
LOG=results/clover_int4.log; GGUF=/mnt/raid6/models/gguf/clover/Diffuspace-Clover-35B-A3B-Preview-Q4_K_M.gguf
BIN=/mnt/ssdraid/git/llama.cpp-clover/build/bin/llama-server; PORT=8081
until grep -q "=== dev2 done" $LOG 2>/dev/null; do sleep 20; done
sleep 10
CUDA_VISIBLE_DEVICES=5 setsid nohup $BIN -m $GGUF -ngl 999 -c 8192 -b 512 -np 1 --host 127.0.0.1 --port $PORT -ctk bf16 -ctv bf16 --jinja --chat-template-file /mnt/raid6/models/gguf/clover/chat_template.jinja > results/clover_int4_server_plain_followup.log 2>&1 < /dev/null &
SPID=$!
for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:$PORT/health | grep -q '"ok"' && break; sleep 5; done
echo "=== dev2 plain_1024 (thinking off, max 1024) $(date +%H:%M:%S)" >> $LOG
.venv/bin/python bench/run_api.py --base-url http://127.0.0.1:$PORT/v1 --model clover --prompts bench/prompts_gsm8k_dev2.json --n 150 --max-tokens 1024 --extra '{"chat_template_kwargs": {"enable_thinking": false}}' --tag plain_1024 --timeout 1200 --out results/clover_int4_dev2_plain_1024.json >> $LOG 2>&1
echo "=== dev2 plain_think (thinking on, max 2048) $(date +%H:%M:%S)" >> $LOG
.venv/bin/python bench/run_api.py --base-url http://127.0.0.1:$PORT/v1 --model clover --prompts bench/prompts_gsm8k_dev2.json --n 150 --max-tokens 2048 --extra '{"chat_template_kwargs": {"enable_thinking": true}}' --tag plain_think --timeout 1800 --out results/clover_int4_dev2_plain_think.json >> $LOG 2>&1
kill $SPID; for i in $(seq 1 30); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
echo "=== followup done $(date +%H:%M:%S)" >> $LOG
