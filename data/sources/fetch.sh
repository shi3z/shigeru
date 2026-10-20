#!/bin/bash
# Prompt sources for Phase A (not committed). GSM8K train split (MIT), from the HF mirror of openai/gsm8k.
cd "$(dirname "$0")" && curl -s -4 -L -m 600 -o gsm8k_train.parquet "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet" && ls -la gsm8k_train.parquet
