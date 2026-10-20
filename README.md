# Open Mercury — an open-source diffusion LLM and fast inference engine

Goal: reproduce the *techniques* behind Inception Labs' Mercury (fast diffusion-based language models) from public papers and
open-source code, and reach ~1000 tokens/s single-request on A100s without losing autoregressive-level quality.
No Mercury code, weights or private architecture details are used or assumed.

Phases: (1) baseline with existing diffusion LLMs (LLaDA-8B-Instruct, LLaDA2.1-mini) and a fair AR comparison (Qwen3);
(2) our own inference engine (block diffusion, confidence-aware parallel decoding, KV/dual cache, dynamic steps,
selective remasking, CUDA graphs, Triton kernels, batching); (3) our own model (Qwen3 -> diffusion conversion + training);
(4) OpenAI-compatible API server; (5) full performance and quality comparison.

Status log, measurements (JSON) and reproduction steps live in `docs/` and `results/`. Everything measured is marked as
measured; everything else is marked as an estimate.

## Environment
See `docs/env.json` (8x A100 80GB PCIe, driver 580 / CUDA 13 driver, toolkits 11.8-12.9, Python 3.12 venv with PyTorch 2.6 cu124, Triton 3.2).
On this machine GPUs 0-3 and 4/6/7 are taken by other services; experiments start on the shared GPU 5.

## Setup
    uv venv --python 3.12 .venv && source .venv/bin/activate
    uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    uv pip install "transformers==4.57.*" einops  # 5.x breaks LLaDA remote code; einops for Fast-dLLM v2 accelerate safetensors sentencepiece huggingface_hub numpy tqdm fastapi uvicorn
