# Phase 1 — baselines on one A100 (measured values only)

## Method
* GPU: one A100 80GB PCIe (GPU 5 of this box, shared with two idle services holding ~20 GB; `docs/env.json`).
* Prompts: `bench/prompts.json` (5 English + 5 Japanese: explanation, code, reasoning, story, summary).
* Diffusion runner `bench/run_llada.py`: the official LLaDA sampler (semi-autoregressive blocks, low-confidence remasking,
  greedy) instrumented per denoising step. No KV cache (the official loop recomputes the whole sequence each step).
* AR runner `bench/run_ar.py`: Hugging Face `generate`, greedy, bf16, thinking off.
* Metrics (per prompt, then mean/min/max):
  * **TTFT**: AR = time to the first generated token (prefill); diffusion = time until the first token of the answer is
    unmasked (first denoising step of the first block). Both include the prompt forward.
  * **E2E latency**: wall time of the whole generation.
  * **effective tokens**: AR = generated tokens up to EOS; diffusion = answer tokens before the first EOS (the sampler
    always fills `gen_length` positions, the rest are EOS padding). tokens/s is reported on effective tokens
    (and "raw" = gen_length / E2E for the diffusion runner).
  * **forward passes / step time**: diffusion only (steps = forward passes without KV cache).
  * **peak GPU memory**: `torch.cuda.max_memory_allocated` during generation (weights included).
* Every run writes a JSON under `results/` with the full config, GPU state and the generated text.

## Results

Measured on GPU 5 (one A100 80GB PCIe), 256 generated tokens, 10 EN/JA prompts, greedy; mean over prompts. Updated as runs complete.

| result file | engine | model | config | tok/s effective | tok/s raw | decode tok/s | TTFT ms | E2E s | fwd/req | eff tokens | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fastdllm_v1_dual_cache_t0.9.json | fast-dllm-v1/dual_cache | LLaDA-8B-Instruct | mode=dual_cache steps=256 block_length=32 threshold=0.9 gen_length=256 | 47.5 | 69.4 | - | 47 | 4.09 | 133 | 184 | 15.7 |
| fastdllm_v1_dual_cache_tnone.json | fast-dllm-v1/dual_cache | LLaDA-8B-Instruct | mode=dual_cache steps=256 block_length=32 gen_length=256 | 23.2 | 32.2 | - | 46 | 7.95 | 256 | 184 | 15.7 |
| fastdllm_v1_prefix_cache_t0.9.json | fast-dllm-v1/prefix_cache | LLaDA-8B-Instruct | mode=prefix_cache steps=256 block_length=32 threshold=0.9 gen_length=256 | 51.0 | 70.9 | - | 46 | 3.99 | 133 | 192 | 15.9 |
| llada8b_g256_s128_b32.json | llada-official-sampler | LLaDA-8B-Instruct | steps=128 block_length=32 gen_length=256 | 42.3 | 49.0 | - | 41 | 5.22 | - | 222 | 15.2 |
| llada8b_g256_s256_b32.json | llada-official-sampler | LLaDA-8B-Instruct | steps=256 block_length=32 gen_length=256 | 20.3 | 24.6 | - | 40 | 10.41 | - | 212 | 15.2 |
| llada8b_g256_s32_b32.json | llada-official-sampler | LLaDA-8B-Instruct | steps=32 block_length=32 gen_length=256 | 133.1 | 196.4 | - | 41 | 1.30 | - | 174 | 15.2 |
| llada8b_g256_s64_b32.json | llada-official-sampler | LLaDA-8B-Instruct | steps=64 block_length=32 gen_length=256 | 64.9 | 98.2 | - | 41 | 2.61 | - | 169 | 15.2 |
| qwen3_8b_hf_256.json | hf-transformers-generate | Qwen3-8B | max_new_tokens=256 | 51.4 | - | 51.5 | 22 | 4.23 | - | 217 | 15.3 |

Notes:
* LLaDA official sampler: speed scales with 256/steps (1 token per step at 256 steps) because every step is a full forward over prompt+256 positions (40.6 ms); effective tokens drop at fewer steps because answers end earlier (more EOS padding) - quality is measured separately on GSM8K-50 (see results/gsm8k50_*.json).
* Fast-dLLM v1: the dual/prefix KV cache alone (256 forwards) gives 1.3x over the official loop; the confidence threshold (0.9) cuts the forwards to ~133 and gives ~70 raw tok/s.
