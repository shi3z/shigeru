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

Measured on GPU 5 (one A100 80GB PCIe), 256 generated tokens, 10 EN/JA prompts, greedy; mean over prompts. GSM8K-50 accuracy from results/gsm8k50_*.json. Updated automatically as runs complete.


### GSM8K-50 accuracy (greedy)
* gsm8k50_fastdllm_v1_dual_t0.9.json: 80% (50), 101.2 tok/s effective
* gsm8k50_fastdllm_v2_t0.9.json: 86% (50), 148.9 tok/s effective
* gsm8k50_llada8b_s128.json: 78% (50), 43.7 tok/s effective
* gsm8k50_llada8b_s256.json: 84% (50), 22.0 tok/s effective
* gsm8k50_llada8b_s64.json: 68% (50), 84.2 tok/s effective
* gsm8k50_ours_t0.9_b32_cache.json: 86% (50), 157.4 tok/s effective
* gsm8k50_ours_t0.9_b32_graph_eos.json: 82% (50), 145.0 tok/s effective
* gsm8k50_phaseB_pilot_4b.json: 92% (50), 52.6 tok/s effective
* gsm8k50_phaseB_pilot_8b.json: 94% (50), 50.4 tok/s effective
* gsm8k50_qwen3_4b_hf.json: 90% (50), 30.5 tok/s effective
* gsm8k50_qwen3_4b_idle2.json: 90% (50), 53.5 tok/s effective
* gsm8k50_qwen3_4b_idle.json: 90% (50), 48.6 tok/s effective
* gsm8k50_qwen3_8b_hf.json: 84% (50), 51.7 tok/s effective
* gsm8k50_qwen3_8b_idle.json: 84% (50), 50.3 tok/s effective

### HumanEval pass@1
* humaneval_fastdllm_v1_t0.9.json: 39.0% (164 problems)
* humaneval_llada_official.json: 41.5% (164 problems)
* humaneval_ours_t0.9_graph.json: 37.8% (164 problems)
* humaneval_phaseB_pilot_4b.json: 76.2% (164 problems)
* humaneval_phaseB_pilot_8b.json: 83.5% (164 problems)
* humaneval_qwen3_4b.json: 81.7% (164 problems)
* humaneval_qwen3_8b_hf.json: 85.4% (164 problems)
