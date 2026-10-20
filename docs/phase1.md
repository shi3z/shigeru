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
(filled in from `results/*.json` as runs complete)
