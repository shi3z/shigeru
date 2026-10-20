# SHIGERU — 1M-token long-context extension: Phase 0 analysis and design

Scope: extend the Open Mercury diffusion-LLM engine (`engine/`) to 1,048,576-token contexts *that the model actually uses*,
without regressing the Phase-1 benchmarks and without touching GPUs 0/1 or other services (`docs/gpu_policy.md`).
Everything below is read from the code as of commit `fcb71b1`+ (2026-10-20); numbers marked *estimate* are computed, not measured.

## 1. What the current engine does (engine/llada.py, engine/sampler.py, engine/graphs.py)
| item | where | finding |
|---|---|---|
| Attention | `Block.forward` (llada.py) | `F.scaled_dot_product_attention(q, K, V, attn_mask=key_mask[None,None,None,:], is_causal=False)` — **full bidirectional** attention over every cache position (prompt, decoded blocks, still-masked future blocks). 32 query heads = 32 KV heads (no GQA), head_dim 128. SDPA picks the flash kernel without a mask and the memory-efficient kernel with the (padding) mask; no L×L score matrix is materialized. |
| Positions / RoPE | `rope_cos_sin`, `apply_rope`, `LLaDA.forward` | Absolute positions `pos` (prompt 0..P-1, generation P..L-1) index fp32 cos/sin tables built for `cfg.max_seq` = **4096** rows (`max_sequence_length` in the checkpoint config), rotate-half convention, theta 5e5, full-precision rotation (matches the reference). **Any position ≥ 4096 raises an index error — this is the hard current limit on prompt+generation length.** |
| KV cache ownership | `LLaDA.new_cache` (per request, sampler) / `GraphRunner.get_cache` (per 64-token bucket, graphs recaptured on resize) | One (K, V) pair per layer, `[1, 32, L, 128]` bf16, **all L positions mutable**: `Block.forward` scatters the computed span with `index_copy_(2, pos, ·)`. A block-start *full forward* rewrites every position; a *block forward* rewrites only the active block's 32 positions and attends over the whole cache (Fast-dLLM "dual cache" semantics). There is no frozen/committed prefix: the prefix K/V is **recomputed at every block start**. |
| Block diffusion loop | `sampler.generate` | For each 32-token block: full forward over all L tokens → repeat {select tokens by confidence (threshold or fixed-k), block forward (T=32) against the cache} until the block has no [MASK]. Early stop on an all-EOS block (`stop_on_eos`), optional selective remasking. **Cost per block start = one forward over the entire sequence** (a prefill per block) — at long L this dominates everything. |
| GPU memory | `new_cache`, `GraphRunner`, `LLaDA.forward` | Cache: 2 × 32 layers × 32 heads × 128 × 2 B = **512 KiB per token** (estimate: 128K → 64 GiB, 256K → 128 GiB, 1M → 512 GiB). Full-forward logits `[1, L, 126464]` fp32 = **494 KiB per token** (128K → 62 GiB) — the LM head is applied to all positions although only the block is needed. Hidden states 8 KiB/token bf16 are negligible. Graph static buffers hold the full-length logits per bucket. No explicit frees: tensors are released by Python refcount; the graph runner keeps one cache for the current bucket. |
| Trained context | checkpoint configs (`/mnt/ssd/models/hf/*/config.json`) | LLaDA-8B-Instruct **4096** (RoPE θ 5e5, no scaling); LLaDA2.1-mini **32768** (θ 6e5, `sliding_window 4096` field but `use_sliding_window false`); Fast-dLLM v2 7B **32768** (Qwen2.5-7B-Instruct base, θ 1e6, GQA 28/4, block size 32); Qwen3-8B 40960 (θ 1e6, GQA 32/8; YaRN ×4 → 131072 per its card). |
| Attention mask | `key_mask` [L] bool | Only marks padded cache slots of the bucketed cache; otherwise every token sees every token. The sampler never masks future blocks: LLaDA's training lets masked future tokens participate (they are [MASK] embeddings). |
| Current max input | `cfg.max_seq` (4096) + full-length logits + per-request cache | 4096 tokens of prompt+generation for LLaDA-8B; the server (`engine/server.py`) caps generation at `--max-gen 1024` and never validates the prompt length. |

## 2. Theoretical maximum context per model (before any new training)
| model | trained | RoPE | without training (position interpolation/YaRN, quality unverified) | notes |
|---|---|---|---|---|
| LLaDA-8B-Instruct | 4,096 | θ 5e5, full attention, trained with random-length masking | ~8–16K with PI/YaRN at best (*estimate*) | bidirectional MDM: a frozen prefix is the Fast-dLLM approximation, not exact |
| LLaDA2.1-mini (16B-A1B MoE) | 32,768 | θ 6e5 | 64–128K with YaRN (*estimate*) | HF remote code, no cache, Python expert loop — slow; engine port needed |
| Fast-dLLM v2 7B (Qwen2.5 base) | 32,768 | θ 1e6 | 128K with YaRN factor 4 (Qwen2.5 documents this for the base) | **block-causal by construction**: blocks attend causally to earlier blocks and bidirectionally within the block → a frozen prefix KV is *exact*, the natural base for Phase 1–2 |
| Qwen3-8B (AR baseline) | 40,960 | θ 1e6 | 131,072 with YaRN ×4 (documented) | reference for fairness |

Conclusion: the 1M target cannot be met by any available checkpoint without continued training (Phase 5); the engine work
(Phases 1–4) is validated at ≤32K on Fast-dLLM v2 / LLaDA2.1-mini and at ≤4K (plus extrapolation tests) on LLaDA-8B.

## 3. Memory and compute budget at 1M (estimates, A100 80GB ×4 available = 320 GB VRAM, 1.7 TB host RAM)
| item | per token | 128K | 1M |
|---|---|---|---|
| KV, LLaDA-8B bf16 (32 layers, 32 KV heads) | 512 KiB | 64 GiB | 512 GiB |
| KV, Fast-dLLM v2 bf16 (28 layers, 4 KV heads, hd 128) | 56 KiB | 7 GiB | 56 GiB |
| KV, Fast-dLLM v2 FP8 | 28 KiB | 3.5 GiB | 28 GiB |
| per-step attention read, full attention (one 32-token block, v2 bf16) | KV bytes | 7 GiB → ~5 ms at 1.5 TB/s | 56 GiB → ~37 ms/step from HBM; impossible from host RAM (~250 GB/s → 220 ms) |

Consequences that drive the design: (1) GQA models (Fast-dLLM v2 class) are the only ones whose 1M cache fits in 4 GPUs even
in FP8; LLaDA-8B's 512 GiB cache does not fit anywhere → LLaDA-8B is a mechanism test bed, not the 1M vehicle. (2) At 1M,
full attention per step is bandwidth-bound on the KV read; sparse/retrieval attention (Phase 3) is required for decode
speed, and must be evaluated against full attention, not assumed equal. (3) The LM head must run only on block positions.

## 4. Design
### Phase 1 — block-causal KV cache (engine/cache.py, new backend `engine/backends/block_causal.py`)
* **Frozen prefix KV**: K/V of committed tokens (prompt + completed blocks) are written once and never recomputed.
  Prefix prefill runs in chunks (e.g. 8K) so a 128K prompt never needs a 128K logits tensor; positions are explicit.
* **Mutable diffusion block**: only the active block's K/V (and, for LLaDA-style models, optionally the masked future
  blocks') are rewritten per step; `index_copy_` into the same storage, as today.
* **Explicit position ids** everywhere (already tensors); a `Positions` helper owns prompt/generation offsets and any
  RoPE scaling (Phase 4).
* **Prefix cache reuse** across requests sharing a prompt prefix (hash of token ids per 512-token page).
* **Block commit**: after a block is fully unmasked its K/V are marked frozen; for LLaDA (bidirectional) a block-start
  refresh of *only* the still-masked future region remains optional (off by default, measured).
* **Cache validation**: a test compares block-causal results against the current full-recompute path on the Phase-1
  prompts (identical tokens for block-causal models; measured divergence for LLaDA) and reruns the Phase-1 benchmarks.
* Compatibility: Fast-dLLM v2 (block-causal) → exact; LLaDA-8B → approximation (document the GSM8K delta); if the
  delta is unacceptable, plan the block-causal fine-tune (Phase 5).

### Phase 2 — paged long-context memory (engine/paged.py)
Pages of 512 tokens per layer; a GPU page pool with a configurable budget; CPU pinned-memory pool for cold pages; async
prefetch of the pages the attention plan will touch; FP8 (e4m3 with per-page scales) / INT8 KV with an accuracy check
against bf16; OOM-safe allocation (reserve before use, never allocate in the hot loop). Expansion order 128K → 256K →
512K → 1M, each gated by the Phase-6 checks.

### Phase 3 — attention variants (engine/attn/)
A full (reference), B sliding window, C local+global (window + sink/anchor blocks), D block-sparse (per-query-block
selection of the top-k key pages by page-summary similarity), E retrieval-augmented sparse (page index over the cache,
query-dependent page retrieval + local window). Each reports exactness vs A (logit/argmax agreement, NIAH accuracy) and
FLOPs/bytes per step.

### Phase 4 — RoPE
Native / position interpolation / YaRN / LongRoPE as `Positions` strategies, compared by NIAH at 2×–8× the trained length;
a RoPE-only change is never counted as long-context capability.

### Phase 5 — training (only if needed; estimates first)
Continued training of a block-causal model (Fast-dLLM v2 recipe on Qwen3-8B/1.7B or the v2 checkpoint) with gradient
checkpointing and FSDP/sequence parallel on GPUs 4–7 (short runs) or ABCI (≥128K, 1M); memory/time estimates precede any job.

### Phase 7 — API
`max_model_len=1048576`, streaming, cache budgets, overflow handling (reject or truncate with a clear error), progress and
memory telemetry during long prefills.

## 5. Where the current limits live (for the implementation PRs)
* `engine/llada.py: rope_cos_sin(cfg)` — table length `cfg.max_seq`; make it position-strategy driven and lazily extended.
* `engine/llada.py: LLaDA.forward` — applies `lm_head` to all positions; add `logits_positions`.
* `engine/sampler.py: generate` — block-start full forward; replace with prefix-prefill + commit.
* `engine/graphs.py: GraphRunner.get_cache` — one dense cache per bucket; becomes a view over the page pool.
* `engine/server.py` — no prompt-length validation; add `max_model_len` and overflow handling.
