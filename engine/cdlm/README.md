# engine/cdlm — causal diffusion LM groundwork (SHIGERU Phase C, steps 1–5 of `docs/distill_design.md` §7)

A self-contained, CPU-testable port of WeDLM's dual-stream training mechanism (reference: `third_party/WeDLM/finetune/src/`,
analysed in `docs/wedlm_analysis.md` §2–4). Nothing here is copied from the reference; it is re-implemented against the
analysis and validated against the reference's mask rule in the tests.

## Implemented

| file | what |
|---|---|
| `data.py` | `build_example(tokens, prompt_len, mask_id, block=32, gamma_sampler, rng, ar_on_prompt)` → physical rows `[x0 clean \| xt]`: per 32-token block γ~U(0,1], `round(n_response·γ)` response tokens masked (prompt never), observed-first reordering, `position_ids` = original positions shared with x0, `mlm_labels` (true id on masked rows, −100 elsewhere), `mlm_weights` (1/γ on masked rows), `ar_labels` (next token on x0 rows, −100 when the target is a prompt token, −100 on all xt rows). `collate(examples, pad_id)` right-pads to `[N, T]` and attaches the dense `[N,1,T,T]` mask. Mask-token choice for Qwen3 is documented in the module docstring (free rows 151669…151935; `<|mask|>` = 151669). |
| `mask.py` | `allowed(q, k, L, B)` — the one rule (x0 causal; xt block k → x0 blocks < k and same-block xt rows at smaller physical index; nothing else; padded rows self-only). `dense_mask` / `batch_mask` for sequences ≤ ~4K rows, `make_mask_mod(seq_lens, B)` with FlexAttention's `(b, h, q_idx, kv_idx)` signature built on the same function, `to_additive` for eager. The leakage argument is in the module docstring. |
| `loss.py` | `mlm_loss(logits, labels, weights)` = Σ wᵢ·CEᵢ / Σ wᵢ over masked rows (WeDLM's self-normalised 1/γ weighting, `loss.py:38-45`; `normalize="tokens"` alternative), `ar_loss` (plain mean CE; the shift is folded into `ar_labels`), `total_loss = (mlm + w·ar)/(1+w)`. |
| `decoder.py` | `generate(model, tok, prompt_ids, max_new_tokens, window=16, tau=None, lam=0.02, mask_id, stop_ids, greedy=True, temperature, predict="mask"\|"ar")` → `(new_ids, stats)`: WeDLM's streaming window (analysis §5–6) on a HF `DynamicCache`. Prefill into the cache; per step forward the window rows observed-first with their original position ids and a causal mask over `[cache \| window]` (`window_attention_mask`, dict passthrough as in `model_glue`); commit the leftmost filled run by **cropping** the cache to `[prefix \| run]` (`DynamicCache.crop(-(T-m))`, no K/V copy or recomputation — the run sits at physical rows 0..m-1, so its K/V were computed at the right positions attending exactly the prefix + earlier run tokens; argument in the module docstring); fill masks with `entropy + λ·(slot − first_mask_slot) < τ` (argmin fallback, `select_positions`), `τ=None` = exactly the leftmost mask; refill; stop on a stop id in the committed run (kept in the output, later tokens dropped) or `max_new_tokens` (window and refill are budget-capped). `stats`: steps, forwards (+1 prefill), window_rows, tokens_per_step, `filled_per_step`/`committed_per_step` histograms, finish_reason, and the cache (reusable for a continuation). `predict="ar"` lets the leftmost mask take the next-token logits of the preceding filled row (exact AR at τ=None, the free speculative proposal of analysis §11 step 7). Single sequence only. |
| `model_glue.py` | `forward_rows(model, input_ids, position_ids, mask, rows)` runs a HF `Qwen3ForCausalLM` (transformers 5.20) on the dual stream and returns logits only for the gathered `(batch, row)` indices; `forward_batch(model, batch)` does one forward for the MLM and AR rows of a `collate()` batch. The 4-D mask is passed as `attention_mask={"full_attention": m, "sliding_attention": m}`, which `Qwen3Model.forward` hands to the layers untouched (bool for `sdpa`, additive float for `eager`; a plain 4-D tensor also passes through, `flash_attention_2` cannot take it). `add_mask_token(model, mask_id)` resizes embedding + lm_head when needed and initialises the row from the mean embedding, returning the ids it added. |

## Tested (`engine/cdlm/tests/test_cdlm.py`, plain asserts, CPU, tiny random Qwen3Config model in float64, no downloads)

```
cd /mnt/ssdraid/project/open-mercury
CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -m engine.cdlm.tests.test_cdlm
```
(~7 s; the script itself defaults `CUDA_VISIBLE_DEVICES` to empty, so it never touches a GPU.)

1. `test_mask_dense_vs_mask_mod` — dense mask == `mask_mod` materialised by Python loops (with padded rows), and both equal
   the reference rule from `masking.py:86-90` written out independently; pad rows attend only themselves.
2. `test_no_leakage` (a) — sdpa and eager: perturbing any physical token leaves every row that must not see it bit-for-bit
   unchanged (< 1e-9 in float64 through 2 layers) and changes every row that may see it.
3. `test_single_block_masks_visibility` (b) — γ > 0 only in block k: each MLM row's visible set is exactly x0 positions
   < k·B ∪ observed xt rows of its block ∪ earlier masks of its block; its own position is visible in neither stream; the
   information test by perturbation agrees row by row.
4. `test_collate_padding` (c) — padded batch equals per-example forwards on the real rows; pad rows carry pad id, position 0,
   −100 labels, 0 weights; per-example invariants (xt is a permutation of x0, masked labels restore x0, prompt never masked,
   observed-first & ascending inside blocks, weights = 1/γ, AR labels shifted and prompt-gated, `ar_on_prompt`).
5. `test_losses` (d) — shapes, exact WeDLM normalisation, weights actually change the loss, batch-shaped inputs, all-ignored
   gives a differentiable zero, `ar_loss` == `F.cross_entropy(ignore_index=-100)`, `total_loss`.
6. `test_tau_none_equivalence` (e) — with zero masks, the x0 rows of the dual-stream forward reproduce the plain HF causal
   forward to 1e-9 (sdpa and eager) and so do the all-observed xt rows (sdpa 1e-9; eager 1e-4 because HF eager runs
   softmax in float32 and the xt rows sum the same keys in a different order).

Last run: `6 passed, 0 failed`.

## Decoder tests (`engine/cdlm/tests/test_decoder.py`, same style; float64 tiny model unless stated, the `[MASK]` row is
re-randomised after `add_mask_token` so nothing relies on the mean init)

```
cd /mnt/ssdraid/project/open-mercury
CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -m engine.cdlm.tests.test_decoder
```
(~14 s, CPU.)

1. `test_tau_none_equals_hf_greedy` (1) — `predict="ar"`, `τ=None`: output `==` HF `generate(do_sample=False)` token for
   token on 6 random prompts × {no stop id, random stop id} × W ∈ {1, 3, 16} × {sdpa, eager} (72 cases); one token per
   step (`filled_per_step == {1: n}`, steps = n + 1).
2. `test_tau_none_mask_predictor_matches_naive` (1b) — `predict="mask"` (faithful WeDLM), `τ=None`: output `==` an uncached
   loop that forwards `[prompt | out | MASK]` from scratch and takes the MASK row's argmax (4 prompts × 3 windows × 2
   impls); and the two predictors do differ on an untrained model.
3. `test_cache_matches_full_forward` (2) — after `generate()` with (τ, predict, W) ∈ {(None, ar, 16), (None, mask, 4),
   (0.5, mask, 16), (2.0, mask, 8), (1e9, mask, 16), (1.0, ar, 16)}, every layer's cached K and V for `[prompt | output]`
   equal a fresh full causal forward of that sequence: float32 < 1e-5 (sdpa and eager), float64 sdpa < 1e-9, float64
   eager < 1e-4 (HF eager softmaxes in float32). With τ=0.5 on random weights the argmin fallback fills non-leftmost
   masks first, so the gap path (filled-after-a-gap rows re-encoded, zero-commit steps) is covered.
4. `test_tau_large_terminates_and_respects_budget` (3) — `τ=1e9`: every mask filled each step, `steps == ceil(n/W) + 1`,
   output length exactly `max_new_tokens` for (W, n) ∈ {(16, 37), (16, 16), (4, 50), (1, 7), (8, 3)}; with a stop id
   the output ends at it and the cache length matches; sampling path (`greedy=False`) and `max_new_tokens=0`.

Last run (verbatim):
```
PASS test_tau_none_equals_hf_greedy
PASS test_tau_none_mask_predictor_matches_naive
  max |cache - full forward|: float32/sdpa 3.81e-06, float32/eager 2.86e-06, float64/sdpa 1.78e-15, float64/eager 7.53e-07
PASS test_cache_matches_full_forward
PASS test_tau_large_terminates_and_respects_budget
4 passed, 0 failed
```
Illustration on the random model (W=16, 64 tokens, λ=0.02): τ=None → 65 steps, 968 window rows; τ=1e9 → 5 steps
(`filled_per_step {16: 4}`, 12.8 tokens/step). Speed is meaningless here (random weights, CPU); the honest metric on a
trained model is `generated / steps` and wall-clock against the HF AR baseline.

**Caveats of this decoder.** (a) The "τ=None exactness gate" of the design doc (§7 step 6) is a *trained-model*
property: a WeDLM mask row predicts its own position, an AR row predicts the next one, so on untrained weights they differ
— test 1 checks the mechanical exactness of the cache/position/mask plumbing with `predict="ar"`, test 2 checks the
faithful predictor against an uncached reference; the gate itself must be run on the converted checkpoint with
`predict="mask"`. (b) `τ=None` fills the leftmost mask, as specified; the reference takes `argmin(entropy + λ·d)`, which
is the leftmost only when λ ≥ ln V. (c) HF `DynamicCache` + a dense `[1,1,T,P+T]` mask per step: correct, not fast (no
paged cache, no CUDA graphs, one sequence at a time); the port to our engine's paged prefix cache is the next step.

## Own-engine decoder (`fast_model.py`, `fast_decoder.py` — Phase E groundwork)

`fast_model.FastQwen3` is our own Qwen3 (RMSNorm, per-head q/k RMSNorm before RoPE, GQA, rotate-half RoPE with
explicit position ids, SwiGLU, tied or untied `lm_head`) loaded from an HF directory (`config.json` + safetensors;
`from_hf` takes an in-memory `Qwen3ForCausalLM`), in the style of `engine/llada.py` / `engine/qwen_block.py`, with
transformers' numerics (float32 RMSNorm, float32 cos/sin cast to the activation dtype, `repeat_kv` GQA, SDPA with an
explicit boolean mask).  It owns one preallocated K/V cache `[layers, 1, Hkv, max_len, hd]` and the caller owns an
integer `cache_len`:

* `forward(tokens[T], pos_ids[T], cache_len, write=True, rows=None, key_mask=None)` rotates the window rows with their
  own positions, writes their K/V at slots `cache_len..cache_len+T-1` and attends over `[cache[:cache_len] | window]`
  with the causal window mask (row i sees the prefix and rows ≤ i).  Logits (float32) only for `rows`.  Committing the
  leftmost filled run is `cache_len += m` — the DynamicCache.crop of `decoder.py` without any copy; the rows after the
  run sit beyond `cache_len` and are overwritten by the next step.  `write=False` attends over the same context without
  touching the cache.
* `prefill(prompt_ids, cache_len=0, chunk=512)` fills the cache in chunks and returns the last token's logits.
* `forward_static(tokens[Wb], pos[Wb], slots[Wb], key_mask[Wb, max_len], rows[Wb])` is the CUDA-graph shape of the same
  step: the window padded to a bucket Wb ∈ {1, 4, 8, 16, 32}, the whole cache as context, key validity in the boolean
  mask, write slots and logit rows as index tensors, a static `[Wb, V]` output — no Python-side shape depends on the
  data.  `run(..., static=True)` pads/fills the static buffers and runs it eagerly on any device; `use_graph=True`
  captures and replays one graph per bucket (engine/graphs.py recipe) **only when the model is on CUDA**; CPU never
  captures.
* `fast_decoder.generate(model, tok, prompt_ids, max_new_tokens, window, tau, lam, mask_id, stop_ids, greedy,
  temperature, predict, generator, static, use_graph)` is `decoder.generate` on this model: same loop, same arguments,
  same `(new_ids, stats)` (`cache_len`/`prompt_len` replace the DynamicCache in `stats`); it gathers logits only for
  the mask rows plus, for `predict="ar"`, the last committed row.

### Tested (`engine/cdlm/tests/test_fast.py`, plain asserts, CPU, float32, tiny random Qwen3 saved to a temp dir)

```
cd /mnt/ssdraid/project/open-mercury
CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -m engine.cdlm.tests.test_fast
```
(~33 s, CPU.)  The HF side gets its `[MASK]` row from `add_mask_token` (re-randomised), is saved with
`save_pretrained`, and the fast model loads that directory (so the mask row and `cdlm_mask_token_id` come from the
checkpoint); tied and untied heads both.

1. `test_logits_exact_vs_hf` — vs `Qwen3ForCausalLM` (sdpa): plain causal forward, chunked prefill (chunk 1/3/512) +
   continuation with gathered rows, the observed-first window forward with original positions over the cache (vs HF
   with the decoder's dict mask on a DynamicCache, logits and the written K/V), `write=False`, and the bucketed static
   path; max abs logit diff 1.8e-5 (< 1e-4).
2. `test_fast_decoder_matches_hf_decoder` — `fast_decoder.generate == decoder.generate` token for token and in stats
   on 4 prompts × {no stop, random stop} × W ∈ {1, 4, 16} × τ ∈ {None, 0.5, 1e9} × predict ∈ {mask, ar} × {untied,
   tied} = 288 cases; the static path gives the same tokens; default `mask_id`; τ=1e9 fills W per step; sampling;
   `max_new_tokens=0`.
3. `test_cache_len_bookkeeping` — after `generate`, slots `[0, cache_len)` equal a fresh HF causal forward of
   `[prompt | output]` (7.8e-6 < 1e-5), garbage written beyond `cache_len` changes nothing (continuation of one token,
   exact and static paths), and over-filling the cache is an assertion, not silent.

Last run (verbatim):
```
  max |fast - HF| logits: 1.81e-05
PASS test_logits_exact_vs_hf
  288 decoder cases identical
PASS test_fast_decoder_matches_hf_decoder
  max |cache - full forward| / continuation: 7.75e-06
PASS test_cache_len_bookkeeping
3 passed, 0 failed
```

### Not done (needs a GPU)

* **Graph capture validation**: `_capture` / `run(use_graph=True)` are written from the engine/graphs.py recipe but
  have never executed (CPU only here); the first GPU step is `fast_decoder.generate(..., static=True)` vs
  `use_graph=True` token equality on the real checkpoint, then capture cost per bucket.
* **Timing**: tokens/step and tok/s of `fast_decoder` vs `decoder.py` (HF DynamicCache) vs the HF AR baseline on the
  converted checkpoint, and the share of the step that is the `[Wb, max_len]` whole-cache attention (a context bucket
  instead of `max_len` is the obvious follow-up).
* **bf16**: the model loads in any dtype, but exactness is only established in float32; the bf16 drift vs HF bf16 and
  the τ=None exactness gate on the converted checkpoint are not measured.
* Single sequence, no batching, no paged cache; `rope_type` must be `default` (no YaRN), no sliding-window layers.

## Training loop (`distill/train_cdlm.py`, Phase C conversion trainer)

Implemented on top of the modules above: JSONL → chat template (thinking disabled) + response + eos → `build_example` per
sample and per epoch (prompt never masked, per-block γ ~ U(gamma.min, gamma.max]) → `collate` → `forward_rows` on the
gathered MLM + AR rows → `mlm_loss` (1/γ self-normalised) and `ar_loss`, combined as `total_loss` (WeDLM weighting).
Full fine-tune with a plain torch loop: AdamW (fused on CUDA), cosine schedule with warmup, grad accumulation, grad
clipping, bf16 weights (optional fp32 master + bf16 autocast), gradient checkpointing, `<|mask|>` registered in the
tokenizer and `add_mask_token` on the model (`config.cdlm_mask_token_id`, saved with `final/`), periodic checkpoints with
automatic resume, `log.jsonl`, `run.json`, single-process or `torchrun` DDP, CUDA-OOM skip-and-halve. Command line, memory
estimate and the test are in `docs/distill_design.md` §7; `distill/tests/test_train_cdlm.py` (CPU, tiny model, real
subprocesses incl. kill/resume and 2-process gloo DDP) last ran `4 passed, 0 failed`.

## Not yet implemented

* **Decoder in our engine** (`docs/distill_design.md` §7 step 5/6 on real hardware): the own-engine port exists
  (`fast_model.py` / `fast_decoder.py` above, CPU-exact vs HF) but has not run on a GPU — graph capture, timing, bf16,
  batching across sequences, and the τ=None exactness gate on the converted checkpoint against the AR student on the
  eval prompts, tokens/step and tok/s versus the HF AR baseline, are all open.
* **Training on GPUs**: `distill/train_cdlm.py` has only run on CPU (tiny model). The 4B pilot config
  (`distill/configs/phaseC_pilot_4b.yaml`) is not launched; bf16 vs `master_fp32` and the ZeRO/FSDP question for the
  4B full fine-tune are open (DDP replicates the optimizer states, see §7 memory estimate). `<|im_end|>` tail tokens
  (WeDLM `num_learnable_im_end`) are not appended; one example = one padded row (no packing, WeDLM `cum_seqlens`).
* **FlexAttention execution**: `make_mask_mod` is written and checked for semantics only; `create_block_mask` +
  `attn_implementation="flex_attention"` are not wired or benchmarked (dense `[N,1,2L,2L]` masks are fine at ≤ 4K rows).
* Dropping the prompt-only xt blocks from the forward (they are all-observed and unscored; the reference also computes them).
