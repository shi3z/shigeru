# engine/cdlm — causal diffusion LM groundwork (SHIGERU Phase C, steps 1–4 of `docs/distill_design.md` §7)

A self-contained, CPU-testable port of WeDLM's dual-stream training mechanism (reference: `third_party/WeDLM/finetune/src/`,
analysed in `docs/wedlm_analysis.md` §2–4). Nothing here is copied from the reference; it is re-implemented against the
analysis and validated against the reference's mask rule in the tests.

## Implemented

| file | what |
|---|---|
| `data.py` | `build_example(tokens, prompt_len, mask_id, block=32, gamma_sampler, rng, ar_on_prompt)` → physical rows `[x0 clean \| xt]`: per 32-token block γ~U(0,1], `round(n_response·γ)` response tokens masked (prompt never), observed-first reordering, `position_ids` = original positions shared with x0, `mlm_labels` (true id on masked rows, −100 elsewhere), `mlm_weights` (1/γ on masked rows), `ar_labels` (next token on x0 rows, −100 when the target is a prompt token, −100 on all xt rows). `collate(examples, pad_id)` right-pads to `[N, T]` and attaches the dense `[N,1,T,T]` mask. Mask-token choice for Qwen3 is documented in the module docstring (free rows 151669…151935; `<|mask|>` = 151669). |
| `mask.py` | `allowed(q, k, L, B)` — the one rule (x0 causal; xt block k → x0 blocks < k and same-block xt rows at smaller physical index; nothing else; padded rows self-only). `dense_mask` / `batch_mask` for sequences ≤ ~4K rows, `make_mask_mod(seq_lens, B)` with FlexAttention's `(b, h, q_idx, kv_idx)` signature built on the same function, `to_additive` for eager. The leakage argument is in the module docstring. |
| `loss.py` | `mlm_loss(logits, labels, weights)` = Σ wᵢ·CEᵢ / Σ wᵢ over masked rows (WeDLM's self-normalised 1/γ weighting, `loss.py:38-45`; `normalize="tokens"` alternative), `ar_loss` (plain mean CE; the shift is folded into `ar_labels`), `total_loss = (mlm + w·ar)/(1+w)`. |
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

## Not yet implemented

* **Decoder** (`docs/distill_design.md` §7 step 5): the streaming window over our engine's prefix KV cache, entropy +
  λ·distance selection, commit/refill, and the τ=None exactness gate (step 6) against the AR student on real prompts.
* **Training loop**: data loading/packing from `data/distill/*.jsonl`, `<|im_end|>` tail tokens (`num_learnable_im_end`),
  optimizer/scheduler, bf16/ZeRO-2 launch on GPUs 4–7, logging of AR vs MLM loss. `forward_batch` + `loss.py` are the
  per-step core; everything around them is missing.
* **FlexAttention execution**: `make_mask_mod` is written and checked for semantics only; `create_block_mask` +
  `attn_implementation="flex_attention"` are not wired or benchmarked (dense `[N,1,2L,2L]` masks are fine at ≤ 4K rows).
* Packing several samples into one row (WeDLM `cum_seqlens`); here one example = one padded row.
* Dropping the prompt-only xt blocks from the forward (they are all-observed and unscored; the reference also computes them).
* Tokenizer registration of `<|mask|>` (only the model side is handled by `add_mask_token`).
