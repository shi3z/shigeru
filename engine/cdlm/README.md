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
| `decoder.py` | `generate(model, tok, prompt_ids, max_new_tokens, window=16, tau=None, lam=0.02, mask_id, stop_ids, greedy=True, temperature, predict="mask"\|"ar")` → `(new_ids, stats)`: WeDLM's streaming window (analysis §5–6) on a HF `DynamicCache`. Prefill into the cache; per step forward the window rows observed-first with their original position ids and a causal mask over `[cache \| window]` (`window_attention_mask`, dict passthrough as in `model_glue`); commit the leftmost filled run by **cropping** the cache to `[prefix \| run]` (`DynamicCache.crop(-(T-m))`, no K/V copy or recomputation — the run sits at physical rows 0..m-1, so its K/V were computed at the right positions attending exactly the prefix + earlier run tokens; argument in the module docstring); fill masks with `entropy + λ·(slot − first_mask_slot) < τ` (argmin fallback, `select_positions`), `τ=None` = exactly the leftmost mask; refill; stop on a stop id in the committed run (kept in the output, later tokens dropped) or `max_new_tokens` (window and refill are budget-capped). `stats`: steps, forwards (+1 prefill), window_rows, tokens_per_step, `filled_per_step`/`committed_per_step` histograms, `filled_positions` (the absolute logical positions filled in each step), finish_reason, and the cache (reusable for a continuation). `predict="ar"` lets the leftmost mask take the next-token logits of the preceding filled row (exact AR at τ=None, the free speculative proposal of analysis §11 step 7). Optional fill-selection rules `margin_tau`, `no_adjacent`, `max_fill`, `fill_mode="leftmost_run"` (all off by default; section "Fill-selection rules" below). `verify=True`: verified (speculative) decoding, output exactly HF greedy (section "Verified (speculative) decoding"). Single sequence only. |
| `model_glue.py` | `forward_rows(model, input_ids, position_ids, mask, rows)` runs a HF `Qwen3ForCausalLM` (transformers 5.20) on the dual stream and returns logits only for the gathered `(batch, row)` indices; `forward_batch(model, batch)` does one forward for the MLM and AR rows of a `collate()` batch. The 4-D mask is passed as `attention_mask={"full_attention": m, "sliding_attention": m}`, which `Qwen3Model.forward` hands to the layers untouched (bool for `sdpa`, additive float for `eager`; a plain 4-D tensor also passes through, `flash_attention_2` cannot take it). `add_mask_token(model, mask_id)` resizes embedding + lm_head when needed and initialises the row from the mean embedding, returning the ids it added. |

## Fill-selection rules (`decoder.select_positions`, shared by both decoders; all off by default)

**Why.** `docs/distill_design.md` §10.3: after the CPT conversion the one-mask-per-step path (τ=None) reaches 80 %
GSM8K (20 prompts) but parallel filling with the reference rule `entropy + λ·d < τ` collapses to 0–20 % at only
1.1–1.3 tokens/step; a larger λ (0.3) recovers 55 % at 1.07 tokens/step. The failure is one of *conditional
independence*: masks filled in the same step are predicted independently given the context, so a multi-token number
whose tokens all clear the threshold is split ("#### 10" for "#### 100"). The rules below restrict *which* of the
passing masks are filled in a step. They change neither the forward, nor the cache or commit logic, nor the fallback
"≥ 1 fill per step" guarantee; with all of them off the selection is byte-for-byte the reference rule (test (a)).

**Candidate set C** (first stage; `margin_k = p_top1 − p_top2` of `softmax(logits_k)` in float32, no temperature, from
the same logits as the entropy, so `predict="ar"` affects both alike):

| `tau` | `margin_tau` | C | fallback when C is empty |
|---|---|---|---|
| given | None | `{k : entropy_k + λ·d_k < tau}` (reference) | argmin of `entropy + λ·d` (reference) |
| given | given | `{k : entropy_k + λ·d_k < tau and margin_k ≥ margin_tau}` — both must hold | argmin of `entropy + λ·d` |
| None | given | `{k : margin_k ≥ margin_tau}` — the margin replaces the entropy rule | the leftmost mask (index 0) |
| None | None | `{0}`, the leftmost mask (the exact-AR step) | — |

So `margin_tau` is an *additional* gate when `tau` is given and the *only* gate when `tau=None`. The margin is a
different confidence measure from the entropy: a row can have low entropy with two near-tied top tokens (small margin),
which is exactly the "1 vs 10" ambiguity, and vice versa.

**Post-filters** (second stage, applied in this order; each keeps the leftmost elements, so the mandatory fill survives):

* `fill_mode="leftmost_run"` — keep the run `0, 1, 2, …` of mask indices that are all in C and sit at consecutive
  logical slots, anchored at the leftmost mask (slot 0 of the window after the commit = logical position `cur`, whose
  left context is entirely committed). If the leftmost mask is not in C it is filled alone (the τ=None step). Everything
  right of the first failing mask waits. Consequences: the window never holds filled-after-a-gap rows, every filled
  token is committed in the very next step (`committed_per_step == filled_per_step + {0: 1}`), no token is ever
  re-encoded, and the filled run is always conditioned on a fully known left context (the tokens *inside* the run are
  still conditionally independent of each other — combine with `no_adjacent` or `max_fill` to limit that).
  `fill_mode="all"` (default) keeps C as is (the reference behaviour: fills anywhere in the window).
* `no_adjacent=True` — of every maximal run of candidates at consecutive logical slots keep only the leftmost
  (slots `0,1,2,3,5,6,8 → 0,5,8`): two masks adjacent in logical position are never filled in the same step, so the
  second token of a number is always predicted after the first one is known. Degenerate case: τ=1e9 (everything
  passes, masks contiguous) becomes one fill per step; with `leftmost_run` it is always exactly one fill.
* `max_fill=n` — keep the `n` leftmost remaining candidates. `max_fill=1` with a threshold is one token per step but
  not the τ=None step (it fills the best-scoring mask, wherever it is).

`stats["filled_positions"]` (both decoders) lists, per fill step, the absolute logical positions that were filled —
the instrument the tests use for the adjacency / contiguity invariants. `bench/run_cdlm.py` exposes the rules as
`--no-adjacent`, `--max-fill`, `--margin-tau`, `--fill-mode` (recorded in the saved JSON under `args`); they are not
yet measured on the converted checkpoint — the design-doc table is the baseline to beat (`τ=0.4, λ=0.3`: 55 % at 1.07
tokens/step; the target is ≥ 80 % at > 1.0 tokens/step).

## Verified (speculative) decoding (`verify=True`, both decoders; loop `decoder.generate_verified`)

**Why.** `docs/distill_design.md` §10.3, "Fill-selection rules on the CPT checkpoint": on the converted checkpoint the
frontier-only path (τ=None) reaches 80 % but *any* token filled ahead of the frontier is unreliable and is never
revised — accuracy collapses to 25–35 % even at 1.01 tokens/step (`no_adjacent`), so the fill-selection rules above
cannot fix it. With `verify=True` every ahead-of-frontier fill is a **draft** that the AR logits of the same forward
verify, as in speculative decoding with the mask head as the drafter: the output is exactly what the AR rows produce
greedily (HF `generate(do_sample=False)`), whatever the drafts are, and tokens/step exceeds 1 exactly when the drafts
are right. The quality of the AR student is kept by construction; the mask head's accuracy is turned into speed.

**Semantics** (`decoder.py` module docstring, "Verified (speculative) decoding"; `fast_decoder.generate(verify=True)`
runs the very same function with its own forward/commit callbacks, so the two decoders cannot drift apart):

* State: committed output `out` (`cur = prompt_len + len(out)`), a window of `W` slots (slot `i` ↔ logical position
  `cur + i`, each a draft or a mask), and `pending`: the last committed token while its K/V are not in the cache yet.
* Per step: (1) refill the window to `W` masks (budget-capped); (2) forward `[pending | drafts | masks]`
  observed-first with original positions; (3) **verify**: walk from the frontier: `g = argmax` of the AR logits of
  position `cur+j` (the pending row for `j=0` — the prefill logits on the first step — else the row of the draft at
  slot `j-1`, which attended exactly the committed prefix plus the drafts before it); the draft at slot `j` is
  accepted iff it equals `g`; at the first mask or mismatch the walk stops and `g` is the correction. Accepted drafts
  are committed (cache crop / `cache_len`, their K/V are final: they are the leading physical rows); `g` is committed
  as the new `pending` (in `out` now, its K/V written by the next forward, which carries it as its first observed
  row — this extra row is what exactness costs: `W+1` rows per forward instead of `W`, and `W ≤ 31` on the static
  bucket path); a stop id or the budget inside the accepted run ends the sequence without a correction; a
  correction that is a stop id or exhausts the budget triggers one extra one-row forward so that the cache holds
  `[prompt | output]` as in the plain mode (`stats["completion_forward"]`, counted in `forwards`). (4) Window
  = slots `j+1..`; on a mismatch every draft to the right is re-masked (it was conditioned on the wrong token), on a
  mask-stop the drafts to the right stay. (5) **Draft**: the rows forwarded as masks and still in the window are
  filled with the usual rule — `τ`/`λ`, `margin_tau`, `no_adjacent`, `max_fill`, `fill_mode` all apply — but without
  the mandatory ≥ 1 fill (`select_positions(..., min_fill=0)`: the fallbacks return `[]`); a step may draft nothing,
  since progress is guaranteed by the correction. `predict="ar"` only affects a leftmost mask preceded by a kept
  draft; the mask right after the correction keeps its own logits (the corrected token's next-token logits exist
  only after the next forward, where they do the verification).
* Every step commits ≥ 1 token, so `tokens_per_step ≥ 1`; a step commits at most `W` (`W-1` accepted drafts + the
  correction) and the average is bounded by `(W+1)/2` (a fully accepted step empties the window, so the next one
  starts from fresh masks and commits only the correction). `W=1` is plain greedy decoding. `greedy=False` only
  randomises the drafts. τ=None drafts one token per step (the frontier mask), so tokens/step ∈ [1, 2].
* Stats added to the usual ones: `verify=True`, `accepted_per_step` ({accepted drafts: steps}), `drafted`,
  `accepted`, `draft_acceptance_rate` (= accepted / drafted, `None` without drafts), `completion_forward`;
  `filled_per_step` may contain 0, `committed_per_step` never does, `forwards = steps + 1 + completion_forward`.

**Guarantee.** By induction `out` is a prefix of the greedy AR output: every committed token is the argmax of logits
computed on the exact committed prefix (the pending row, or an accepted draft's row whose context is the prefix plus
drafts already found equal to the greedy tokens), and a draft is only committed after it was found equal to that
argmax. Tested exactly (below): every τ, W, predict, with/without stop ids, every fill rule, both decoders.

**Tests** (`test_decoder.py` 9–10, `test_fast.py` 5–6, same fixtures):

9. `test_verify_equals_hf_greedy` — HF decoder, `verify=True`: output `==` HF `generate(do_sample=False)` for
   5 prompts (sdpa) + 2 (eager) × {no stop, random stop} × τ ∈ {None, 0.5, 1e9} × W ∈ {1, 4, 16} × predict ∈ {mask,
   ar} and 5 fill-rule sets × W ∈ {4, 16} × τ (588 cases); stats consistency (`check_verify_stats`: forwards, every
   step commits 1..W, histograms vs totals, tokens/step ≥ 1); the final cache `==` a fresh causal forward of
   `[prompt | output]` (float64 sdpa, 2.7e-15, incl. the completion forward); W=1 is one token per step with
   `2n-1` window rows; drafts are both accepted and rejected on the random model (1970 / 4870) and steps committing
   ≥ 2 exist; sampling drafts keeps the greedy output; budget 0 / budget < W; `select_positions(min_fill=0)` is
   the documented rule minus its fallbacks on 8 800 random selections (2 038 empty).
10. `test_verify_good_drafter_speeds_up` — a deliberately good mask head (`OracleMaskHead`: a mask row gets a one-hot
   of the greedy token at its position, the AR rows are the real model, the mask id is suppressed as an output):
   every draft accepted, W=16/τ=1e9 reaches 6.7 tokens/step over 40 tokens (the (W+1)/2 ceiling minus the budget
   tail; W tokens in a fully accepted step), τ=None ≈ 2; a drafter wrong at every third position is still exact
   (1.5 tokens/step) while the unverified mode with it is not; a stop id inside an accepted run is honoured. The
   "tie the mask row to the AR distribution" construction was not practical on a random transformer (a mask row
   never sees the token before it, so a 2-ahead prediction cannot be built by weight tying) — the oracle head is the
   substitute, it exercises exactly the decoder path that a good trained mask head would.
5. `test_fast_decoder_matches_hf_decoder_verify` — `fast_decoder.generate(verify=True) == decoder.generate(verify=True)`
   in tokens and every stat (incl. `filled_positions` and the verify stats) and both `==` HF greedy, on 3 seeds
   (tied/untied) × 4 prompts × {no stop, random stop} × W ∈ {1, 4, 16} × τ ∈ {None, 0.5, 1e9} × predict + 4 fill
   rules × W × τ = 816 cases, 168 of them also on the static path; the fast cache `[0, cache_len)` `==` a fresh
   causal forward (3.8e-6).
6. `test_fast_verify_good_drafter` — the oracle drafter on the fast model (`OracleFast.run`), exact and static paths:
   exact output, perfect drafter 6.7 tokens/step at W=16/τ=1e9, every-third-wrong 1.5.

(Both test files now pass an explicit all-ones `attention_mask` to HF `generate`: without it, a prompt containing
token 0 — a real id here — is treated as padding by `generate` with `pad_token_id=0`, which is not the greedy
continuation; the float64 suite's seeds happened not to contain it.)

Last run (verbatim, `test_decoder.py` ~55 s, `test_fast.py` ~150 s, CPU):
```
$ CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -m engine.cdlm.tests.test_decoder
PASS test_tau_none_equals_hf_greedy
PASS test_tau_none_mask_predictor_matches_naive
  max |cache - full forward|: float32/sdpa 3.81e-06, float32/eager 2.86e-06, float64/sdpa 1.78e-15, float64/eager 7.53e-07
PASS test_cache_matches_full_forward
PASS test_tau_large_terminates_and_respects_budget
  39600 selection cases == documented semantics, 48 generate() cases == frozen reference rule
PASS test_fill_options_default_to_reference
  27 steps filled >= 2 masks, none adjacent (without the rule: 346 adjacent pairs)
PASS test_no_adjacent_never_fills_adjacent
PASS test_max_fill_respected
PASS test_margin_tau_and_leftmost_run
  588 verify cases == HF greedy; drafts accepted 1970/4870; max |cache - full forward| 2.66e-15; 8800 min_fill=0 selections checked (2038 empty)
PASS test_verify_equals_hf_greedy
  108 good-drafter cases exact; W=16, tau=1e9, 40 tokens: 6.67 tokens/step with the perfect drafter, 1.48 with the every-third-wrong drafter
PASS test_verify_good_drafter_speeds_up
10 passed, 0 failed

$ CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -m engine.cdlm.tests.test_fast
max |fast - HF| logits: 1.81e-05
PASS test_logits_exact_vs_hf


  288 decoder cases identical
PASS test_fast_decoder_matches_hf_decoder

  384 decoder cases with fill options identical (189 differ from the reference rule)
PASS test_fast_decoder_matches_hf_decoder_fill_options

  max |cache - full forward| / continuation: 7.75e-06
PASS test_cache_len_bookkeeping



  816 verify cases identical in both decoders and == HF greedy (168 static); drafts accepted 2528/18062; max |cache - full forward| 3.81e-06
PASS test_fast_decoder_matches_hf_decoder_verify

  48 good-drafter cases exact on the fast model; W=16, tau=1e9, 40 tokens: 6.67 tokens/step with the perfect drafter, 1.54 with the every-third-wrong drafter
PASS test_fast_verify_good_drafter
6 passed, 0 failed
```

**Bench.** `bench/run_cdlm.py --verify` (recorded in the JSON `args`; the per-run records carry `drafted`,
`accepted`, `draft_acceptance_rate`, and the summary adds `draft_acceptance_rate` and `accepted_per_step_mean` for
the configuration). **GPU measurement pending**: accuracy must equal the AR rows (90 % on the §10.3 prompts) by
construction; the open numbers are tokens/step and tok/s on the CPT and stage-2 checkpoints for τ ∈ {None, 0.4,
1.0} × the fill rules (the draft acceptance rate of the mask head is the quantity to maximise), and the cost of the
`W+1`-row forward against the HF AR baseline.

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
(~55 s, CPU.)

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
5. `test_fill_options_default_to_reference` (a) — `select_positions` with every option off `==` a frozen copy of the
   pre-option function kept in the test, on 300 random (entropy, margin, gapped mask slots) × τ ∈ {None, 0.5, 2, 1e9}
   × λ ∈ {0, 0.02, 0.3}; every option combination (11 sets) `==` an independent pure-Python transcription of the
   documented semantics (39 600 cases); `generate()` with default arguments gives identical tokens and stats to the
   same run with the frozen function patched into the module (3 prompts × W ∈ {4, 16} × τ ∈ {None, 0.5, 2.5, 1e9} ×
   predict, 48 cases); invalid `fill_mode` / `max_fill=0` / `margin_tau>1` / `margin_tau` without a margin raise.
6. `test_no_adjacent_never_fills_adjacent` (b) — from `stats["filled_positions"]`: no step fills two adjacent
   positions, for τ ∈ {2.5, 3.2, 1e9}, margin-only and τ+margin, W ∈ {4, 16}, 4 prompts; non-vacuous (steps with ≥ 2
   fills exist, and the same runs without the rule fill hundreds of adjacent pairs); τ=1e9 degenerates to `{1: n}`.
7. `test_max_fill_respected` (c) — `max(filled_per_step) ≤ max_fill` for max_fill ∈ {1, 2, 3, 5} × W ∈ {4, 16} × four
   rules; with τ=1e9 every step fills exactly `min(max_fill, W)` and `steps == ceil(n / min(max_fill, W)) + 1`.
8. `test_margin_tau_and_leftmost_run` — a spy on the module's `select_positions` checks for every step that the
   filled set is exactly the masks with `margin ≥ margin_tau` (ANDed with `entropy + λd < τ` when τ is given), or the
   documented fallback when none passes (both branches hit); the margin equals `margin_from_logits` of a fresh forward;
   `leftmost_run`: every step's positions are exactly `cur, cur+1, …`, `committed_per_step == filled_per_step + {0: 1}`,
   τ=1e9 is `W` per step; a stop token inside a run truncates the output and the cache consistently.

Last run: `10 passed, 0 failed` (verbatim output, with tests 9–10, in the section "Verified (speculative) decoding").
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
  temperature, predict, generator, static, use_graph, prefill_chunk, margin_tau, no_adjacent, max_fill, fill_mode)`
  is `decoder.generate` on this model: same loop, same arguments, same `(new_ids, stats)` (`cache_len`/`prompt_len`
  replace the DynamicCache in `stats`); it gathers logits only for the mask rows plus, for `predict="ar"`, the last
  committed row. The fill-selection rules are literally `decoder.select_positions` (imported), so both decoders
  cannot drift apart there.

### Tested (`engine/cdlm/tests/test_fast.py`, plain asserts, CPU, float32, tiny random Qwen3 saved to a temp dir)

```
cd /mnt/ssdraid/project/open-mercury
CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -m engine.cdlm.tests.test_fast
```
(~150 s, CPU.)  The HF side gets its `[MASK]` row from `add_mask_token` (re-randomised), is saved with
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
3. `test_fast_decoder_matches_hf_decoder_fill_options` (d) — `fast_decoder.generate == decoder.generate` in tokens
   and stats (including `filled_positions`) for 7 option sets (`no_adjacent`, `max_fill` ∈ {1, 3}, `margin_tau=0.3`,
   `leftmost_run`, `no_adjacent+max_fill=2+margin_tau=0.2`, `margin_tau=0.5+leftmost_run`) × W ∈ {4, 16} × τ ∈
   {0.5, 2.5, 1e9} (+ τ=None for the margin-only sets) × predict ∈ {mask, ar} × 4 prompts (two with a random stop id)
   = 384 cases; the static path agrees on the W=16 cases; 189 of the 384 outputs differ from the reference rule (the
   options are exercised, not vacuous).
4. `test_cache_len_bookkeeping` — after `generate`, slots `[0, cache_len)` equal a fresh HF causal forward of
   `[prompt | output]` (7.8e-6 < 1e-5), garbage written beyond `cache_len` changes nothing (continuation of one token,
   exact and static paths), and over-filling the cache is an assertion, not silent.

Last run: `6 passed, 0 failed` (verbatim output, with tests 5–6, in the section "Verified (speculative) decoding").

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
* **Fill-selection rules on the converted checkpoint**: `no_adjacent` / `max_fill` / `margin_tau` / `leftmost_run`
  are CPU-verified for semantics and decoder equality only; their accuracy / tokens-per-step trade-off against the
  §10.3 table (`bench/run_cdlm.py --no-adjacent --margin-tau … --fill-mode leftmost_run`) is unmeasured.
* **Verified decoding on the converted checkpoint**: `--verify` is CPU-exact in both decoders (section above) but
  its tokens/step, draft acceptance rate and tok/s on the CPT / stage-2 checkpoints are unmeasured.

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
