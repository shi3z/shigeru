# SHIGERU Phase E — 4-token block diffusion (K = 4, two denoising forwards, no AR verification)

Brief (2026-10-24): set the 32-token problem aside; can a block of **4** future tokens be generated in **two** denoising
forwards at near-AR accuracy, giving ~2 tokens per forward without the AR head?  Companion documents:
docs/pure_diffusion_research.md (Phases A–D: diagnostics, block/self-correcting decoders, on-policy training) and
docs/distill_design.md (the model).  Everything here is **measured** unless marked *estimated* / *not measured*.
Dev data: `bench/prompts_gsm8k_dev.json` items 0–49 (GSM8K train, never in any training file) for all exploration;
items 50–199 are held back for confirmation; GSM8K test (1 319) is used once, at the end, for the final rows.

## 1. The problem in one table (from pure_diffusion_research.md, final checkpoint runs/phaseC_sft3_4b/final)

| decoding, dev 50 | GSM8K | tokens/forward |
|---|---|---|
| AR greedy | 95–96 % | 1.0 |
| K=32 block, 1 token per forward, left to right | 94 % | ~0.9 (block overhead) |
| K=32 block, 1 token per forward, most-confident-first | 40 % | ~0.9 |
| K=32 block, 4 tokens per forward (S=8) | 10 % | 3.9 |
| K=32 anchored adaptive (leftmost + conf ≥ 0.99) | 80 % | 1.28 |
| K=32 + re-masking self-correction (any strategy) | 32–54 % | ≤ 1.26 |

Teacher-forced recovery (§2.1 there): a mask 1 position after the last observed token is recovered at 97–99 %, at
distance 2: 84 %, 3: 73 %, 4: 61 %.  Blocks of 4 therefore live in the part of the reach curve where the model is still
mostly right — the brief's premise — and the open question is whether a second forward can repair the rest.

## 2. Why K = 4 (design rationale)

* The first forward of a block predicts 4 marginals; their expected per-position accuracy on this checkpoint is
  0.97 / 0.84 / 0.73 / 0.61 (teacher-forced, §1), i.e. about one wrong token per block.  A second forward that sees
  the other three tokens can, in principle, fix that one.
* Two forwards per 4 tokens = 2 tokens/forward, 2× the AR step count, with K² = 16 rows per probe forward — small
  enough that a forward costs about the same as an AR step on an A100 (launch-bound regime, distill_design.md §10).
* Nothing in the architecture needs to change: the probe layout is the training layout (§4).

## 3. Decoder (`engine/cdlm/k4_decoder.py`, CLI `bench/run_block.py --decoder k4 --k4-strategy …`)

Per block: forward 1 = all K masks (merged with the natural-order re-encoding of the previous block, so committing costs
nothing); then one of
`s1` (place all), `conf2` (place the most confident half incl. the leftmost, then the rest in a standard forward),
`refine` (place all, one leave-one-out probe forward, adopt changes), `refineN` (probe rounds until stable),
`conf2refine`, `seq` (leftmost one per forward = AR, the ceiling).  The probe forward (§4) re-predicts every position
given the other K−1 placed tokens in a single launch of K² rows and never writes the cache.  Tests
`engine/cdlm/tests/test_k4.py` (6): key-mask structure, no leakage of a position's own token into its probe, probe ==
training layout, `seq` == AR, cache validity after generation, no dependence on any non-mask-row logits.

## 4. Attention check

* Training (engine/cdlm/data.py, mask.py): inside an xt block, rows are observed-first; observed rows are causal among
  themselves and see the clean prefix; a mask row sees the prefix, **all** observed tokens of its block (left and
  right, original RoPE positions) and the masks before it.  Masks never see later masks; no x0 token of the same or a
  later block is ever visible (leakage test in tests/test_cdlm.py by perturbation).
* Inference probe: group i = [the K−1 placed tokens ≠ i, ascending, causal | mask at position i].  This is exactly a
  training block with position i masked, so the conditional the probe reads was trained (in random-mask mode).  A
  naive single set of placed rows would leak t_i into m_i through the causal encoding of t_{j>i}: measured on the
  tiny test model, probe 2's logits moved by 0.41 when t_2 changed; with separate groups they do not move (1e-5).
* KV cache: probe forwards are write=False; placed/mask rows written by the first forward are overwritten by the
  natural-order re-encoding before anything attends them as committed context (test_cache_valid_after_generation).
* Mask-to-mask attention within a block: absent in training and in the probe; carries no token information.

## 5. Results on existing checkpoints
*running*

## 6–11. Self-correction, training, speed, VRAM, failures, next
*pending*
