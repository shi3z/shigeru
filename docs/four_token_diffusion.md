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

## 5. Results on the final checkpoint (runs/phaseC_sft3_4b/final), dev items 0–49, no AR verification

### 5.1 Free generation (`bench/run_block.py --decoder k4`, GPU 7 shared with a training job: tok/s are not comparable)

AR greedy on the same 50 prompts: 96 %.  "forwards" counts every model launch including the merged commit.

| K | strategy | GSM8K dev | tokens / forward | corrections / 100 tok | repetition (4-gram) | premature EOS |
|---|---|---|---|---|---|---|
| 4 | s1 (all 4 in one forward) | 20 % | 3.83 | – | 0.25 | 13 / 50 |
| 4 | conf2 (2 + 2) | 58 % | 1.94 | – | 0.08 | 3 |
| 4 | refine (place 4, one leave-one-out probe) | 52 % | 1.94 | 34 | 0.07 | 12 |
| **4** | **refineN (probe rounds until stable, ≤ 3)** | **88 %** | **1.33** | 54 | 0.09 | 4 |
| 4 | conf2 + 1 probe | 70 % | 1.30 | 15 | 0.07 | 2 |
| 2 | s1 | 76 % | 1.95 | – | 0.09 | 1 |
| 2 | refine | 88 % | 0.99 | 10 | 0.11 | 1 |
| 8 | conf2 (4 + 4) | 18 % | 3.79 | – | 0.23 | 5 |
| 8 | refine | 2 % | 3.83 | 36 | 0.31 | 17 |
| 8 | refineN (≤ 3) | 18 % | 1.98 | 147 (oscillates) | 0.05 | 7 |
| 16 | conf2 (8 + 8) | 0 % | 7.66 | – | 0.83 | 3 |
| 16 | refineN (≤ 3) | 4 % | 5.19 | 90 | 0.41 | 10 |

(files results/pure/k4_sft3_K*_*.json)

### 5.2 Teacher-forced diagnostic (`bench/diag_k4.py`, 2 255 blocks of 4 from the model's own answers, 50 prompts)

| strategy | forwards / block | first-forward acc by distance 1/2/3/4 | final acc by distance | token acc | block exact (first → final) | correction success | regression |
|---|---|---|---|---|---|---|---|
| probe ceiling (true neighbours) | – | – | .958 .962 .959 .977 | 96.4 % | – | – | – |
| s1 | 1.0 | .969 .840 .708 .577 | same | 77.3 % | 48.1 % | – | – |
| conf2 | 2.0 | same | .969 .893 .841 .714 | 85.4 % | 48.1 → 67.9 % | 44.6 % | 2.6 % |
| refine (1 probe) | 2.0 | same | .950 .934 .827 .710 | 85.5 % | 48.1 → 64.2 % | 53.6 % | 5.1 % |
| **refineN (≤ 3 probes)** | 2.84 | same | **.955 .924 .888 .878** | **91.1 %** | 48.1 → **82.3 %** | **72.0 %** | 3.3 % |
| conf2 + 1 probe | 3.0 | same | .949 .934 .870 .836 | 89.7 % | 48.1 → 77.6 % | 66.1 % | 3.4 % |
| seq (AR, ceiling) | 4.0 | same | .969 .951 .940 .922 | 94.5 % | 48.1 → 90.4 % | 80.2 % | 1.3 % |

## 6. What the second forward does (answers to the brief's three questions, measured)

1. **Does restricting to 4 tokens recover non-AR accuracy?**  Partly by itself (one forward: 77 % of tokens, 48 % of
   blocks exact, 20 % GSM8K) and substantially with refinement: 88 % GSM8K dev vs 96 % AR, at 1.33 tokens per forward.
   This is the first pure-diffusion setting on this checkpoint above the Phase 1 bar (80 %) and above Phase D's 85 %
   dev target; it is below the Phase E1 speed target (≈ 2 tokens/forward) because the fixed point needs ~1.8 probe
   rounds per block on average.
2. **Does the second denoising fix first-step errors?**  Yes, when the other tokens are right: the leave-one-out probe
   with TRUE neighbours is 96–98 % accurate at every distance (ceiling row), i.e. the model's conditional given both-side
   context is as good as its AR conditional.  With the model's own first-step neighbours, one probe round fixes 54 % of
   the wrong tokens and breaks 5 % of the right ones; iterating to a fixed point fixes 72 % and breaks 3 %.  The gain is
   concentrated at distance 3–4 (.71 → .89, .58 → .88).  In free generation a single round is harmful (52 % vs 58 %
   for conf2) because simultaneous changes at neighbouring positions were each conditioned on the other's old token;
   the fixed-point iteration removes those inconsistencies.  Adoption rules that forbid adjacent simultaneous changes
   (`--adopt nonadjacent` / `one`) are being measured (§8).
3. **2 tokens/forward without AR verification?**  Not yet at ≥ 85 %: conf2 and refine give 1.94 tokens/forward at
   58 / 52 % dev accuracy; refineN gives 88 % at 1.33.  The teacher-forced numbers (85.5 % token accuracy at exactly
   2 forwards) say the gap is error propagation across blocks in free generation, not the per-block conditional.

Why it stops at K = 4: K = 8 and K = 16 oscillate (147 and 90 corrections per 100 tokens) because the first-forward
accuracy at distance ≥ 5 is below 50 % (pure_diffusion_research.md §2.1) — more than half the neighbours a probe
conditions on are wrong, so the fixed point is not the reference text.  The leave-one-out probe costs K² rows
(64 / 256 for K = 8 / 16), which is also where the plain-forward engine stops being launch-bound.

## 7–11. Training for K = 4, speed, VRAM, failures, next
*pending: K = 4-specialised stage 2 (brief §5) is justified by §6 only if the on-policy / random-mask checkpoints now
training do not already lift the distance-3/4 first-forward accuracy; evaluated next on the same 50 dev items.*

