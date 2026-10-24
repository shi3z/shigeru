# SHIGERU Phase E — 4-token block diffusion (K = 4, two denoising forwards, no AR verification)

Brief (2026-10-24): set the 32-token problem aside; can a block of **4** future tokens be generated in **two** denoising
forwards at near-AR accuracy, giving ~2 tokens per forward without the AR head?  Companion documents:
docs/pure_diffusion_research.md (Phases A–D: diagnostics, block/self-correcting decoders, on-policy training) and
docs/distill_design.md (the model).  Everything here is **measured** unless marked *estimated* / *not measured*.
Dev data: `bench/prompts_gsm8k_dev.json` items 0–49 (GSM8K train, never in any training file) for all exploration;
items 50–199 are held back for confirmation; GSM8K test (1 319) is used once, at the end, for the final rows.

Follow-ups: **docs/k_scaling_failure_analysis.md** (why K = 5…8 fail: non-convergence of Jacobi adoption, not model
capability or a bug; held-out numbers), **docs/dependency_aware_refinement.md** (Phase E3: dependency measurement,
adoption rules, early stopping, adaptive K; the frontier stays K = 4), **docs/discrete_diffusion_b_series.md** (the
block-bidirectional B-series: K = 16 at 84 % after 30 min of LoRA).

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


## 8. Ablations on the refinement (dev items 0–49, final checkpoint; AR 96 %)

| K | rounds ≤ | adoption rule | GSM8K dev | tokens / forward | corrections / 100 tok |
|---|---|---|---|---|---|
| 4 | 3 | always | 88 % | 1.33 | 54 |
| 4 | 6 | always | 92 % | 1.18 | 65 |
| 4 | 6 | nonadjacent | 90 % | 1.22 | 40 |
| 4 | 6 | conf (adopt only if the probe is more confident) | 86 % | 1.35 | 33 |
| **4** | **8** | **one (a single change per round)** | **96 %** | **1.17** | 35 |
| 8 | 3 | always | 18 % | 1.98 | 147 |
| 8 | 8 | nonadjacent | 72 % | 1.16 | 128 |
| 8 | 12 | one | 90 % | 0.94 | 82 |
| 8 | – | conf2 + 1 probe (nonadjacent) | 36 % | 2.53 | 24 |
| 4 (random-mask ckpt phaseC_sft_4b) | 3 | always | 70 % | 1.14 | 95 |

Reading: adopting **one** change per round (Gauss–Seidel) reaches the AR accuracy (96 % = AR on the same prompts) at
K = 4 and 90 % at K = 8 — the probe's conditional is reliable, simultaneous changes were the problem.  The cost is
rounds: ~2.4 probe forwards per 4-token block (1.17 tokens per forward).  Whether that is faster than AR depends on
the price of a 16-row forward relative to a 1-row AR step; on the A100 the engine is launch-bound up to ~32 rows
(distill_design.md §10.8), so *estimated* ≈ 1.17 × (forwards/s of the graph engine) — to be measured on an idle GPU
with the static/graph probe path (§9).  The random-mask checkpoint's weaker first forward (.97/.75/.48/.38, §5.2
diag_k4_sftrandom) gives it a worse fixed point (70 %): the first-forward reach still matters.

### 8.1 Held-out confirmation (dev items 50–199, never used for any selection; 150 prompts; same checkpoint)

| decoding | GSM8K (150) | tokens / forward | paired vs AR (bootstrap 95 % CI of the difference) |
|---|---|---|---|
| AR greedy (HF) | **90.0 %** | 1.00 | – |
| K=4 refineN, adopt one, ≤ 8 rounds | 84.7 % | 1.13 | −5.3 pts [−10.0, −1.3]; both right 125, only K4 2, only AR 10, neither 13 |
| K=4 refineN, adopt always, ≤ 6 rounds | 84.7 % | 1.17 | −5.3 pts [−10.7, −0.7]; both 123, only K4 4, only AR 12, neither 11 |

The 96 % / 92 % of the 50 exploration items were optimistic (selection over ~20 settings on 50 prompts, and those 50
are easier: AR 96 % there vs 90 % here).  The honest estimate for the best pure-diffusion K = 4 setting on this
checkpoint is **about 5 points below AR** (85 % vs 90 %) at 1.13–1.17 tokens per forward, with no AR verification.
Phase E1's 80 % is met; Phase E2's 85 % is at the edge; the 2 tokens/forward target is not met.

## 9. Speed on an otherwise idle GPU (zeta RTX PRO 6000 Blackwell 96 GB, bf16, batch 1, 384-token budget, dev items 0–49)

Model runs/zeta_sft_final_e5/final (same 1 B CPT, 5-epoch stage 2; its AR is 92 % on these 50 items).  Graph engine =
engine/cdlm/fast_model.py with CUDA graphs for both the standard forward and the K-group probe (`run_masked`).

| decoding | GSM8K dev 50 | tokens / forward | **tok/s** | time split (50 prompts) |
|---|---|---|---|---|
| AR greedy, HF generate | 92 % | 1.00 | 79.2 | – |
| AR via the graph engine (τ=None streaming) | 88 %* | 0.99 | 84.4 | – |
| **K=4 refineN, Jacobi, ≤ 3 rounds, graph** | **90 %** | 1.35 | **104.6** | first 28.0 s / probe 54.3 s / prefill 0.9 s |
| K=4 refineN, Jacobi, ≤ 6 rounds, graph | **94 %** | 1.23 | **95.6** | first 27.4 / probe 61.0 |
| K=4 refineN, one change per round, ≤ 8, plain engine | 90 % | 1.24 | 81.5 | first 32.7 / probe 71.5 |

\* the streaming decoder's τ=None differs from HF greedy on 3/50 prompts (bf16 numerics, see distill_design.md §11.6).
(files results/pure/speed_zeta_*.json; the one_r8 graph run shared the GPU with a stale process for part of its run and
is omitted.)

Reading: with CUDA graphs a 16-row probe forward costs about the same as a 1-row AR step (launch-bound), so the
refinement's 1.2–1.35 tokens per forward turn into **1.2–1.3× the HF AR throughput at the same accuracy** (90–94 % vs
92 % on these prompts) — pure diffusion, no AR verification.  The probe forwards are ~2/3 of the decode time; the
remaining lever on the engine side is fusing the K-group probe into fewer, wider launches and the per-step host work.
Not measured here: tok/s on the A100 (all three were busy with trainings; the plain-engine A100 numbers in §5 were taken
on a shared GPU).

## 10. Final GSM8K test (1 319 problems, setting fixed beforehand on dev/held-out, never tuned on test)

| decoding (no AR verification), A100 80 GB, bf16, batch 1, 384-token budget | GSM8K test | tokens / forward | tok/s | rounds / block | blocks converged / cycle / budget |
|---|---|---|---|---|---|
| **K = 4 leave-one-out refinement, Jacobi ≤ 6 rounds, CUDA-graph engine** (runs/phaseC_sft3_4b/final) | **84.2 %** (1 111 / 1 319) | 1.18 | **89.9** (88.0 total) | 2.32 | 55 249 / 3 667 / 2 303 |
| K = 4 leave-one-out refinement, Jacobi ≤ 3 rounds, CUDA-graph engine (same checkpoint) | 79.5 % (1 049 / 1 319) | 1.32 | 99.2 (98.0 total) | 1.97 | 46 221 / 1 042 / 14 182 |
| AR greedy, HF generate, same checkpoint (distill_design.md §11.6) | 88.9 % | 1.00 | 52.5 | – | – |
| AR via the graph engine (τ=None), same checkpoint (§2 of dependency_aware_refinement.md, dev 50) | – | 0.99 | 72.2 | – | – |
| AR-verified speculative decoding (the A-series headline that the pure-diffusion briefs excluded) | 87.6 % | 2.15 | 156.0 | – | – |

Pure diffusion decoding of the 4B — no AR logits read, no draft model — reaches **84.2 % on the full GSM8K test at 1.7×
the HF-AR throughput** (89.9 vs 52.5 tok/s on the same GPU class), 4.7 points below AR.  50 of 1 319 outputs hit the
384-token budget (vs 68 premature stops); 6 % of blocks ended in a detected cycle and 4 % in the 6-round budget.
Timing split over the whole run: probe forwards 1 887 s, first forwards 773 s, prefill 29 s.  (results/pure/
final_test_k4_refineN_r6_graph.json; GPU 6 while GPU 7 ran another job — not strictly idle.)  Capping the Jacobi
rounds at 3 trades 4.7 points for 10 % more throughput (79.5 % at 99.2 tok/s, 1.32 tokens/forward): 23 % of blocks then
end on the round budget instead of 4 %, so the 6-round setting chosen on dev is the right one for accuracy.

## 7. Training for K = 4 (brief §5) — measured on the stage-2 ablation checkpoints, dev items 0–49

The brief's K = 4-specialised stage 2 was justified by §6 only if a different mask recipe lifts the distance-3/4
first-forward accuracy.  None of the four stage-2 variants trained for pure_diffusion_research.md §8 does:

| stage-2 checkpoint | first-forward accuracy by distance 1 / 2 / 3 / 4 | K = 4 refineN ≤ 3 / ≤ 6 |
|---|---|---|
| suffix masks (phaseC_sft3_4b, used everywhere above) | .97 / .84 / .71 / .58 | 88 / 92 % |
| random masks (pure_sft3_random) | .98 / .83 / .65 / .52 | 80 / 82 % |
| mixed suffix + random (pure_sft3_mixed) | .97 / .84 / .69 / .55 | 82 / 82 % |
| self-generated responses, random masks (pure_sft3_self_random) | .98 / .84 / .68 / .57 | 82 / 86 % |
| on-policy states as context (pure_sft3_onpolicy, config D) | .88 / .76 / .54 / .45 | 10 / 20 % (collapse) |

The distance-3/4 conditionals are a property of the data and the causal-within-block layout, not of the mask recipe;
the suffix checkpoint stays the best K = 4 model, and a K = 4-specialised stage 2 (every block exactly one group of 4
with all leave-one-out rows, which the training layout already contains) was therefore not run.  Self-generated
targets improve the adaptive sampler (+6 points) but not this decoder.

## 11. VRAM, failures, next

Inference VRAM of the K = 4 graph engine: 9.4 GB (4B bf16 weights 8.1 GB + static KV/probe buffers, batch 1, 4 096
positions) on the A100; the HF AR path uses 10.8 GB.  Failures recorded in this document and its companions: K ≥ 5
Jacobi non-convergence (k_scaling_failure_analysis.md), dependency-aware adoption limited to ~1.1 tokens/forward
(dependency_aware_refinement.md), on-policy training collapse (§7 above), position-free future prediction gate
(future_token_set_research.md), continuous latent diffusion (latent_diffusion_research.md), block-bidirectional
B-series below the K = 4 decoder at every step count (discrete_diffusion_b_series.md §11).  Next: nothing in the
pure-diffusion line is left that the measurements suggest would beat 84.2 % / 89.9 tok/s without AR verification; the
remaining lever is batching several problems per forward in the graph engine, which is an engineering rather than a
research item.

