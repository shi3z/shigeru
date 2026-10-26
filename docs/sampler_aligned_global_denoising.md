# Phase B11 — Sampler-Aligned Global Denoising (K = 16, few parallel denoising steps, no left-to-right commitment)

Brief: "SHIGERU Phase B11 — Sampler-Aligned Global Denoising" (2026-10-26).  Main hypothesis under test: the denoiser's
*training states* (random masks over gold tokens, visible tokens always correct) do not match the states a global
sampler visits (self-generated, partly wrong visible tokens; mask layouts shaped by the sampler), and this mismatch —
not denoising capacity — is why every global refinement attempt (B9) failed and why the B10.5 gains did not transfer to
a non-monotonic sampler.  Competing hypotheses kept open: AR-pretrained dependency structure, the noise distribution,
capacity, and the sampling transition itself.  Constraints: no AR verification, no speculative/sequential drafts, no
visible-token correction CE (B2 failure), no left-to-right commitment hidden inside a "global" sampler.

## 1. Audit — how this differs from what failed before

| earlier attempt | what it did | why it failed (measured) | what B11 does differently |
|---|---|---|---|
| B9 full-block Jacobi (sync) | re-forwarded the all-visible self-generated block | the model never saw wrong visible tokens: 6–9 rewrites/step, drift to repetition | states with wrong visible tokens become *training* states, but only re-masked positions are targets |
| B9 soft-token inputs | fed probability-weighted embeddings | out of distribution, 4–6 % | not used; stays in token space |
| B9/B10.5 non-monotonic re-masking | any token may return to [MASK] (62 % at S = 8 on dev-50) | the denoiser was trained on random masks over gold, never on its own re-masked states | the sampler's own states are collected and trained on (masked CE), with the joint teacher supplying consistent targets |
| B2 self-correction training | CE on corrupted *visible* positions | collapsed free generation | **not** reintroduced: visible positions are never targets; a wrong visible token is corrected only by explicitly re-masking it in the training state |
| B7 trust-aware re-masking | re-mask by a confidence/trust score | the model's opinion of a visible token is uninformative (AUROC 0.5) | re-masking rules are re-examined only with a sampler-aligned model, whose opinion of visible tokens may become informative because it has seen wrong ones |
| B10 / B10.5 | masked CE on left-to-right S = 8 states + teacher labels: +5 points on test, no transfer to the global sampler | the states were left-to-right states | the same recipe on the *global sampler's* states (and mixed) |

Data: dev items 0–49 (reference = the base model's own greedy answer, results/pure/dev_refs_e5.json) for the state
analysis and teacher-correction diagnostics; bench/prompts_gsm8k_dev2.json (150 fresh items) for decisions; GSM8K test
only for a fixed final setting; training states only from GSM8K-train prompts disjoint from all of these.

## 2. B11-A — sampler state distributions (bench/diag_sampler_states.py)

(results in §2.1 when the dev-50 runs finish)

## 4. B11-C — sampler-aligned training states (distill/gen_global_states.py)

States are collected by running the *student sampler itself* on GSM8K-train prompts disjoint from every evaluation set
(bench/prompts_gsm8k_pairtrain.json items 800–1099 for the non-monotonic re-masking sampler, 1100–1399 for the
left-to-right sampler; 300 prompts each, K = 16, S = 8), following its own choices (on-policy).  Every state x_s — the
block before the s-th forward, with its [MASK]s and its possibly wrong visible tokens — is stored twice:

| variant | input state | targets | what it teaches |
|---|---|---|---|
| plain | x_s as visited | masked positions ← the teacher's sequential completion *conditioned on the current visible tokens* (one forward per mask, left to right over the masks) | fill the masks consistently with whatever is visible, including wrong tokens — the capability a sampler needs at every step; visible positions are never targets |
| corrected | x_s with every visible token ≠ T re-masked (T = the teacher's sequential block for this prefix, 16 forwards) | all masked positions ← T | correct a wrong visible token only by making it a mask: the B2 failure (CE on visible positions) is not reintroduced |

The all-mask state (step 1), random-looking states (the re-masking sampler at mid-ratio), non-monotonic states
(masks with visible tokens to their right) and teacher-corrected states are all present in the collected set; the
"structured mask" and "global refinement" states of the brief are the re-masking sampler's own states at steps 2–8.
Per state the number of visible tokens differing from T is recorded (the wrong-visible statistic of free generation).
Cost: ~2 × n_mask teacher forwards per state (plain + the shared T): ~40 s per prompt.

### 2.1 State geometry and visible-token correctness per step (dev items 0–49, 296 reference blocks, S = 8; results/pure/b11_states_*.json)

"islands" = runs of [MASK]; "prefix" = longest visible run from position 0; "right-visible" = some mask has a visible
token to its right (impossible under left-to-right unmasking); "wrong visible" = visible tokens ≠ REF.

| sampler, step | mask ratio | islands | prefix | right-visible states | wrong visible / state | wrong-visible fraction | states with ≥ 1 wrong visible |
|---|---|---|---|---|---|---|---|
| **training distribution** (random masks on gold), s2 / s5 / s8 | .875 / .5 / .125 | 2.7 / 4.4 / 1.9 | 0.1 / 0.9 / 4.8 | 99 / 100 / 99 % | **0 / 0 / 0** | 0 | 0 % |
| non-monotonic re-masking (global), s2 / s5 / s8 | .875 / .5 / .125 | 1.3 / 2.3 / 1.7 | 1.7 / 3.4 / 6.8 | 27 / 82 / 85 % | 0.3 / 2.9 / 6.4 | 15 / 36 / **46 %** | 24 / 63 / 80 % |
| full-block synchronous (Jacobi), s2 / s5 / s8 | 0 / 0 / 0 | 0 | 16 | 0 % | 10.5 / 14.4 / 14.7 | 66 / 90 / 92 % | 100 % |
| left-to-right monotonic, s2 / s5 / s8 | .875 / .5 / .125 | 1 / 1 / 1 | 2 / 8 / 14 | 0 / 0 / 0 % | 0.3 / 2.5 / 6.0 | 14 / 32 / 43 % | 25 / 59 / 77 % |
| confidence monotonic, s2 / s5 / s8 | .875 / .5 / .125 | 1.3 / 1.7 / 1.4 | 1.7 / 6.3 / 11.5 | 27 / 53 / 54 % | 0.3 / 2.6 / 5.9 | 15 / 32 / 43 % | 24 / 60 / 78 % |

Against REF, the left-to-right and the global re-masking samplers accumulate wrong visible tokens at the *same* rate
(0.3 → 2.5 → 6.0 vs 0.3 → 2.9 → 6.4 per state): how much of this is real error and how much paraphrase of the reference
is the subject of §3.2.

**Gate 1 — what differs between the training states and the sampler states.**  Not the mask geometry: the random
training masks have *more* islands and *always* have visible tokens to the right of masks, so the global sampler's
layouts (1.3–2.3 islands, right-visible in 27–85 % of states) are inside the training support, and it is the
left-to-right sampler whose contiguous-prefix layout is the rare one.  The difference is the **visible tokens**: in
training every visible token is correct; in the global sampler 15 % of the visible tokens are wrong at step 2 and
46 % at step 8 (80 % of the late states contain at least one wrong token); in full-block Jacobi 66–92 % of the block
is wrong and nothing is masked.  The denoiser's conditional accuracy at the masked positions shows the consequence
directly: 51–94 % masked accuracy on training-distribution states vs 34–41 % on the re-masking sampler's states at
the same mask ratios (§3).

## 3. B11-B — teacher correction from the visited states (Gate 2)

From each visited state: `fill` = one denoiser forward at the masks; `seqfill` = masks filled one per forward, left to
right over the masks (the chain-rule teacher generalised to any mask layout); `oracle` = visible tokens ≠ REF are
re-masked first (uses REF: an upper bound, never an inference method), then seqfill.  Token accuracy of the whole
16-token block vs REF after the update (W→C per block in brackets).

| state source, step | fill | seqfill (teacher) | oracle re-mask + seqfill |
|---|---|---|---|
| all-mask (every sampler, s1) | 34.4 % (5.5) | **63.3 %** (10.1) | 63.3 % |
| training distribution, s2 / s5 / s8 | 57.5 / 91.6 / 99.3 % | 77.3 / 94.9 / 99.5 % | = seqfill (no wrong visible) |
| re-masking sampler, s2 / s5 / s8 | 40.6 / 49.7 / 52.5 % | 60.3 / 55.2 / 52.5 % | **65.2 / 72.5 / 74.6 %** |
| full-block Jacobi, s2 / s5 / s8 | 34.4 / 9.7 / 7.9 % (no masks: nothing to fill) | same | **77.2 / 71.7 / 70.7 %** |
| left-to-right, s2 / s5 / s8 | 40.1 / 51.3 / 54.2 % | 58.6 / 56.6 / 54.2 % | 64.5 / 72.9 / 74.7 % |
| confidence, s2 / s5 / s8 | 40.6 / 51.2 / 54.4 % | 60.3 / 57.0 / 54.5 % | 65.2 / 71.6 / 74.1 % |

Reading: (1) the sequential teacher is strong where the context is right (training states: 95–99 % at ≤ 50 % masks;
all-mask: 63 %), (2) on the global sampler's states its gain shrinks step by step (60 → 52 %) because the masks it fills
are conditioned on more and more wrong visible tokens — **the teacher cannot correct what it is not allowed to touch**,
(3) if the wrong visible tokens are re-masked (oracle), 65–75 % of the block is recovered from the same states, and
even the fully wrong Jacobi blocks come back to 71–77 %.  So a consistent correction *is reachable* from global states,
and the whole question of global refinement reduces to one capability the current model lacks: deciding which visible
tokens to re-mask (B7 measured its current opinion of visible tokens as uninformative — because it never saw a wrong
one in training).  This is exactly the capability the sampler-aligned states of §4 expose the model to, and the
"corrected" variant is its supervised form.

### 3.1 What the "oracle 70–77 %" numbers are, exactly (brief v2 §3)

- **Quantity**: token accuracy of the whole 16-token block against REF (the base model's own greedy answer for the
  reference prefix) after the update — *not* block exact match, *not* a correction success rate.  Denominator: 16 × 296
  blocks per step.
- **Procedure**: from the state the sampler visited, every visible token that differs from REF is re-masked (this
  reads REF — it is an upper bound, not an inference method), then the masks are filled one per forward left to right
  by the frozen model (n_mask' extra forwards, up to 16).  The teacher sees REF only through the choice of which
  positions to re-mask; it never sees the reference tokens themselves.
- **Difference to B7's oracle** (trust_aware_diffusion.md §3.2): B7 re-masked wrong tokens under the *left-to-right
  monotonic* sampler with a per-step budget and refilled them with the plain sampler's remaining forwards (30 % repair
  success).  Here the refill is the sequential teacher with as many forwards as masks, from states of the *global*
  sampler, and the number is accuracy after the update, which includes the 50–85 % of tokens that were already right.
- **What it does and does not say**: it says that from the global sampler's states a block ≈ 73 % identical to REF is
  reachable *if* the wrong visible tokens are known and *if* the refill is sequential (22+ forwards); it does not say
  that any inference-time rule finds those tokens (B7: AUROC ≈ 0.5 for the current model), nor that mismatch with REF
  is error (§3.2 separates the two).

### 3.2 Reference mismatch is not error (brief v2 §2; bench/diag_mismatch.py, free generation on dev-50, baseline adapter)

Five quantities, measured separately on the same runs (T = the sequential teacher block for the same prefix, 16 extra
forwards per block, research only; REF only where the generated prefix still equals the reference's):

| left-to-right S = 8, free generation (561 blocks) | step 2 | step 5 | step 8 | final block |
|---|---|---|---|---|
| A reference mismatch of visible tokens (51 reference-aligned blocks) | 11.8 % | 25.0 % | 42.9 % | 48.9 % |
| B self-trajectory mismatch (visible tokens vs the run's own final block) | 0 | 0 | 0 | – |
| C teacher-trajectory mismatch (visible vs T) | 10.9 % | 30.5 % | 42.2 % | **45.2 %** |
| D adjacent duplicate tokens per block | 0.06 | 0.36 | 0.70 | 0.84 |
| E task failure | – | – | – | GSM8K dev-50 76 % correct |
| **paraphrase check**: final-block mismatch with T in task-CORRECT answers / in WRONG answers | | | | **43.5 % / 49.7 %** |

Reading: a correct answer differs from the teacher's sequence at 43.5 % of its tokens; a wrong answer at 49.7 %.  So
of the ~45 % "wrong visible" tokens reported in §2.1–§3 against REF or T, roughly **43 points are wording freedom and
about 6 points are the error signal** — the mismatch counts cannot be used as an error rate, and a correction scheme that
"fixed" every token that differs from the teacher would mostly be rewriting correct answers into the teacher's wording.
What *does* separate correct from wrong blocks is structural: adjacent duplicates (0.84 per final block overall; §3.3
gives the split per outcome once the remask run is in) and the GSM8K result itself.  This also recasts §3.1: the
oracle's 70–77 % "recovery" is to a large extent re-wording into REF, not repair.  Consequences for the training data of
§4: the *corrected* variant (re-mask every visible token ≠ T and target T) imposes the teacher's wording on the student
— it should be read as "teacher-trajectory imitation", not as "error correction"; the *plain* variant (keep the visible
tokens, complete consistently with them) does not have this problem.  Both are evaluated (arms C vs E).

## 7. Denoising dynamics of the global sampler (brief v2 §5; baseline adapter, free generation, dev-50, 560 blocks)

Non-monotonic confidence re-masking, S = 8 (GSM8K dev-50 62 %; left-to-right on the same prompts 76 %):

| step | visible / 16 | positions changed vs previous guess | visible later overturned (B) | mismatch with T of visible (C) | adjacent duplicates | Hamming to T (argmax block) | Hamming to own FINAL | W→C / C→W vs T | W→C / C→W vs FINAL |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 2 | 5.7 | 3.7 % | 13.5 % | 0.06 | 9.73 | 8.02 | 1.61 / 0.64 | 2.36 / 0.53 |
| 3 | 4 | 4.9 | 3.8 % | 23.3 % | 0.18 | 9.08 | 6.23 | 1.28 / 0.64 | 2.26 / 0.47 |
| 5 | 8 | 3.2 | 2.6 % | 36.3 % | 0.46 | 8.31 | 3.07 | 0.70 / 0.40 | 1.81 / 0.27 |
| 7 | 12 | 1.5 | 1.0 % | 45.6 % | 0.84 | 8.03 | 0.71 | 0.32 / 0.17 | 1.13 / 0.08 |
| 8 | 14 | 0.7 | 0 | 48.0 % | 0.99 | 7.97 | 0 | 0.14 / 0.08 | 0.71 / 0 |

Final blocks: mismatch with T 49.8 % (task-correct answers 48.2 %, wrong 51.6 %), adjacent duplicates **1.23 per block**
(left-to-right 0.84), 4-gram repetition 0.087.  Positions whose argmax changed value at least twice during the run:
40 % (left-to-right 52 % — most of that is the masked argmax moving before placement).

Answers to the four questions of the brief:
- **A. Correction gain.**  Against T, each step repairs about twice as many positions as it breaks (W→C / C→W 1.6 / 0.6
  → 0.14 / 0.08), but the repairs are overwhelmingly *newly filled* positions, not revisits: the fraction of visible
  tokens that the sampler later overturns is 3.7 % at step 2 and falls to 0 — the re-masking budget is spent on
  low-confidence *fresh* placements that get re-predicted to the same or a similar token.  Revisiting exists but is
  almost inert.
- **B. Global consistency.**  Hamming distance to the teacher block stops improving after step 3 (9.1 → 8.0 and flat)
  while the distance to the run's own final block falls to 0: the process converges to *its own* fixed point, which is
  as far from T at step 8 as at step 4.  Adjacent duplicates *grow* with the steps (0.06 → 0.99 per block) — the
  degenerate pairs are created by the placements, not removed by the iterations.  So the honest answer to "is the
  whole block improving?" is **no**: masks are being filled, and the quality of the filled tokens is set at placement.
- **C. Order sensitivity.**  From the *same* mid-run state (step 5, 560 states), finishing with 4 more non-monotonic
  steps or with 4 more left-to-right monotonic fills gives completions that agree with T equally (50.2 vs 50.7 %) and
  with each other at 83.9 % of positions: the order of the remaining placements barely matters; what matters is the
  state they start from.
- **D. Convergence.**  Stable, not oscillating (no position flips back and forth once visible: B → 0), but stable at
  a point with 1.2 duplicates per block and 12 points less GSM8K accuracy than left-to-right.  The failure mode is
  *premature lock-in with no mechanism to undo*, not instability.

Taken with §3.2, the picture for the undistilled model is: the global sampler's extra freedom (which positions to
place, which to revisit) buys nothing because the denoiser's output at a visible position is a copy and at a masked
position depends on a context that is 40–48 % different from the teacher's wording either way; the only measurable
damage of the non-left-to-right order is structural (duplicates 1.23 vs 0.84, GSM8K 62 vs 76).  §6 tests whether
training on exactly these states changes any of it.

## 5. B11-D — sampler-aligned masked CE arms (distill/train_states.py; all: new LoRA r 64 on the merged B-series model, 1 000 × 8 states, lr 1e-4, targets at masked positions only)

| arm | training states | targets | dev-50 remask S4 / S8 / S16 | dev-50 left S4 / S8 / S16 | dev2 remask S8 | dev2 left S8 |
|---|---|---|---|---|---|---|
| B1 baseline (lr1e-4 adapter) | – | – | 32 / 62 / 74 | 32 / 76 / 92 | – / 62 (B9 dev-50 only) | 70.0 |
| B10.5 arm B (left pair-states, teacher-followed) | left S = 8 pair states | final block of the teacher-followed trajectory | 28 / 62 / 62 | 56 / 82 / 92 | – | 83.3 |
| B11-B left states, plain | left S = 8 on-policy states | teacher completion given the visible tokens | __B__ | | | |
| B11-C remask states, plain | non-monotonic re-masking S = 8 on-policy states | same | __C__ | | | |
| B11-D mixed, plain | remask + left states | same | __D__ | | | |
| B11-E remask states, plain + corrected | remask states; half the updates use the corrected state (visible ≠ T re-masked) with T as target | | __E__ | | | |

## 6. B11-E — true global refinement with the sampler-aligned models

(filled from the arm evaluations: the non-monotonic sampler's S = 4 / 8 / 16 accuracy, duplicates, overturn rate and
W→C / C→W per step, side by side with the monotonic samplers on the same models)

## 8. Training distribution vs inference distribution (bench/diag_state_dist.py)

Features per state (mask ratio, mask islands, visible prefix length, right-visible flag, fraction of visible tokens ≠ T);
histogram intersection (0–1) between each training set and the 4 480 states the non-monotonic sampler visits in free
generation on dev-50 (results/pure/b11_state_dist_remask.json).

| training set | n states | islands (mean) | prefix | right-visible | wrong-visible frac | overlap: islands / prefix / right-visible / wrong-visible |
|---|---|---|---|---|---|---|
| inference (remask sampler, dev-50, free) | 4 480 | 1.77 | 3.76 | 59 % | 29.7 % | – |
| **arm C/E states** (remask sampler on training prompts, on-policy) | 27 872 | 1.76 | 3.76 | 60 % | 30.0 % | **0.998 / 0.967 / 0.996 / 0.974** |
| original recipe (random masks over gold) | 4 536 | 3.18 | 1.29 | 87 % | 0 % | 0.50 / 0.64 / 0.72 / **0.46** |
| arm B/D left states | __LEFTDIST__ | | | | | |

**Gate 1 (state alignment): the on-policy states cover the inference distribution** (overlap ≥ 0.97 on every feature;
the sampler is the same, the prompts differ), and the original recipe's states do not: half the mass is elsewhere on
islands and prefix, and the wrong-visible dimension — the one that matters (§2.1, §3) — overlaps 0.46 because the
training states never contain a wrong visible token.  Whether closing this gap changes generation is the separate
question of §5–§6 (brief v2 §7: "distribution closer" and "quality better" are verified separately).
