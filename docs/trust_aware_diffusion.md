# Phase B7 — Trust-Aware Diffusion Decoding (K = 16, 8 forwards per block)

Brief: "SHIGERU Phase B7 — Trust-Aware Diffusion Decoding" (2026-10-25).  Hypothesis under test: the B-series' S = 8
weakness is not denoising capacity but the inability to decide *which placed tokens to trust and which to re-mask*.
Constraints kept throughout: no AR verification, no draft model, exactly 8 network calls per 16-token block (plus the
prefix pass), no reference tokens read at inference, checkpoints never modified, the 4B never re-trained with the B2
visible-correction loss.

## 1. Audit of what is already measured (no experiment repeated)

- B-series recipe and limits: discrete_diffusion_b_series.md §10–§11.  Best adapters on the e5 base: **lr 1e-4, no
  time embedding** (held-out S = 16 / 8 = 82.0 / 60.7 %, S = 8 left-to-right 69.3 %; GSM8K test 78.6 / 72.9 %) and
  **lr 2e-4, no time** (76.0 / 66.0 %; test 77.3 / 63.4 %).  Both were trained with the mask-only loss (λ_vis = 0 —
  no corrupted visible tokens), so a visible token attends itself and is simply copied: the one-step correction test
  (B3, §7 there) found random-replaced visible tokens are *not* corrected (errors 4.0 → 9.3 per block).
- The existing "remask" heuristic (re-mask visible tokens whose argmax disagrees with the token, budget 1/2/4) did not
  help: S = 16 84 → 84 / 82 / 76 %, S = 8 64 → 62 / 60 / 50 % (dev 50, B1 adapter, §10 there) — and it added a forward.
- Self-correction training (B2, B3 on-policy states, B8 mask patterns, A-series config D) improves teacher-forced
  correction but collapses free generation (§6, §10.4, §10.11 there) — excluded here; only a frozen 4B is used.
- A-series dependency diagnostic (dependency_aware_refinement.md): dependency between block positions is distance-1
  and left-sided; simultaneous Jacobi updates of adjacent positions are the ones that go wrong.
- Development data: items 0–49 of bench/prompts_gsm8k_dev.json for exploration.  Items 50–199 were used to *select*
  adapters in the B-series round, so a **new** held-out set was made: bench/prompts_gsm8k_dev2.json — 150 GSM8K-train
  questions absent from every training file and from the old dev set (bench/make_gsm8k_dev.py --exclude).  GSM8K
  test (1 319) only for the final fixed setting.

## 2. Trust estimation without extra training (brief §2)

Code: engine/bdiff/trust.py (features, logistic trust score, AUROC/AUPRC/Brier/ECE), bench/diag_trust.py.
Protocol: teacher-forced trajectories — gold prefix, 16 reference tokens, the fixed-budget sampler run from all masks
for 8 forwards without re-masking; at every step every position's features are logged with its correctness.  Visible
positions: label = placed token equals the reference ("should it be re-masked?").  Masked positions: label = current
argmax equals the reference ("safe to unmask?").  Features come from the forward the sampler already makes:
conf, margin, entropy, p_tok (probability of the placed token), agree (argmax == placed token), stable (argmax same as
the previous step), age (steps since placement), conf_at (confidence at placement), dconf = conf_at − p_tok, changes
(argmax changes so far), nb_changed (a neighbour changed last step).  Oracle (research only, +K forwards per step on
40 blocks): leave-one-out — re-mask the position alone and ask the model again.

### 2.1 Results (dev items 0–49, 294 blocks × 8 steps; e5 base's own greedy answers as reference, results/pure/dev_refs_e5.json, 47/50 correct)

Adapters: **lr1e-4** = runs/bdiff_e5_lr1e4 (no time), **lr2e-4** = runs/bdiff_e5_notime.  Visible rows: 16 576
(positions placed at an earlier step; 42 % of them differ from the reference by step 8).  Masked rows: 21 000.
AUROC is for predicting a WRONG token (0.5 = no information); Brier/ECE treat the feature as P(correct).
Files: results/pure/diag_trust_e5refs_{lr1e4,notime}.json (the same runs against the sft3 references,
diag_trust_{lr1e4,notime}.json, give the same picture within ±0.02).

| feature (no extra forward) | visible AUROC, lr1e-4 conf / left | lr2e-4 conf / left | Brier · ECE as P(correct) (lr1e-4 left) |
|---|---|---|---|
| conf (max prob of the current forward) | 0.507 / 0.499 | 0.489 / 0.493 | 0.329 · 0.305 |
| margin (top-1 − top-2) | 0.505 / 0.497 | 0.487 / 0.491 | 0.339 · 0.319 |
| entropy | 0.509 / 0.502 | 0.491 / 0.495 | – |
| p_tok (prob. of the placed token now) | 0.494 / 0.486 | 0.486 / 0.491 | 0.376 · 0.364 |
| agree (argmax == placed token) | 0.472 / 0.473 | 0.487 / 0.488 | – |
| stable (argmax same as last step) | 0.489 / 0.485 | 0.494 / 0.495 | – |
| age (steps since placement) | 0.418 / 0.417 | 0.417 / 0.416 | – |
| **conf_at (confidence when placed)** | **0.703 / 0.709** | **0.711 / 0.714** | 0.231 · 0.182 |
| dconf = conf_at − p_tok | 0.318 / 0.308 | 0.320 / 0.315 | – |
| **changes (argmax changes before placement)** | **0.668 / 0.693** | **0.666 / 0.681** | – |
| nb_changed (neighbour changed last step) | 0.526 / 0.558 | 0.524 / 0.558 | – |
| **logistic trust score (fit on prompts 0–24, scored on 25–49)** | **0.774 / 0.794** (AUPRC 0.654 / 0.652, base rate 0.42) | **0.781 / 0.798** | 0.169 · **0.037** |
| oracle: leave-one-out re-mask + 1 extra forward, argmax ≠ token (40 blocks) | 0.576 / 0.614 | 0.582 / 0.611 | – |
| oracle: leave-one-out confidence in the placed token | 0.627 / 0.643 | 0.652 / 0.674 | 0.295 · 0.275 |

Masked positions (is the current argmax right?): conf / entropy AUROC **0.765 / 0.772** (lr1e-4), margin 0.75; the
logistic score on masked features 0.776–0.794; conf as P(correct) has Brier 0.21, ECE 0.14–0.15 (over-confident).

### 2.2 What this says

1. **Every signal read from the current forward at a visible position is uninformative (AUROC 0.47–0.51).**  The
   adapters were trained with the mask-only loss, a visible token attends itself, and the model simply copies it: the
   argmax disagrees with the placed token at only 7–10 % of the tokens that end up wrong (`final_wrong_detectable`),
   p_tok < 0.5 at 7–11 %.  "Confidence-based" and "stability-based" re-masking therefore have almost nothing to act
   on, and the existing B-series remask heuristic (§10 of the B-series doc) was blind for this reason.
2. **The trust that exists is historical**: how confident the model was when it placed the token (conf_at, 0.70–0.71)
   and how often its prediction at that position changed before placement (changes, 0.67–0.69).  Age has AUROC 0.42
   in the wrong direction: tokens placed *early* (high confidence) are right more often, not less.
3. Combining the features linearly gives a usable but moderate detector: **AUROC 0.77–0.80, AUPRC 0.65 at a 42 %
   base rate, ECE 0.04** — well calibrated, mediocre ranking.  At any threshold a re-masking rule that re-masks the
   least-trusted tokens will re-mask many correct ones (Gate 2 is about whether the refill still nets positive).
4. Asking the model again with an extra forward (leave-one-out oracle) is *worse* than conf_at (0.58–0.67): the
   model's second opinion on a visible token is not better than its first, so no extra-forward trust scheme is worth
   building on these adapters.
5. The label itself is noisy at the token level: once a block diverges from the reference wording, later positions
   are "wrong" even when fluent (block exact 18–20 %, token error 42–45 % at step 8 against the model's own AR
   answer).  GSM8K accuracy of free generation is therefore the decisive metric for §3–§6.
6. Unmasking order has real signal (masked conf AUROC 0.77), which is why confidence order works at S = 16; the
   logistic score adds little (+0.01–0.02).

**Gate 1 verdict: a trust score exists (0.77–0.80 AUROC) but it is not sharp, and it cannot come from the
model's current opinion of a visible token.**  Re-masking rules are therefore evaluated under the fixed 8-forward
budget directly on GSM8K (§3), with the teacher-forced W→C / C→W accounting alongside.

## 3. Selective re-masking under a fixed budget of 8 forwards (brief §3)

Code: engine/bdiff/sampler2.py (`sample_block_fixed`), bench/eval_trust.py.  Exactly 8 network calls per 16-token
block (plus the one prefix pass per block that every B-series number includes); the 8th call fills every remaining
mask, so re-masking never adds a forward — the harness counts and reports forwards.  Re-masking never touches a token
placed in the same step, starts at step 2, and is capped by a budget per step (default 2).  Rules:

| rule | re-mask a visible token when … | signal needed |
|---|---|---|
| A none | never (current B-series monotonic unmasking) | – |
| B disagree | the argmax of the current forward differs from the placed token (most confident disagreement first; the old "remask" heuristic under the fixed budget) | current forward |
| B' lowprob | the placed token's probability under the current forward is < 0.5 | current forward |
| C stability | the argmax disagreed with the token in this step **and** the previous one | two forwards |
| D dependency | the argmax disagrees **and** a neighbour (distance 1) was placed or re-masked in the previous step | current forward + history |
| E trust | the logistic trust score (§2, fit on dev prompts 0–24) is < 0.5, least trusted first | history (conf_at, changes, age …) |

A note on measurement: the first dev-50 sweep was run with a sampler whose argmax used `topk`, which orders bf16
ties differently from the original sampler's `max`; that alone changed 11 of 48 teacher-forced blocks and moved the
S = 8 confidence-order GSM8K dev-50 score from 66 % to 74 %.  Everything below uses the original tie-breaking (the
two samplers now agree block for block, 48/48), and a ±8-point swing from tie-breaking is the scale of noise a
50-item set carries — which is why candidates are decided on the 150 fresh items of dev2.

### 3.1 Teacher-forced accounting (dev items 0–49, 296 reference blocks, lr1e-4 adapter, 8 forwards per block)

For every re-masked position: the token it had before re-masking vs the token it ends with, both against the
reference (W→C = a wrong token was repaired, C→W = a correct token was destroyed, W→W' = still wrong).  Baseline
error: 7.3 wrong tokens per 16-token block at step 8 (results/pure/b7_tf_dev50_lr1e4*.json).

| unmask | remask | budget / step | re-masks / block | W→C | C→W | W→W' | C→C | net (W→C − C→W) / block | final token error | block exact |
|---|---|---|---|---|---|---|---|---|---|---|
| confidence | none | – | 0 | – | – | – | – | 0 | 45.6 % | 19.3 % |
| confidence | disagree | 2 | 2.23 | 14 | 14 | 135 | 497 | 0.000 | 45.8 % | 18.6 % |
| confidence | lowprob | 2 | 2.31 | 13 | 15 | 138 | 519 | −0.007 | 45.8 % | 18.9 % |
| confidence | stability | 2 | 1.05 | 6 | 5 | 45 | 255 | +0.003 | 45.4 % | 19.3 % |
| confidence | dependency | 2 | 1.12 | 15 | 12 | 71 | 235 | +0.010 | 45.5 % | 18.6 % |
| confidence | trust | 2 | 4.12 | 78 | 44 | 689 | 407 | +0.115 | 44.6 % | 17.6 % |
| left | none | – | 0 | – | – | – | – | 0 | 45.8 % | 20.9 % |
| left | disagree | 1 / 2 / 4 | 2.15 / 2.65 / 2.79 | 16 / 24 / 25 | 18 / 31 / 34 | 115 / 149 / 157 | 489 / 581 / 609 | −0.007 / −0.024 / −0.030 | 46.1 / 46.5 / 46.3 % | 19.3 % |
| left | lowprob | 2 | 2.78 | 23 | 35 | 159 | 605 | −0.041 | 46.6 % | 19.3 % |
| left | stability | 2 | 1.14 | 8 | 12 | 54 | 262 | −0.014 | 46.1 % | 20.3 % |
| left | dependency | 2 | 1.44 | 17 | 24 | 102 | 282 | −0.024 | 46.3 % | 20.3 % |
| left | trust | 1 / 2 / 4 | 3.29 / 4.78 / 5.61 | 90 / 112 / 134 | 43 / 58 / 80 | 525 / 811 / 961 | 315 / 434 / 486 | +0.159 / +0.182 / +0.182 | 45.5 / 45.0 / 45.0 % | 20.6 / 20.3 / 20.9 % |
| trust (unmask by trust score) | none | – | 0 | – | – | – | – | 0 | 46.8 % | 16.9 % |
| trust | trust | 2 | 3.93 | 78 | 42 | 713 | 329 | +0.122 | 46.3 % | 14.9 % |

Reading: the rules that read the model's current opinion (disagree, lowprob, stability, dependency) re-mask 1–3
tokens per block and repair about as many tokens as they destroy (net −0.04 … +0.01 per block; correction success
9–17 %, damage 2–11 %).  The trust rule re-masks 3–6 tokens per block, 60–70 % of them *already wrong*
(AUROC 0.79 at work), yet 85 % of the re-masked wrong tokens come back wrong (W→W' 525–961 vs W→C 90–134): the
model has no better answer for those positions in the remaining steps.  Best net gain: +0.18 of the 7.3 wrong
tokens per block (2.5 %), final token error 45.8 → 45.0 %.  **Gate 2 (net repair) is passed only nominally by the
trust rule and failed by every rule built on the current forward; the refill, not the detection, is the limit.**
Unmasking by the trust score instead of confidence is worse (block exact 20.9 → 16.9 %).

### 3.2 Ceiling: oracle re-masking (research diagnostic, reads the reference — never usable at inference)

Same protocol, but the rule re-masks exactly the visible tokens that differ from the reference (lowest placement
confidence first), i.e. a perfect trust detector under the same 8-forward budget.

| unmask | oracle budget / step | re-masks / block | W→C | W→W' | correction success | final token error | block exact |
|---|---|---|---|---|---|---|---|
| left | 2 | 4.98 | 446 | 1 029 | 30.2 % | 45.8 → 37.6 % | 20.9 → 23.3 % |
| left | 4 | 6.56 | 542 | 1 401 | 27.9 % | 36.4 % | 23.3 % |
| left | 16 (unlimited) | 6.93 | 542 | 1 510 | 26.4 % | 36.4 % | 23.3 % |
| confidence | 2 | 4.96 | 297 | 1 170 | 20.3 % | 45.6 → 39.2 % | 19.3 → 21.6 % |
| confidence | 16 | 7.25 | 394 | 1 751 | 18.4 % | 37.8 % | 21.6 % |

**Even perfect detection repairs only 26–30 % of the tokens it re-masks**; the block exact-match rate rises by 2.4
points at most and 36 % of the tokens are still wrong after 8 forwards.  The refill — the model's prediction for a
re-masked position given its (partly wrong) neighbours, with 1–5 forwards left — is the bottleneck, not the
decision of what to re-mask.  For scale: simply spending 16 forwards (S = 16, left-to-right, no re-masking) gives
block exact 38.5 % and token error 36.7 % on the same blocks (results/pure/b7_tf_dev50_lr1e4_s16.json) — eight
more forwards of ordinary unmasking are worth more than perfect error detection within eight; and at S = 16 the
stability rule is already harmful (W→C 5 vs C→W 11).  This is the reason the Trust Head of brief §5 was not trained (§5 below): a head
could at best approach these oracle numbers, which do not move GSM8K (the trust rule's net +0.18 tokens per block
vs the oracle's +1.5–1.8 would have to be multiplied by the refill's 30 % success).

### 3.3 Free generation under the fixed budget — GSM8K accuracy

Setting: lr1e-4 adapter, K = 16, S = 8, 384-token budget, greedy, batch 1; every row is exactly 8 forwards per block
plus the prefix pass (tokens/forward 1.70–1.72 for every row; re-masking changes nothing in the count).  dev-50 =
exploration items (±8 points of tie-breaking noise, §3); **dev2 = 150 fresh items, the decision set**; AR greedy of
the same base model on dev2 = 92.0 % (dev2_refs_e5.json), B-series S = 16 on dev2 = 82.0 % (confidence) /
**90.0 % (left-to-right)**.  Files: results/pure/b7_gen_dev50_lr1e4_*.json, b7_gen_dev2_lr1e4_*.json (zeta rows in the
same names there).  McNemar = exact paired test against left/none on the same 150 items.

| unmask | remask (budget) | dev-50 | dev2 (150) | dev2 vs left/none: wins / losses, p |
|---|---|---|---|---|
| **left** | **none** (baseline, = B-series S = 8 left) | 76 | **70.0** | – |
| left | disagree (2) | 80 | 64.7 | +17 / −25, 0.28 |
| left | lowprob 0.5 (2) | 88 | 70.0 | – |
| left | **stability (2)** | 78 | **76.7** | **+21 / −11, 0.11** |
| left | stability (1) | 80 | 74.7 | |
| left | stability (4) | 76 | 76.7 | |
| left | dependency (2) | 84 | 70.0 | +20 / −20, 1.0 |
| left | dependency (1) | – | 73.3 | |
| left | trust 0.5 (2) | 68 | 66.7 | +16 / −21, 0.51 |
| left | lowprob 0.3 (2) | – | 74.0 | |
| confidence | none (= B-series S = 8 confidence) | 66 | 70.0 | +24 / −24, 1.0 |
| confidence | disagree (2) | 68 | 57.3 | +14 / −33, **0.008** (worse) |
| confidence | lowprob 0.5 (2) | 66 | 62.7 | |
| confidence | stability (2) | 72 | 66.7 | +21 / −26, 0.56 |
| confidence | dependency (2) | 72 | 64.0 | +18 / −27, 0.23 |
| confidence | trust 0.5 (2) | 62 | 61.3 | +23 / −36, 0.12 |
| trust-score order | none / stability / trust | 64 / 64 / 60 | – | |
| left, cosine mask schedule | none / stability (2) | – | 68.0 (1.93 tok/fwd) / 64.7 | |

Reading: of fifteen re-masking variants, **one** has a positive effect on the fresh set — left-to-right unmasking
with the stability rule (re-mask a token the model has contradicted in two consecutive forwards; ~1 re-mask per
block): 70.0 → 76.7 % (+10 items of 150, p = 0.11, not significant at 150).  Every rule based on a single forward's
disagreement is neutral or harmful on dev2 (confidence + disagree: −12.7, p = 0.008), the dev-50 "winners" (lowprob
88 %, dependency 84 %) are noise, and the trust-score rules are harmful despite the best detector (§3.1: they re-mask
5–6 tokens per block and most refills come back different-but-wrong, which also breaks the left-to-right structure
that gives S = 8 its 70 %).  **Gate 3 (clear improvement of S = 8 on unused data): not met at the 80 % target — the
best S = 8 point is 76.7 % on dev2; see §7 for the test-set number.**

## 4. Dependency-aware token selection (brief §4; S = 8, K = 16, 2 tokens per forward)

| unmask rule (no re-masking) | dev-50 | dev2 (150) | tokens / forward |
|---|---|---|---|
| confidence top-2 | 66 | 70.0 | 1.70 |
| left-to-right 2 tokens | 76 | 70.0 | 1.71 |
| non-adjacent top-2 (skip neighbours of a chosen position) | 66 | – | 1.70 |
| dependency-aware top-2 (non-adjacent, left neighbour must be visible) | 72 | – | 1.71 |
| adaptive top-M (all positions with conf ≥ 0.9, at least the schedule's 2) | 60 | 67.3 | 1.72–1.73 |
| trust-score order (logistic score on masked-position features) | 64 | – | 1.69 |
| + re-masking on top of these (disagree / dependency) | 52–72 | 66.0 (adaptive + dependency) | |

The A-series finding (simultaneous updates of adjacent positions are the harmful ones) does **not** transfer: forbidding
adjacent pairs or requiring a visible left neighbour does not beat plain confidence order (66–72 vs 66 on dev-50), and
the adaptive rule is worse (it commits more tokens per step exactly when the block is easy, and the 2-per-step
schedule is already the budget).  In the block-bidirectional model the two tokens placed per step are predicted
jointly from the same forward and are coherent with each other; the errors of S = 8 come from positions placed with
*too little context*, not from adjacent collisions — which is why left-to-right (every position sees all its left
context) is as good as confidence order at S = 8 and clearly better at S = 16 (90 vs 82 % on dev2).  No dependency
estimate with extra forwards was built: the A-series oracle (dependency_aware_refinement.md) already costs K
forwards per step and the inference-time proxy (adjacency) above has no effect.

## 5. Trust Prediction Head (brief §5) — not trained, by the gate

The brief allows a trained head only if the training-free trust signals lack discriminative power.  They are
moderate (AUROC 0.77–0.80, §2) — but §3.2 shows that a *perfect* detector would repair only 26–30 % of what it
re-masks and lift block exact match by 2.4 points, and §3.3 shows that the best available detector (the logistic
score) makes generation worse when used for re-masking.  A head on hidden states could improve detection, not the
refill; the measured ceiling says that is not where the S = 8 accuracy is lost.  The 4B stays frozen and no
visible-correction loss was reintroduced, as required.

## 6. The 8-step block, redesigned (brief §6) — what survives

Per-step state of a 16-token block (dev2 average over all blocks; fixed / masked / re-masked after the step).  Every
variant finishes 16 tokens at the 8th forward; nothing reads the reference.

| strategy | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | dev2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 current left-to-right | 2/14/0 | 4/12/0 | 6/10/0 | 8/8/0 | 10/6/0 | 12/4/0 | 14/2/0 | 16/0/0 | 70.0 |
| 2 current confidence order | 2/14/0 | 4/12/0 | 6/10/0 | 8/8/0 | 10/6/0 | 12/4/0 | 14/2/0 | 16/0/0 | 70.0 |
| 3 trust-aware unmasking (logistic score order) | same counts as 2 | | | | | | | | 64 (dev-50) |
| 4 trust-aware re-masking (left, score < 0.5, budget 2) | 2/14/0 | 3.5/12.5/0.5 | 5.4/10.6/0.6 | 7.2/8.8/0.8 | 8.9/7.1/1.1 | 10.7/5.3/1.3 | 12.5/3.5/1.5 | 16/0/0 | 66.7 |
| 5 dependency-aware unmasking (non-adjacent, left neighbour visible) | same counts as 1 | | | | | | | | 72 (dev-50) |
| 6 adaptive mixed (conf ≥ 0.9 → unmask, + dependency re-mask) | 2–4 per step, 1.70–1.73 tok/fwd | | | | | | | | 66.0 |
| **left-to-right + stability re-mask (budget 2)** | 2/14/0 | 4/12/0 | 5.8/10.2/0.17 | 7.9/8.1/0.11 | 9.9/6.1/0.14 | 11.7/4.3/0.30 | 13.75/2.25/0.25 | 16/0/0 | **76.7** |

The surviving design re-masks about one token per block (0.98), almost always in the second half of the block's
life, and otherwise is the plain left-to-right 2-tokens-per-forward schedule.  It is the only variant that beats the
current S = 8 decoders on the fresh set; it does not reach the 80 % target there (76.7 %) and HumanEval shows it does
nothing for code (8.5 % with and without re-masking, §7).  The re-masking rules that touch more tokens (trust, disagree
at every step) lose more than they gain because the refill fails 70–85 % of the time (§3.1–3.2).
