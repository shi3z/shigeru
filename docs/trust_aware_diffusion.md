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
