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
