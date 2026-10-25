# Phase B8 — Pairwise Joint Denoising & Trajectory Distillation (K = 16, 8 forwards per block)

Brief: "SHIGERU Phase B8 — Pairwise Joint Denoising & Trajectory Distillation" (2026-10-25).  Hypothesis under test:
the S = 8 gap (72.9 % GSM8K test vs 84.8 % at S = 16, left-to-right, same adapter) comes from placing two tokens per
forward by *independent* argmax, i.e. from ignoring the dependency between the two positions.  Constraints kept: no
AR verification, no draft model, 8 forwards per block for the practical rules (extra forwards are counted and
reported wherever a research ceiling uses them), frozen 4B, no checkpoint modified, nothing from B2/B3/B8/B7 repeated.

## 1. Audit — what the earlier phases already say about this hypothesis

- At S = 8, left-to-right order places positions (i, i+1) together from one forward; confidence order places the two
  most confident masked positions, which are adjacent 35–45 % of the time.  Both reach 70.0 % on the fresh dev2 set
  and 72.9 / 63.4 % on test (trust_aware_diffusion.md §3.3, §7).
- Spending one forward per token (S = 16, left-to-right) — i.e. the exact chain-rule joint of every pair — gives
  90.0 % dev2 / 84.8 % test.  The gap S = 8 → S = 16 is therefore an upper bound on what *any* pairwise joint
  decision can recover within the 8-forward budget: the joint of (i, i+1) with one extra forward per pair IS S = 16.
- The A-series dependency diagnostic (dependency_aware_refinement.md): within a block, dependency is distance-1 and
  left-sided; the B-series §4 experiment (trust_aware_diffusion.md) found that forbidding adjacent pairs does not help
  the B-series, i.e. adjacent pairs are not obviously the ones that fail.
- Re-masking cannot repair what independent placement breaks (oracle 30 % repair, trust_aware_diffusion.md §3.2),
  so if the pair decision is the problem it has to be fixed at placement time — which is what this phase tests.
- Data: dev items 0–49 (e5 base's own greedy answers as reference, results/pure/dev_refs_e5.json) for the
  teacher-forced diagnostics; bench/prompts_gsm8k_dev2.json (150 fresh items) for decisions; GSM8K test for the
  final setting only.

## 2. Pairwise dependency and joint re-ranking without training (brief §2, §9)

Code: engine/bdiff/pairwise.py (`rerank_pair`), bench/diag_pairwise.py, sampler option `pair=` in
engine/bdiff/sampler2.py, `--pair/--pair-k` in bench/eval_trust.py.
Rules compared at every S = 8 step that places two tokens (i, j), candidates = top-k per position (k = 2 → 4 pairs,
k = 4 → 16 pairs):

| rule | score of pair (a, b) | extra forwards per step | status |
|---|---|---|---|
| A independent argmax | log p(a) + log p(b) (product of marginals — identical to independent argmax, verified: "indep_check" row) | 0 | existing sampler |
| B/C joint re-ranking, chain rule forward | log p(a) + log p(b \| a placed at i) | k | research ceiling |
| B/C joint re-ranking, symmetric | ½[log p(a) + log p(b \| a) + log p(b) + log p(a \| b)] | 2k | research ceiling |
| oracle | the reference pair when it is among the k × k candidates | – | ceiling of any re-ranker |
| D/E pairwise compatibility score / dependency-aware pair selection without extra forwards | learned head on hidden states (§3) | 0 | only if Gate 1 passes |

### 2.1 Teacher-forced pair diagnostic (dev 0–49, left-to-right S = 8, 2 368 two-token steps, lr1e-4 adapter)

Protocol: at every step that places two tokens (i, i+1) the step's marginals are re-ranked post hoc under each rule
and the chosen pair is scored against the reference pair (reference = the base model's own greedy answer).  Extra
forwards are counted per step (results/pure/diag_pairwise_lr1e4_left.json; 40 256 forwards, 15 min on an A100).

| | k = 2 (4 pairs) | k = 4 (16 pairs) |
|---|---|---|
| reference pair among the candidates (ceiling of any re-ranker) | 54.9 % | 59.0 % |
| steps where independent is wrong but a candidate pair is right ("fixable") | 7.2 % | 11.3 % |
| **independent argmax — pair exact / token acc** | **47.8 % / 54.2 %** | 47.8 / 54.2 |
| product of marginals (control; must equal independent) | 47.7 % (differs from argmax in 3 % of steps by bf16 ties) | 47.6 % |
| joint, chain rule forward (k extra forwards) | 50.9 % / 54.2 % (W→C 98, C→W 24) | 51.9 % / 54.8 % (126 / 29) |
| joint, symmetric (2k extra forwards) | 51.4 % / 54.7 % (99 / 13) | 52.6 % / 55.3 % (128 / 14) |
| oracle (reference pair if among candidates) | 54.9 % | 59.0 % |

Dependency between the two positions (k = 2): placing a1 vs a2 at i changes the argmax at j in **67 %** of the steps
(JSD between the two conditionals 0.64 bits) — the dependency is strong.  j's accuracy is 51.4 % from the marginal
and **75.2 % when the reference token is placed at i** (one extra forward).  Yet re-ranking recovers little, because
the problem is not choosing among j's marginal candidates: the reference token for j is in j's top-2 only 60 % of
the time when i is unknown, so in 45 % of the steps no candidate pair is right (k = 2), and the chain-rule joint
fixes about half of the 7 % that are fixable.  **Gate 1 (joint re-ranking beats independent argmax): passed only
marginally — +3.1 to +4.8 points of pair exact match with 2–8 extra forwards per step, against a re-ranking ceiling
of +7.2 / +11.3.**  The large conditional gain (51 → 75 % for j given the right i) is exactly what S = 16 already
buys with one forward per token.

### 2.2 Free generation with the chain-rule joint (research ceiling; extra forwards counted)

| S = 8, left-to-right, lr1e-4 adapter | dev-50 | **dev2 (150)** | forwards / 16-token block | tokens / forward | tok/s (A100, HF path) |
|---|---|---|---|---|---|
| independent argmax (existing) | 76 | **70.0** | 8 (+1 prefix) | 1.71 | 72 |
| joint, chain rule forward, k = 2 | 84 | **89.3** | 22 | 0.67 | 29.5 |
| joint, symmetric, k = 2 | 82 | – | 36 | 0.42 | 18 |
| joint, chain rule forward, k = 4 | 86 | – | 36 | 0.41 | 18 |
| reference: S = 16 left-to-right (one token per forward) | – | 90.0 | 16 | 0.90 | 56 (zeta) |
| reference: AR greedy | – | 92.0 | 1 per token | 1.00 | 52 |

**The hypothesis holds in its ceiling form: deciding the two tokens of a step jointly under the model's own
conditional recovers essentially the whole S = 8 → S = 16 gap (70.0 → 89.3 % on 150 fresh items; S = 16 = 90.0 %).**
The teacher-forced pair-exact gain (§2.1, +3–5 points) understates this because exact match against one reference
wording penalises paraphrases; what the joint choice removes are the *incoherent* pairs (two tokens that are each
plausible but not together), and those are what break arithmetic.  The cost is the problem: 2 extra forwards per
step make 22 forwards per block — more than S = 16 itself (Gate 3 fails for the chain-rule rule).  The practical
question for §3 is therefore whether the dependency term log p(b | a) − log p(b) can be predicted from the step's own
hidden states without the extra forward.

## 3. Decision on the Compatibility Head and Trajectory Distillation (brief §3–§5)

Gate 1 (candidates contain the reference pair): 55 % at k = 2, 59 % at k = 4 (§2.1; k up to 32 in §2.3).  Gate 2
(joint beats independent): yes — marginally on pair exact match (+3–5), decisively on GSM8K (70.0 → 89.3 % dev2).
Gate 3 (compute): the chain-rule joint needs 2 extra forwards per step (22 per block, 0.67 tokens/forward), which
is *more* than S = 16 — the only zero-forward route would be a learned compatibility head approximating
log p(b | a) − log p(b) from the step's hidden states; its data collector was written (distill/gen_pairs.py, collects
on the sampler's own trajectories with the model's own conditional as target) but **the head was not trained**: the
Phase B9 brief received at this point redirects the research away from token-selection fixes ("Pairwise Head …を既定の
解決策にしてはならない") toward denoising the whole future block, so Gates 3–4 and trajectory distillation are left
unexplored here.  The recorded facts carry over: the two tokens of a step are strongly dependent (67 % argmax change),
and conditioning fixes most of the S = 8 loss — a global method has to achieve the same coupling for all 16 positions
without sequential conditioning.

### 2.3 Top-k oracle diagnostic, k = 1 … 32 (brief v2 §2; S = 8 left-to-right, teacher-forced, 2 368 two-token steps on GSM8K dev 0–49, 3 528 on HumanEval 0–63)

Candidates = the step's own top-k at each of the two positions (results/pure/b8_candidates_*.jsonl keep the top-32 and
their log-probs for every step); "pair in candidates" = both reference tokens present = the ceiling of any re-ranker.

| k | GSM8K: token recall | GSM8K: reference pair among k×k | GSM8K: oracle pair exact | HumanEval: token recall | HumanEval: pair among k×k |
|---|---|---|---|---|---|
| 1 (independent argmax) | 54.3 % | 47.8 % | 47.8 % | 92.8 % | 91.4 % |
| 2 | 60.7 % | 55.0 % | 55.0 % | 94.0 % | 92.8 % |
| 4 | 65.2 % | 58.9 % | 58.9 % | 94.8 % | 93.7 % |
| 8 | 69.9 % | 63.0 % | 63.0 % | 95.5 % | 94.4 % |
| 16 | 74.6 % | 66.9 % | 66.9 % | 96.3 % | 95.2 % |
| 32 | 79.5 % | 71.7 % | 71.7 % | 96.9 % | 95.6 % |
| chain-rule joint, k = 2 / 4 (2 / 4 extra forwards) | – | – | 50.6 / 51.6 % | – | 92.3 % (k = 2) |

By distance of the position from the block's correct left context (GSM8K; d = 0: the position right after the longest
correct visible prefix, d = 1: the second token of the pair, d ≥ 2: beyond the first divergence):

| d | n | independent token accuracy | reference in top-2 | in top-8 |
|---|---|---|---|---|
| 0 | 1 195 | **94.5 %** | 99.3 % | 100 % |
| 1 | 1 358 | **74.6 %** | 82.6 % | 89.4 % |
| 2 | 228 | 23.3 % | 36.0 % | 58.8 % |
| 3–7 | ~1 000 | 20–23 % | 27–30 % | 39–50 % |
| ≥ 8 | 934 | 16.5 % | 21.1 % | 34.7 % |

By step: independent pair exact 75 % at step 1, 61 / 55 / 49 / 43 / 35 / 33 / 30 % at steps 2–8; the pair-in-top-2×2
rate follows (88 → 32 %).  HumanEval (teacher-forced code blocks are near-deterministic): the same shape, d = 0 99.5 %,
d = 1 97.2 %, d ≥ 2 13–41 %.

**Reading (Gate 1 of brief v2).**  The reference pair is in the top-2×2 set only 55 % of the time and even top-32×32
reaches 72 % — but that number is dominated by positions *after the first divergence from the reference wording*
(d ≥ 2: 16–23 % accuracy, 35–59 % in top-8), where the reference token is simply not what the model's own context
calls for.  Where the context is still the reference's, the first token of the pair is 94.5 % right and the second
74.6 % right with 82.6 % in its top-2: at the point of placement the problem is neither candidate generation (99 /
83 % containment) nor dramatically combination selection (≤ 8 points available at d = 1).  **The block-level loss
comes from the cascade**: once one token of the pair deviates, every later position is conditioned on a diverged
block and the exact-match metric is lost — which is why teacher-forced code blocks look near-perfect (91 % pair
exact) while free generation of code at S = 8 passes 8.5 % of HumanEval, and why the chain-rule joint is worth +3
points teacher-forced but +19 points of GSM8K accuracy in free generation (§2.2): it prevents the incoherent pair
that starts the cascade.  Confidence order instead of left-to-right: pair exact 46.0 %, pair-in-top-2×2 53.5 %, the
dependency between the two chosen positions decays with their distance (JSD 0.64 bits at distance 1, 0.37 at 2,
0.26 at 3, 0.16 at 4, ≤ 0.1 beyond 6).
