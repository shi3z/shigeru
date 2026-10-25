# Phase B10 — Global Consistency Distillation (K = 16, ≤ 8 forwards per block, no sequential conditioning at inference)

Brief: "SHIGERU Phase B10 — Global Consistency Distillation" (2026-10-25).  Hypothesis: the B-series can produce the
*candidates* for the future tokens but cannot resolve the dependencies among them in one forward; a teacher that
resolves them with extra forwards (the B8 chain-rule joint) can be distilled into a student that resolves them in one.

## 1. What the B8 joint teacher does and why it works (analysis of pairwise_joint_diffusion.md)

- Independent argmax of the two positions placed per S = 8 step: pair exact 47.8 % (teacher-forced), GSM8K dev2 70.0 %.
- Chain-rule joint of the pair (log p(a) + log p(b | a), 2 extra forwards): pair exact 50.6 %, **GSM8K dev2 89.3 %**
  (S = 16 = 90.0 %); 22 forwards per block.
- The resolved dependency is large: conditioning j on the chosen a changes j's argmax in 67 % of steps (JSD 0.64 bits
  at distance 1, 0.37 at 2, 0.26 at 3); j's accuracy 51 → 75 % given the right a.  Teacher-forced exact match
  understates the effect because the reference wording is one of several; what the joint removes are *incoherent*
  pairs (adjacent duplicates, broken numbers) — the all-mask marginals produce 6.8 adjacent duplicates per 16-token
  block (global_block_denoising.md §3) and any such pair starts a cascade (pairwise §2.3: accuracy 94.5 % at the
  first position after the correct context, 74.6 % at the second, 16–23 % after the first divergence).
- A hidden-state probe does not pick the eventual token better than the LM head (global_block_denoising.md §2.4): the
  dependency is not latent in the state, it is produced by the extra forward with the committed token — so the
  student has to *learn* to output the conditional in the same forward, which is the object of this phase.

## 2. Teacher data (B10-A; distill/gen_pairs.py)

States of the student's own K = 16, S = 8, left-to-right trajectories on 800 GSM8K-train prompts
(bench/prompts_gsm8k_pairtrain.json — disjoint from dev, dev2 and every training file) and 300 MBPP prompts
(bench/prompts_mbpp_pairtrain.json; HumanEval is never used for training).  At every two-token step: the block state
(mask pattern + tokens), the two positions (i, j), their top-4 candidates with log-probs, the 4 × 4 matrix
δ(a, b) = log p(b | a) − log p(b) from 4 extra teacher forwards, the hidden states of the two positions, the step, and —
per prompt — the prompt ids and the generated tokens (the trajectory *follows the teacher's joint choice*, so later
states and the final block are the teacher's; the GSM8K correctness of each teacher output is recorded, so imitation
and task accuracy are kept apart).  The pairwise conditional logits of the teacher are recomputed online during
training (one forward with a* placed) rather than stored.

## 3. Global consistency objective (B10-B) — what is actually trained

Stage 1 (no training; the gap to close).  On the teacher-followed trajectories 18 % of the two-token states are
"hard": the chain-rule joint pair differs from the independent-argmax pair (pilot data, 8 000 states; 19 % in the
teacher-forced diagnostic).  On those states the student's independent argmax matches the teacher's pair 0 % of the
time by definition and its second token 8 %; on the other 82 % it already matches.  The free-generation effect of
resolving the 18 % is the +19 points of B8 (dev2 70.0 → 89.3 %).

Stage 2 — **conditional distribution distillation** (distill/train_consist.py; loss A of the brief).  For a state
with positions (i, j) to be placed, the student's single forward (both masked) is trained so that
  position i → CE to a* (the token the teacher's joint picks),
  position j → KL( p_T(· | a* placed) ‖ p_S(·) ): the *conditional* the teacher reaches only after committing i,
  other masked positions → CE to the block the teacher trajectory ends with (λ_mask), or, with the transition term
  (loss C, `--lookahead`), KL( p_T at the next state (a*, b* placed) ‖ p_S now ) — the student predicts at step s what
  the teacher believes at step s + 1 for *every* still-masked position.
Hard states are up-weighted in sampling (×4).  Limit stated in advance: a per-position KL to a conditional makes the
student's marginal at j equal the teacher's conditional *given the teacher's a\**; it does not make the student
represent a joint over (i, j) — if the student's own choice at i deviates from a*, the learned j-distribution is
conditioned on the wrong token.  Whether that matters is measured, not assumed (free-generation GSM8K is the metric).

## 4. Student architecture (B10-C)

The student is the frozen B-series model (e5 base + lr1e-4 mask-only LoRA, merged) plus a new LoRA r = 64 on all
attention and MLP projections (132 M trainable parameters, 3.2 %); block-bidirectional attention, prefix cache, no
time embedding (the B-series found it harmful), one forward per denoising step, K/S positions committed per step by
the existing left-to-right rule; no pairwise re-evaluation at inference, no AR path.  Teacher and student share the
GPU during training (2 × 8.1 GB weights + LoRA state; peak 20.7 GB).  Full fine-tuning and a new global denoiser are
deliberately not started (brief §4, §12).

## 5. Distillation results (Stage 2)

### 5.1 Pilot and the first two variants (12 000 teacher states from 286 GSM8K-train prompts, 1 000 steps × 8, lr 1e-4, LoRA r 64 on top of the merged B-series model; ~25 min on one A100, peak 20.7 GB)

Held-out = states of 7 prompts never seen in training.  "hard" = states where the teacher's joint pair differs from
the independent argmax (18 % of states); "easy" = the rest.

| run | hard-state pair accuracy (student's one forward = teacher's pair) | easy-state pair accuracy | GSM8K dev-50, S = 8 | **GSM8K dev2 (150), S = 8** | forwards / block |
|---|---|---|---|---|---|
| B-series student before distillation (baseline) | 2.9 % | 100 % | 76 % | **70.0 %** | 8 |
| pilot: 400 steps, 8 000 states, no hard up-weighting | (not measured) | – | 76 % | – | 8 |
| **A: pair-KL (CE on a*, KL to p_T(·\|a*) at j), hard ×4** | **47.6 %** | 97.2 % | **82 %** | **81.3 %** (+28 / −11 items vs baseline, McNemar p = 0.0095) | 8 |
| B: A + look-ahead transition KL on the other masked positions | 49.5 % | 98.3 % | 80 % | 79.3 % (+10 items vs baseline) | 8 |
| reference: B8 joint teacher (2 extra forwards per step) | 100 % | 100 % | 84 % | 89.3 % | 22 |
| reference: S = 16 left-to-right, same adapter | – | – | 92 % | 90.0 % | 16 |

**Gate 1 (teacher gap shrinks): yes** — the student reproduces the teacher's resolved pair in half of the states
where it previously never did, without any extra forward, after 25 minutes of LoRA training on 12 k states.
**Gate 3 (free generation): yes on dev2** — S = 8 GSM8K 70.0 → 81.3 % on 150 fresh items (p = 0.01), the first
S = 8 result above the brief's 80 % short-term target; the teacher's 89.3 % marks how much is left.  S = 16 with the
same student, HumanEval, variant B and the full-data run follow in §5.2.

### 5.2 Student A at other budgets, on code, and the ablations

| | S = 8 | S = 16 | HumanEval S = 8 | HumanEval S = 16 |
|---|---|---|---|---|
| B-series baseline (lr1e-4 adapter) | dev2 70.0 % | dev2 90.0 % | 8.5 % | 50.6 % |
| **student A** (12 k GSM8K states, 1 000 steps) | **81.3 %** | 87.3 % (−4 items) | 10.4 % | **54.9 %** |
| student B (A + look-ahead) | 79.3 % | – | – | – |
| student D (hard CE labels instead of KL; held-out hard-pair 48.6 %) | __D150__ | – | – | – |
| student C (all 65 k GSM8K states, 2 000 steps) | __C150__ | __C16__ | – | – |
| student E (65 k GSM8K + 11 k MBPP states, 3 000 steps) | __E150__ | __E16__ | __EH8__ | __EH16__ |

The S = 16 cost of student A is small and within noise (−4 of 150), and its HumanEval at S = 16 is higher
(54.9 vs 50.6 %); at S = 8 code stays near 10 % with GSM8K-only teacher states — the code cascade (pairwise §2.3) is not
touched by a student that has only seen arithmetic states, hence variant E with MBPP teacher data.
