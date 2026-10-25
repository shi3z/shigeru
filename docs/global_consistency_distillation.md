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
