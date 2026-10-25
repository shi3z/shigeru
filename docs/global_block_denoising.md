# Phase B9 — Global Block Denoising (K = 16, few parallel denoising steps)

Brief: "SHIGERU Phase B9 — Global Block Denoising" (2026-10-25).  Goal: generate the next 16 (–32) tokens as one joint
object with few parallel denoising steps — not AR acceleration, not left-to-right filling, no AR verification, no
sequential 2+2+2 initialisation as the main method, no pairwise head; quality per unit of total compute is the metric;
the 4B's trained language ability is the engine; eventual target 8–16 GB VRAM.

## 1. Research question and what the earlier phases fix as facts

- B-series K = 16 (block-bidirectional, mask-only loss, lr1e-4 LoRA on the e5 base): S = 16 left-to-right 84.8 % GSM8K
  test / 50.6 % HumanEval; S = 8 72.9 % / 8.5 %; confidence order worse than left-to-right (78.6 / 29.9 at S = 16).
- Re-masking cannot repair (trust_aware_diffusion.md: oracle 30 % repair); self-correction training collapses free
  generation (B2/B3/B8, A-series config D); the model copies visible tokens (visible-token signals AUROC 0.5).
- The two tokens placed per S = 8 step are strongly dependent and deciding them jointly recovers the S = 16 quality,
  but only with sequential conditioning forwards (pairwise_joint_diffusion.md).
- Hence the hypothesis here: what is missing is the ability to *keep and improve the whole block's coherence during
  generation* rather than to pick the next token.  Monotonic unmasking never revisits a position; the information
  about still-masked positions that each step produces is thrown away except for the positions placed.

## 2. B9-A — generation trajectory analysis (bench/diag_trajectory.py)

Protocol: the K = 16 sampler (S = 16, left-to-right; also S = 8 and confidence order) is run on dev items 0–49
(teacher-forced from the reference prefix: targets REF = the base model's own greedy answer and FINAL = the block the
run itself ends with; and in free mode: FINAL and the GSM8K answer).  At every step, for all 16 positions: top-8
candidates with log-probs, entropy, max-prob, mask pattern, placed positions, final-layer hidden states (saved).
Two kinds of statements are kept apart: *diagnostic* ones that compare with FINAL/REF (known only after the run) and
*inference-time* ones that use only the step's own outputs (entropy, confidence, step-to-step change).
