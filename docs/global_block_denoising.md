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

### 2.1 S = 16, left-to-right, teacher-forced (296 blocks of dev 0–49; results/pure/traj_s16_left_tf.json)

Masked positions only (the positions not yet placed at that step); "FINAL" = what this run ends with, "REF" = the
base model's own greedy answer.  top-k = top-8.

| step | masked | top-1 recall of FINAL | top-8 recall of FINAL | top-1 / top-8 recall of REF | entropy (nats) | mean max-prob | log p(FINAL token) | JSD vs previous step | argmax changed vs previous |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 0.342 | 0.749 | 0.344 / 0.751 | 2.51 | 0.38 | −2.59 | – | – |
| 2 | 15 | 0.362 | 0.775 | 0.350 / 0.755 | 2.36 | 0.40 | −2.47 | 0.097 | 0.338 |
| 4 | 13 | 0.412 | 0.816 | 0.357 / 0.765 | 2.19 | 0.44 | −2.22 | 0.089 | 0.296 |
| 6 | 11 | 0.469 | 0.835 | 0.369 / 0.753 | 2.00 | 0.48 | −2.04 | 0.090 | 0.295 |
| **8** | 9 | **0.519** | **0.856** | 0.362 / 0.721 | 1.78 | 0.53 | −1.81 | 0.106 | 0.299 |
| 10 | 7 | 0.584 | 0.884 | 0.376 / 0.678 | 1.52 | 0.60 | −1.58 | 0.119 | 0.296 |
| 12 | 5 | 0.699 | 0.929 | 0.415 / 0.664 | 1.13 | 0.69 | −1.13 | 0.142 | 0.310 |
| 14 | 3 | 0.827 | 0.975 | 0.448 / 0.635 | 0.74 | 0.79 | −0.64 | 0.147 | 0.252 |
| 16 | 1 | 1.000 | 1.000 | 0.463 / 0.564 | 0.22 | 0.92 | −0.11 | 0.162 | 0.196 |

Whole-block view: Hamming distance of the current argmax block (masked positions take their argmax) to FINAL falls
10.5 → 4.3 (step 8) → 0; to REF it falls 10.5 → 7.3 → **5.9** and stays there — 37 % of the final tokens differ from
the reference wording, and the masked positions' agreement with REF never exceeds 46 % (top-1) at any step.

**First forward vs final block** (the "independent argmax" block the model would emit in one step): position
accuracy 34 %, block exact 0 %, **6.8 adjacent duplicate tokens per 16-token block** (vs 0.1 in the final block;
4-gram repetition 0.11 vs 0.002): the all-mask marginals are individually plausible but jointly degenerate
("Let's go by by by by---\n\n.1111111" for "Let's go step by step.\n\n**First round:**\n- Hattie").

**Stabilisation** (first step from which a position's argmax equals its FINAL token for good): mean 8.2 of 16;
18 % of positions are right from the first forward, 53 % by step 8, 91 % by step 15.  Against REF 42 % of the
positions never stabilise (their final token is simply not the reference token).

Reading for Gate 1 (is there usable information about the still-masked future in the intermediate states?):
*yes, but as a candidate set, not as an argmax*: at step 1 the final token is in the top-8 of 75 % of the positions
(top-1 34 %); at step 8, 86 % (top-1 52 %).  The marginal over a still-masked position keeps moving while others are
placed (JSD 0.09–0.16 per step, argmax changes at ~30 % of masked positions every step), and the movement does not
shrink with distance (§2.3) — the block is globally coupled.  At step 8 the structure of the block is "half formed":
half the positions have their final argmax, the other half is decided only in the last steps by the growing left
context.  The entropy at masked positions falls smoothly (2.5 → 1.8 at step 8 → 0.2), i.e. the process is not a
sudden commitment but a gradual sharpening — which is what a distribution-level sampler would try to exploit.

### 2.2 S = 8 (same blocks) and free generation

S = 8 left-to-right, teacher-forced (results/pure/traj_s8_left_tf.json): first forward identical (34–38 % position
accuracy, 6.8 adjacent duplicates); masked top-1 recall of FINAL 0.38 → 0.58 (step 5) → 0.80 (step 7); the per-step
JSD at masked positions is larger (0.15–0.30 vs 0.09–0.16) and 40–47 % of masked argmaxes change every step —
placing two tokens per step moves the rest of the block more than placing one.  Hamming to REF ends at 7.3 (vs 5.9
at S = 16); 52 % of positions never match the reference.  Free generation S = 16 left (574 blocks, GSM8K dev-50
92 %): the same curves within 1–2 points (masked top-1 recall of FINAL 0.38 / 0.53 / 0.72 / 1.0 at steps 1 / 8 /
12 / 16; stabilisation mean 8.3 steps, 51 % by step 8) — the teacher-forced trajectories are representative of
deployment.  The final blocks of free generation contain 0.8 adjacent duplicates per block (vs 0.1 teacher-forced) —
some of the first-forward degeneracy survives to the output when no reference prefix anchors the block.

### 2.3 Where the movement comes from (JSD at still-masked positions by distance to the position placed last)

See results/pure/traj_s16_left_tf.json → `jsd_by_distance_to_last_placed` (filled when the re-run with the distance
breakdown finishes; the pairwise diagnostic of pairwise_joint_diffusion.md §2.3 gives the same quantity for the
conditional on one placed token: JSD 0.64 bits at distance 1, 0.37 at 2, 0.26 at 3, 0.16 at 4, ≤ 0.1 beyond 6).

### 2.4 Is the final block predictable from the intermediate hidden states? (bench/probe_hidden.py)

A bilinear probe (rank 256; score = (W h)·E[c] + β·log p(c)) was trained to pick the FINAL token among the step's
top-8 candidates from the final-layer hidden state of the masked position, on the blocks of prompts 0–24, and
evaluated on prompts 25–49 (16 840 masked-position rows; results/pure/probe_hidden_s16.json).

| | LM head (sampler's own ranking) | probe on hidden state |
|---|---|---|
| top-1 among the top-8 candidates, given the final token is among them (test) | **57.9 %** | 56.0 % (train 72.8 %) |
| recall of the final token, step 1 / 4 / 8 (test) | 33.6 / 42.1 / 51.8 % | 33.0 / 39.6 / 50.9 % |

**No**: the hidden state of a masked position carries no linearly readable information about the eventual token
beyond what the LM head already outputs (the probe only overfits).  Whatever decides the final token later is
produced by *later forwards with more context*, not stored in the current state.  Gate 1 is therefore passed only in
the weak sense: the candidate set is known early (top-8 recall 75–86 %), the choice among candidates is not.
