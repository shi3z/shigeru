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

S = 16 left-to-right, teacher-forced; for every still-masked position, JSD between its distribution before and after
the previous step's placement, by distance to the placed position (results/pure/traj_s16_left_tf.json):

| distance | 1 | 2 | 3 | 4 | 5 | 6 | 7 | ≥ 8 |
|---|---|---|---|---|---|---|---|---|
| JSD (bits) | 0.152 | 0.171 | 0.147 | 0.119 | 0.100 | 0.086 | 0.072 | 0.049 |
| argmax changed | 19 % | 27 % | 33 % | 33 % | 34 % | 34 % | 31 % | 30 % |

The influence of one placed token on the rest of the block is strongest over the next 2–3 positions and decays
slowly (still 0.05 bits eight or more positions away); the far positions change their argmax as often as the near
ones because their distributions are flat (entropy 2–2.5 nats), not because they are informed.  The pairwise
diagnostic (pairwise_joint_diffusion.md §2.3) gives the same picture for the conditional on a single candidate
token: 0.64 bits at distance 1, 0.37 at 2, 0.26 at 3, 0.16 at 4, ≤ 0.1 beyond 6.  So the block is coupled mostly
locally, and a global step that moves all positions at once mainly redistributes the flat tails of the far positions.

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

### 2.5 Step budget and the drift from the reference (teacher-forced, left-to-right; Hamming to REF of the final block)

| S | forwards / block | final Hamming to REF (of 16) | positions never matching REF | masked top-1 recall of REF at the first / last step |
|---|---|---|---|---|
| 16 | 16 | 5.9 | 42 % | 34 % / 46 % |
| 8 | 8 | 7.3 | 52 % | 34 % / 34 % |
| 4 | 4 | 9.4 | 63 % | 34 % / 20 % |

The first forward is the same in all three (34 % of masked positions already carry the reference token as argmax);
what differs is how many tokens are committed per forward from that flat start.  At S = 4 the still-masked positions'
agreement with the reference *falls* with each step (34 → 20 %): committing four jointly-incoherent tokens drags the
rest of the block away from the reference faster than the added context pulls it back.  The budget S is, for this
model, the number of times the block is re-read with more committed context — and nothing in the intermediate state
(§2.4) substitutes for those re-reads.

## 3. B9-B — joint consistency of the block (brief §3)

What the per-token cross-entropy training buys and what it does not, measured on the same 296 teacher-forced blocks
(S = 16 left-to-right unless stated):

| quantity | independent argmax block (first forward, all 16 masked) | block after 16 monotonic steps |
|---|---|---|
| position-wise accuracy vs FINAL / vs REF | 34.2 % / 34.4 % | 100 % / 63 % |
| block exact match vs REF | 0 % | 38.5 % (S = 8: 20.9 %, S = 4: 4.7 %) |
| adjacent duplicate tokens per block | **6.85** | 0.11 (free generation 0.82) |
| 4-gram repetition inside the block | 0.113 | 0.002 (free 0.035) |
| log p(FINAL token) per masked position | −2.59 | −0.11 (last step) |

- **Pairwise consistency.**  The all-mask marginals are individually sensible (the reference token is in the top-8
  of 75 % of the positions) but jointly incoherent: 6.85 of 15 adjacent pairs repeat a token ("by by by", "1111111"),
  the signature of independent sampling from positionally-smeared marginals.  The pairwise diagnostic
  (pairwise_joint_diffusion.md §2) quantified the coupling: placing one token changes the neighbour's argmax 67 % of
  the time; the chain-rule joint of a pair removes the incoherence and recovers S = 16 quality — with extra forwards.
- **Long-range dependency.**  §2.3: coupling decays with distance (JSD 0.15–0.17 bits at distance 1–2, 0.05 at ≥ 8)
  — the block is a chain of local dependencies, not a set of independent slots and not a globally rigid object.
- **Contradictory predictions.**  Confidence order places the two most confident positions, which are adjacent in
  35–45 % of the steps and then often the *same* token; left-to-right with the pair's second token conditioned on an
  unknown first token is 74.6 % right vs 94.5 % for the first (pairwise §2.3).
- **Can the 16-step result be explained by the step-4 / step-8 marginals?**  Log-probability of the final block under
  the marginals of step s (masked positions): −2.59 (s = 1), −2.22 (4), −1.81 (8), −1.13 (12), −0.11 (16) nats per
  token; top-1 agreement 34 / 41 / 52 / 70 / 100 %.  At step 8 the marginals put the final token first at half the
  remaining positions and somewhere in the top-8 at 86 %: the block is "half explained" at half the budget, and the
  other half is not latent in the state (§2.4) — it is created by the later forwards.
- **Joint probability.**  No estimate of the block's joint distribution is available from the model without
  sequential conditioning (the product of marginals is *not* it — §2's duplicates are exactly what the product of
  marginals produces).  All statements above are about marginals and about the chain-rule joint of pairs.

**Gate 1 verdict (global information in the intermediate state):** present as a candidate set, absent as a
decision; no hidden-state probe extracts more than the LM head.

## 4. B9-C — global refinement without training (bench/eval_global.py, engine/bdiff/global_sampler.py)

All samplers spend exactly S forwards per 16-token block (plus the prefix pass), never read a reference, never call
the AR path.  dev items 0–49: teacher-forced (296 reference blocks: final token accuracy, block exact match,
per-step W→C / C→W of the current argmax block) and free generation (GSM8K accuracy, repetition).  lr1e-4 adapter.

| sampler | what moves per step | S = 4: TF token acc / GSM8K | S = 8 | S = 16 |
|---|---|---|---|---|
| C1 monotonic unmasking, left-to-right (baseline) | K/S positions placed, never revisited | 41.3 % / **32 %** | 54.2 % / **76 %** | 63.3 % / **92 %** |
| C2 full-block synchronous refinement (Jacobi on the all-visible block) | 6–9 of 16 tokens rewritten every step | 9.7 % / 0 % | 7.1 % / 0 % | 7.0 % / 0 % |
| C2r non-monotonic confidence re-masking (any token can return to [MASK]; visible tokens kept, masks filled) | 2–7 changes / step, 0.6–1.9 W→C vs 0.3–0.8 C→W per step | 42.9 % / 32 % | 52.5 % / 62 % | 58.3 % / 74 % |
| C3 distribution-level (soft token embeddings for unplaced positions) | all 16 distributions updated | 5.9 % / 0 % | 4.5 % / 0 % | 5.2 % / 0 % |
| C4 annealed soft (T 1 → 0.2, α 0 → 1) | all 16 | 6.0 % / 0 % | 4.2 % / 0 % | 4.0 % / 0 % |
| C4 annealed soft, mild (T 1 → 0.5, α 0 → 0.5) | all 16 | 6.0 % / 0 % | 4.2 % / 0 % | 4.0 % / 0 % |
| cosine mask schedule for C2r | – | 36.2 % / 2 % | 11.1 % / 2 % | 6.7 % / 10 % |

Per-step dynamics (teacher-forced, S = 8):
- **C2 sync**: step 1 = the independent-argmax block (34 % token accuracy); the forward on the fully visible
  self-generated block then rewrites 9.0 tokens (step 2), 7.8, 7.4, … 6.0 per step and accuracy *falls* 34 → 16 → 11 →
  7 %; W→C 0.95 vs C→W 3.95 at step 2; free generation degenerates into repetition (4-gram repetition 0.87, every
  output hits the length budget).  This is not self-copying: the model trained on mask-only corruption does not
  treat an all-visible block as given; it keeps re-predicting it from a context that is itself wrong, and the
  predictions drift to the degenerate attractor the first forward already showed ("by by by").  Full-block
  synchronous refinement needs a model trained for it.
- **C2r non-monotonic re-masking** behaves like a confidence-order monotonic sampler with small revisiting: W→C exceeds
  C→W at every step (net +1.0, +0.7, +0.5, +0.3, … per step) but the total is below left-to-right monotonic
  (S = 8: 52.5 vs 54.2 % token accuracy, 62 vs 76 % GSM8K) because confidence order itself is the worse order
  (§2, trust_aware_diffusion.md).  With the original "re-forward the all-visible block at the last step" variant it
  collapsed like C2 (token accuracy 42 → 11 % at the last step: 13 of 16 tokens rewritten) — the same lesson.
- **C3 / C4 soft inputs**: the model has never seen a probability-weighted embedding mixture; accuracy drops from the
  34 % of the first (all-mask) forward to 4–6 % and generation is pure repetition at every temperature / mixing
  schedule tried.  Recorded as "fails without training", not as a disproof (brief §5).

**Gate 2 (global correction without training): failed.**  No sampler that updates several positions at once beats
monotonic left-to-right unmasking at any budget; the two that do not collapse (C2r) are at best confidence-order
monotonic sampling in disguise.  The frozen mask-only denoiser supports exactly one operation reliably: predicting
masked positions from a context of committed tokens it has not produced itself under full visibility.

## 5. B9-D — denoising dynamics: is the block converging or being filled left to right?

Answered by §2.1/§2.5 and the table above.  Under monotonic left-to-right unmasking the quantities move smoothly
(masked entropy 2.5 → 0.2 nats, masked top-1 recall of the final token 34 → 100 %, hidden-state relative displacement
0.22 → 0.62 per step at the still-masked positions), but the *information* arrives with the committed left context:
agreement with the reference at the still-masked positions stays flat (34–46 %) until the position itself is reached,
and the first-forward argmax block is a degenerate, repetitive object (6.8 duplicates / 16).  So the honest answer is
"holes are filled left to right, and each filled hole sharpens its right neighbours by 0.1–0.17 bits" — not "the whole
sentence converges".  Correction rate is 0 by construction in C1; in C2r, W→C ≈ 1.5–2× C→W but the total gain is eaten
by the worse order; in C2 and C3/C4, C→W dominates and the block diverges.

## 6. Evaluation (brief §8)

No global-refinement sampler reached the stage of a held-out or test evaluation: every variant without training is
below the monotonic baseline on dev-50 (§4), so the comparison table of the brief reduces to the existing rows
(AR 88.2 % / 68.9 %, A-series K = 4 84.2 % / 40.2 %, B-series S = 16 left 84.8 % / 50.6 %, S = 8 left 72.9 % / 8.5 %
GSM8K test / HumanEval; trust_aware_diffusion.md §7).  The trained successor of this line — the distilled parallel
student of Phase B10, which reaches 81.3 % at S = 8 on dev2 — is evaluated in global_consistency_distillation.md.

## 7. Compute

Diagnostics of this phase: trajectory runs 5 × 3–5 min, probe 2 min, global-sampler sweep 79 min + variants 3 × 15
min, all on single A100s (peak 7.9 GB inference, 404 MB of saved hidden states): about 3 GPU-hours.  No training.

## 8. Failed hypotheses (recorded so they are not repeated)

1. Full-block synchronous (Jacobi) refinement of the all-visible block with the mask-only denoiser: rewrites 6–9 of 16
   tokens per step, accuracy decays 34 → 7 %, output is repetition (0.87).  The model is not "self-copying" — it does
   not trust a fully visible self-generated block and has no attractor other than the degenerate one.
2. Soft-token (probability-weighted embedding) inputs, with or without temperature/mixing annealing: 4–6 % token
   accuracy, GSM8K 0 % — out of the training distribution; a soft state needs its own noise process and objective
   (brief §4 of B10), which was not attempted here.
3. Non-monotonic confidence re-masking (any token may return to [MASK]): converges and nets W→C > C→W per step, but
   lands below left-to-right monotonic unmasking (62 vs 76 % at S = 8) because confidence order is the worse order.
4. A forward on a fully visible block as the *last* step of any sampler: rewrites 13 of 16 tokens (first version of
   C2r) — the last forward must still see masks.
5. Hidden-state probe for the eventual token: no gain over the LM head (§2.4).

## 9. Go / No-Go

| gate | verdict | evidence |
|---|---|---|
| 1 global information in the intermediate state | weak yes | the final token is in the top-8 of 75 % of masked positions at step 1 and 86 % at step 8; top-1 only 34 / 52 %; no probe does better than the LM head |
| 2 global correction without training | **no** | §4: every multi-position update is below monotonic left-to-right; sync and soft collapse |
| 3 training feasibility | deferred to B10 | the frozen denoiser supports one operation (predict masks from committed context); teaching it to resolve dependencies in one forward is B10's object, and its first result is positive (S = 8 dev2 70.0 → 81.3 %) |
| 4 compute efficiency | not reached here | – |

## 10. Next architecture (proposal, from the measurements)

The measurements point away from "refine all 16 positions of a committed block" with the current model and toward
making each parallel forward *resolve more of the dependency it currently leaves open*: (i) distil the resolved
conditionals of the chain-rule teacher into the one-forward student (B10 Stage 2, positive), extending from the two
placed positions to the whole block via the next-state transition term; (ii) only after that, re-introduce
revisiting (C2r-style re-masking) on a student whose predictions at visible positions are trained to be meaningful,
with a noise process that includes wrong visible tokens at low rates and identity targets weighted up — the B2
collapse came from high corruption rates and visible-correction losses on a model that was never asked to keep its
own outputs; (iii) a soft-state denoiser is a separate research line requiring its own forward process (Gaussian or
Dirichlet noise on token distributions) and should not be bolted onto the mask-only model.
