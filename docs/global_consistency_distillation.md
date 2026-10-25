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
| student D (hard CE labels instead of KL; held-out hard-pair 48.6 %) | 82.0 % | – | – | – |
| student C (all 65 k GSM8K states, 2 000 steps; hard-pair 48.3 %) | 77.3 % | 87.3 % | – | – |
| student A' (A with end-of-text pairs excluded from the states; hard-pair 46 %) | **82.7 %** (0 premature EOS) | – | – | – |
| student E (65 k GSM8K + 11 k MBPP states, 3 000 steps; hard-pair 40.7 %) | 80.0 % (0 premature EOS) | 88.7 % | **0.0 %** (132 / 164 IndentationError) | 40.9 % (baseline 50.6, A 54.9) |

The S = 16 cost of student A is small and within noise (−4 of 150), and its HumanEval at S = 16 is higher
(54.9 vs 50.6 %); at S = 8 code stays near 10 % with GSM8K-only teacher states — the code cascade (pairwise §2.3) is not
touched by a student that has only seen arithmetic states, hence variant E with MBPP teacher data.

### 5.3 Quality per forward (dev2, 150 items, left-to-right; student A vs the undistilled adapter)

| S (forwards per 16-token block) | tokens / forward | baseline | student A | Δ items |
|---|---|---|---|---|
| 4 | 3.1 | 27.3 % | **46.0 %** | +28 |
| 8 | 1.71 | 70.0 % | **81.3 %** | +17 |
| 12 | 1.18 | **85.3 %** | 83.3 % | −3 |
| 16 | 0.90 | **90.0 %** | 87.3 % | −4 |
| reference: chain-rule joint at S = 8 (22 forwards) | 0.67 | 89.3 % | – | |

The student is specialised to the states it was trained on — two tokens committed per forward — and helps most where
the independence assumption costs most (S = 4: +19 points, S = 8: +11); at one token per forward or close to it there
is nothing to resolve and the extra LoRA costs 2–3 points.  A deployment therefore switches adapters with the budget
(or trains one student on mixed S), which is a configuration question, not a research one.

### 5.4 GSM8K test (1 319; S = 8 left-to-right fixed on dev2 before the run)

| decoder, 8 forwards per 16-token block | GSM8K test | vs baseline (paired) | tokens / forward | tok/s (A100, HF path) | premature EOS / length-budget hits |
|---|---|---|---|---|---|
| B-series S = 8 (undistilled adapter; trust_aware_diffusion.md §7) | 72.9 % | – | 1.72 | 77.6 | 4 / 100 |
| **student A (conditional distillation, 12 k states, 25 min)** | **77.1 %** (1 017 / 1 319) | **+169 / −113 items, McNemar p = 0.001** | 1.71 | 75.5 | 35 / 54 |
| student A' (A with EOS pairs excluded) | 77.3 % (1 019 / 1 319) | +162 / −104 vs baseline, p < 0.001 | 1.71 | 68.3 | 18 / 48 |
| student E (GSM8K + MBPP states, 3 000 steps) | **78.4 %** (1 034 / 1 319) | +185 / −112 vs baseline, p < 0.0001; +135 / −118 vs A, p = 0.31 | 1.71 | 73.6 | 4 / 61 |
| reference: B8 joint teacher (22 forwards per block) | 80.7 % (1 065 / 1 319) | | 0.67 | 29.8 | 2 / 83 |
| reference: S = 16 left-to-right, undistilled | 84.8 % | | 0.90 | 56 (zeta) | 0 / 67 |

The test gain is +4.2 points (significant), smaller than the +11.3 on dev2 — the 150-item set was on the favourable
side for everything: the teacher itself is 89.3 % on dev2 but **80.7 % on test** — so the students recover about half
(A) to two thirds (E, 78.4 %) of the teacher's test-set gain over the baseline (72.9 → 80.7), and the teacher does not
reach the S = 16 result (84.8 %) on test either.  Below the 80 % short-term target on the test set.  The student ends 35 answers early (premature EOS) where
the baseline ended 4: the distilled pair decisions include end-of-answer pairs the teacher chose on its own
trajectories; this is a visible failure mode to address in the next student (e.g. excluding EOS pairs from the
distillation set), not a tuning knob turned on the test set.

## 6. Global refinement results (B10-D, methods C–F of the brief)

Methods C (8-step global synchronous refinement) and D (8-step distribution refinement) were measured without
training in Phase B9 (global_block_denoising.md §4): both collapse (GSM8K 0 %, repetition 0.87–0.94) and were not
re-tried with the distilled student — the student changes what one forward predicts, not the fact that the frozen
backbone has never seen soft inputs or fully visible self-generated blocks.  Method E (8-step distilled denoiser) is
the student above (dev2 81.3 %, test 77.1 %); method F (4-step distilled) is student A at S = 4: dev2 46.0 % vs
27.3 % undistilled (+19 points, still unusable).  Tokens once committed stay fixed in all of these (monotonic
unmasking); the "all positions jointly updated" family remains open for a model trained with its own forward process.

## 7. Denoising dynamics of the student (B10-E)

Teacher-forced on 7 held-out training prompts (states the teacher trajectory visited): on the 18 % of states where the
undistilled model's independent pair contradicts the teacher's joint pair, the student's single forward produces the
teacher's pair in 41–50 % of cases (A 47.6, B 49.5, C 48.3, D 48.6, E 40.7 %) and the teacher's second token in
48–57 %; on the other 82 % it keeps the pair at 92–98 %.  In free generation this moves the whole trajectory: at
S = 8 the premature-EOS count rises (35 vs 4 on test) while length-budget hits fall (54 vs 100) — the student commits
to endings earlier, both correctly and incorrectly.  The question of the brief — "can the student resolve
multi-position contradictions without a sequential forward?" — is answered *for pairs, partly* (half of the
contradictions) and *not at all beyond pairs*: the distilled signal is pairwise by construction, and the block-level
coherence metrics (adjacent duplicates of the first forward, 6.8 per block) are unchanged because the first forward of
a block is still an all-mask forward (the student is trained on states with ≥ 2 committed tokens; a first-forward
objective is the obvious next addition).

## 8. GSM8K · HumanEval · speed · memory (B10 rows added to the series table; same adapter family, A100 HF path)

| decoder (no AR verification, no draft) | forwards / 16 tokens | GSM8K test | HumanEval | tok/s | peak VRAM | train GPU-h |
|---|---|---|---|---|---|---|
| AR greedy (e5 base) | 16 | 88.2 % | 68.9 % | 52.5 | 8.1 GB + cache | – |
| A-series K = 4 graph engine (sft3) | ~13.6 | 84.2 % | 40.2 % | 89.9 | 9.4 GB | – |
| B-series S = 16 left | 16 | 84.8 % | 50.6 % | 39.5 (A100) / 56 (zeta) | 7.8 GB | 0.8 |
| B-series S = 8 left | 8 | 72.9 % | 8.5 % | 77.6 | 7.8 GB | 0.8 |
| B8 joint teacher, S = 8 | 22 | 80.7 % | 23.2 % | 29.8 | 7.9 GB | – |
| **B10 student A, S = 8** | **8** | **77.1 %** | 10.4 % | 75.5 | 7.9 GB | 0.8 + 0.4 |
| B10 student A, S = 16 | 16 | – (dev2 87.3) | 54.9 % | 36.9 | 7.9 GB | |
| B10 student E (GSM8K + MBPP), S = 8 | 8 | 78.4 % | 0.0 % | 73.6 | 7.9 GB | 0.8 + 1.3 |

Latency per GSM8K problem at S = 8: 2.46 s (student) vs 2.4–2.6 s (baseline) — the student adds no inference cost
(same LoRA count after merging).  Teacher-data generation: 89 min (GSM8K, 65 k states, 343 k forwards) + 25 min (MBPP)
on zeta; training 25 min (A, D), 40 min (B), 60 min (C), 80 min (E) on one A100 each (peak 20.7–21 GB with the frozen
teacher resident).  Whole phase ≈ 9 GPU-hours including evaluations.

## 9. Failed hypotheses and limits (recorded)

1. **More teacher data does not help**: 12 k → 65 k states, 1 000 → 2 000 steps: hard-state pair accuracy stays at
   ~48 % and dev2 S = 8 moves 81.3 → 77.3 (noise band 77–82 across A/B/C/D).  The remaining half of the hard states
   is not learnable from the state alone by this LoRA — the information the teacher uses (one committed token) is
   genuinely absent from the student's input.
2. **The distribution does not matter, the label does**: KL to the teacher conditional (A) and hard CE to the
   teacher's token (D) give the same result; a per-position KL to a conditional does not build a joint (as the brief
   warned), it moves the marginal to the conditional's mode.
3. **Look-ahead transition KL** (student predicts the teacher's next-state distributions for all masked positions):
   no gain (79.3 vs 81.3 on dev2).
4. **Budget specialisation**: the S = 8-trained student loses 2–4 items at S = 12/16.
5. **Code**: the teacher itself reaches only 23.2 % HumanEval at S = 8 (vs 50.6 % at S = 16): resolving the pair does
   not undo the code cascade; the student with GSM8K states only: 10.4 %; **with MBPP states: 0.0 %** — the code it
   writes is logically close but 132 of 164 programs fail with `IndentationError: unexpected indent` (the function
   header re-emitted with extra indentation, bodies mis-indented): the distilled pair decisions on code states made
   the student commit to indentation/newline pairs that are wrong in the HumanEval prompt format (MBPP prompts have
   no code header; HumanEval's do).  Code at S = 8 needs its own teacher states in the evaluation's prompt format
   and, more fundamentally, a longer sequential context than a pair gives.
6. The dev2 → test shrinkage (+11.3 → +4.2) is a reminder that 150 items select favourably even when the choice is
   principled; the test number is the one to quote.

## 10. Go / No-Go and next model structure

| gate | verdict |
|---|---|
| 1 teacher gap shrinks with conditional distillation | **yes** — hard-state pair agreement 3 → 48 %, KL 0.27 → 0.12 on held-out states |
| 2 global consistency: joint prediction more coherent than independent argmax | **partly** — for the pair placed per step; block-level first-forward incoherence untouched |
| 3 free generation improves | **yes** — GSM8K S = 8: dev2 70.0 → 81.3, test 72.9 → 77.1 (p = 0.001); HumanEval S = 8 no (10 %) |
| 4 compute efficiency | **yes** — same 8 forwards, same latency, +4.2 test points; 25 min of training |

Short-term target (S = 8, GSM8K ≥ 80 %, HumanEval ≥ 30 %): reached on dev2 for GSM8K, **not on the test set
(77.1 %)** and not on code.  What the measurements say to build next: (i) extend the distilled decision from the pair
to the *first forward* of a block (the all-mask step), whose independent argmax carries 6.8 duplicates per block — a
teacher that resolves the first 2–4 positions sequentially and a student trained to emit them jointly; (ii) make the
student's training states cover S = 4 (4-token steps) so the 4-forward budget inherits the gain (+19 points already
without targeting it); (iii) for code, the sequential teacher at S = 8 is itself only 23 %: the block size, not the
pair, is the limit — K = 8 with S = 4 (same forwards per token) should be measured before any K = 32 work; (iv) the
EOS artefact (35 early endings) is removable by excluding end-of-answer pairs from the distillation set.  None of this
needs a new backbone; all of it stays within "one LoRA, eight forwards, no sequential conditioning at inference".
