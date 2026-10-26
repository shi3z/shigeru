# Phase B12 — Counterfactual Global Denoising

Brief: "SHIGERU Phase B12 — Counterfactual Global Denoising" (2026-10-26).  Question: from the global sampler's
intermediate states, which re-masking / joint-update interventions improve the *final* generation under the *same*
remaining forward budget — measured by counterfactual experiments on real trajectories, not by mismatch with a teacher
sequence.  Starting point: the B11 best global student (arm E: masked CE on the non-monotonic sampler's own states with
teacher-corrected targets) under non-monotonic confidence re-masking, K = 16, S = 8 (dev2 65.3 %; left-to-right 78.7 %).
Constraints: no AR verification, no speculative decoding, no sequential pairwise teacher at inference; oracle search
costs reported separately from the sampler's cost; the GSM8K gold answer is used only to score outcomes, never to pick
interventions except in the explicitly labelled oracle rows.

## 1. What B11 established that this phase builds on

- The frozen and the sampler-aligned denoisers both converge, under the global sampler, to a fixed point: ≤ 5 % of
  visible tokens are ever overturned, duplicates per block grow to 1.2–1.4, and the completion order from a given state
  barely matters (sampler_aligned_global_denoising.md §7, §7.1).
- Mismatch with the teacher's or the reference's wording is 43 % even in task-correct answers; it is not an error label
  (§3.2 there).  Hence this phase scores interventions by *outcome* (final GSM8K correctness) plus dense structural
  auxiliaries (duplicates, repetition, whether/how much the block changed), never by token agreement.
- The B7 oracle (re-mask the tokens that differ from the reference, refill) repaired 30 % and the B11 oracle reached 73 %
  token agreement — both *reconstruction* numbers.  The quantity defined here is different: **intervention value** =
  quality(intervention) − quality(control) from the same state with the same remaining forwards.

## 2. Counterfactual intervention dataset (bench/diag_counterfactual.py)

For each dev-50 prompt the global sampler (arm E, S = 8) generates the answer; for blocks 0, 1, 2 the state after step 4
(8 visible, 8 masked positions on average) is the intervention point.  From that state, with the remaining 4 forwards of
the block and then the unmodified sampler for the remaining blocks, the following are run to the end of the answer:

| arm | intervention at the step-4 state | candidates per state |
|---|---|---|
| control | none | 1 |
| single | one visible position → [MASK] | every visible position (≤ 8) |
| pair, adjacent | two adjacent visible positions | all adjacent pairs (≤ 7) |
| pair, non-adjacent | two non-adjacent visible positions | 4 random |
| multi-3 / multi-4 | 3 / 4 visible positions | 2 random sets each |
| random-k | k ∈ {1, 2, 3, 4} random visible positions | 1 each (comparison baseline for the policies) |

Greedy argmax everywhere, so control and interventions differ only by the re-masked positions.  Per candidate: final
correctness, block changed (yes/no), tokens changed in the block, adjacent duplicates and 4-gram repetition deltas, the
state confidence of the re-masked positions.  Oracle = per state the best candidate by final correctness (search cost:
~25 candidates × ~110 forwards ≈ 2 900 forwards per state, 25× the control's).  Inference-feasible policies evaluated
on the same candidates: lowest-confidence single, lowest-confidence adjacent pair, random-k.
