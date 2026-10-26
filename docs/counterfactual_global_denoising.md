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

## 3. Control vs intervention (arm E, dev-50, 150 intervention states: 50 prompts × blocks 0–2, after step 4 of 8)

Control outcome of the 150 states: 93 correct, 57 wrong (62 %).  "Repair" = control wrong → intervention correct;
"harm" = control correct → intervention wrong; "net" = expected change of accuracy per applied intervention at this
62/38 mix.  results/pure/b12_cf_E_{a,b}.json.

| intervention at the step-4 state (same remaining 4 forwards + unmodified continuation) | candidates | repair rate (of 456–114 wrong-state candidates) | harm rate (of correct-state candidates) | net | block changed | tokens changed |
|---|---|---|---|---|---|---|
| single visible position | 1 200 | **29.8 %** | 12.8 % | +0.034 | 56 % | 1.9 |
| pair, adjacent | 843 | 39.6 % | 16.9 % | +0.046 | 67 % | 2.5 |
| pair, non-adjacent | 600 | 39.0 % | 15.6 % | +0.052 | 72 % | 2.7 |
| multi-3 | 300 | 39.5 % | 13.4 % | +0.067 | 83 % | 3.7 |
| multi-4 | 300 | 49.1 % | 19.4 % | +0.067 | 89 % | 4.3 |
| random-1 / -2 / -3 / -4 | 150 each | 31.6 / 33.3 / 54.4 / 52.6 % | 11.8 / 23.7 / 19.4 / 18.3 % | +0.047 / −0.020 / +0.087 / +0.087 | | |

Three facts.  (1) **Re-masking almost anything mid-block and regenerating with the same budget is a net-positive
move at this accuracy level**: a wrong answer is repaired 30–55 % of the time, a correct one broken 12–19 %, so every
kind except random-2 has positive expected value (+3 to +9 points per applied intervention).  (2) The targeted
candidates (lowest confidence, adjacency) are *not* better than random sets of the same size: the regenerated
continuation after a re-mask is a different sample of the answer, and whether it lands on the right answer depends on
the state, not on which position was re-masked.  (3) The effect decays with position in the answer: single re-masks in
block 0 repair 35 % / harm 17 %, block 2 24 % / 9 % — early states are more malleable in both directions.

## 4. Oracle repair upper bound

| | value |
|---|---|
| wrong states repairable by at least one candidate | **50 / 57 (88 %)** |
| smallest repairing set: 1 / 2 / 3 / 4 positions | 38 / 9 / 1 / 2 states |
| correct states robust to *every* candidate | 32 / 93 (34 %) |
| oracle accuracy if the best candidate were chosen for every state (reads the outcome) | (93 + 50) / 150 = **95 %** |
| control accuracy at the same states | 62 % |
| search cost of the oracle | 2 206 forwards per state vs 85 for the control (26 ×) |

So within the 8-forward budget the *state* after step 4 contains, in 88 % of the wrong cases, a one- or two-position
re-mask whose regeneration ends in the right answer — but 66 % of the correct states contain a re-mask that ends in a
wrong one.  The oracle is therefore not "repair" in the sense of fixing a known defect: it is *selection among
stochastic continuations* with knowledge of the outcome, i.e. best-of-N over 25 restarts of the second half of the
block, and its 95 % is the best-of-25 ceiling of this sampler on dev-50, not a quantity an intervention policy can
approach without the outcome.  **Gate 1 (repair feasibility within the budget): yes for the existence of improving
transitions; the improving transition is a regeneration, not a targeted correction.**

## 5. Intervention value analysis — is it predictable at inference time?

| single re-mask, by the state's confidence in the re-masked token | repair rate (control wrong) | harm rate (control correct) |
|---|---|---|
| confidence < 0.5 | **51.7 %** (15 / 29) | 14.6 % |
| 0.5–0.9 | 31.3 % | 16.9 % |
| ≥ 0.9 | 27.4 % | 11.5 % |

Confidence of the re-masked token predicts the repair rate (52 vs 27 %) but not the harm rate (15 vs 12 %), and the
low-confidence bucket is small (29 of 456 wrong-state candidates).  Feasible policies on the same candidates (per state):
lowest-confidence single: repair 12 %, harm 11 % of states, net +1.3 points; lowest-confidence adjacent pair: net +10.7
on part a (n = 75); random-3: net +8.7 to +16.  These are expected values of *one* applied intervention; they are all
within the noise of 150 states but consistently positive.  **Gate 2 (predictability): weakly — confidence is a usable
but coarse signal; the main lever is the number of positions regenerated, not which ones.**

The zero-cost consequence is tested directly in §6: the global sampler with a forced re-mask of the k
lowest-confidence earlier-placed visible tokens at step 4 (no extra forwards), on dev-50 and dev2.

## 6. Zero-cost policy: forced re-mask of k earlier-placed visible tokens at step 4 inside the global sampler (arm E, S = 8)

`remask_vis` mode of engine/bdiff/global_sampler.py: after the normal step-4 update, k additional visible tokens placed at
an earlier step are re-masked (lowest confidence under the current forward, or random) and regenerated by the remaining
4 forwards — exactly the intervention of §3 applied to every block, no extra forward.

| sampler (arm E, S = 8) | dev-50 | **dev2 (150)** |
|---|---|---|
| plain non-monotonic re-masking (reference) | 62 | **65.3** |
| + forced re-mask k = 1, lowest confidence | 76 | 63.3 |
| + k = 2, lowest confidence | 56 | 64.7 |
| + k = 3, lowest confidence | 68 | 59.3 |
| + k = 1, random | – | 65.3 |
| + k = 2, random | 52 | 63.3 |
| + k = 3, random | 70 | – |
| left-to-right monotonic, same model | 86 | 78.7 |

The per-intervention expectation of §3 (+3 to +9 points at a state with 62 % control accuracy) does **not** turn into a
better sampler: dev-50 scatters 52–76 % around 62 (the spread of a resampling step on 50 items) and dev2 is 59–65 %
against 65.3 %.  The reason is in §3 itself: the intervention's value is positive only because it re-rolls a block whose
outcome is already known to be wrong 38 % of the time; applied blindly to every block it re-rolls the 62 % correct
blocks as well, and across the ~10 blocks of an answer the harm (12–19 % per correct block) compounds while a wrong
block is rescued at most once.  Without an outcome signal (which is what the oracle uses) a forced re-mask is a
variance-adding restart, not a correction.  **Gate 3 (learned/heuristic policy better than random or confidence
re-masking): not reached — no feasible policy is better than not intervening.**

## 7. True global denoising — verdict on the brief's success conditions

| condition | measured | met? |
|---|---|---|
| changes improve the final quality | an *individual* re-mask improves expectation by +3–9 points at a 62 % state; applied as a policy, −2 to −6 points on dev2 | no |
| W→C exceeds C→W | per applied intervention yes (repair 30–55 % of wrong vs harm 12–19 % of correct), but only because wrong states are a minority and regeneration is a coin flip | formally yes, mechanistically no |
| block-level consistency improves | duplicates per block unchanged by interventions (Δ 0.00–0.07) | no |
| no dependence on left-to-right commitment | the global sampler stays 13–15 points behind left-to-right on the same model (dev2 65.3 vs 78.7) | no |
| same forward budget | yes for every row above (oracle excluded) | yes |
| transfers beyond GSM8K | not tested for interventions; the sampler itself: HumanEval unchanged (B11) | – |

The counterfactual data answer the brief's central question negatively for the *current* denoiser: the improving
state transitions that exist (88 % of wrong states have one) are not identifiable from the state — they are
alternative samples whose value is known only from the outcome.  A re-masking policy trained on measured intervention
values (brief §5) would therefore be trained to predict a quantity that the state does not determine (the oracle's 95 %
is best-of-25 selection with the answer in hand), and joint-transition learning (brief §6) would learn from transitions
whose "improvement" is a lucky draw.  Neither is started (brief §4: "もし Oracle 介入でも改善しない場合…" — the oracle
improves, but only with the outcome; the pre-registered stopping rule applies to policies, which do not).

## 8. GSM8K / HumanEval / MBPP (unchanged rows carried from B11; no new sampler is proposed by this phase)

| decoder (K = 16, no AR verification) | GSM8K dev2 | GSM8K test | HumanEval | MBPP |
|---|---|---|---|---|
| left-to-right S = 8, undistilled | 70.0 | 72.9 | 8.5 | 1.9 |
| left-to-right S = 8, B10.5 arm B / B11 arm B | 83.3 / 84.0 | 77.8 / 77.2 | – | – |
| global re-masking S = 8, undistilled | 58.0 | – | – | – |
| global re-masking S = 8, B11 arm E | 65.3 | __ETEST__ | 8.5 (left-to-right mode) | – |
| global + forced re-mask (this phase) | 59.3–64.7 | – | – | – |
| counterfactual oracle (best of 25, outcome known; 26 × forwards) | 95 (dev-50 states) | – | – | – |

## 9. Forwards, speed, VRAM

Interventions cost nothing at inference (same S forwards per block; 1.70 tokens/forward at S = 8, 69–75 tok/s on the
A100 HF path, 7.9 GB).  The counterfactual search cost 2 206 forwards per state (26 × the control's 85), 150 states on
two A100s in 70 min each (plus the baseline-adapter replica on zeta); the forced-re-mask sweep 5 × 4 min (dev-50) and
5 × 7 min (dev2).  Phase total ≈ 5 GPU-hours, no training.

## 10. Failed hypotheses (recorded)

1. "Which position to re-mask" is the question: targeted re-masks (lowest confidence, adjacent pairs) are no better
   than random sets of the same size; the lever is how many positions are regenerated (regeneration = resampling).
2. A positive expected intervention value implies a useful policy: it does not — the value exists only conditional on
   the outcome being wrong, which the state does not reveal (confidence < 0.5 raises the repair rate to 52 % but covers
   6 % of candidates and does not lower harm).
3. Oracle repair rates (88 % of wrong states, 95 % accuracy) measure the sampler's best-of-N potential, not a reachable
   policy; they repeat, in outcome space, what B7's and B11's oracles showed in token space.
4. Interventions do not reduce structural defects (duplicates Δ ≈ 0): the degenerate pairs are re-created by the
   regeneration as often as removed.

## 11. Research decision

Gates 1 (existence of improving transitions) passes trivially via resampling; Gates 2–4 fail: intervention value is not
predictable from the state, no feasible policy beats non-intervention, and joint multi-position updates improve
nothing beyond what a random restart does.  Per the brief's stopping rule, no re-masking policy and no joint-transition
student are trained.  What the two phases B11–B12 leave as the honest state of the global line: the denoiser can be
aligned to any sampler's states and gets better *first placements* from it (B11), but it has no error signal about its
own visible tokens — not from confidence (B7), not from teacher mismatch (B11), not from counterfactual outcome (B12,
unpredictable) — and without such a signal "global refinement" is restarting.  The one source of an error signal that
has not been tried and that the data do support is *structural*: duplicates and syntax breaks are observable in the
state itself and are the defects that separate wrong from correct blocks (B10.5 §3, B11 §3.2); a re-masking rule
restricted to those (and verified counterfactually on the same harness) is the smallest next experiment, and the
left-to-right decoder with its on-policy training remains the production path at 8 forwards per block.
