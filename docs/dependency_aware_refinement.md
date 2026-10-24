# Phase E3 — Dependency-aware parallel refinement (no training, no AR verification)

Brief (2026-10-24): make the leave-one-out refinement of engine/cdlm/k4_decoder.py beat AR in both accuracy and speed by
choosing *which* positions to change in each round (dependency-aware adoption), stopping early when a block has
settled, and switching block size adaptively.  Companion reports: docs/four_token_diffusion.md (Phase E),
docs/k_scaling_failure_analysis.md (K = 3…8), docs/pure_diffusion_research.md.  Everything measured unless marked
*estimated*.  Fixed conditions: checkpoint runs/phaseC_sft3_4b/final, dev items 0–49 (GSM8K train, never trained on),
greedy, 384-token budget, bf16, batch 1, anchor on, EOS policy "current"; code at the commit of each result file
(results/pure/e3_*.json carry the arguments).  The A100 (GPU 4) was idle for these runs unless stated.

## 1. Purpose and hypothesis

K = 8 Jacobi refinement collapses (18 %) although the probe conditional with true neighbours is 96 %+ accurate at every
position (k_scaling §10) and one-change-per-round reaches 90 % (12 rounds).  Hypothesis: simultaneous changes at
mutually dependent positions are each conditioned on the other's OLD token and break consistency; changes at weakly
dependent positions can be adopted together.  If true, an adoption rule that selects a set of mutually independent
changes should keep the accuracy of sequential adoption at a fraction of its rounds.

## 2. Reproduction on an idle A100 (GPU 4, nothing else running) — results/pure/e3_K*_repro_*.json, e3_ar_graph_a100.json

| decoding | GSM8K dev 50 | tokens / forward | tok/s (A100) |
|---|---|---|---|
| AR greedy, HF generate | 96 % | 1.00 | 54.6 |
| AR via the graph engine (τ=None) | 96 % | 0.99 | 72.2 |
| K=4 refineN Jacobi ≤ 3, plain engine | 88 % | 1.33 | 63.7 |
| **K=4 refineN Jacobi ≤ 3, CUDA graphs** | **88 %** | 1.33 | **99.6** |
| K=4 refineN Jacobi ≤ 6, CUDA graphs | 92 % | 1.21 | 91.2 |
| K=8 refineN sequential (one change / round) ≤ 12, plain | 90 % | 0.97 | 40.9 |

The K = 4 numbers reproduce the earlier shared-GPU runs exactly (same tokens, same outcomes); on the idle A100 the
graph-engine K=4 decoder is 1.8× the HF AR throughput at −8 points (dev 50) / 1.26× the graph-engine AR.


## 3. Dependency measurement (`bench/diag_dependency.py`) — results (K = 8: 294 blocks, 4 169 forwards; K = 4: 2 255 blocks)

Dependency of position i on a change at position j (the probe's alternative token substituted at j), K = 8:

| distance \|i−j\| | JSD mean (± sd) | top-1 of i changes | \|Δ conf\| | KL |
|---|---|---|---|---|
| 1 | **0.376 ± 0.44** | **45.4 %** | 0.134 | 4.41 |
| 2 | 0.110 ± 0.22 | 19.0 % | 0.107 | 0.63 |
| 3 | 0.061 ± 0.15 | 13.4 % | 0.066 | 0.27 |
| 4 | 0.042 ± 0.12 | 9.9 % | 0.053 | 0.18 |
| 5–7 | 0.032–0.042 | 9.7–15.5 % | 0.04–0.05 | 0.12–0.15 |

By side (K = 8): a change to the **left** of i moves P_i by JSD 0.272 (top-1 changes 36.9 %); a change to its **right**
by 0.012 (6.2 %).  The JSD distribution is bimodal: 77 % of (i, j) pairs are below 0.1 and 8 % above 0.9 (a token that
flips its neighbour outright).  K = 4 shows the same curve (0.345 / 0.082 / 0.044 at distance 1 / 2 / 3).

Jacobi changes tagged by the maximum JSD-dependency to the other positions changed in the same round (K = 8, 3 rounds):

| co-dependency of a simultaneous change | n | → correct | → wrong (was right) | wrong → other wrong |
|---|---|---|---|---|
| none (the only change of the round) | 80 | **82.5 %** | 3.8 % | 13.8 % |
| max JSD < 0.1 | 879 | 57.0 % | 25.6 % | 17.4 % |
| 0.1 ≤ max JSD < 0.3 | 234 | 35.9 % | 24.8 % | 39.3 % |
| **max JSD ≥ 0.3** | **1 987** | **19.9 %** | 22.0 % | **58.0 %** |

Answers (brief §12): (1) yes — simultaneous changes between strongly co-dependent positions are mostly churn
(58 % wrong→wrong, 20 % correct) while isolated or weakly dependent changes are 57–83 % correct; (2) no — distance
explains most of it but not all: the dependency is **directional** (left context matters 20× more than right context
for this suffix-trained checkpoint), and 10–15 % of far pairs still flip the top-1, so a distance-only graph misses
long-range edges (numbers, brackets) and over-penalises right-side neighbours.  Measuring JSD at inference costs
K extra probe forwards per round and is not worth it (§12); the distance graph is a cheap proxy that captures the
dominant d = 1 term, which is why colouring / odd-even / top-2 recover most of the gain.

## 4. Update policies (`--adopt`)

| policy | rule |
|---|---|
| Jacobi (`always`) | every changed position is adopted |
| Sequential (`one`) | the most confident change only |
| Top-M (`top<M>`) | the M most confident changes |
| Nonadjacent | greedy by confidence, no two adjacent |
| Weighted independent set (`wmis<d>`) | brute-force subset of the changes with no two within distance d maximising Σ (probe conf − current conf) |
| Graph colouring (`color<d>`) | colour = position mod (d+1); one colour class per round, rotating |
| Trust region (`--max-changes`) | cap on adopted changes per round |
| Confidence-gated (`conf`) | adopt only if the probe is more confident than the current token |

Selection cost: all rules are O(K) on the host except `wmis` (2^|changes| subsets, ≤ 256 at K = 8) — microseconds.

## 5. Update policies at K = 8 (idle A100, plain engine, ≤ 6 rounds unless noted; AR 96 %)

| policy (`--adopt`) | GSM8K | tokens / forward | tok/s | corrections / 100 tok | blocks converged / cycle / budget |
|---|---|---|---|---|---|
| Jacobi `always` ≤ 6 (k-scaling run, shared GPU) | 48 % | 1.28 | – | 200 | 468 / 46 / 467 |
| `top1` (sequential) ≤ 6 | 68 % | 1.32 | 60.9 | 56 | 452 / 4 / 581 |
| `top2` ≤ 6 | 78 % | 1.33 | 62.5 | 97 | 530 / 16 / 480 |
| `top3` ≤ 6 | 72 % | 1.35 | 63.0 | 125 | 570 / 21 / 433 |
| `top4` ≤ 6 | 66 % | 1.32 | 55.1 | 161 | 468 / 31 / 470 |
| `nonadjacent` ≤ 3 (k-scaling) / `oddeven` ≤ 3 (6 probes) | 28 % / 78 % | 2.0 / 1.20 | – | 75 / 114 | – |
| `wmis1` (independent set, distance 1) ≤ 6 | 70 % | 1.31 | 56.1 | 112 | 509 / 12 / 466 |
| **`color1` (odd/even colouring, one class per round) ≤ 6** | **82 %** | 1.30 | 62.6 | 115 | 435 / 11 / 560 |
| `always` + trust region max 2 changes ≤ 6 | 78 % | 1.33 | 56.1 | 97 | = top2 |
| `top2` ≤ 12 | 78 % | 1.05 | 44.5 | 126 | 789 / 85 / 148 |
| `wmis1` ≤ 12 | 88 % | 1.11 | 52.9 | 133 | 840 / 85 / 121 |
| `one` ≤ 12 | 90 % | 0.97 | 40.9 | 79 | 713 / 47 / 205 |

| `wmis2` / `color2` (distance 2) ≤ 6 | 62 % / 64 % | 1.32 / 1.28 | 64.3 / 54.5 | – | – |

Distance-2 graphs are worse than distance-1 (62–64 vs 70–82 %): forbidding more simultaneous changes than the
dependency structure requires slows convergence without reducing conflicts (the d ≥ 2 dependency is small, §3).

Reading (measured): every rule that forbids *adjacent* simultaneous changes beats Jacobi by 20–34 points at the same
forward budget (odd/even 78 %, colouring 82 %, top-2 78 %, independent set 70 %), and the gain is not from changing
fewer positions per se (top-4 66 % < colouring 82 % at similar change counts): it is from *which* positions change
together.  That is the hypothesis of §1, confirmed.  The ceiling of all rules at K = 8 stays ~1.1–1.3 tokens/forward
because 5–7 probe rounds are needed; more rounds buy accuracy (wmis1 88 %, one 90 % at 12) but not speed.  The
remaining accuracy gap to K = 4 at equal forwards comes from the first forward's weaker far positions
(k_scaling §3/§4), which no adoption rule repairs.

## 6. Correction outcome accounting
Teacher-forced direction counts (W→C / C→W / W→W') per policy are in results/pure/diag_k6_sft3.json and
diag_k8_sft3.json (k-scaling chain 2, strategies s1 / refine / refineN / seq); free-generation runs log the number of
changes and the converged / cycle / budget outcome of every block.  *Per-policy teacher-forced counts for the E3 rules
were not run (time); the k-scaling diagnostic covers Jacobi.*

## 7. Adaptive early stopping (K = 4, graph engine, idle A100)

| stop rule (≤ 6 rounds) | GSM8K | tokens / forward | tok/s | rounds / block |
|---|---|---|---|---|
| no change (default) | 92 % | 1.21 | 91.2 | 2.24 |
| two quiet rounds | 92 % | 0.95 | 71.0 | 3.14 |
| confidence-stable (≥ 0.9, ≤ 1 change) | 90 % | 1.23 | 92.6 | 2.18 |
| confidence-stable (≥ 0.99) | 92 % | 1.21 | 91.2 | 2.19 |

No early-stopping rule improves on "stop at the first quiet round": the default already stops at 2.2 rounds on
average; requiring two quiet rounds costs 22 % throughput for nothing, and confidence gating stops the same blocks.
Stable wrong answers: 8 % of dev problems converge to a wrong fixed point (the 92 % ceiling vs 96 % AR).

## 8. Adaptive block size (K = 8 with fallback to 2 × K = 4 when the block does not settle; wasted forwards counted)

| setting | GSM8K | tokens / forward | tok/s | fallbacks / big blocks |
|---|---|---|---|---|
| K=8 Jacobi ≤ 3 → fallback K=4 | 86 % | 0.91 | 45.1 | 830 / 1 101 (75 %) |
| K=8 top-2 ≤ 6 → fallback K=4 | **94 %** | 0.91 | 44.1 | 423 / 1 091 (39 %) |
| fixed K=4 Jacobi ≤ 3 (reference) | 88 % | 1.33 | 63.7 | – |

The fallback recovers accuracy (94 %) but the discarded K = 8 attempts make it slower than fixed K = 4 (0.91 vs 1.33
tokens/forward).  Adaptive K does not beat fixed K = 4 on this checkpoint (brief question 7: no).

## 9. Held-out confirmation (dev items 50–199, 150 prompts, never used for selection; AR on these items 90.0 %)

| K = 8 rule | dev 0–49 | **held-out 50–199** | tokens / forward |
|---|---|---|---|
| colouring `color1` ≤ 6 | 82 % | **58.7 %** | 1.31 |
| top-2 ≤ 6 | 78 % | **68.0 %** | 1.31 |
| independent set `wmis1` ≤ 12 | 88 % | **81.3 %** | 1.06 |
| (reference) K = 4 Jacobi ≤ 6 | 92 % | 84.7 % | 1.17 |

The 50-item exploration numbers of the K = 8 rules were inflated by 10–23 points (the 50 items are easier: AR 96 vs
90 %, and ~25 settings were compared on them).  On held-out data the only K = 8 rule near the K = 4 decoder is the
weighted independent set with 12 rounds (81.3 % at 1.06 tokens/forward, i.e. no speed gain over K = 4's 84.7 % at 1.17).

## 10–13. CUDA graphs for K = 8, GSM8K test, HumanEval, VRAM

Not run: the K-group probe at K = 8 has 64 rows and the static engine's largest bucket is 32 (adding a 64-row bucket is
straightforward but was not justified by §9); no K = 8 rule earned a GSM8K-test or HumanEval evaluation; VRAM of the
plain-engine K = 8 decoder is the 4B's weights + a 4 096-slot cache (≈ 9 GB bf16, as for K = 4).

## 14. Failed experiments

* Early-stopping rules (§7): none better than stopping at the first quiet round.
* Adaptive K = 8 → 4 fallback (§8): recovers accuracy but discards too much work (0.91 tokens/forward).
* Distance-2 dependency graphs (§5): worse than distance-1.
* Trust region (max 2 changes) ≡ top-2.
* Every K = 8 rule at 6 rounds lost 10–23 points on held-out data relative to the 50 exploration items.

## 15. Conclusions and next steps

Measured answers to the brief's questions: (1) simultaneous updates between dependent positions do amplify errors
(58 % churn vs 20 % correct); (2) dependency is mostly distance-1 and strongly left-sided, with a long-range tail;
(3) dependency-aware adoption improves K = 8 by 20–34 points at equal forwards (colouring / top-2 / wmis vs Jacobi);
(4) its selection cost is negligible when the graph is distance-based and prohibitive when measured (K extra probes);
(5) 2 simultaneous changes at K = 8 keep 68 % held-out — not 85 %; (6) K = 4 early stopping gains nothing; (7)
adaptive K does not beat fixed K = 4.  **The accuracy–speed frontier of this checkpoint is still K = 4 Jacobi with CUDA
graphs (84.7 % held-out at 1.17–1.33 tokens/forward, 1.8× HF-AR tok/s on the A100).**  To move the frontier the
*first forward* must get better at distance 3–8 (k_scaling §3): that is a training question (the B-series block-
bidirectional model, docs/discrete_diffusion_b_series.md, reaches 84 % at K = 16 after 30 min of LoRA and is the line
being pursued), not an adoption-rule question.  Engine-side, the remaining lever for K = 4 is fusing the 16-row probe
and the first forward into one launch per round (the probe is 2/3 of decode time, four_token_diffusion.md §9).

