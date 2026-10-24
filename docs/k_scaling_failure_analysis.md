# K-scaling failure analysis — why leave-one-out refinement works at K = 4 and collapses at K = 8

Brief (2026-10-24): no training; final checkpoint runs/phaseC_sft3_4b/final; dev items 0–49 of bench/prompts_gsm8k_dev.json
(GSM8K train, never trained on); greedy; 384-token budget; the K-group leave-one-out probe of engine/cdlm/k4_decoder.py.
Everything measured unless marked *estimated*.  Companion: docs/four_token_diffusion.md (Phase E), which this report
extends; docs/pure_diffusion_research.md.

## 1. Purpose

Separate three candidate causes of the collapse from K = 4 (88–96 % GSM8K dev) to K = 8 (18 %) and K = 16 (4 %):
(a) the model's conditional predictor is too weak at larger K, (b) the all-mask initialisation is too poor to iterate
from, (c) the Jacobi-style update rule oscillates.

## 2. Environment

tsuginosuke A100 80 GB (GPU 7, **shared with a stage-2 training job during these runs — tok/s are not comparable
across runs and are reported only as measured**), bf16, batch 1, plain-forward engine (no CUDA graphs) unless stated.
Files: results/pure/kscale_*.json (generation), results/pure/diag_kscale_sft3.json (teacher-forced B/C/10),
results/pure/audit_k_cache_sft3.json (§10).  All scripts: bench/run_block.py --decoder k4, bench/diag_kscale.py,
bench/audit_k_cache.py.

Missing files noted: none of the brief's listed documents were missing (four_token_diffusion.md, pure_diffusion_research.md
exist); "現行LOO Decoder / refineN" = engine/cdlm/k4_decoder.py; EOS handling = k4_generate (a stop id in the final block
cuts the block there); GSM8K scoring = bench/common.py.

## 3. Step 1 — reproduction, and Experiment A — K scaling (refineN, Jacobi "always" adoption, ≤ 3 rounds, anchor, greedy)

Reproduction (same 50 prompts, same checkpoint, new code path with cycle detection and timing): K = 4 → 88 %,
1.33 tokens/forward (previous run 88 %, 1.33); K = 8 → 18 %, 1.98 (previous 18 %, 1.98).  Reproduced exactly.

| K | GSM8K dev 50 | tokens / forward | mean forwards / block | corrections / 100 tok | blocks: converged / cycle / budget (≤ 3 rounds) | budget share | premature EOS | max-length |
|---|---|---|---|---|---|---|---|---|
| 3 | **94 %** | 1.15 | 2.56 | 31 | 2529 / 46 / 183 | 7 % | 1 | 1 |
| 4 | 88 % | 1.33 | 2.94 | 53 | 1636 / 32 / 478 | 22 % | 4 | 1 |
| 5 | 62 % | 1.45 | 3.35 | 79 | 878 / 13 / 681 | 43 % | – | 1 |
| 6 | 54 % | 1.57 | 3.67 | 108 | 419 / 27 / 791 | 64 % | – | 1 |
| 7 | 34 % | 1.77 | 3.78 | 132 | 203 / 5 / 816 | 80 % | – | 2 |
| 8 | 18 % | 1.98 | 3.86 | 147 | 118 / 1 / 840 | 88 % | – | 3 |

(files results/pure/kscale_K*_refineN_r3.json; "converged" = a round with no change, "cycle" = a block state seen
before, "budget" = 3 rounds exhausted with changes still happening.)  AR on the same prompts: 96 %.

Reading: the decline is **continuous**, not a cliff — it starts between K = 4 and K = 5 (88 → 62) and loses ~15 points
per extra token.  True oscillation (revisiting an earlier block state) is rare at every K (0.1–1.6 % of blocks); the
dominant failure is **non-convergence within the 3-round budget**: the share of blocks cut off while still changing
rises from 7 % (K = 3) to 88 % (K = 8), in lock-step with the accuracy loss, and the number of corrections per token
grows 5× from K = 3 to K = 8.  K = 6 does not hold high accuracy under this update rule (54 %).  The question the rest
of the matrix answers is whether the extra rounds are needed because the first forward is poorer (B, C, D) or because
Jacobi adoption fights itself (E).

## 4. Experiment D — split initialisation (refineN, Jacobi, ≤ 3 rounds; initialisation forwards counted)

| K | initialisation | GSM8K dev 50 | tokens / forward | corrections / 100 tok | blocks converged / cycle / budget |
|---|---|---|---|---|---|
| 6 | all 6 masked (A) | 54 % | 1.57 | 108 | 419 / 27 / 791 |
| 6 | 3 + 3 (B) | 48 % | 1.36 | 71 | 595 / 15 / … |
| **6** | **2 + 2 + 2 (C)** | **80 %** | 1.19 | 41 | 871 / 14 / … |
| 8 | all 8 masked (A) | 18 % | 1.98 | 147 | 118 / 1 / 840 |
| 8 | 4 + 4 (B) | 30 % | 1.65 | 100 | 269 / 6 / … |
| 8 | 2 + 2 + 2 + 2 (C) | 42 % | 1.27 | 51 | 563 / 6 / … |

Reading: initialising in pairs (each pair conditioned on the previous ones) raises K = 6 from 54 to 80 % and halves
the corrections, at the price of the initialisation forwards (1.57 → 1.19 tokens/forward — the same cost regime as
K = 4).  3 + 3 does not help (48 %): a group of 3 from scratch already contains the distance-3 error that Jacobi then
has to clean up.  For K = 8 the pairwise init doubles the accuracy (18 → 42 %) but stays far from usable: the probe
rounds, not the initialisation alone, limit K = 8.

## 10. Attention / KV-cache audit (`bench/audit_k_cache.py`, 8 prefixes, bf16, GPU 7 shared)

| K | probe rows | cache vs no-cache: max |Δlogit| | argmax agreement | each group alone vs the K-group probe: max |Δ| | agreement | static/graph path vs plain |
|---|---|---|---|---|---|---|
| 4 | 16 | 0.38 | 100 % (32/32) | 0.44 | 100 % | 0.0 (identical), 100 % |
| 6 | 36 | 0.50 | 100 % (48/48) | 0.47 | 100 % | n/a (36 rows exceed the 32-row static bucket) |
| 8 | 64 | 0.77 | 100 % (64/64) | 0.44 | 100 % | n/a |

The probe computed through the cache equals a from-scratch forward of [prefix | probe rows] under the equivalent dense
mask up to bf16 accumulation noise (the max logit difference grows with the number of rows, 0.4 → 0.8 on logits of
magnitude ~20, never flipping an argmax here), and every group of the K-group probe equals the same position probed
alone in a standard observed-first forward (the group-independence test at K = 6 and 8, i.e. no leakage between
groups).  Position ids, RoPE and the K/V written by the merged commit are the same code path for every K (tests
engine/cdlm/tests/test_k4.py run K = 3…8 on CPU).  **No K-dependent implementation defect was found**; the collapse
is algorithmic.

## 4b. Experiment B — first-forward quality by K (teacher-forced, results/pure/diag_kscale_sft3.json)

| K | blocks | first-forward acc by position | token acc | block exact | tokens correct per block (mode / ≥ K−1) | longest error run ≥ 3 | duplicate rate | mean error position | oracle conditional (true neighbours) |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 3 016 | .97 .83 .70 | 83.4 % | 65.1 % | 3 / 86.6 % | 1.5 % | 5.6 % | 1.6 | .96 .96 .98 |
| 4 | 2 255 | .97 .84 .71 .58 | 77.3 % | 48.1 % | 4 / 72.3 % | 9.3 % | 9.0 % | 2.3 | .96 .96 .96 .98 |
| 5 | 1 795 | .97 .83 .70 .58 .48 | 71.1 % | 34.1 % | 5 / 56.1 % | 19.7 % | 13.9 % | 3.0 | .96 .96 .96 .96 .97 |
| 6 | 1 497 | .97 .83 .70 .57 .46 .39 | 65.3 % | 23.0 % | 6 / 41.3 % | 31.9 % | 18.1 % | 3.7 | .95 .96 .96 .96 .96 .98 |
| 7 | 1 277 | .97 .83 .67 .58 .47 .37 .34 | 60.5 % | 15.1 % | 4 / 29.1 % | 42.8 % | 22.7 % | 4.3 | .91 .96 .95 .97 .96 .97 .97 |
| 8 | 1 117 | .97 .83 .69 .58 .47 .36 .33 .29 | 56.5 % | 9.4 % | 4 / 20.1 % | 53.3 % | 25.4 % | 5.0 | .92 .95 .95 .96 .95 .97 .96 .97 |

Calibration is stable across K (max-prob ≥ 0.99 → 99.9 % correct at every K; ≥ 0.9 → 94–96 %; < 0.5 → 27–39 %).
Reading: the per-position curve is the *same* at every K (the model's reach does not depend on the block size); what
changes is how many far positions a block contains.  At K = 8 the first forward leaves 3.5 errors per block, half the
blocks contain a run of ≥ 3 consecutive errors, and a quarter of the predictions duplicate another position's token.
The oracle conditional is ≥ 92 % at every position of every K — the restorer is not the limit (question 3: yes).

## 4c. Experiment C — oracle seed injection (teacher-forced, refineN ≤ 6 rounds; free positions scored only)

Free-position accuracy / block exact match, Jacobi adoption (the "one change" rule gives the same numbers ± 1):

| K | seeds n → | 0 | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|---|---|
| 6 | left | 86 / 71 % | 89 / 77 | 92 / 83 | 94 / 89 | 95 / 93 | – |
| 6 | even / random | 86 / 71 % | 89–90 / 77–78 | 91–93 / 82–84 | 94–95 / 88 | 95 / 92 | – |
| 6 | **wrong-first** | 86 / 71 % | **93 / 83** | **95 / 88** | 96 / 91 | 97 / 95 | – |
| 8 | left | 81 / 57 % | 84 / 65 | 87 / 72 | 90 / 79 | 91 / 83 | 95 / 93 |
| 8 | even / random | 81 / 57 % | 84–85 / 65–66 | 86–88 / 69–71 | 90–91 / 76–79 | 93–94 / 83–84 | 95–96 / 91–92 |
| 8 | **wrong-first** | 81 / 57 % | **88 / 71** | **91 / 77** | 94 / 84 | 96 / 89 | 97 / 95 |

Rounds to settle fall with the seeds (K = 8: 4.6 → 2.6 with two wrong-first seeds; budget-exhausted blocks 39 % → 8 %).
Reading: fixing the **one or two positions the first forward got wrong** lifts K = 8 block exact match from 57 to
71–77 % and K = 6 from 71 to 83–88 %; fixing arbitrary positions helps less (random two seeds: 69–71 %).  So the
refinement fails to repair a *specific minority* of first-forward errors, and once those are right the rest converges
fast.  This is the "initial candidate quality" cause (brief §4 expectation): it is real but secondary to the update
rule — with the true neighbours fixed the rule converges, and with the sequential rule the same starting state already
reaches 90 % (§5).  Oracle seeds are diagnostic only.

## 5. Experiment E — update policies (K = 6 and K = 8; GPU 7 shared, tok/s not comparable)

| K | rule (rounds) | GSM8K dev 50 | tokens / forward | corrections / 100 tok | converged / cycle / budget |
|---|---|---|---|---|---|
| 6 | Jacobi (3) | 54 % | 1.57 | 108 | 419 / 27 / 791 |
| 6 | nonadjacent (3) | 58 % | 1.62 | 60 | 537 / 2 / … |
| 6 | confidence-gated (3) | 50 % | 1.70 | 59 | 909 / 1 / 377 |
| 6 | one change (3) / two (3) | 52 % / 56 % | 1.60 / 1.62 | 38 / 63 | 528 / 0 / 816 |
| 6 | damped (3) | 48 % | 1.61 | 81 | – |
| 6 | **odd-even (3 rounds = 6 probes)** | **78 %** | 1.03 | 76 | – |
| 6 | Jacobi (6) | 62 % | 1.19 | 139 | 996 / 87 / 282 |
| 6 | **one change (12)** | **94 %** | 1.11 | 56 | 1 220 / … |
| 8 | Jacobi (3) | 18 % | 1.98 | 147 | 118 / 1 / 840 |
| 8 | nonadjacent / confidence / one / two / damped (3) | 28 / 28 / 18 / 32 / 28 % | 2.0–2.2 | 30–91 | – |
| 8 | **odd-even (3 = 6 probes)** | **78 %** | 1.20 | 114 | 402 / 14 / 579 |
| 8 | Jacobi (6) | 48 % | 1.28 | 200 | 468 / 46 / 467 |
| 8 | one change (12) | 90 % | 0.97 | 79 | 713 / 47 / 205 |
| 8 | two (6) | 78 % | 1.33 | 97 | 530 / 16 / 480 |

(The Phase E3 matrix on an idle A100 adds top-M, independent-set and colouring rules at K = 8:
docs/dependency_aware_refinement.md §5.)

## 6. Experiment G — EOS policy (K = 8, Jacobi, 3 rounds; K = 6 in results/pure/kscale_K6_eos_*.json)

| policy | GSM8K | tokens / forward | premature EOS | max-length |
|---|---|---|---|---|
| current (a stop id in the final block cuts it) | 18 % | 1.98 | – | 3 |
| suppress EOS in the first forward only | 16 % | 1.97 | – | 3 |
| verify: one more probe must leave the EOS position unchanged | 24 % | 1.97 | – | – |

Early EOS is a *consequence* of broken blocks, not a cause: suppressing it does not help and the Jacobi failure mode
(unconverged blocks) dominates; at K = 8 the output length (151–190 tokens) is similar to K = 4 (170).

## 7. Teacher-forced refinement statistics (bench/diag_k4.py at K = 6 / 8; blocks of the model's own answers)

| K | strategy | forwards / block | first-forward acc by position | final acc by position | token acc | block exact | success | regression |
|---|---|---|---|---|---|---|---|---|
| 6 | oracle conditional (true neighbours) | – | – | .95 .96 .96 .96 .96 .98 | 96.0 % | – | – | – |
| 6 | first forward only | 1 | .97 .83 .70 .57 .46 .39 | same | 65.3 % | 23.0 % | – | – |
| 6 | refineN Jacobi ≤ 3 | 3.41 | | .94 .90 .86 .83 .73 .67 | 82.1 % | 56.8 % | 60.8 % | 6.6 % |
| 6 | sequential (AR ceiling) | 6 | | .97 .95 .94 .92 .90 .88 | 92.6 % | 85.8 % | 81.9 % | 1.7 % |
| 8 | oracle conditional | – | – | .92 .95 .95 .96 .95 .97 .96 .97 | 95.5 % | – | – | – |
| 8 | first forward only | 1 | .97 .83 .69 .58 .47 .36 .33 .29 | same | 56.5 % | 9.4 % | – | – |
| 8 | refineN Jacobi ≤ 3 | 3.74 | | .93 .89 .85 .82 .73 .65 .56 .47 | 73.7 % | 34.2 % | 53.6 % | 10.9 % |
| 8 | sequential (AR ceiling) | 8 | | .97 .96 .94 .93 .90 .88 .88 .85 | 91.4 % | 82.0 % | 82.7 % | 1.9 % |

Brief §16 question 3: **yes** — with correct neighbours every position of a K = 8 block is recovered at 92–97 %
(no worse than K = 4's 96–98 %).  The conditional restorer is not the bottleneck; reaching a correct configuration is.
The first forward of K = 8 has 3.5 wrong tokens per block (block exact 9 %) and Jacobi fixes 54 % of them while
breaking 11 % of the right ones; the same model with one change per round fixes 83 % and breaks 2 %.

## 8. Oscillation summary (free generation, all K)

Cycles (a block state seen before) are 0.1–1.6 % of blocks for Jacobi at every K; the dominant non-convergence is
budget exhaustion (§3), and among sequential rules cycles rise to 4–8 % at 12 rounds (e.g. K = 8 one ≤ 12: 47 of 965
blocks).  Per-change direction counts (§7) show that Jacobi's extra changes are mostly wrong→wrong churn.

## 9. Held-out confirmation (dev items 50–199; AR 90.0 %)

| setting | dev 0–49 | held-out | tokens / forward |
|---|---|---|---|
| K = 6, 2+2+2 init + Jacobi ≤ 3 | 80 % | **58.7 %** | 1.18 |
| K = 8, colouring ≤ 6 / top-2 ≤ 6 / wmis1 ≤ 12 (E3) | 82 / 78 / 88 % | 58.7 / 68.0 / 81.3 % | 1.31 / 1.31 / 1.06 |
| K = 4 Jacobi ≤ 6 (reference) | 92 % | 84.7 % | 1.17 |

## 11. Measured cost (idle A100, docs/dependency_aware_refinement.md §2; zeta: four_token_diffusion.md §9)

AR HF 54.6 tok/s; K = 4 Jacobi ≤ 3 graph 99.6 tok/s @ 88 %; K = 6 (plain, zeta) 103.6 tok/s @ 60 %; K = 8 Jacobi ≤ 3
(plain, zeta) 123.3 @ 32 %; K = 8 one ≤ 12 (plain, A100) 40.9 @ 90 %.  Probe rows K² = 16 / 36 / 64; the probe is
~2/3 of decode time at every K.  VRAM: 4B weights + cache ≈ 9 GB for all K (plain engine).

## 12. Conclusion on the cause of the K = 8 failure (measured, in order of weight)

1. **Update rule** (dominant): Jacobi adoption of all changes at once churns strongly co-dependent neighbours
   (E3 §3: 58 % wrong→wrong when co-changed with a JSD ≥ 0.3 neighbour) and needs more rounds than any fixed budget
   gives; replacing it by sequential / colouring / independent-set adoption turns 18 % into 68–90 % at the same model.
2. **Initial prediction** (second): the first forward is right on 56 % of K = 8 tokens (9 % blocks exact) vs 77 % at
   K = 4 (48 %); pairwise initialisation (2+2+2+2) doubles K = 8 to 42 % under Jacobi but is itself nearly sequential.
3. **Model capability**: not the cause — the oracle conditional is 92–97 % at every K = 8 position.
4. **EOS**: a symptom, not a cause.
5. **Implementation bug**: none found (§10).

## 13. Summary table (dev 0–49, Jacobi ≤ 3 unless noted; "initial" = first-forward token accuracy, teacher-forced)

| K | GSM8K | initial | correction success (Jacobi / sequential) | regression (Jacobi / seq) | cycle rate | early EOS | tok/s (idle) |
|---|---|---|---|---|---|---|---|
| 3 | 94 % | – | – | – | 1.7 % | 1 | – |
| 4 | 88 % | 77.3 % | 72 % / 80 % | 3.3 / 1.3 % | 1.5 % | 4 | 99.6 (graph) |
| 5 | 62 % | – | – | – | 0.8 % | – | – |
| 6 | 54 % | 65.3 % | 61 % / 82 % | 6.6 / 1.7 % | 2.2 % | – | 103.6 (plain, zeta, 60 %) |
| 7 | 34 % | – | – | – | 0.5 % | – | – |
| 8 | 18 % | 56.5 % | 54 % / 83 % | 10.9 / 1.9 % | 0.1 % | – | 123.3 (plain, zeta, 32 %) |

Experiments B and C: §4b–4c (results/pure/diag_kscale_sft3.json).

