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

## 4–13. Results (continued)
*filled as the runs complete (D → E → G → B/C/10 → audit → held-out)*
