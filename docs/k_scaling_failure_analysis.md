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

## 3–13. Results
*filled as the runs complete (Step 1 reproduction → A → B → C → D → E → F/G → audit → held-out)*
