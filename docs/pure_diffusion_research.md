# SHIGERU: pure diffusion decoding without AR verification — research log

Brief (2026-10-24, "Mercury 3型・純粋拡散言語モデルへの進化"): turn the 4B causal-diffusion student into a decoder that
fills several tokens of a block in parallel, refines them iteratively, and never consults the AR head at inference.
Everything below is **measured** unless marked *estimated* or *not measured*.  GSM8K **test** (1 319) is reserved
for final rows; strategy and hyper-parameter choices use `bench/prompts_gsm8k_dev.json` (200 GSM8K **train**
questions not present in any training file, `bench/make_gsm8k_dev.py`, seed 0; 6 773 of 7 473 train questions
qualified).

## 1. Audit (Phase A) — what the repository actually does

### 1.1 Starting numbers (verified in results/, all GSM8K test 1 319, one A100 80 GB, bf16, batch 1, 384 new tokens)

| Row | Checkpoint | Decoding | GSM8K | tok/s | tokens/step | File |
|---|---|---|---|---|---|---|
| AR | runs/phaseC_sft3_4b/final | HF greedy | 88.9 % | 52.5 | 1.0 | results/gsm8k_full_phaseC_sft3_4b_ar.json |
| AR-verified diffusion | same | streaming window W=16, drafts verified by the AR rows, τ=∞, CUDA graphs | 87.6 % | 156.0 | 2.15 | results/cdlm_sft3_4b_full_graph_tauinf.json |
| Unverified diffusion | zeta snapshot-6000 + 1-epoch stage 2 (suffix masks) | streaming window, entropy threshold τ=0.4, no verification | 77.3 % | – (zeta) | 1.20 | docs/distill_design.md §11.2 |
| Unverified diffusion | runs/phaseC_sft3_4b/final, 20 dev-like test prompts | HF-cache decoder τ=0.4 / 0.7 | 70 % / 65 % | 62.0 / 75.3 | 1.39 / 1.59 | results/cdlm_sft3_4b_plain.json |

The 77.3 % was never re-measured on the final checkpoint on the full set; the 20-prompt numbers are the only
unverified measurements of the final model (not measured: full-set unverified accuracy of the final checkpoint).
Note the unverified rows run at 1.2–1.6 tokens/step — barely parallel — and already lose 10–20 points.

### 1.2 What the model was trained to do (engine/cdlm/data.py, mask.py, loss.py)

* Dual stream `[x0 | xt]`, blocks of B = 32, per block γ ~ U(0,1], `round(n·γ)` response tokens masked.  **Attention
  inside an xt block is causal in observed-first physical order**: a mask row sees the clean prefix (earlier blocks,
  x0 rows), **every observed token of its block — left and right of it in logical position** (RoPE keeps the original
  positions), and the masks physically before it.  Mask rows never see later masks.  So the trained conditional is
  p(x_p | prefix, observed subset of the block), the MaskGIT/LLaDA-style "bidirectional over observed tokens"
  conditional, restricted to one block.  No architecture change is needed for a blockwise masked-diffusion sampler;
  only mask→mask attention is missing, which carries no token information.
* Losses: 1/γ-weighted CE on the mask rows (`mlm_loss`, WeDLM normalisation) + plain next-token CE on the x0 rows
  (`ar_loss`), equal weight.  Nothing trains block consistency, multi-step refinement or remasking; every mask is
  predicted independently given the observed set (conditional-independence assumption of one denoising step).
* `mask_mode`: the 1 B-token CPT (runs/phaseC_cpt3_4b) used **random** subsets (default); the final stage 2
  (runs/phaseC_sft3_4b, zeta_best_28000) used **suffix** masks — a mask never had an observed token on its right.
  Right-context conditioning is therefore *out of the stage-2 distribution* for the final checkpoints (the CPT saw it).
  Random-mode stage-2 checkpoints exist (runs/phaseC_sft_4b/final on the 60 M CPT; zeta snapshot models).

### 1.3 What the inference code does (engine/cdlm/decoder.py, fast_decoder.py, fast_model.py)

* The streaming-window decoder (W masks after the frontier, observed-first rows, causal mask over [cache | window]) is
  a WeDLM-style *sliding* sampler: each step fills the masks whose `entropy + λ·distance < τ`, commits only the
  leftmost filled run, never revises a filled token, and the K/V of committed rows are reused (they attended only the
  prefix and earlier committed rows).  A filled token after a gap is re-encoded each step but never re-predicted.
* The headline numbers use `verify=True`: the mask predictions are drafts and the **AR logits of the same forward
  decide**; the output is the AR greedy output.  This is speculative decoding with a diffusion drafter — explicitly
  excluded by the brief.  The brief's three starting figures are confirmed: 88.9 (AR), 87.6 @ 156 (AR-verified),
  77.3 (unverified, different checkpoint).
* The CUDA-graph engine (`fast_model.FastQwen3`) exposes `run(tokens, pos, cache_len, rows)` = one forward of
  arbitrary rows with explicit positions over [cache | rows] under a causal window mask, with static buckets
  (1/4/8/16/32 rows) and cache buckets of 256.  This is enough for a block denoiser: a step of a block of K is one
  `run` of K rows (filled asc | masks asc).

### 1.4 Problems found, by priority

1. **No pure parallel sampler exists.**  Unverified mode = the sliding sampler with an entropy threshold; it fills
   ahead of the frontier only when confident, so it is nearly sequential (1.2–1.6 tok/step) and still loses accuracy,
   because (a) tokens filled in one step are conditionally independent (multi-token numbers split, §10.3 of
   distill_design.md), (b) a wrong early fill is never revised, (c) a mask never benefits from tokens to its right.
2. **Training/inference mismatch of the final checkpoints**: suffix-mask stage 2 vs any sampler that fills
   non-contiguous positions.  Cheap to fix (stage 2 is 3 h 50 on one A100; random/mixed-mode checkpoints exist).
3. **No training signal for refinement**: the model never saw its own partial generations (no on-policy states), was
   never trained to re-predict a token it filled, and the loss is per-token.
4. Denoising-step count is not a training variable: every training block is one step from one random γ.
5. The AR-verified path hides all of this because the AR head corrects everything.

## 2. Diagnostics on the existing checkpoint (Phase A/B, dev set, no retraining)

### 2.1 Teacher-forced mask recovery (`bench/diag_mask_recovery.py`)
*running — results go here*

### 2.2 Pure block diffusion decoder (`engine/cdlm/block_decoder.py`, `bench/run_block.py`)
*implemented; CPU tests `engine/cdlm/tests/test_block.py`; sweeps go here*

## 3. Implemented solution

`engine/cdlm/block_decoder.py` — blockwise masked diffusion on the existing FastQwen3 engine: K masks per block,
S denoising steps (linear / cosine / geometric fill schedules, or adaptive confidence threshold), fill order by
confidence / entropy / margin / random / leftmost, optional low-confidence remasking, one natural-order re-encoding
per committed block (merged into the next block's first forward).  No AR logits are read (test
`test_no_ar_head_dependence` perturbs every non-mask row's logits and checks the output is unchanged).

## 4. Theory and hypotheses
*to be written with the diagnostics*

## 5–7. Experimental conditions, GSM8K results, tokens/sec
*tables are added as runs complete; every JSON under results/pure/ records model, prompts, GPU, dtype, torch, seed,
batch size 1, max_new_tokens, warm-up*

## 8. Ablations · 9. Failed experiments · 10. Open questions
*pending*
