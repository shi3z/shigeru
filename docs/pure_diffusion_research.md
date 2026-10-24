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

### 2.1 Teacher-forced mask recovery (`bench/diag_mask_recovery.py`, final checkpoint runs/phaseC_sft3_4b/final)

100 dev prompts, the model's own greedy AR answers as references (AR dev accuracy 95 %), blocks of 32, one forward per
(block, layout, γ) in observed-first order on the clean prefix; 6 060 forwards, 144 s on GPU 7.  Token accuracy = argmax
of the mask row equals the reference token.  results/pure/diag_recovery_sft3.json.

| layout : γ | token acc | block exact | acc by distance to the nearest observed token on the left: 1 / 2 / 3 / 4 / 5 / 6 / 7 / ≥8 |
|---|---|---|---|
| random : 0.10 | 90.3 % | 75.7 % | .91 .77 .50 – – – – – |
| random : 0.25 | 80.1 % | 26.2 % | .83 .71 .62 .41 .50 – – – |
| random : 0.50 | 59.3 % | 5.4 % | .67 .59 .47 .39 .26 .26 .36 .00 |
| random : 0.75 | 40.8 % | 2.8 % | .54 .49 .41 .33 .27 .25 .21 .14 |
| suffix : 0.10 | 84.9 % | 70.0 % | .98 .85 .70 – – – – – |
| suffix : 0.25 | 59.7 % | 21.3 % | .97 .86 .73 .61 .51 .37 .36 .27 |
| suffix : 0.50 | 38.1 % | 5.6 % | .97 .78 .67 .56 .47 .37 .31 .20 |
| suffix : 0.75 | 29.4 % | 3.3 % | .99 .85 .68 .57 .43 .37 .33 .15 |
| any : 1.00 (block from scratch) | 24.6 % | 2.8 % | .97 .84 .73 .61 .49 .35 .31 .13 |

Calibration (random, γ=0.5; suffix in brackets): max-prob ≥ 0.99 → 99.7 % (100 %) correct, n = 913 of 4 282 masked
tokens (734 of 6 347); ≥ 0.95 → 97.6 % (97.1 %); ≥ 0.90 → 92.2 % (95.0 %); ≥ 0.80 → 85.3 % (87.1 %); < 0.5 → 33.9 % (20.4 %).

Reading (measured):
1. **Reach, not noise, is the limit.**  The token right after an observed token is recovered at 97–99 % (the AR
   conditional, as it should be — the model is the AR model on that row).  Accuracy falls almost linearly with the
   distance to the last observed token: 84 % at 2, 73 % at 3, 61 % at 4, 49 % at 5, ~13 % at ≥ 8.  A block of 32
   predicted from scratch gets 25 % of its tokens right and essentially never the whole block (2.8 %).  Therefore any
   schedule that fills many tokens per step (S ≪ K) must lose accuracy on this checkpoint; one-step or two-step
   denoising of a 32-block is out of reach without retraining.
2. **Right context barely helps this suffix-trained checkpoint.**  At equal γ the random layout beats suffix on
   token accuracy only because its masks are closer to observed tokens; at the same left distance the suffix layout is
   as good or better (distance 1: .97–.99 vs .54–.91).  Splitting by "has an observed token on the right": γ=0.10
   90 % vs 97 % (right context *hurts*), γ=0.25 80 vs 82, γ=0.50 60 vs 56, γ=0.75 42 vs 31 (helps only when the left
   context is far).  Expected: stage 2 of this checkpoint never showed it a right-side observed token (§1.2).  The
   same diagnostic on the random-mode checkpoint runs/phaseC_sft_4b/final and on runs/zeta_best_28000/final is queued.
3. **Confidence is well calibrated**, so confidence-adaptive filling is sound: at γ=0.5 about 21 % of the masked tokens
   carry ≥ 0.99 and are 99.7 % correct, 36 % carry ≥ 0.95 at 97.6 %.  This predicts that an adaptive sampler
   (fill every mask with confidence ≥ τ, τ≈0.95) can commit ~2–4 tokens per step at near-AR accuracy *(estimated from
   the table; measured in §2.2)*, while fixed schedules with ≥ 8 tokens per step cannot.
4. The remaining question for training (§5) is how far the reach can be pushed: the CPT saw random masks (reach is a
   property the 1 B tokens did not buy — the reach curve is the same shape as in distill_design.md §10.9/11.1), so the
   lever is a loss that targets distant positions and block consistency, not more tokens of the same objective.

### 2.2 Pure block diffusion decoder on the final checkpoint (`bench/run_block.py`, 50 dev prompts, GPU 4, bf16, plain engine)

All rows: runs/phaseC_sft3_4b/final, K = block size, S = denoising steps per block (fixed linear schedule unless noted),
greedy argmax, 384-token budget, no AR logits read.  AR greedy on the same 50 prompts: **95 %** (HF, 52.5 tok/s).
tok/s are wall-clock with the un-fused plain forward (no CUDA graphs); they are upper-bounded by the step count.

| K | S | fill order | GSM8K dev (50) | tokens/step | tok/s | file (results/pure/) |
|---|---|---|---|---|---|---|
| 32 | 32 | **leftmost** (1/step, = AR conditional) | **94 %** | 0.92 | 41.5 | block_sft3_dev_K32_S32_leftmost.json |
| 32 | 32 | confidence (1/step, any position) | 40 % | 0.91 | 41.9 | block_sft3_dev_K32_conf_linear.json |
| 32 | 16 | confidence | 32 % | 1.86 | 83.4 | same |
| 32 | 16 | entropy / margin / random | 28 / 18 / 4 % | 1.85 / 1.90 / 1.99 | 86 / 86 / 90 | block_sft3_dev_K32_S16_*.json |
| 32 | 16 | confidence, cosine / geometric schedule | 22 / 26 % | 1.88 / 1.89 | 86 / 86 | block_sft3_dev_K32_S16_conf_*.json |
| 32 | 8 | confidence | 10 % | 3.86 | 171 | block_sft3_dev_K32_conf_linear.json |
| 32 | 4 / 2 / 1 | confidence | 0 / 0 / 0 % | 7.95 / 16 / 32 | 344 / 688 / 1287 | same |
| 8 | 8 | confidence (1/step) | 72 % | 0.98 | 41.7 | block_sft3_dev_K8_16_conf_linear.json |
| 8 | 4 / 2 / 1 | confidence | 40 / 12 / 0 % | 1.96 / 3.95 / 8.0 | 83 / 161 / 327 | same |
| 32 | adaptive, conf ≥ 0.90 / 0.95 / 0.99 (fallback: most confident mask) | confidence | 48 / 46 / 44 % | 1.12 / 1.06 / 0.98 | 52 / 50 / 46 | block_sft3_dev_K32_adaptive_tau*.json |

Reading (measured):
1. The decoder is correct: left-to-right one token per step reproduces the AR accuracy (94 vs 95 %, the 1-point gap is
   bf16 numerics of the different forward shapes, cf. distill_design.md §11.6).
2. **Order alone costs 54 points at equal speed** (94 → 40 % at one token per step).  Filling the most confident
   position first — typically a formatting token far ahead ("\n\n", "**", "---") — fixes anchors that the suffix-trained
   model then cannot write towards (it never learned to use right context, §2.1); the typical failures are duplicated
   fragments across adjacent masks ("attatt", "jumps jumps", "= =") and dropped tokens ("-attie").  Random order (4 %)
   is the floor; confidence > entropy > margin for choosing positions; cosine/geometric schedules do not help.
3. Any genuine parallelism collapses: 2 tokens/step → 32–40 %, 4 → 10–12 %, 8 → 0 %.  Smaller blocks help at equal
   tokens/step (K=8, S=8: 72 % vs K=32, S=32: 40 %) because the far positions are simply absent.
4. The adaptive threshold does not rescue it on this checkpoint: ≥ 0.99 confidence is 99.7 % *correct* (§2.1) but
   those tokens are the far formatting tokens whose early commitment breaks the text, and the mandatory fill of a step
   was the most confident mask rather than the leftmost one — fixed by `anchor_leftmost` (§3), measured next.
5. Conclusion for Phase B: on the existing suffix-trained checkpoint no sampler reaches the brief's Phase 1 (80 %)
   with more than ~1 token per step; the limiting factor is the model, not the sampler (§2.1: reach and right context).
   Phase C therefore starts immediately (stage-2 re-training with random / mixed masks on the final CPT, §5), while the
   anchored adaptive sampler and the random-mask checkpoint runs/phaseC_sft_4b/final are measured on GPU 4.

Recovery diagnostic on other checkpoints (same protocol as §2.1, 100 dev prompts):

| checkpoint (stage-2 mask mode) | random γ=.25 | random γ=.5 | random γ=.1 block exact | suffix γ=.25 | suffix γ=.5 |
|---|---|---|---|---|---|
| phaseC_sft3_4b (suffix, 1 B CPT) | 80.1 % | 59.3 % | 75.7 % | 59.7 % | 38.1 % |
| zeta_best_28000 (suffix, 670 M CPT) | 81.6 % | 62.8 % | 76.8 % | 60.3 % | 37.5 % |
| **phaseC_sft_4b (random, 60 M CPT)** | **89.3 %** | **76.3 %** | **91.4 %** | 43.1 % | 27.3 % |

The random-mask checkpoint recovers scattered masks far better (+9 / +17 points) and contiguous suffixes worse (−17):
right-context use is learned or lost in stage 2, independent of CPT volume.  This is the training lever for the block
sampler (§5).

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
