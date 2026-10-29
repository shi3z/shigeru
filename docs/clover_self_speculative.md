# Clover (DiffuSpace Clover-35B-A3B-Preview) and SHIGERU — what transfers, and the acceptance-rule experiment

Date: 2026-10-29.  Sources: github.com/DiffuSpace/Clover-35B-A3B-Preview, huggingface.co/Diffuspace/Clover-35B-A3B-Preview
(config.json, modeling_clover_moe.py, sampling_clover_moe.py), diffuspace.ai/blog/clover-35b, SGLang PR 46927.

## 1. What Clover is (from the released code and cards)

- Qwen3.6-35B-A3B (40 layers, 256 routed experts top-8 + 1 shared, 3 Gated-DeltaNet layers per full-attention layer,
  vocab 248,320, 128K) converted to **block diffusion**.  Apache-2.0, BF16 69 GB.  Training recipe not released
  (`sft/` "coming soon"; the modeling file raises NotImplementedError on `labels`).
- Inference = two passes of the same weights: `preview=True` (bidirectional attention inside the block, previous
  blocks read from the KV cache / GDN recurrent state) and `preview=False` (**causal commit** of the resolved block,
  one forward per block that writes the cache).  This is exactly the B-series layout (engine/bdiff: prefix cache +
  bidirectional block rows, block rows never appended to the cache until final).
- GDN bidirectionality = two extra weight files: a zero-initialised right-side convolution branch
  (`bd_gdn_conv_right_weight`, [channels, 1, kernel-1]) and three per-head softmax calibration scalars
  (`bd_softmax_logit_scale` 10, `bd_softmax_history_bias` 5, `bd_softmax_history_c` 1) that mix the block's
  softmax logits with a "history" logit depending on log(block size).  Only relevant for hybrid (linear attention)
  bases; SHIGERU's Qwen3 dense base has none.
- Sampler (HF reference): block_size 32, **LowConfidence**: fill every pending position whose confidence
  (probability of the sampled token) > 0.95, else the single most confident; **no re-masking** of filled positions;
  at most 32 forwards per block + 1 commit.  Penalties: presence 0.9, repetition 1.05, a ramp bias on the stop token
  after `dllm_length_penalty_start` (`dllm_length_penalty` 10) and a bias on `</think>` (`dllm_think_end_bias` 10).
- Production path (SGLang, llama.cpp): **self-speculative decoding** — the preview pass drafts a block (width 16 in
  the PR), the causal pass of the same weights verifies it in one forward, the accepted prefix is committed.
  Acceptance rules: `exact` (draft == penalty-adjusted verify argmax, stop at the first mismatch) and the default
  `threshold`: accept when p_target(draft) >= min(ratio, delta * exp(-H_target)) and > 0, delta 0.6, ratio 1.0.
  Neither is distribution-preserving (drafts are argmax; temperature/top-p only sample the bonus token).
  Reported: H200, MATH500 96.6 % (483/500), 450.6 tok/s per request at DP=1; 4268.7 aggregate at DP=4 (192 per
  request); GB300 680 tok/s; RTX 5070 Ti INT4 101.8 tok/s.  SWE-bench Verified 73.6, Terminal-Bench 2.1 51.7,
  GPQA Diamond 83.8, MMLU-Pro 84.8 (blog, settings not given).

## 2. Mapping onto SHIGERU

| Clover piece | SHIGERU counterpart | transferable? |
|---|---|---|
| preview / commit two-pass block diffusion | engine/bdiff (K = 16 block, prefix cache, LoRA mask-only adaptation) | already the same design; Clover confirms it at 35B |
| LowConfidence 0.95, no re-masking, fill >= 1 | sampler2 threshold / confidence modes; B7 found re-masking never beats the baseline | consistent; nothing new except the EOS length ramp and the think-end bias |
| self-speculative decoding (draft = diffusion block, verify = causal pass of the same weights) | engine/cdlm verified decoding (A-series headline: 87.6 % GSM8K, 156 tok/s A100, exact walk) | same mechanism; the B-series briefs excluded AR verification as the research default, Clover ships it as the product path |
| threshold acceptance rule | exact walk only (`verify_drafts`) | **this experiment**: `verify_drafts_threshold`, `--accept-rule threshold --accept-delta --accept-ratio` |
| GDN bidirectionalisation (zero-init right conv + calibration scalars) | none (dense attention) | only if SHIGERU moves to a Qwen3-Next/3.6-style hybrid |
| the model as a diffusion teacher | distill/gen_global_states.py-style on-policy trajectories | text / unmask-order level only: vocab 248,320 vs Qwen3 151,936 (no logit KL) |
| the model inside SHIGERU's engine | engine/cdlm, engine/bdiff are Qwen3-dense | no (GDN + MoE; SGLang already serves it with CUDA graphs) |

Hardware here: BF16 needs 69 GB — no free A100 while the H3 fleet holds GPUs 4/6/7; an INT4 GGUF (~18 GB) fits
GPU 5 with llama.cpp self-speculative, usable for a Phase F comparison row or trajectory generation.

## 3. Experiment — Clover's acceptance rule in SHIGERU's verified decoder

Hypothesis: the exact walk accepts only 15.5 % of drafts on GSM8K (A-series, tau = 1e9, W = 16: 2.15 tokens per
step, 1.16 accepted per step); an entropy-calibrated rule should accept far more at a bounded accuracy cost
(Clover keeps 96.6 % on MATH500 with it).

Implementation (commit 421bc2e): `engine/cdlm/decoder.py: verify_drafts_threshold(ar_logits, slots, is_mask,
delta, ratio)` — the AR rows that the exact walk already reads ([pending | drafts] rows of the verify forward),
softmax in float32, entropy H_j, accept slot j iff p_j(draft_j) >= min(ratio, delta * exp(-H_j)) and > 0, stop at
the first rejection, correction g = argmax of row j (the bonus token is greedy, as everything here is).
`generate_verified(..., accept_rule, accept_delta, accept_ratio)`, threaded through `decoder.generate`,
`fast_decoder.generate` (CUDA-graph engine) and `bench/run_cdlm.py`.  Tests: threshold(delta = inf) == exact;
delta = 0 accepts every draft; the 588-case `test_verify_equals_hf_greedy` is unchanged.

Protocol (zeta, RTX PRO 6000, model runs/phaseC_sft3_4b/final, fast_graph engine, W = 16, tau = 1e9, 384 new
tokens, greedy): dev2 (150 items, decisions) — exact + AR, threshold delta in {0.3, 0.6, 1.0} at ratio 1.0, and
delta 0.6 at ratio 0.5; then GSM8K test (1 319) for exact and the chosen setting.  Metrics: accuracy, tokens per
step (= per verify forward), draft acceptance rate, tok/s (same GPU for every row).

## 4. Results

### 4.1 dev2 (150 items, zeta RTX PRO 6000, W = 16, tau = 1e9, 384 new tokens, greedy; results/clover_dev2_*.json)

| decoding | accuracy | tokens / step | draft acceptance | accepted / step | tok/s |
|---|---:|---:|---:|---:|---:|
| AR greedy (HF generate) | 92.0 % | 1.00 | – | – | 78.4 |
| verified, exact walk | 93.3 % | 2.22 | 16.5 % | 1.23 | 186.3 |
| verified, threshold delta 0.3, ratio 1.0 | 92.7 % | 2.33 | 17.9 % | 1.34 | 194.1 |
| verified, threshold delta 0.6, ratio 1.0 (Clover default) | 93.3 % | 2.29 | 17.4 % | 1.30 | 190.7 |
| verified, threshold delta 1.0, ratio 1.0 | 92.7 % | 2.24 | 16.7 % | 1.25 | 187.1 |
| verified, threshold delta 0.6, ratio 0.5 | 93.3 % | 2.29 | 17.4 % | 1.30 | 190.8 |

- The rule changes little: +1.4 points of draft acceptance and +4 % tok/s at the loosest delta, accuracy within
  the noise of 150 items (one item = 0.67 points).  ratio 0.5 reproduces ratio 1.0 exactly: the cap never binds,
  i.e. the rule is governed by delta * exp(-H) alone, and delta from 0.3 to 1.0 moves acceptance by ~1 point.
- Why so small: the exact walk already rejects mostly drafts that are *implausible* under the AR distribution, not
  near-misses with an uncertain target (4.2).  Clover's rule can only rescue the latter.  On this drafter (the
  Phase C 4B conversion, whose mask rows were never trained on their own trajectories) the acceptance bottleneck is
  the drafter, as the B10/B11 result (on-policy trajectory data lifts the few-step student) already said.

### 4.2 Why the rule cannot do more — what the exact walk rejects (dev2, exact walk, `--draft-diag`, A100 GPU 5; results/clover_dev2_exact_diag_gpu5.json)

Of the drafts the walk examines (the leading drafted run plus the first rejection; 22,549 on dev2), **72.9 % are
accepted**.  The 16.4 % "draft acceptance rate" counts every drafted token, and most drafts are never examined:
the first rejection re-masks every draft to its right (step 5 of `generate_verified`), so they are discarded
unverified and re-drafted next step.

| rejected drafts (6,114) by p_AR(draft) | < 0.01 | 0.01–0.05 | 0.05–0.1 | 0.1–0.3 | 0.3–0.5 | >= 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| share | 83.3 % | 3.3 % | 1.9 % | 6.8 % | 4.6 % | 0.0 % |

| by AR entropy (nats) | < 0.5 | 0.5–1 | 1–1.5 | 1.5–2 | >= 2 |
|---|---:|---:|---:|---:|---:|
| accepted (16,435) | 84.7 % | 9.7 % | 3.7 % | 1.6 % | 0.3 % |
| rejected (6,114) | 65.4 % | 19.2 % | 10.0 % | 3.8 % | 1.7 % |

83 % of the rejections are drafts the AR model gives < 1 % probability, mostly where it is confident (H < 0.5):
plain misses of the drafter, not near-misses of an uncertain target.  Only 11.5 % of rejections have p >= 0.1 and
4.6 % have p >= 0.3 — the ceiling of any probability-threshold rule — which is exactly the +1.4 points the Clover
rule bought.  The lever is therefore the drafter (on-policy trajectory training, B10/B11), not the acceptance rule.

### 4.3 Negative: keeping the unverified right side after a rejection (`--keep-right`, A100 GPU 5)

Hypothesis: the first rejection re-masks every draft to its right; keeping them as drafts (still verified against
the AR rows conditioned on the correction before anything is committed — the 588-case harness still reproduces
greedy AR) would let the ones that survive the correction be accepted without re-drafting.
Result (dev2, exact walk): **1.07 tokens per step, 75.7 tok/s, acceptance 6.3 %** vs 2.22 / 155 / 16.4 % without.
Kept drafts are stale (drafted under the old left context: 25,213 of the examined drafts had p_AR < 0.01) and a
draft row is not a mask row, so it is never re-drafted — each stale draft costs one step to reject, and the loop
degenerates to AR.  Re-drafting the right side with the corrected context (what the exact walk does) is right;
the flag stays available, default off.

### 4.4 GSM8K test (1 319 items, zeta, same settings; results/clover_test_*.json)

| decoding | accuracy | tokens / step | draft acceptance | accepted / step | tok/s |
|---|---:|---:|---:|---:|---:|
| verified, exact walk (output == greedy AR) | 88.0 % | 2.15 | 15.5 % | 1.16 | 183.4 |
| verified, Clover threshold delta 0.3 | 88.6 % | 2.28 | 17.2 % | 1.29 | 191.1 |
| verified, Clover threshold delta 0.6 (Clover default) | 88.3 % | 2.24 | 16.7 % | 1.25 | 188.6 |

(The exact row reproduces the A-series A100 measurement: 87.6 % / 2.15 / 15.5 %, 156 tok/s there.)  Accuracy does
not drop (+7 and +4 items, within noise), throughput +4 % / +3 %.

## 5. Conclusions

1. Clover is the B-series design at 35B-A3B; its released code confirms the preview/commit two-pass block diffusion
   and a no-remask confidence sampler, and ships AR self-verification as the product path.  Nothing in the released
   artefacts tells how the diffusion ability was trained; the SFT release is the thing to watch.
2. Its acceptance rule is a small, safe win here: +1.4–1.7 points of acceptance, +3–4 % tok/s, accuracy unchanged on
   dev2 and test.  `--accept-rule threshold --accept-delta 0.3` is the setting to use if the 0.5-point accuracy
   wobble is acceptable; the exact walk stays the lossless default.
3. The rule cannot do more because the exact walk already accepts 73 % of what it examines and 83 % of its
   rejections are < 1 % probability under the AR model: the drafter is the bottleneck.  Keeping the unverified right
   side (keep_right) makes it worse (1.07 tokens per step).  The path to Clover-class acceptance is a better drafter
   (on-policy trajectory training, B10/B11), not the walk.
4. Clover itself can serve as a comparison row (INT4 on GPU 5 via llama.cpp) or as a text-level trajectory teacher
   (vocab differs from Qwen3, so no logit distillation); running it inside SHIGERU's engines is not worth it.
