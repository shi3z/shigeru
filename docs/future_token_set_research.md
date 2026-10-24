# SHIGERU Phase F — Position-Free Future Prediction (PFFP): diagnostic result

Brief (2026-10-24): before building a Future Token Set Head and a set-guided 16-token diffusion decoder, measure on the
current SHIGERU whether the *content* of the next 16 tokens is predicted much better than their *positions*.  Gate
(brief §3/§11, Phase F1): go on only in **Case A** — position accuracy low, multiset F1 high, robust to duplicates and
across genres.  Everything below is measured; nothing was trained.  Companion docs: docs/pure_diffusion_research.md
(Phases A–D), docs/four_token_diffusion.md (Phase E).

## 1. Hypothesis

H-PFFP: "the model cannot place the next 16 tokens but it knows which tokens are coming", so generation could be split
into content prediction (a multiset of future tokens) and sequence reconstruction (ordering them).  Prediction: at
K = 16, multiset F1 ≫ position accuracy, with few duplicate predictions.

## 2. Relation to prior work (brief §12.2)

Set / bag-of-words prediction as an auxiliary or planning signal appears in non-autoregressive MT (fertility / bag-of-
words losses), in "plan-then-realise" generation, and in order-agnostic / MaskGIT-style parallel decoding where
confidence-based unmasking is the reconstruction step.  Our measurement asks whether the existing causal-diffusion
rows already carry such a signal; it does not reproduce any of those systems.

## 3. Method (`bench/diag_pffp.py`, no training, no AR logits)

Teacher-forced: clean prefix in the cache, K [MASK] rows at the next K positions, one forward (the first step of any
block decoder).  Scores per block, top-1 per position unless stated:
position accuracy (and by distance); **multiset F1** (multiset intersection / K; with K predictions and K references
P = R = F1); optimal assignment accuracy (maximum bipartite matching pred→gold, equal to the multiset intersection —
reported as a check, identical in every cell); top-k future recall as a **matching** (each position offers its top-k,
each gold occurrence matched at most once: no credit for duplicates, no credit for a token twice); fixed-candidate
recall (the K or 2K most probable (position, token) pairs of the whole block, matched the same way — so the candidate
count does not grow with k); duplicate rate (1 − unique top-1 / K); count error (Σ_t |count_pred − count_gold| / K);
block exact match.  Data (never used for training): GSM8K dev (50 items, the model's own greedy answers as reference,
so "right" = what the AR path writes); code (57), English (37: en_general + reasoning) and Japanese (28) teacher
responses from data/distill/phaseA_medium_{test,val}.jsonl and phaseA_codecomp_test.jsonl.  Blocks at stride K, ≤ 6 per
item; K = 4, 8, 16, 32.  GPU 6, bf16; 3 908 forwards, 164 s per checkpoint.  results/pure/diag_pffp_{sft3,sftrandom}.json.

## 4. Results

### 4.1 Final checkpoint runs/phaseC_sft3_4b/final (1 B CPT + suffix stage 2)

| genre | K | position acc | **multiset F1** | gap | duplicate rate | count err / tok | top-5 recall | top-20 recall | fixed K / 2K cands | block exact |
|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 4 | 71.1 % | 73.7 % | +2.6 | 7.7 % | 0.53 | 91.9 % | 97.3 % | 71.8 / 86.2 % | 33.7 % |
| gsm8k | 8 | 51.4 % | 61.4 % | +10.0 | 26.5 % | 0.77 | 87.3 % | 95.9 % | 61.0 / 76.5 % | 3.3 % |
| **gsm8k** | **16** | **35.4 %** | **50.5 %** | **+15.1** | **49.0 %** | 0.99 | 79.3 % | 92.9 % | 51.3 / 65.7 % | 0.0 % |
| gsm8k | 32 | 24.2 % | 39.1 % | +14.9 | 70.2 % | 1.22 | 70.9 % | 89.4 % | 41.8 / 54.1 % | 0.0 % |
| code | 4 | 73.9 % | 78.6 % | +4.7 | 7.5 % | 0.43 | 89.1 % | 95.5 % | 78.7 / 84.4 % | 53.8 % |
| code | 16 | 35.6 % | 48.4 % | +12.7 | 47.4 % | 1.03 | 72.8 % | 86.4 % | 49.7 / 61.9 % | 0.3 % |
| en | 4 | 41.8 % | 45.7 % | +3.9 | 16.2 % | 1.09 | 67.6 % | 80.3 % | 46.9 / 57.6 % | 10.4 % |
| en | 16 | 16.4 % | 26.6 % | +10.2 | 62.1 % | 1.47 | 46.9 % | 63.8 % | 28.7 / 36.3 % | 0.0 % |
| ja | 4 | 38.4 % | 43.4 % | +5.1 | 22.6 % | 1.13 | 64.9 % | 77.4 % | 45.1 / 55.6 % | 6.6 % |
| ja | 16 | 12.4 % | 19.4 % | +7.0 | 68.8 % | 1.61 | 37.7 % | 57.3 % | 21.9 / 27.9 % | 0.0 % |

Position accuracy by distance at K = 16 (gsm8k): .97 .79 .69 .57 .44 .32 .27 .24 .27 .18 .20 .21 .15 .13 .13 .12.

### 4.2 Random-mask checkpoint runs/phaseC_sft_4b/final (60 M CPT + random stage 2) — same protocol

gsm8k K=16: position 26.4 %, multiset F1 36.7 % (+10.3), duplicates 63.5 %, top-20 recall 87.0 %, fixed 2K 51.8 %.
code K=16: 20.3 / 29.5 %; en K=16: 13.6 / 20.8 %; ja K=16: 10.7 / 16.1 %.  Uniformly below the final checkpoint.

## 5. Verdict against the brief's three cases

* **Not Case A.**  Ignoring positions recovers +10 to +15 points at K = 16, but multiset F1 is only 50 % on GSM8K (the
  model's own text), 48 % on code, 27 % on English prose and 19 % on Japanese.  Half of the content is not predicted
  even when any order is accepted.
* **Case C is present:** 49 % of the K = 16 top-1 predictions duplicate another prediction in the block (70 % at K = 32;
  on prose 62–69 %).  The far mask rows collapse onto the same high-frequency tokens (newline, "the", digits, "**"),
  so the multiset the model proposes is a frequency prior, not a plan; count error ≈ 1 token per position.
* **Case B dominates:** the per-distance curve of position accuracy (.97 → .12) is the same curve as the recovery
  diagnostic of pure_diffusion_research.md §2.1, and the multiset score inherits it: the rows beyond distance ~4 do not
  know *what* comes, only what is frequent.  Top-k recall looks high (93 % at k = 20) only because 16 × 20 = 320
  candidates are offered; with a fixed budget of 32 candidates per block the matched recall is 66 % (54 % at K = 32),
  and 36 % on English prose.
* Genre robustness fails: the gap shrinks to +7–10 on prose and Japanese, where both numbers are far lower.

**Gate not passed.**  Per brief §3 ("Case A に相当する十分な結果が出ない場合、大規模な追加学習へ進まない") no Future Token
Set Head and no set-guided decoder are trained on this evidence.

Caveat (stated, not measured): this diagnostic reads the *positional* mask rows as set predictors.  A head trained
explicitly for the multiset could score higher than 50 % — but the brief's gate is about the current model, and the
duplicate pattern shows that the information a set head would have to extract from the prefix hidden state (the
content 5–16 tokens ahead) is exactly what the distance curve says the model does not represent reliably.  A cheap
pre-test, if this line is to be pursued later: a 1-layer projection + shared LM head trained with multi-label BCE on
~50 k prefixes (minutes on one A100) and the same matching-based recall at a fixed 32-candidate budget; the bar to
beat is 66 % (gsm8k) / 36 % (en).

## 6. Consequence for decoding (what the numbers do support)

* Near-deterministic positions exist and are well calibrated (§2.1 of pure_diffusion_research.md): the anchored
  adaptive sampler (leftmost + conf ≥ 0.99) is the only pure-diffusion setting that holds ≥ 80 % dev accuracy, at
  1.28 tokens/step.
* K = 4 (docs/four_token_diffusion.md) is the block size where position accuracy and multiset F1 nearly coincide
  (71 vs 74 %) and block exact match is 34 % (54 % on code): ordering is not the problem there; the fourth token is.
* The failure is content at distance ≥ 3–4 on *this* training distribution.  The training-side experiments that
  target it are running (random / mixed masks; on-policy states; self-generated low-entropy responses, H1 in
  pure_diffusion_research.md §2.4) and are the right next measurement before any set head.

## 7–11. Head structure · set-guided decoder · ablations · GSM8K/HumanEval · speed/VRAM
Not built (gate failed).  Failed experiment recorded above; next research questions in §5–6.
