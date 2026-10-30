# Phase B13 — auxiliary-conditioned drafter (PLM-style) for verified decoding

Brief (2026-10-30, following the Clover study): the exact verified decoder (engine/cdlm, window W = 16, "Quadratic
Decoding") is drafter-bound — 83 % of its rejected drafts have p_AR < 0.01 (docs/clover_self_speculative.md §4), and the
B-series showed that neither the acceptance rule nor more denoising steps move it.  PLM ("Parallel Language Models",
arXiv 2610.26693, justuswill.com/plm) attacks the drafter itself: the parallel decoder's input at a block position is not
an uninformative [MASK] but an **auxiliary token** carrying a coarse, sampled summary of the AR model's predictive
distribution there, and a **representation-alignment** loss ties the decoder's hidden states to the AR encoder's.  Their
150M-scale ablation (Table 4): tokens per call 1.44 (masks) → 1.705 (quantile auxiliaries) → 1.785 (Voronoi) → 1.85
(+ alignment).  B13 ports the idea to SHIGERU's dual-stream stage-2 recipe and measures tokens/step and tok/s of the
exact verified decoder on GSM8K dev2 and test.

## 1. What the auxiliary is here

PLM's auxiliaries are 512 fixed unit vectors assigned by a weighted Voronoi partition of the AR head's output
embeddings (the sampled target token's cell).  B13 uses the simpler **quantile / rank** family in its exact form:

    aux(t) = rank of the true token x_t under Q_t = p_AR(· | x_<t), the model's OWN AR distribution
    rank 0            = "x_t is the AR argmax"        (id cdlm_aux_base)
    rank k < 128      = "x_t is the (k+1)-th choice"  (id cdlm_aux_base + k)
    rank >= 128       = plain [MASK]                  (no information)

Q_t comes from a no-grad causal pass over the x0 (clean) stream of the same dual-stream example at every micro-step,
so the auxiliary always refers to the current model (PLM's encoder shares the weights).  The xt rows that build_example
marks as [MASK] are forwarded with the aux id instead; labels, 1/γ weights, suffix masks, AR loss and the attention
mask are unchanged (engine/cdlm/aux.py `assign_aux`, distill/train_cdlm.py `DualStreamForward`).

Why rank and not Voronoi: the production decoder is **exactly greedy**.  Feeding rank 0 on every mask row at inference
is the statement "every token of this block is the greedy continuation" — precisely the hypothesis the AR verifier
checks.  The drafter at position t+1 can then read x_t from the aux row at t (the argmax of that row's own prediction)
instead of marginalising over it; the marginalisation is what makes mask-trained drafts incoherent two positions out.
A Voronoi auxiliary would need a previous-step sample to condition on (two steps to carry information, and the
re-masking after a rejection discards it).  Training sees rank 0 on ~70 % of the masked rows (log field
`aux_rank0_frac`), the other ranks keep the input informative about the model's errors.

Alignment (PLM §3.3): cos similarity between the decoder's hidden state at layer ⌊L/3⌋ = 12 on a mask row (position t)
and the encoder's hidden state at layer ⌊L/2⌋ = 18 on the x0 row t−1 (the AR row that predicts x_t), through a learned
linear map on the decoder side (identity-initialised), stop-grad on the encoder side; loss = mean(1 − cos), weight 1
(PLM: 5; chosen lower because the mlm loss here is ~0.25 at convergence).  The map is a training-only module
(`model.cdlm_align_proj`, saved in resume checkpoints, dropped before the final HF save).

## 2. Implementation

| piece | where |
|---|---|
| auxiliary rows (128 ids above [MASK], mean + random unit direction × median row norm, trained), config `cdlm_aux_base` / `cdlm_aux_n` | engine/cdlm/aux.py `add_aux_tokens` |
| rank assignment (one no-grad causal pass over the x0 rows per micro-batch, ranks in 256-position chunks) | engine/cdlm/aux.py `assign_aux` |
| alignment hooks + loss | engine/cdlm/aux.py `AlignHooks`, `align_loss` |
| trainer: config keys `aux_tokens`, `aux_base`, `align_weight`, `align_dec_layer`, `align_enc_layer`; log fields `loss_align`, `aux_rank0_frac` | distill/train_cdlm.py |
| decoders: `exclude_from` (ids ≥ [MASK] removed from every argmax — verification and drafts — with a large finite negative logit, −inf would make the draft entropy NaN) | engine/cdlm/decoder.py `generate_verified`, fast_decoder.py |
| benchmark: `--draft-input mask | aux0 | mask_excl` | bench/run_cdlm.py |
| tests (CPU, tiny model): rank assignment checked against the causal logits by hand, forced-argmax → rank 0, alignment backward, aux0 verified decoding == greedy | engine/cdlm/tests/test_aux.py |
| configs / chain | distill/configs/b13_auxalign_4b.yaml, b13_aux_4b.yaml, distill/chain_zeta_b13.sh |

Cost: the extra no-grad pass is ~1/6 of a training step (forward only over L of the 2L rows); zeta logs 1,540 tokens/s
at step 1 vs 1,424 for the mask recipe on an A100, so the run time is the same ~3.7 h for 3 epochs.

## 3. Recipe

Exactly the stage-2 recipe of runs/phaseC_sft3_4b (docs/distill_design.md §11.4: base runs/phaseC_cpt3_4b/final, data
data/distill/phaseA_stage2c_train.jsonl 11,169 examples, block 32, max_len 2048, batch 1 × accum 16, lr 3e-5, 3 epochs,
suffix masks) plus the B13 keys, so phaseC_sft3_4b is the control:

| run | aux | align | GPU |
|---|---|---|---|
| runs/phaseC_sft3_4b (control) | [MASK] | — | A100 (2026-10) |
| runs/b13_auxalign_4b | 128 ranks | λ = 1, layers 12 → 18 | zeta RTX PRO 6000, 2026-10-30 |
| runs/b13_aux_4b | 128 ranks | — | zeta, after the first |

Evaluation (bench/run_cdlm.py, fast_graph engine, `--verify --tau 1e9 --window 16`, exact acceptance): dev2 (150) with
`--draft-input aux0` and, on the same checkpoint, `--draft-input mask_excl` (the B13 model fed [MASK]: how much of the
change is the input vs the training), AR rows for accuracy/tok/s; GSM8K test (1,319) with aux0.  Reference numbers of
the control on the same GPU: results/clover_dev2_exact.json — 2.22 tokens/step, 186 tok/s, 93.3 %, draft acceptance
16.5 %; test 88.0 % / 183 tok/s (docs/clover_self_speculative.md).

## 4. Results

GSM8K, fast engine with CUDA graphs, verified decoding W=16, tau=1e9 (exactly greedy); zeta (RTX PRO 6000).
Control = phaseC_sft3_4b, the same recipe without auxiliaries and without alignment.

dev2 (150 prompts):

| model | draft input | tok/step | tok/s | accuracy | draft acceptance |
|---|---|---|---|---|---|
| control phaseC_sft3_4b | [MASK] | 2.22 | 186 | 93.3 % | 16.5 % |
| B13 aux + align | rank-0 auxiliary | 2.26 | 190 | 92.0 % | 16.9 % |
| B13 aux only | rank-0 auxiliary | 2.28 | 192 | 92.7 % | 17.2 % |
| B13 aux + align | [MASK] | 1.18 | 99.5 | 92.0 % | 2.5 % |
| B13 aux only | [MASK] | 1.16 | 97.2 | 92.7 % | 2.1 % |
| B13 aux only, AR reference | — | 1.00 | 79.1 | 91.3 % | — |

GSM8K test (1319 prompts):

| model | draft input | tok/step | tok/s | accuracy |
|---|---|---|---|---|
| control phaseC_sft3_4b | [MASK] | 2.22 | 183 | 88.0 % |
| B13 aux + align | rank-0 auxiliary | 2.21 | 187 | 88.3 % |
| B13 aux only | rank-0 auxiliary | 2.21 | 188 | 88.9 % |

- **Negative result.**  On the full test set both B13 models draft 2.21 tokens per step against the control's 2.22.
  The dev2 edge (2.26 / 2.28 vs 2.22) does not survive 1319 prompts.  Accuracy is unchanged (88.3 / 88.9 vs 88.0,
  and exact greedy decoding guarantees the verified output equals the AR output of the same checkpoint).
- **Alignment adds nothing.**  aux only is 2.28 on dev2 and 2.21 on test; aux + align is 2.26 and 2.21.  The
  representation-alignment loss fell from 0.64 to 0.16 during training without moving a decoding metric.
- **The auxiliary does carry information during training, and none at inference.**  Fed [MASK] rows, both B13
  models collapse to 1.16-1.18 tokens per step (2 % acceptance): they learned to read the auxiliary instead of
  marginalising over the previous token.  But the auxiliary they are fed at inference is the constant "rank 0",
  whose content ("this is the AR argmax") is exactly what the verifier already enforces, so the drafter gains
  nothing over the mask-trained control.  This is PLM Table 6's conversion caveat in practice: the auxiliary is
  informative because it is *sampled* per position from Q_t, and sampling is incompatible with exact greedy
  verification, which is what this engine ships.
- Training-side signals (same seed, same examples as the control): mlm loss below the control from step ~500 on
  (3.12 vs 3.35 at 500, 1.88 vs 1.97 at 713, 1.51 at 1500), aux_rank0_frac 0.72 to 0.83.  The lower masked loss is
  the auxiliary leaking the target's rank, not a better drafter.
- Cost: 4 h 04 min per run on the RTX PRO 6000 (5.9-6.3 s/step), peak 48 GB, against 4 h 08 min / 35.7 GiB for the
  control on an A100: the no-grad AR pass per micro-batch and the alignment hooks cost ~12 GB and no wall-clock
  here (different GPU).

B13 is closed.  The engine keeps the auxiliary machinery (engine/cdlm/aux.py, `--draft-input aux0`, the
`exclude_from` argument of the decoders) because it is the only way to measure this family, and the two trained
checkpoints stay in runs/ for the sampled-auxiliary follow-up in §5 should the acceptance rule ever stop being
exact.

## 5. Follow-ups considered

* PLM's relaxed acceptance (ε = 2e-3 / ρ = 0.94) is the Clover threshold rule with other constants — available as
  `--accept-rule threshold`, not part of B13's exactness claim.
* Voronoi auxiliaries with a previous-step sample (PLM's best) if the rank version shows the mechanism works.
* Converting a trained AR model is worse than training from scratch in PLM (Table 6); B13 is a conversion of a
  CPT'd Qwen3-4B, so the absolute gain may undershoot the paper's ratios.
