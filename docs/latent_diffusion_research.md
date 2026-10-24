# SHIGERU-LDM (Phase H) — continuous latent diffusion / flow matching over 16-token blocks

Brief (2026-10-24): build a Stable-Diffusion-like continuous latent generator on top of the frozen SHIGERU-4B — encode a
block of K = 16 future tokens as continuous latents, learn a prefix-conditioned denoiser (Gaussian diffusion or flow
matching) that produces those latents from noise, decode them to tokens in parallel; no AR verification.  Decision on
the running jobs (brief preamble: interrupt them only if this is more promising): **not interrupted** — the stage-2
re-trainings on GPUs 4/6/7 are half done, cheap (≈ 4 h each) and test the measured bottleneck directly; the LDM pilot
runs on zeta (RTX PRO 6000, free) in parallel.  Everything below is measured unless marked *estimated* / *not measured*.

## 1. Audit

* docs present: distill_design.md, pure_diffusion_research.md, future_token_set_research.md, four_token_diffusion.md.
  **docs/latent_guided_diffusion.md does not exist** (listed in the brief; nothing by that name was ever written).
* Model: Qwen3-4B (hidden 2560, 36 layers, vocab 151 936 incl. the `<|mask|>` row 151669, untied LM head) with the
  causal-diffusion conversion (engine/cdlm); embeddings and LM head unchanged by the conversion except the mask row.
* Attention: observed-first causal (engine/cdlm/mask.py); the LDM does not use the dual-stream mask — the frozen 4B
  is run as a plain causal encoder of the prefix (its last hidden states are the condition).
* Checkpoints: runs/phaseC_sft3_4b/final (tsuginosuke), runs/zeta_sft_cpt3_final_e5/final (zeta, same CPT, 5-epoch
  stage 2); the LDM pilot conditions on the zeta checkpoint (frozen).
* GPUs: tsuginosuke 4/6/7 busy with Phase C/D trainings and Phase E sweeps; zeta: the 5-epoch evaluation is finishing
  (≈ 57 GB used), 39 GB free — enough for the pilot (4B bf16 8 GB + 120 M denoiser).
* What previous phases established (pure_diffusion_research.md §2, future_token_set_research.md §4): the mask rows
  know the next 1–3 tokens and little beyond (position accuracy at distance 5–16 is 0.44 → 0.12; ignoring order gives
  only +15 points; half the far predictions are duplicates of frequent tokens).  The LDM hypothesis is that iterating
  in a continuous space resolves the joint continuation progressively instead of predicting independent marginals.

## 2. Pilot design (`engine/ldm/`, `distill/train_ldm.py`, `bench/eval_ldm.py`)

* **Codec (H1, option A)**: z(x) = standardize(E[x] · P), P a fixed random orthonormal 2560 → d projection of the
  frozen input embedding; decode = nearest codebook row.  No parameters, so "reconstruction" only measures the
  invertibility of the projection at d (the trivial-copy caveat of the brief §3.3 applies by construction: this codec
  does not compress; 8/4-slot compression needs a learned decoder and is a later step).
* **Denoiser (H2/H3)**: pre-LN transformer over the K slots (self-attention), cross-attention to the frozen 4B's prefix
  hidden states (last ≤ 768 positions), sinusoidal time embedding; predicts the rectified-flow velocity v = z0 − ε with
  z_t = (1 − t) ε + t z0; sampled by Euler (Heun optional) in S steps.  d_model 1024 × 8 layers ≈ 120 M parameters
  (reported exactly by the trainer).  Ablations wired: predict z0 instead of v, no conditioning, token CE on ẑ0.
* **Training**: stage-2 mix (11 227 examples, the data every stage-2 checkpoint saw), random cut points inside the
  response, 4B run once per sequence (bf16, no grad) → conditions for all cuts.
* **Evaluation**: the Phase F protocol (teacher-forced blocks on GSM8K dev own-AR references + code / EN / JA held-out
  responses): oracle latents, noisy-oracle latents from t = 0.6 / 0.3, generated from noise with S = 16 / 8 / 4 / 2,
  and a zero-condition control; position accuracy by distance, multiset F1, duplicate rate, block exact match — the
  same numbers as masked diffusion (position acc 35 %, multiset F1 50 % at K = 16 for the final checkpoint).

## 3. Results — pilot 1 (zeta, runs/ldm_pilot_k16: v-prediction, 6 000 steps × 32 latents, 120 M denoiser, 53 min)

Training: flow MSE 1.52 → 0.69 (EMA), one-step ẑ0 decode accuracy 0.6 → 0.83 at random t.  Evaluation
(`bench/eval_ldm.py`, results/pure/ldm_pilot_k16_eval.json, 50 GSM8K dev own-AR references + code/EN/JA held-out):

| genre, K = 16 | oracle z0 → decode | noisy oracle t = 0.6 → 6 Euler steps | noisy oracle t = 0.3 → 11 steps | **generated from noise, 16 steps** | 8 / 4 / 2 steps | zero condition, 16 steps |
|---|---|---|---|---|---|---|
| gsm8k | 100 % | 99.8 % (block exact 96 %) | 77.2 % (exact 4 %) | **2.0 % position, 18.8 % multiset F1** | 1.9 / 1.5 / 1.1 % | 0.6 % (F1 4 %) |
| code | 100 % | 99.4 % | 55.9 % | 0.7 % (F1 7.4 %) | 0.4 / 0.4 / 0.1 % | 0.1 % |
| en | 100 % | 99.9 % | 59.4 % | 0.8 % (F1 8.1 %) | 0.4 / 0.6 / 0.2 % | 0.3 % |
| ja | 100 % | 99.8 % | 48.8 % | 0.2 % (F1 2.9 %) | 0.3 / 0.2 / 0.0 % | 0.1 % |

Reading:
* H1 (codec) and H2 (denoising near the data) hold trivially: the projection is exactly invertible and latents
  corrupted at t = 0.6 are recovered 99.8 %; from t = 0.3 (mostly noise) 49–77 %.
* **H4 fails in this pilot**: generated from pure noise, position accuracy is 2 % on GSM8K (masked diffusion: 35 %;
  AR distance-1: 97 %) and even distance 1 is 2.4 %.  The condition is used a little (multiset F1 18.8 % vs 4.1 % with
  the condition zeroed: the denoiser learned the prefix's vocabulary, not its continuation) but the generated latents
  land on tokens unrelated to the reference.  The noisy-oracle rows show why this is not a decoding bug: the model can
  only "snap" a latent that is already near the right codebook row; from noise it has to *decide* the content from the
  condition, and 6 000 steps of velocity regression did not learn that.
* Compared with masked diffusion on the same protocol (position 35 %, multiset F1 50 % at K = 16), pilot 1 is far
  worse; compared with its own cost (one 4B forward per 16 tokens + 16 cheap denoiser steps) it would be attractive
  only if H4 held.

Pilot 2 (running on zeta, runs/ldm_pilot_k16_z0tok): z0-prediction instead of velocity (better behaved at high noise),
token cross-entropy on ẑ0 (λ = 0.5, the brief's reconstruction loss), 12 000 steps.  If distance-1 accuracy does not
rise to the AR level (~0.97), the conditioning pathway itself is the problem (cross-attention to raw hidden states
has to learn in a few thousand steps what the LM head already computes), and the next design would feed the
denoiser the prefix's **next-token logits / LM-head features** rather than hidden states, or initialise the denoiser
from the 4B's own layers.  Verdict so far: **continuous latent generation is not ahead of masked diffusion on this
model; the running stage-2 trainings were correctly left alone.**

### 3.2 Pilot 2 (runs/ldm_pilot_k16_z0tok: z0-prediction + token CE λ = 0.5, 12 000 steps, 1 h 25)

Training: flow MSE → 1.0 (EMA), token CE 106 → 7, one-step ẑ0 accuracy 0.9 at random t.  Evaluation
(results/pure/ldm_pilot_k16_z0tok_eval.json): oracle 100 %; noisy oracle t = 0.6 → 91.4 % (pilot 1: 99.8 %), t = 0.3 →
42.6 % (77.2 %); **generated from noise, 16 steps: 0.6 % position accuracy, multiset F1 7.4 %** (pilot 1: 2.0 % / 18.8 %);
code / en / ja ≤ 0.5 %.  Worse than pilot 1 on every row: the token loss sharpened the one-step decode near the data
but did not make the condition informative.

### 3.3 Verdict (Phase H, measured)

* H1, H2 hold (trivially and near the data); **H4 fails twice**: with ~250 k and ~380 k training latents, two
  parameterisations and two losses, the prefix-conditioned denoiser does not learn the continuation — not even the
  next token (distance-1 accuracy ≤ 2.4 % against the AR path's 97 %).  The generated blocks are fluent-looking
  frequent tokens unrelated to the reference (duplicate rate ≈ 12–15 %, like masked diffusion's far positions).
* Interpretation: a 120 M denoiser reading raw 2560-d hidden states through cross-attention has to re-learn what the
  4B's LM head already computes; the regression targets at high noise are dominated by the noise; and the data
  (11 k conversations) is tiny for learning a conditional generator from scratch.  None of these are fixed by more
  steps alone (pilot 2 had 2× the steps and did worse).
* Against the brief's own gates: H3 ("Prefix だけから意味のある未来 16 トークン") is not met, so H4–H8 (85 % GSM8K,
  4-step distillation, VRAM targets) are not attempted.  Masked diffusion with the K = 4 leave-one-out refinement
  (docs/four_token_diffusion.md: 85 % held-out dev, AR 90 %) remains the best pure-diffusion path on this model.
* If this line is resumed, the design changes with a chance are (not measured): (i) condition on the 4B's next-token
  **logit / LM-head features** at the cut (and at the K−1 following positions through the mask rows) instead of raw
  hidden states, (ii) initialise the denoiser from the 4B's last layers, (iii) train on ≥ 10 M latents from the CPT
  corpus, (iv) hybrid: use the masked-diffusion first forward as the latent initialisation and the continuous
  denoiser only as a refiner.

## 4–12. Phases H2–H8, ablations, GSM8K/HumanEval, VRAM, failures, next
Not pursued: §3.3.  Failed experiments: §3.1–3.2.  Next research questions: §3.3 last bullet.
