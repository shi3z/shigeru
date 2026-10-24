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

## 3. Results
*running*

## 4–12. Phases H2–H8, ablations, GSM8K/HumanEval, VRAM, failures, next
*pending on §3*
