# SHIGERU B-series — true discrete diffusion over 16-token blocks (block-bidirectional, time-conditioned, self-correcting)

Brief (2026-10-24): an independent line next to the A-series (the leave-one-out K = 4 decoder of docs/four_token_diffusion.md,
untouched here): absorbing-mask diffusion over a block of K = 16 future tokens with **bidirectional attention inside the
block**, a time (mask-ratio) conditioning, a reverse sampler that is a reverse process rather than a fixed number of
completions, and self-correction training on the model's own wrong states.  Gates 0–4 (brief §10) decide whether any
larger training happens.  Code: engine/bdiff/ (masks, corrupt, model, sampler, tests), distill/train_bdiff.py,
bench/eval_bdiff.py.  Everything measured unless marked *estimated*.

## 1. Audit — how bidirectional is the current SHIGERU? (code level)

* engine/cdlm/mask.py: inside an xt block, rows are reordered observed-first and attention is **causal in that
  physical order**: an observed row sees the prefix and the observed rows before it; a [MASK] row sees the prefix,
  *all* observed rows of the block and the masks before it; **a mask never sees a later mask and an observed row never
  sees any mask**.  So the current model is bidirectional only in the sense "masks see observed tokens on both sides";
  the K-group probe of the A-series (k4_decoder.py) is the same conditional computed one position at a time.  A true
  block-bidirectional pass (every block row attends every block row) is a new attention pattern for these weights.
* The prefix (x0 stream) never attends the block (previous_x0_block rule), so prefix hidden states are independent of
  the block — the same property the B-series needs for prefix K/V reuse.
* RoPE positions are the logical positions in both designs; the B-series keeps them (block rows at P..P+K−1).
* No time conditioning exists in the A-series: the mask ratio is implicit in the input.

## 2. Formulation

Forward process q(x_t | x_0, t): each block token independently → [MASK] with probability γ(t); γ_linear(t) = t,
γ_cosine(t) = 1 − cos(πt/2); "uniform" draws the ratio itself.  Network p_θ(x_0 | x_t, c, t), trained with the
absorbing-diffusion weighted cross-entropy (1/γ on masked positions) plus — for self-correction — a cross-entropy on
*visible* positions whose token was corrupted (target: the clean token) and a small identity term on correct visible
positions.  **Reverse transition actually implemented by the sampler (engine/bdiff/sampler.py)**: at step s of S the
mask count is driven from K·γ(t_{s−1}) to K·γ(t_s); the positions to reveal are chosen by confidence (or randomly /
left to right) and receive the network's argmax — the standard MaskGIT/LLaDA approximation of p(x_{t−Δt} | x_t), in
which revealed tokens are final.  The "remask" variant re-masks visible positions whose x0-prediction disagrees with
their token (budgeted per step): a heuristic *on top of* the forward process, not derived from it; it is reported
separately.

## 3. Gate 0 — attention audit (engine/bdiff/tests/test_bdiff.py, 6/6 pass, CPU tiny Qwen3)

prefix logits invariant to any block change; block row i's logits change when block row j changes (both directions);
a masked position's gold token influences nothing (its input is [MASK]); a visible token changes its own and other
rows (copy path exists by design); cache path == full forward (1e-4) and the cache length is restored after every
denoise call; shifting the block's position ids changes the logits (RoPE is in effect); time embedding changes the
block logits and a no-time model is invariant to t.  HF 4D bool mask == additive mask, inputs_embeds path checked.

## 4. Time conditioning

Option A (brief §4): a 16-bucket learned embedding of the mask ratio added to the block rows' input embeddings,
zero-initialised; 41 k parameters.  LoRA (r 64, α 128, all projections) carries the attention-pattern adaptation:
264 M trainable parameters (6 % of the 4B), base weights untouched.  Time-off ablation not yet run (*not measured*).

## 5. Gate 1 and Stage B1 (zeta RTX PRO 6000, base runs/zeta_sft_final_e5/final, K = 16, linear schedule)

Gate 1 (32 fixed blocks, 300 steps × 4): masked accuracy on the fixed blocks 100 % at γ < 0.6, 97 / 93 / 83 % at
γ = 0.4–0.6 / 0.6–0.8 / 0.8–0.95 after 300 steps → passes (the implementation can fit multi-rate reconstruction).

B1 (2 000 steps × 8 blocks = 16 k blocks from the stage-2 mix, 29 min, peak 16.9 GB): teacher-forced masked-token
accuracy on 294 held-out blocks of the model's own GSM8K answers (results/pure/bdiff_b1_eval.json):

| mask rate | 10 % | 25 % | 50 % | 75 % | 90 % | 100 % |
|---|---|---|---|---|---|---|
| masked-token accuracy | 94.6 % | 92.5 % | 82.1 % | 64.2 % | 49.5 % | 31.1 % |
| block exact (16 tokens) | 90.5 % | 76.5 % | 32.3 % | 6.8 % | 1.0 % | 0.0 % |

100 % masked, by position 1…16: .88 .77 .57 .46 .41 .27 .25 .19 .27 .17 .16 .16 .13 .11 .11 .09 — the same reach curve as
the A-series first forward (pure_diffusion_research.md §2.1: .97 .84 .73 .61 …), a little lower at distance 1–2
(bidirectional LoRA after 16 k blocks vs the full stage-2 training) and the same beyond.

## 7. Stage B3 — one-step correction effectiveness of the B1 model (before self-correction training)

| input state (294 blocks) | errors/block before → after | W→C | C→W | W→W' | correction success | error introduction |
|---|---|---|---|---|---|---|
| A random replacement of 25 % (no masks) | 4.0 → **8.5** | 329 | **1 645** | 689 | 28 % | 47 % |
| B 50 % masked, rest clean | 8.0 → **1.65** | 1 914 | 47 | 438 | 81 % | 2 % |
| C on-policy state of the A-series decoder (masks + its wrong tokens) | 10.8 → 8.9 | 671 | 130 | 1 744 | 21 % | 8.5 % |

Reading: the mask-only model is a good *in-filler* (B) and a bad *corrector*: given wrong visible tokens it rewrites
correct ones too (A: it introduces more errors than it fixes, because training never showed it a wrong visible token,
so "visible" carried no information about trustworthiness).  On the model's own states (C) it removes 17 % of the
errors per step.  This is exactly the gap Stage B2 (visible-token corruption + on-policy states in training) targets;
B2 is training now with the same budget.

## 8. Stage B4 — free generation, K = 16 from all masks (50 dev prompts, confidence-order unmasking, greedy)

| steps S | GSM8K dev 50 | tokens / network call | tok/s (zeta, HF model + LoRA merged, no graphs) | finish at 384 |
|---|---|---|---|---|
| 16 | **84 %** | 0.91 | 57.0 | 4 |
| 8 | 64 % | 1.72 | 107.4 | 5 |
| 4 | 28 % | 3.11 | 188.4 | 11 |

(AR of the base model on these prompts: 92 %; the A-series K=4 refineN: 88–90 %.)  **Gate 3 passes at S = 16**: the first
true block-bidirectional reverse-diffusion sampler on this model produces correct GSM8K answers 84 % of the time after
30 minutes of LoRA training, with no AR head and no refinement heuristics.  Halving the steps costs 20 points, which
is where B2 (correction) and the remask sampler have to deliver.  The tokens/forward column counts the prefix pass per
block (1 of 17 calls at S = 16); the network call here is an HF forward of 16 rows on a 16-row-free prefix cache.

## 6. Stage B2 — self-correction training with random replacement + A-series states (zeta, 2 000 × 8 blocks, 31 min)

Config: as B1 plus 30 % of blocks with 25 % of the visible tokens replaced by random tokens (λ_vis = 1.0, λ_id = 0.1)
and 20 % of blocks drawn from the A-series decoder's recorded states (distill/gen_onpolicy.py, K = 32 states cut to 16).

| | B1 (mask only) | B2 (+ corruption) |
|---|---|---|
| teacher-forced masked acc 10 / 50 / 100 % | 94.6 / 82.1 / 31.1 % | 91.7 / 80.2 / 31.2 % |
| B3-A random replacement, errors before → after | 4.0 → 8.5 (introduces 47 %) | **4.0 → 3.26** (success 87 %, introduces 23 %) |
| B3-B 50 % masked | 8.0 → **1.65** | 8.0 → 2.54 |
| **free generation K = 16, S = 16 / 8 / 4** | **84 / 64 / 28 %** | **40 / 18 / 2 %** (mean length 207–362 tokens, 10–44 of 50 hit the budget) |

B2 did what it was trained for on the diagnostic (it now repairs replaced tokens instead of destroying their
neighbours) and **collapsed in free generation**: the sampler never changes visible tokens, so the collapse comes from
the masked-position predictions becoming worse in the *clean-context* regime the sampler actually visits (the identity
term was weak and 30 % of training blocks taught "visible context may be wrong"), and from degenerate long outputs
(repetition until the budget).  Gate 2 is passed only in the narrow sense (fewer errors after one step on corrupted
inputs); the brief's own condition — it must not come at the expense of generation — fails.  **Decision (brief
"B2 の結果を確認してから優先順位を決める"): random-replacement self-correction is deprioritised; B4 (remasking) and B6
(step distillation) now build on the B1 adapter; B3 is re-planned with the B-sampler's OWN intermediate states (not the
A-series decoder's, not random tokens), a lower λ_vis (0.3) and initialisation from B1.**

## 9. Parallel plan (GPU assignment, 2026-10-24 13:30 UTC)

| GPU | experiment | base | status |
|---|---|---|---|
| tsuginosuke GPU 4 (A100) | B1 control on the A-series base (runs/phaseC_sft3_4b) → B5 ablations: no time conditioning; cosine schedule | sft3 | queued behind the E3 held-out runs |
| tsuginosuke GPU 6 (A100) | B1 ×3 steps (6 000 × 8 blocks) — does more mask-only training lift S = 8? | sft3 | queued behind the on-policy checkpoint evaluation |
| tsuginosuke GPU 7 (A100) | B3: B-sampler on-policy states from the B1-sft3 adapter → training from B1 (λ_vis 0.3, 30 % states) | sft3 | queued behind the random/mixed evaluations and the GPU 4 B1 control |
| zeta (RTX PRO 6000) | B4 remask budgets 1/2/4 at S = 16/8, cosine sampling schedule, random/left orders; B1 held-out (items 50–199); B6 distillation (B1 teacher S = 16 → student from B1, t ≥ 0.5) | e5 | running |

## 10–11. 16→8→4 steps, gates 4, VRAM
*pending the runs above*
