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

## 3. Attention audit (Gate 0) — engine/bdiff/tests/test_bdiff.py
*(CPU, tiny Qwen3; the HF 4D-mask path was also checked to equal the additive-mask path and to accept inputs_embeds)*

## 4–11. Time conditioning, B1, B2, B3, B4, steps, gates, GPUs
*filled as the runs complete; Gate 1 (overfit) and B1 run on zeta (RTX PRO 6000), the A100s being busy with the
A-series experiments and the stage-2 trainings.*
