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
| zeta (RTX PRO 6000) | B4 remask budgets 1/2/4 at S = 16/8, cosine sampling schedule, random/left orders; B1 held-out (items 50–199); B6 distillation (B1 teacher S = 16 → student from B1, t ≥ 0.5) | e5 | done (§10) |

## 10. Results of the parallel round (first batch)

### 10.1 Base-checkpoint control (GPU 4, A100): B1 recipe on the A-series base runs/phaseC_sft3_4b/final

| | B1 on zeta base (e5, 5-epoch stage 2) | B1 on sft3 base (3-epoch, the A-series checkpoint) |
|---|---|---|
| masked acc 10 / 50 / 100 % | 94.6 / 82.1 / 31.1 % | 93.7 / 81.4 / 30.7 % |
| B3-B 50 % masked, errors 8.0 → | 1.65 | 1.64 |
| B3-A random replacement, 4.0 → | 8.5 | 9.3 |
| **free generation S = 16 / 8 / 4** | **84 / 64 / 28 %** | **76 / 62 / 14 %** |
| tok/s (HF + merged LoRA, no graphs) | 57 / 107 / 188 (zeta) | 40 / 74 / 135 (A100) |

Same recipe, same data, same steps: the e5 base is 8 points better at S = 16 and 14 at S = 4.  The 5-epoch stage 2
(docs/distill_design.md §11.4, trained for its drafter) gives a better starting point for block-bidirectional LoRA than
the 3-epoch one; the e5 checkpoint will be copied to tsuginosuke so that the remaining B-series runs share one base.

### 10.2 B4 — remasking and sampling variants on the B1 adapter (zeta, dev 50)

| sampler | S = 16 | S = 8 |
|---|---|---|
| monotonic unmasking, confidence order (baseline) | 84 % | 64 % |
| + remask budget 1 / step | 84 % | 62 % |
| + remask budget 2 / step | 82 % | 60 % |
| + remask budget 4 / step | – | 50 % |
| cosine sampling schedule (few reveals first) | 82 % (1.03 tok/fwd) | 50 % (1.92) |
| random unmasking order | 14 % | 10 % |
| **left-to-right unmasking order** | **84 %** | (see JSON) |

**Held-out (dev items 50–199, 150 prompts, AR 90.0 %): B1 S = 16 → 76.0 %, S = 8 → 63.3 %** (dev-50: 84 / 64).  The
50-item number was 8 points optimistic at S = 16 and exact at S = 8.  Against the A-series on the same held-out items
(K = 4 refineN 84.7 % at 1.17 tokens/forward): the B-series is behind at S = 16 (76 %, 0.9 tokens/forward) and offers a
different trade-off at S = 8 (63 % at 1.7 tokens/forward).  Confidence order and left-to-right order tie at S = 16; random
order collapses (14 %), i.e. the order matters as much as in the A-series (pure_diffusion_research.md §2.2).

Re-masking visible tokens whose x0-prediction disagrees does not help a mask-only model: its visible-position
predictions were trained only by the weak identity term, so the disagreement signal is noise (consistent with B3-A).
Remasking can only pay off together with a self-correction objective that does not break generation (B2 broke it).

### 10.3 B6 — step distillation (zeta): B1 teacher samples at S = 16 (5 410 blocks) → student from the B1 adapter, t ≥ 0.5

| S | B1 | B6 student |
|---|---|---|
| 16 | 84 % | 80 % |
| 8 | 64 % | 58 % |
| 4 | 28 % | **34 %** |

The high-noise emphasis moves accuracy from the many-step to the few-step regime (+6 at S = 4, −4/−6 at S = 16/8); the
total is not a gain.  A real step-distillation signal needs the teacher's *intermediate* decisions (which positions to
reveal at which step), not only its final block — *not implemented*.

### 10.4 B5 — time conditioning ablation (GPU 4, sft3 base, same 2 000 × 8 blocks)

| | with time embedding (B1-sft3) | **without** (`--no-time`) |
|---|---|---|
| masked acc 10 / 50 / 100 % (train-log buckets at the end) | .83 / .67 / .34 | .75 / .65 / .33 |
| free generation S = 16 / 8 / 4 | 76 / 62 / 14 % | **82 / 66 / 28 %** |

The mask-ratio embedding (option A, added to the block rows) does not help and the no-time model generates better at
every step count.  The ratio is already visible to the network (it sees how many [MASK] rows it attends), so the extra
input only adds a train/inference mismatch (training buckets vs the sampler's continuous schedule).  Time conditioning is
dropped from the recipe.  The other B5 arm — **cosine mask schedule** for training *and* sampling (γ = 1 − cos(πt/2):
few masks early, many late) — gives 58 / 18 / 6 % at S = 16 / 8 / 4 vs 76 / 62 / 14 % for linear: fewer high-mask-rate
training examples and a sampler that reveals little in its first steps both hurt.  Linear stays.

### 10.5 B1 ×3 steps (zeta, e5 base, 6 000 × 8 blocks = 48 k blocks, 88 min)

| | B1 (2 000 steps) | B1 ×3 (6 000 steps) |
|---|---|---|
| dev 50, S = 16 / 8 / 4 | 84 / 64 / 28 % | 74 / 56 / 26 % |
| held-out 150, S = 16 / 8 | 76.0 / 63.3 % | **68.0 / 52.0 %** |
| masked acc at γ ≥ 0.95 (train log) | 0.33 | 0.17 |

Three times the steps on the same 11 227 conversations is *worse* (−8 / −11 held-out): the LoRA over-fits the small
stage-2 mix (≈ 4 passes), and the all-mask regime degrades first.  More passes are not the lever; more diverse text is —
the next run (zeta) mixes raw FineWeb blocks from the 1 B-token CPT stream (data/cpt/fineweb_1b.bin) 50/50 with the
conversations, no time conditioning.

### 10.6 B1 ×3 on the sft3 base (GPU 6, 6 000 × 8 blocks, 2 h 19)

dev S = 16 / 8 / 4: 80 / 50 / 20 % (B1-sft3 2 000 steps: 76 / 62 / 14 %).  Same picture as on e5: more passes over the
11 k conversations buy a little at S = 16 and lose at S = 8.

### 10.7 B7 — data diversity: 50 % raw FineWeb blocks from the CPT token stream (zeta, e5, no time, 4 000 × 8)

dev S = 16 / 8 / 4: 86 / 50 / 14 %; **held-out 75.3 / 54.7 %** (B1: 76.0 / 63.3).  No gain at S = 16, −9 at S = 8, and
10 of 50 dev outputs end prematurely (raw-text blocks teach a different end-of-answer behaviour; outputs 200+ tokens).
More diverse *text* is not the lever either, at least not mixed in at this ratio.

### 10.8 B3 — self-correction with the B-sampler's OWN intermediate states (GPU 7, sft3 base, from the B1 adapter,
λ_vis 0.3, 30 % of blocks are recorded states, 1 500 × 8)

States: 10 820 recorded (7 298 with ≥ 1 wrong placed token) from 2 739 training conversations × 2 cuts, 16-step sampler.
dev S = 16 / 8 / 4: 76 / 50 / 18 % (B1-sft3: 76 / 62 / 14).  Generation did **not** collapse this time (the states are
the sampler's own and the corruption weight is low), but it did not improve either, and S = 8 lost 12 points.  One-step
correction on the B-sampler's own states (teacher-forced, 294 blocks): B3 10.8 → 8.5 errors per block (success 22.5 %,
introduction 2.6 %) vs B1 11.2 → 9.5 (18.8 %, 8.1 %); random replacement 4.0 → 6.2 (B1 9.3); 50 % masked 8.0 → 1.52 (B1 1.64).
So the gentler self-correction recipe (own states, λ_vis 0.3, from B1) improves every teacher-forced correction metric
without collapsing generation — and still does not improve generation.  Gate 2 in the brief's sense (fewer errors after
one step on the model's own states) passes narrowly; the gain does not transfer to the sampler because the sampler never
revisits visible tokens (the remask variant would have to be re-tested on B3: *not run*).

### 10.9 Where the B-series stands (held-out items 50–199, AR 90 %)

| run | base | recipe | S = 16 | S = 8 |
|---|---|---|---|---|
| B1 | e5 | mask-only, time, 2 k steps | **76.0 %** | **63.3 %** |
| B1 ×3 | e5 | same, 6 k steps | 68.0 | 52.0 |
| B7 | e5 | no time, +50 % FineWeb, 4 k | 75.3 | 54.7 |
| B1-sft3 / no-time / cosine / ×3 (dev 50 only) | sft3 | | 76 / 82 / 58 / 80 | 62 / 66 / 18 / 50 |
| B3 (dev 50) | sft3 | own states, from B1 | 76 | 50 |
| B6 (dev 50) | e5 | distilled from B1 | 80 | 58 |

The S = 8 target (80 % held-out) is 17 points away and none of the recipe variants moved it; the first B1 run is still the
best S = 8 model.  Then running (results in §10.10–§10.16): no-time ×3 (GPU 4), no-time on e5 (GPU 6), **B8** — on-policy *mask patterns* (which
positions the sampler reveals late) with clean tokens, i.e. matching the inference mask distribution without any
corruption (GPU 7) — and a LoRA rank-256 capacity ablation (zeta).

### 10.10 Training length and base (no-time recipe)

| run | base | steps | dev S = 16 / 8 / 4 | held-out S = 16 / 8 |
|---|---|---|---|---|
| no-time (B5) | sft3 | 2 000 | 82 / 66 / 28 % | – |
| no-time ×3 | sft3 | 6 000 | 70 / 34 / 18 % | 71.3 / 43.3 % |
| **no-time** | **e5** | 2 000 | 80 / **68 / 36 %** | 76.0 / 66.0 % |
| no-time, LoRA r = 256 (α 512, lr 1e-4) | e5 | 2 000 | 74 / 66 / 22 % | 76.7 / 58.7 % |
| no-time | e5 | 1 000 | 80 / 64 / 20 % | 68.0 / 54.7 % (23 / 150 hit the length budget: under-trained) |

Learning curve on the e5 base (held-out S = 8): 1 000 steps 54.7 % → **2 000 steps 66.0 %** → 6 000 steps 52.0 %.  The
budget is a sharp optimum at ≈ 1.4 passes over the 11 k conversations; B8's held-out confirmation (56.0 / 45.3 %)
closes that line.
| B1 (with time) | e5 | 2 000 | 84 / 64 / 28 % | 76.0 / 63.3 % |

Third confirmation that 6 000 steps over-train (−12 / −32 at S = 16 / 8 on dev; 71 / 43 held-out); the useful budget on
this data is ≤ 2 000 steps (≈ 1.4 passes).  On the e5 base the no-time recipe gives the best S = 8 / S = 4 dev numbers
so far (68 / 36 %).  A 1 000-step point (GPU 4) and the rank-256 run (zeta) complete the capacity/length picture.

### 10.11 B8 — on-policy mask *patterns* with clean tokens (GPU 7, sft3, no time, 50 % of blocks use a recorded
16-step sampler state's mask pattern)

dev S = 16 / 8 / 4: 60 / 46 / 18 % (no-time baseline 82 / 66 / 28).  Negative: the recorded patterns come from the
16-step sampler's late states (few masks left, the easy positions already revealed), so half the training signal moved to
low mask ratios (training masked-loss 0.77 vs 3.3) and the all-mask / high-ratio regime the sampler starts from was
starved.  Matching the sampler's mask *distribution* needs the early-step patterns weighted by their frequency, not a
uniform sample of recorded states — *not retried*.

### 10.12 Capacity (LoRA rank 256 vs 64, zeta, e5, no time): no gain (held-out 76.7 / 58.7 vs B1's 76.0 / 63.3).

### 10.13 Block size K = 8 (GPU 6, e5, no time, 2 000 × 8)

| K = 8 blocks | S = 8 (1 token / call) | S = 4 (2 / call) | S = 2 (4 / call) |
|---|---|---|---|
| dev 50 | 78 % (0.88 tok/forward) | 48 % (1.58) | 12 % (2.66) |

At equal tokens per network call the 16-token block is better than the 8-token block (K = 16 S = 8: 68 % at 1.70 vs
K = 8 S = 4: 48 % at 1.58): the bidirectional model benefits from the wider block even though its far positions are
weaker.  K = 16 stays the block size.

K = 8 held-out: S = 8 70.7 % (0.88 tok/call), S = 4 40.7 % (1.58).

### 10.14 Optimiser ablations (e5, no time, 2 000 steps) — held-out items 50–199

| | lr 2e-4, batch 8 (reference) | lr 1e-4, batch 8 | lr 2e-4, batch 16 |
|---|---|---|---|
| dev S = 16 / 8 / 4 | 80 / 68 / 36 % | 78 / 66 / 32 % | (JSON) |
| **held-out S = 16 / 8** | 76.0 / **66.0 %** | **82.0** / 60.7 % | 72.7 / 54.7 % |

Lower lr buys S = 16 (82 %, the best held-out K = 16 number of the series) and costs S = 8; doubling the batch (2.8
passes) over-trains like the longer runs.  The S = 8 frontier stays at 66 % held-out; the full fine-tune (GPU 4, fp32
master, lr 1e-5, gradient checkpointing, 80 GB) is the last capacity lever in this round.

### 10.15 Sampler points on the two best adapters (held-out items 50–199, K = 16)

| adapter | S = 16 | S = 12 | S = 8 confidence | **S = 8 left-to-right** | S = 6 |
|---|---|---|---|---|---|
| e5 no-time lr 2e-4 | 76.0 % | 70.7 % (1.18 tok/call) | 66.0 % (1.70) | 66.7 % | 51.3 % (2.20) |
| e5 no-time lr 1e-4 | 82.0 % | 73.3 % | 60.7 % | **69.3 %** | 46.7 % |

Left-to-right unmasking (the sampler reveals the K/S leftmost masks each step, i.e. a block-local sequential order with
bidirectional conditioning) matches or beats confidence order at S = 8 (+8.6 points for the lr 1e-4 adapter) at the same
cost.  **Best held-out K = 16, S = 8 so far: 69.3 % at 1.70 tokens per network call** (AR on the same items 90.0 %;
A-series K = 4: 84.7 % at 1.17 tokens/forward).  The 80 % target at S = 8 is 11 points away.

### 10.16 Full fine-tune (GPU 4, e5, no time, fp32 master + bf16 autocast, gradient checkpointing, lr 1e-5, 2 000 × 8;
80.7 GB peak, 2.4 s/step)

The first run completed but its weights could not be saved (root filesystem full: safetensors "No space left on
device"); the repeat (after freeing 23 GB of an ablation's optimizer checkpoint) gives dev S = 16 / 8 / 4 = **68 / 50 / 18 %**
— worse than the LoRA runs (80 / 68 / 36 %), with a loss spike late in training.  Held-out (items 50–199) S = 16 / 8 = **67.3 / 50.7 %** vs
76.0 / 66.0 % for the LoRA with the same data and steps.  Full fine-tuning at this learning rate and budget is not the
capacity lever either (results/pure/bdiff_e5_full_heldout.json; 76 min, 75.5 GB peak).

### 10.17 Second batch (finished)

GPU 4: B5 ablations (no time conditioning; cosine training schedule), no-time ×3, e5 no-time 1 000 steps, full
fine-tune.  GPU 6: B1 ×3 steps, e5 no-time, e5 K = 8.  GPU 7: B3 with the B-sampler's own intermediate states
(λ_vis 0.3), B8 mask patterns, e5 lr 1e-4.  zeta: the B4 sweep, B1 held-out, B6 distillation, B1 ×3, B7 mixed data,
r = 256, accumulation 16, and the GSM8K-test rows.  The A-series training ablations finished in parallel: random-mask
stage 2 gives K = 4 refineN 80 / 82 % (3 / 6 rounds) vs 88 / 92 % for the suffix checkpoint, K = 8 colouring 78 % (82),
K = 6 2+2+2 74 % (80) — random masks help the adaptive sampler and recovery (pure_diffusion_research.md) but not the
leave-one-out decoders.  **The on-policy-state stage 2 (config D of pure_diffusion_research.md) collapsed exactly like
B2**: mask recovery is fine (random γ = 0.5: 84 %) but free generation degenerates (K = 4 refineN 10–20 %, anchored
adaptive 6–10 %, 4-gram repetition 0.67, 41 of 50 outputs hit the length budget).  Two independent implementations
(A-series dual-stream, B-series block-bidirectional) of "train with the decoder's own wrong tokens as observed context"
both destroy generation while improving teacher-forced correction — the mechanism is the same: the model learns that
observed context is unreliable and stops committing, which in a sampler that never revisits tokens shows up as
repetition.  Self-correction training has to be re-designed (e.g. only mask-side targets, a separate "trust" signal, or
much smaller corruption rates with the identity term weighted up) before it is tried again.

## 11. Summary of the round: 16→8→4 steps, Gate 4, VRAM, GPU time

### 11.1 Every B-series training run (K = 16 block-bidirectional, LoRA r 64 unless stated; dev = items 0–49, held-out = items 50–199; accuracy at S = 16 / 8 / 4 steps per block, confidence order, greedy)

| run | base | what changed vs B1 | GPU | train time | peak VRAM | dev 16 / 8 / 4 | held-out 16 / 8 |
|---|---|---|---|---|---|---|---|
| B1 (§5) | e5 | – (time emb., linear, 2 000 × 8) | zeta | 29 min | 16.9 GB | 84 / 64 / 28 | 76.0 / 63.3 |
| B1 ×3 steps | e5 | 6 000 steps | zeta | 88 min | 16.9 GB | 74 / 56 / 26 | 68.0 / 52.0 |
| B2 self-correction (§6) | e5 | random replacement + A-states | zeta | 31 min | 23.5 GB | collapsed (repetition) | – |
| B6 step distillation | e5 | B1 S = 16 samples as targets, from B1 | zeta | 22 min | 16.8 GB | 80 / 58 / 34 | – |
| B7 mixed data | e5 | gold + self samples, 4 000 steps | zeta | 64 min | 16.8 GB | 86 / 50 / 14 | 75.3 / 54.7 |
| r = 256 | e5 | LoRA rank 256 | zeta | 33 min | 23.8 GB | 74 / 66 / 22 | 76.7 / 58.7 |
| accumulation 16 | e5 | effective batch ×2 | zeta | 58 min | 16.9 GB | 82 / 70 / 44 | 72.7 / 54.7 |
| B1 control | sft3 | A-series base | GPU 4 | 48 min | 17.2 GB | 76 / 62 / 14 | – |
| B5 no time | sft3 | time embedding removed | GPU 4 | 45 min | 17.2 GB | 82 / 66 / 28 | – |
| B5 cosine | sft3 | cosine mask schedule | GPU 4 | 47 min | 17.2 GB | 58 / 18 / 6 | – |
| no time ×3 | sft3 | 6 000 steps | GPU 4 | 136 min | 17.2 GB | 70 / 34 / 18 | 71.3 / 43.3 |
| B1 ×3 | sft3 | 6 000 steps | GPU 6 | 139 min | 17.2 GB | 80 / 50 / 20 | – |
| B3 on-policy states | sft3 | B-sampler states, λ_vis 0.3, from B1 | GPU 7 | 46 min (+66 min states) | 17.2 GB | 76 / 50 / 18 | – |
| B8 mask patterns | sft3 | sampler's own mask patterns as corruption | GPU 7 | 44 min | 17.1 GB | 60 / 46 / 18 | 56.0 / 45.3 |
| **e5 + no time** | e5 | time embedding removed | GPU 6 | 44 min | 17.2 GB | 80 / 68 / 36 | **76.0 / 66.0** |
| e5 + no time, 1 000 steps | e5 | half the steps | GPU 4 | 23 min | 17.1 GB | 80 / 64 / 20 | 68.0 / 54.7 |
| **e5 + no time, lr 1e-4** | e5 | half the learning rate | GPU 7 | 49 min | 17.2 GB | 78 / 66 / 32 | **82.0 / 60.7** |
| e5 + no time, K = 8 | e5 | block 8 (S = 8 / 4 / 2) | GPU 6 | 45 min | 17.1 GB | 78 / 48 / 12 | 70.7 / 40.7 |
| e5 + no time, full fine-tune | e5 | all weights, lr 1e-5, fp32 master | GPU 4 | 76 min | 75.5 GB | 68 / 50 / 18 | 67.3 / 50.7 |

Reading the table: the only changes that moved the held-out numbers were removing the time embedding (+0–3 points) and
halving the learning rate (S = 16: 76 → 82).  More steps (×3), more data (mixed, 1 000 vs 2 000), more capacity (rank 256,
full fine-tune), a larger batch, a smaller block, step distillation, and every form of on-policy training (B2, B3, B8)
were neutral or negative.  **The S = 8 ceiling of this recipe is 60–66 % held-out; the K = 16 S = 8 ≥ 80 % and S = 4 ≥ 85 %
targets of the brief were not reached by any run.**

### 11.2 Sampler side (held-out 50–199, best two adapters; §10.15)

| sampler | e5 + no time | e5 + no time, lr 1e-4 |
|---|---|---|
| S = 16 confidence | 76.0 | 82.0 |
| S = 12 confidence | 70.7 | 73.3 |
| S = 8 confidence | 66.0 | 60.7 |
| S = 8 left-to-right | 66.7 | **69.3** |
| S = 6 confidence | 51.3 | 46.7 |

Remask budgets (1/2/4 re-masked tokens per step) and the cosine sampling schedule did not help on dev (§10.x); random
order collapses (14 %).  Left-to-right order at S = 8 is the best S = 8 point (69.3 %) — the block-bidirectional model
is used most reliably when it is unmasked like a 2-tokens-per-step AR model.

### 11.3 GSM8K test (1 319 problems; settings fixed on held-out, never tuned on test)

| decoding (no AR verification) | adapter | test | tokens / forward | tok/s (GPU) |
|---|---|---|---|---|
| K = 16, S = 16, confidence | e5 + no time | 77.3 % (1 020 / 1 319) | 0.90 | 57.1 (zeta) |
| K = 16, S = 16, confidence | e5 + no time, lr 1e-4 | **78.6 %** (1 037 / 1 319) | 0.90 | 39.5 (A100) |
| K = 16, S = 8, confidence | e5 + no time | 63.4 % | 1.71 | 106.5 (zeta) |
| K = 16, S = 8, left-to-right | e5 + no time, lr 1e-4 | **72.9 %** (961 / 1 319) | 1.70 | 77.6 (A100) |
| K = 16, S = 16, left-to-right (Phase B7, §12) | e5 + no time, lr 1e-4 | **84.8 %** (1 118 / 1 319) | 0.91 | 56.2 (zeta) |
| K = 16, S = 8, left-to-right + stability re-mask (Phase B7) | e5 + no time, lr 1e-4 | 71.5 % | 1.72 | 73.4 (A100) |
| A-series K = 4 leave-one-out, Jacobi ≤ 6, graph engine (four_token_diffusion.md §10) | – | 84.2 % | 1.18 | 89.9 (A100) |
| AR greedy (HF) | – | 88.9 % | 1.00 | 52.5 (A100) |

### 11.4 Gate 4 (speed), VRAM

The B-series sampler runs on the HF model with a 4-D mask (no CUDA graphs, no static buffers), so its tok/s is not a
fair speed comparison with the graph-engine A-series decoder; tokens per forward is the architecture-level number.  At
S = 8 the B-series emits 1.7 tokens per forward (vs 1.18 for K = 4 refinement and 1.0 for AR) but at 60–69 % held-out and
63.4 % (confidence) / 72.9 % (left-to-right) on test, 11 points below the K = 4 decoder; at S = 16 it is below one token per forward (0.90) and below AR accuracy.  **Gate 4 fails: there is no
(S, order) point where the block-bidirectional model is both faster per forward than the A-series K = 4 decoder and
within 5 points of it in accuracy.**  Inference VRAM: 4B bf16 + LoRA 11.9 GB (A100, 384-token budget, batch 1; the
merged full fine-tune 10.8 GB); training 17.2 GB for LoRA r 64 at 1 024 tokens (23.8 GB for r 256, 23.5 GB with the
B2 replacement forward, 75.5 GB for the full fine-tune with fp32 master weights and gradient checkpointing).

### 11.5 GPU time of the round

Training: 20 runs, 19.1 GPU-hours (tsuginosuke A100s: 13.6 h including the first full fine-tune whose save failed on
the full disk; zeta: 5.5 h).  Evaluation and diagnostics (dev, held-out, sampler sweeps, B3 tests, state and sample
generation, five GSM8K-test runs): about 17 GPU-hours.  All runs on GPUs 4/6/7 and
zeta only; the DeepSeek teacher (GPUs 0–3) and GPU 5 were not touched.

### 11.6 Decision

The B-series stays a negative result at the targets set by the brief.  What transfers: time conditioning is harmful
for a model that can see the mask pattern, lower learning rates help the bidirectional LoRA, left-to-right unmasking
is the safest S = 8 order, and every on-policy / self-correction recipe tried so far (B2, B3, B8, A-series config D)
degrades free generation.  The A-series K = 4 leave-one-out decoder remains the accuracy/speed frontier of this model
(84.2 % test at 1.7× HF-AR throughput, no AR verification).

## 12. Phase B7 — trust-aware decoding (summary; full record in trust_aware_diffusion.md)

Hypothesis tested: the S = 8 gap is a *trust* problem (which placed tokens to re-mask), not a denoising-capacity
problem.  Answer: **no.**  (i) Every signal from the current forward at a visible position is uninformative
(AUROC ≈ 0.5) because the mask-only adapters copy visible tokens; the usable trust is historical (confidence at
placement, prediction changes before placement; logistic score AUROC 0.77–0.80).  (ii) Under a fixed budget of 8
forwards per block, fifteen re-masking / selection rules were evaluated on 150 fresh held-out items
(bench/prompts_gsm8k_dev2.json): one (left-to-right + stability re-mask) gained +6.7 points there and lost 1.4 on the
GSM8K test (72.9 → 71.5 %); all others were neutral or harmful.  (iii) An oracle that re-masks exactly the wrong
tokens repairs only 30 % of them (block exact +2.4 points) — the refill, not detection, is the limit, so no trust
head was trained.  Side result worth adopting: **left-to-right unmasking** — S = 16 dev2 90.0 % (confidence 82.0),
GSM8K test **84.8 %** (vs 78.6 %; the A-series K = 4 decoder: 84.2 %), HumanEval 50.6 % (vs 29.9 %; A-series K = 4 40.2 %, AR 68.9 %).  S = 8 remains at
72.9 % test and 8 % HumanEval; the K = 16 / 8-step ≥ 80 % target stands unmet.

## 13. Phase B8 — pairwise joint denoising (summary; full record in pairwise_joint_diffusion.md)

The two tokens an S = 8 step places are strongly dependent (placing a1 vs a2 at i changes the argmax at j in 67 % of
steps; j's accuracy 51 → 75 % given the right i).  Re-ranking within top-k marginal candidates has a low ceiling on
pair exact match (reference pair among top-2×2 candidates 55 %, top-4×4 59 %), but choosing the pair by the model's
own chain-rule joint recovers essentially the whole S = 8 → 16 gap in free generation: **dev2 70.0 → 89.3 %**
(S = 16: 90.0 %) — at 2 extra forwards per step (22 per block), i.e. no compute advantage.  The zero-forward
compatibility head was not trained; the Phase B9 brief redirected the work to global block denoising.

## 14. Phase B9 — global block denoising (summary; full record in global_block_denoising.md)

Trajectory analysis of the K = 16 sampler: the final token is in the top-8 of 75 % of still-masked positions already
at the first forward (86 % at step 8) but top-1 only at 34 % (52 %); the independent-argmax block of the first forward
has 6.8 adjacent duplicate tokens per 16 — individually plausible, jointly degenerate marginals; coupling between
positions decays with distance (JSD 0.15–0.17 bits at distance 1–2, 0.05 at ≥ 8); a hidden-state probe does not pick
the eventual token better than the LM head.  Global refinement without training fails in every form (full-block
Jacobi: collapse to repetition; soft-token inputs: 4–6 % token accuracy; non-monotonic confidence re-masking: 62 %
vs 76 % for left-to-right monotonic at S = 8).  Conclusion: the frozen mask-only denoiser does one thing — predict
masks from committed context — and a parallel sampler has to be *taught* to resolve dependencies in one forward
(Phase B10).

## 15. Phase B10 — global consistency distillation (summary; full record in global_consistency_distillation.md)

The B8 chain-rule joint (2 extra forwards per S = 8 step; dev2 89.3 %) was used as a teacher: a new LoRA on the merged
B-series model is trained so that one forward emits, for the two positions a step places, the token the joint picks
and the *conditional* distribution the teacher reaches only after committing the first.  Result at 8 forwards per
block, no sequential conditioning at inference: hard-state pair agreement 3 → 48 %, GSM8K dev2 70.0 → 81.3 %
(p = 0.01), **GSM8K test 72.9 → 77.1 % (p = 0.001)**, same latency; S = 4 27 → 46 % on dev2.  Label-only CE works as
well as the KL; 5× data, more steps and a look-ahead term do not help; the student costs 2–4 items at S = 12/16; code
stays near 10 % at S = 8 (the teacher itself is 23 %).  First S = 8 improvement of the B-series on the test set.

## 16. Phase B10.5 — ablation and structural consistency (summary; full record in distillation_ablation_structural_consistency.md)

Same data, steps, rank and lr: masked CE alone on the teacher-trajectory states gives dev2 S = 8 83.3 % / S = 16 90.0 %
(GSM8K test 77.8 %), CE + KL 81.3 / 87.3, KL alone 75.3 / 81.3, ordinary data 74.7 / 82.0 — **the B10 gain is the
on-policy teacher-trajectory data, not the conditional KL.**  The code "indentation collapse" of student E was the
first pair of the answer mixing two formats (132 / 133 errors at block 0 position 3) plus an MBPP dialect shift
(colon accuracy 96 → 17 % teacher-forced); baseline and student A fail code through distributed pair-seam token drops.
Teacher-forced code pairs are 90–92 % right vs 47–49 % for prose; structural seams cost only 2–5 points — code dies
because every pair error is fatal.  The same masked-CE recipe with code states in the evaluation's prompt format lifts
HumanEval S = 8 from 8.5 to 19.5 % (teacher ceiling 23.2 %) at a 7-point GSM8K cost in the shared LoRA.  Gains do not
transfer to a non-monotonic sampler.  All at 8 forwards per block, no AR verification.

## 17. Phase B11 — sampler-aligned global denoising (summary; full record in sampler_aligned_global_denoising.md)

The training states of the B-series recipe (random masks over gold, visible tokens always correct) overlap the states the
non-monotonic global sampler visits by only 0.46–0.72 (0.13–0.44 for the left-to-right sampler); on-policy states of
the sampler itself overlap 0.97–1.0.  Mismatch with the teacher sequence is mostly wording freedom (43.5 % of tokens in
task-correct answers vs 49.7 % in wrong ones).  The frozen model's global sampler converges to its own fixed point:
revisits overturn ≤ 4 % of visible tokens, duplicates grow to 1.2 per block, completion order from a given state barely
matters.  Masked CE on the global sampler's own states with teacher-corrected targets (arm E) lifts the non-monotonic
sampler on dev2 from 20.0 / 58.0 / 63.3 to 43.3 / 65.3 / 74.0 % at S = 4 / 8 / 16 (significant at 4 and 16) and the
left-to-right sampler 70.0 → 78.7 at S = 8, with HumanEval unchanged; plain targets alone (arms C/D) do not help the
global sampler.  The gains come from better first placements under the global mask layouts, not from iterative
repair — revisits stay inert — so Gate 4 (true non-autoregressive correction) is not met; the global sampler remains
13 points behind left-to-right on the same model.  Arm B (left states only): dev2 left-to-right S = 8 **84.0 %** (best of the series), global sampler 50.0 % (−8): each sampler is improved by its own states and only by them.

## 18. Phase B12 — counterfactual global denoising (summary; full record in counterfactual_global_denoising.md)

From 150 mid-block states of the B11 global sampler, re-masking 1–4 visible positions and regenerating with the same
remaining budget repairs 30–55 % of wrong answers and breaks 12–19 % of correct ones; targeted positions (lowest
confidence, adjacency) are no better than random sets of the same size; an outcome-aware oracle (best of 25) reaches
95 % at 26 × the forwards.  Applied blindly as a zero-cost sampler rule (forced re-mask of k tokens at step 4), the
same intervention gives 59–63 % on dev2 vs 65.3 % without it.  Intervention value is not predictable from the state, so
no re-masking policy or joint-transition student was trained; the global line's remaining lever is a structural error
signal (duplicates, syntax), not confidence or teacher mismatch.

## 19. Clover study — self-speculative acceptance rule (summary; full record in clover_self_speculative.md)

DiffuSpace Clover-35B-A3B-Preview (2026) is the B-series block design (preview = bidirectional block over the prefix
cache, commit = causal pass) on Qwen3.6-35B-A3B, served by self-speculative decoding with an entropy-calibrated
acceptance rule (accept a draft when p_AR >= min(ratio, delta * exp(-H_AR))).  Implemented in engine/cdlm verified
decoding (`--accept-rule threshold`).  dev2: acceptance 16.5 -> 17.9 %, 186 -> 194 tok/s (zeta), accuracy unchanged;
GSM8K test: 88.0 -> 88.6 % (delta 0.3), 183 -> 191 tok/s.  Diagnostic: the exact walk accepts 73 % of the drafts it
examines and 83 % of its rejections have p_AR < 0.01, so no acceptance rule can add more than a few points — the
drafter is the bottleneck (consistent with B10/B11).  Keeping the unverified right side after a rejection is a
negative (1.07 tokens per step).


## 20. Phase B13 — auxiliary-conditioned drafter + representation alignment (summary; full record in b13_auxiliary_drafter.md)

PLM (arXiv 2610.26693) replaces the parallel decoder's uninformative [MASK] input with an AUXILIARY token carrying a
coarse summary of the AR model's predictive distribution at that position, and adds a cosine alignment loss between a
decoder layer's mask-row hidden state and an encoder layer's hidden state for the row before it; the paper's 150M
ablation goes 1.44 -> 1.785 tokens per call from the auxiliary and 1.85 with alignment.  Ported here as: aux(t) = the
RANK of the true token under the model's own AR distribution (128 auxiliary ids added after [MASK], rank >= 128 keeps
[MASK]), computed in one no-grad causal pass over the x0 rows per micro-batch; at inference every mask row is fed the
rank-0 id ("every token is the greedy continuation"), which is the conditioning the AR verifier checks.  Two runs on
the stage-2 recipe: auxiliaries + alignment, and auxiliaries alone.

Result: no gain.  GSM8K test (1319) 2.21 tokens per step for both against the control's 2.22; dev2 2.26 / 2.28 against
2.22 is inside the noise of 150 prompts.  Accuracy unchanged (88.3 / 88.9 vs 88.0).  Alignment moved no decoding
metric although its loss fell 0.64 -> 0.16.  Fed [MASK] instead, both B13 checkpoints collapse to 1.16-1.18 tokens per
step (2 % draft acceptance vs the control's 16.5 %): the drafter did learn to read the auxiliary, so the auxiliary
carries information during TRAINING — but the constant rank-0 auxiliary fed at inference says only "this is the AR
argmax", which the verifier already enforces, and the mask-trained drafter already drafts that mode.  PLM's auxiliary
is informative because it is SAMPLED per position from Q_t, and sampling is incompatible with the exactly-greedy
verification this engine ships (same wall the Clover study hit in §19: the drafter, not the acceptance rule, is the
bottleneck).  Kept: engine/cdlm/aux.py, `bench/run_cdlm.py --draft-input aux0`, the decoders' `exclude_from`
argument, and both checkpoints.
