# Phase B11 — Sampler-Aligned Global Denoising (K = 16, few parallel denoising steps, no left-to-right commitment)

Brief: "SHIGERU Phase B11 — Sampler-Aligned Global Denoising" (2026-10-26).  Main hypothesis under test: the denoiser's
*training states* (random masks over gold tokens, visible tokens always correct) do not match the states a global
sampler visits (self-generated, partly wrong visible tokens; mask layouts shaped by the sampler), and this mismatch —
not denoising capacity — is why every global refinement attempt (B9) failed and why the B10.5 gains did not transfer to
a non-monotonic sampler.  Competing hypotheses kept open: AR-pretrained dependency structure, the noise distribution,
capacity, and the sampling transition itself.  Constraints: no AR verification, no speculative/sequential drafts, no
visible-token correction CE (B2 failure), no left-to-right commitment hidden inside a "global" sampler.

## 1. Audit — how this differs from what failed before

| earlier attempt | what it did | why it failed (measured) | what B11 does differently |
|---|---|---|---|
| B9 full-block Jacobi (sync) | re-forwarded the all-visible self-generated block | the model never saw wrong visible tokens: 6–9 rewrites/step, drift to repetition | states with wrong visible tokens become *training* states, but only re-masked positions are targets |
| B9 soft-token inputs | fed probability-weighted embeddings | out of distribution, 4–6 % | not used; stays in token space |
| B9/B10.5 non-monotonic re-masking | any token may return to [MASK] (62 % at S = 8 on dev-50) | the denoiser was trained on random masks over gold, never on its own re-masked states | the sampler's own states are collected and trained on (masked CE), with the joint teacher supplying consistent targets |
| B2 self-correction training | CE on corrupted *visible* positions | collapsed free generation | **not** reintroduced: visible positions are never targets; a wrong visible token is corrected only by explicitly re-masking it in the training state |
| B7 trust-aware re-masking | re-mask by a confidence/trust score | the model's opinion of a visible token is uninformative (AUROC 0.5) | re-masking rules are re-examined only with a sampler-aligned model, whose opinion of visible tokens may become informative because it has seen wrong ones |
| B10 / B10.5 | masked CE on left-to-right S = 8 states + teacher labels: +5 points on test, no transfer to the global sampler | the states were left-to-right states | the same recipe on the *global sampler's* states (and mixed) |

Data: dev items 0–49 (reference = the base model's own greedy answer, results/pure/dev_refs_e5.json) for the state
analysis and teacher-correction diagnostics; bench/prompts_gsm8k_dev2.json (150 fresh items) for decisions; GSM8K test
only for a fixed final setting; training states only from GSM8K-train prompts disjoint from all of these.

## 2. B11-A — sampler state distributions (bench/diag_sampler_states.py)

(results in §2.1 when the dev-50 runs finish)

## 4. B11-C — sampler-aligned training states (distill/gen_global_states.py)

States are collected by running the *student sampler itself* on GSM8K-train prompts disjoint from every evaluation set
(bench/prompts_gsm8k_pairtrain.json items 800–1099 for the non-monotonic re-masking sampler, 1100–1399 for the
left-to-right sampler; 300 prompts each, K = 16, S = 8), following its own choices (on-policy).  Every state x_s — the
block before the s-th forward, with its [MASK]s and its possibly wrong visible tokens — is stored twice:

| variant | input state | targets | what it teaches |
|---|---|---|---|
| plain | x_s as visited | masked positions ← the teacher's sequential completion *conditioned on the current visible tokens* (one forward per mask, left to right over the masks) | fill the masks consistently with whatever is visible, including wrong tokens — the capability a sampler needs at every step; visible positions are never targets |
| corrected | x_s with every visible token ≠ T re-masked (T = the teacher's sequential block for this prefix, 16 forwards) | all masked positions ← T | correct a wrong visible token only by making it a mask: the B2 failure (CE on visible positions) is not reintroduced |

The all-mask state (step 1), random-looking states (the re-masking sampler at mid-ratio), non-monotonic states
(masks with visible tokens to their right) and teacher-corrected states are all present in the collected set; the
"structured mask" and "global refinement" states of the brief are the re-masking sampler's own states at steps 2–8.
Per state the number of visible tokens differing from T is recorded (the wrong-visible statistic of free generation).
Cost: ~2 × n_mask teacher forwards per state (plain + the shared T): ~40 s per prompt.
