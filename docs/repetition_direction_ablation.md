# Repetition has a direction: finding it, removing it, and what that is worth (2026-11-03 to 11-08)

A side study, on an external model, of a question that keeps coming back in this project: **can a degenerate
generation habit be removed from the weights, instead of suppressed at sampling time?**  Our own server fights
repetition with `DSV41_LOOP_DETECT`, `ban_cycles` and a repetition penalty, and B13's drafter work keeps running
into the same wall from the other side.  The trigger was the "Sparks of Weight-Space Programming" note; the method
is not from that note but from Heretic-style difference-of-means ablation, which is what the note's practical
core reduces to at our scale.

Subject: `Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B`, a reasoning distillation into Qwen3.5-9B (32 layers, hidden 4096,
hybrid linear/full attention), and its decensored variant `xtools-at/...-UNCENSORED-heretic`, which is itself a
Heretic product and so a reference point for what such an edit costs.  Hardware: one RTX PRO 6000 (zeta).
Everything greedy, GSM8K, repetition scored with the detector the first measurement used: **a 12-word window
occurring 4 or more times in the answer**.

## 1. Result

Removing a single direction at half strength from what layers 16-31 write to the residual stream, on 600 questions
that neither the direction nor the configuration had seen:

| | before | after |
|---|---|---|
| accuracy | 93.17 % | 94.17 % |
| looping answers | 47 (7.83 %) | **27 (4.50 %)** |
| errors | 41 | 35 |
| errors that loop | 20 | 13 |
| answers at the 2048-token cap | 28 | 20 |
| mean tokens | 491.4 | 462.3 |

Paired: 26 answers stopped looping, 6 started (sign test p = 0.0005); 6 right answers became wrong and 12 wrong
became right (McNemar two-sided p = 0.238).

So repetition **is** carried by a direction that a difference of means finds, removing it works, and it costs
nothing measurable in accuracy.  It also only takes away **half** the loops, and the accuracy gain is not
significant — the honest headline is "loops down 43 %, accuracy unchanged", not "accuracy up".

## 2. How the direction was found

450 generations, labelled by the detector above, taking the onset to be the first token position whose prefix
satisfies it.  Two probes per layer, split by question:

- **descriptive**: positions inside the repetition against positions in generations that never repeat.
- **predictive**: the 24 positions *before* onset against matched positions elsewhere.  Only this one could justify
  an edit, because by the time the descriptive signal exists the loop has already started.

With negatives drawn uniformly the predictive AUC was 0.93 — and meaningless, because loop onset happens deep in a
generation while uniform negatives sit at half the depth; a direction encoding "how far in am I" scores that by
itself.  With **every negative matched to a positive's absolute position** (median position 604 on both sides):

| layer | descriptive (diff of means) | predictive | position alone |
|---|---|---|---|
| 4 | 0.637 | 0.755 | 0.456 |
| 8 | 0.693 | 0.858 | 0.456 |
| 12 | 0.699 | 0.862 | 0.456 |
| 16 | 0.724 | 0.888 | 0.456 |
| **20** | 0.720 | **0.895** | 0.456 |
| 24 | 0.709 | 0.892 | 0.456 |
| 28 | 0.704 | 0.881 | 0.456 |
| 31 | 0.629 | 0.802 | 0.456 |

Three things worth keeping:

- The **predictive** signal is stronger than the descriptive one (0.895 against 0.720).  The state that precedes a
  loop is represented more cleanly than the loop itself.
- A **single direction beats a trained hyperplane** (ridge 0.842 at layer 20), which is the shape an ablation needs.
- Position explains nothing (0.456, below chance).

## 3. The sweep

`W <- (I - alpha d d^T) W` on the residual writers (`linear_attn.out_proj`, `self_attn.o_proj`, `mlp.down_proj`),
screened on 200 questions, baseline 95.0 % accuracy and 5.5 % loops:

| arm | accuracy | loops | loop fixed | new loop | broke | fixed | mean tokens |
|---|---|---|---|---|---|---|---|
| alpha 1.0, L9-24 | 93.5 % | 1.0 % | 11 | 2 | 6 | 3 | 398 |
| alpha 0.25, L9-24 | 96.5 % | 4.0 % | 5 | 2 | 0 | 3 | 401 |
| alpha 0.5, L9-24 | 95.0 % | 1.5 % | 10 | 2 | 3 | 3 | 397 |
| alpha 0.75, L9-24 | 93.5 % | 1.5 % | 9 | 1 | 7 | 4 | 404 |
| alpha 0.5, L16-24 | 96.0 % | 4.0 % | 6 | 3 | 1 | 3 | 402 |
| **alpha 0.5, L16-31** | **96.5 %** | **2.0 %** | 7 | **0** | 1 | 4 | **415** |
| alpha 0.5, L9-31 | 95.5 % | 1.0 % | 11 | 2 | 2 | 3 | 382 |
| alpha 0.5, mlp only | 95.0 % | 5.5 % | 5 | 5 | 3 | 3 | 418 |
| alpha 0.5, attention only | 95.5 % | 2.5 % | 7 | 1 | 2 | 3 | 394 |
| alpha 0.5, direction from layer 16 | 95.0 % | 2.5 % | 8 | 2 | 3 | 3 | 389 |

- **Strength trades monotonically against capability.**  Loops fall as alpha rises (5.5 to 4.0 to 1.5 to 1.0 %)
  while accuracy peaks early and then falls (96.5, 95.0, 93.5 %).  Full removal is simply too strong.
- **The loops live on the attention side.**  Editing `mlp.down_proj` alone changes nothing (5 fixed, 5 created);
  the attention output projection alone gets 2.5 %.
- **Editing the early layers shortens the output.**  L9-31 reaches the lowest loop rate (1.0 %) at 382 tokens
  against the baseline's 430: it avoids loops partly by stopping sooner.  From layer 16 up that does not happen.
- **AUC does not pick the configuration.**  The layer-16 direction scores 0.888 against layer 20's 0.895 and is
  clearly worse to ablate with (2.5 %, 3 answers broken).

## 4. What repetition is not

Three explanations were tested and ruled out before the direction was looked for, each because the measurement
contradicted it:

- **Not quantization.**  Same harness, same questions, only the quantization changing: Q4_K_M 8.5 % loops at
  92.80 % accuracy, Q6_K 7.7 % at 91.66 %, bf16 9.3 % at 94.67 %.  Raising the quantization leaves the loops and
  costs nothing but time; accuracy, on the other hand, does improve with precision (bf16 is ~2 points above Q4).
- **Not sampling.**  The llama.cpp server at `temperature: 0` is exactly reproducible, 12 of 12 runs identical.
- **Not the prompt.**  The GGUF's embedded template and the HF tokenizer's produce the same 44-token string,
  byte for byte, `<think>` opener included.

What the data does show is that **which** question loops is not a property of the question.  Loops barely overlap
across paths: 65 of (112, 101) between Q4 and Q6, 64 of (109, 112) between the original and the decensored model.
Greedy decoding here is chaotic — Q4 and Q6 diverge from bf16 at the *same* token positions (75, 40, 66 on the first
three questions), which is what near-ties look like: any perturbation tips them the same way.  Loops are a
trajectory outcome, and yet the state that precedes them is linearly separable.  Those two facts sitting together
is the interesting part of this study.

## 5. Mistakes made on the way

Recorded because each one cost hours and each one is easy to repeat:

- **The first attempt labelled with the wrong detector.**  Exact token-level period-8 repetition fires on 0.7 % of
  these generations where the benchmark's word-level rule fires on 9.3 %, so 450 generations yielded 2 positives
  and the probes had one class.  The labels have to come from the same rule the phenomenon was defined by.
- **The first probe had a position confound** and reported 0.93 for what was partly "how deep am I".  Matched
  negatives brought it to 0.895 and the descriptive probe from 0.91 to 0.72.
- **The first run saved no text**, so the detector question could not be settled without regenerating.
- **"bf16 does not loop" was reported, and was wrong** — an artefact of the detector above.
- **The direction was assumed not to exist**, twice, on the grounds that loops are trajectory-dependent.  The
  measurement says otherwise.

## 6. For this project

- The method transfers in principle to the 748B server, and the recipe is now concrete: collect ~450 generations
  with hooks, label with the word-level detector, take the difference of means of the pre-onset window at a middle
  layer, and project it out of the residual writers of the upper half at half strength.  What does not transfer
  cheaply is the setting: fp8 weights sharded over four GPUs with expert parallelism, so the edit targets have to
  be identified again, and the server already suppresses repetition at sampling time.  The useful experiment there
  is a comparison, not an addition: does the edit let the penalties come off?
- For the 9B itself the ranking is clear and the weight edit is second: **run it in bf16** (+1.9 points over Q4_K_M)
  and then ablate (loops 7.8 to 4.5 %, accuracy flat).  Q6_K is the one option to skip — slower than Q4_K_M
  (160 against 190 tok/s) and no better.
- The 27B long-context model in `docs/` terms: `summerAIO/Qwen3.8-27B-Longbook27-v2` collapsed into exactly this
  failure mode at 57K tokens, and the same probe would say whether its loops carry the same direction.

Data and code: `results/loopdir/` (per-question rows for every run: `confirm_ablation.json` is the 600-question
before/after, `ablate_sweep.json` the nine arms, `loop_direction3.json` the probes, `results_full.json` and
`results_q6.json` the quantization comparison, `loop_texts3.jsonl` the 450 labelled generations) and
`bench/loopdir/` (the scripts, in the order they were used: `ds9b_bench.py`, `bf16_looprate.py`,
`determinism.py`, `compare_prompt.py`, `three_paths.py`, `loop_direction3.py`, `ablate_loop.py`,
`ablate_sweep.py`, `confirm_ablation.py`).  The activations (348 MB) stay on zeta at `~/loopdir/loop_acts.npz`.
