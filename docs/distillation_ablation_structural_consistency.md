# Phase B10.5 — Distillation Ablation & Structural Consistency

Brief: "SHIGERU Phase B10.5 — Distillation Ablation & Structural Consistency" (2026-10-25).  Two questions: (1) *why*
did the B10 student improve S = 8 (teacher data, the CE on the teacher's first token, or the conditional KL)?  (2) why
does the code student collapse into indentation errors, and is that a structural-dependency failure of parallel
placement?  No new large training before both are answered.

## 1. The B10 results being explained (global_consistency_distillation.md §5)

| student (all: new LoRA r 64 on the merged e5 + lr1e-4 B-series model, lr 1e-4, accum 8, hard states ×4) | states | steps | loss | dev2 S = 8 | dev2 S = 16 | HumanEval S = 8 / 16 | GSM8K test S = 8 |
|---|---|---|---|---|---|---|---|
| baseline (no further training) | – | – | – | 70.0 | 90.0 | 8.5 / 50.6 | 72.9 |
| A | 12 k GSM8K | 1 000 | CE(a*) + KL(p_T(·\|a*) ‖ p_S at j) + mask CE (final tokens) | 81.3 | 87.3 | 10.4 / 54.9 | **77.1** |
| B | 12 k GSM8K | 1 000 | A + look-ahead KL on the other masked positions | 79.3 | – | – | – |
| C | 65 k GSM8K | 2 000 | as A | 77.3 | 87.3 | – | – |
| D | 12 k GSM8K | 1 000 | CE(a*) + CE(b*) + mask CE (labels only) | 82.0 | – | – | – |
| E | 65 k GSM8K + 11 k MBPP | 3 000 | as A | 80.0 | 88.7 | **0.0** / 40.9 | (running) |

A and D differ only in the j-term (KL to the teacher's conditional vs CE to the teacher's token): same data, steps,
rank, lr, sampling.  Their equal results (81.3 / 82.0) already say the *distribution* is not what matters; what remains
open is whether the *teacher data* (states of teacher-followed trajectories + the teacher's pair) or the *pair terms*
carry the gain.  The ablation in §2 separates them.

## 2. CE / KL / data ablation (brief §2) — design

All arms: the same 12 000 teacher-trajectory states (286 GSM8K-train prompts, 7 held out), 1 000 steps × 8, lr 1e-4,
LoRA r 64 / α 128 on the same merged model, hard states ×4 in sampling where pair terms exist, left-to-right S = 8 and
S = 16 on dev2 (150 fresh items).

| arm | what is trained | pair terms | teacher data |
|---|---|---|---|
| A  baseline | nothing | – | – |
| B  masked CE only (`--loss-mode mask_only`) | standard masked CE on *all* masked positions incl. the pair, targets = the teacher-trajectory final block (pseudo-label SFT on the same states) | none | yes |
| C  teacher-label CE (= student D, `label_ce`) | CE(a*) + CE(b*) + mask CE | labels | yes |
| D  conditional KL only (`kl_only`) | KL at j + mask CE, no CE at i | distribution at j | yes |
| E  CE + KL (= student A, `full`) | CE(a*) + KL(j) + mask CE | both | yes |
| F  ordinary data, same budget | continuation of the lr1e-4 adapter on the original B-series data (phaseA_stage2c, mask-only loss, 1 000 × 8, lr 1e-4; distill/train_bdiff.py --init-adapter) | none | no |

Reading rule fixed in advance: if B ≈ C ≈ E ≫ A and F ≈ A, the gain is the *teacher data* (states and pseudo-labels),
not the pair objective; if C ≈ E ≫ B, the pair terms matter; if D ≈ E, the KL alone suffices; if F ≈ E, it is just
more training.

## 3. HumanEval syntax collapse of student E (brief §3; bench/diag_code_structure.py, results/pure/code_struct_gen_e.json)

Classification of the 164 programs (ast.parse, then the HumanEval tests), S = 8 vs S = 16, same student, same prompts:

| | pass | test_fail (AssertionError) | runtime_error | timeout | incomplete (length budget / no fence) | SyntaxError | **IndentationError** |
|---|---|---|---|---|---|---|---|
| student E, S = 8 | 0 | 1 | 0 | 0 | 30 | 0 | **133 (81 %)** |
| student E, S = 16 | 67 | 51 | 33 | 1 | 11 | 0 | 1 |

Where the 133 IndentationErrors come from (error line → generated token → block, position, denoising step):
**132 of 133 are the same token: block 0, position 3, placed at step 2**; the outputs begin
`defpython\n has_close_elements(` instead of ```` ```python\ndef has_close_elements( ```` or `def has_close_elements(`.
The first forward (all 16 masks) places positions 0–1 independently: position 0 takes `def` (the mode of the unfenced
answer style), position 1 takes `python` (the mode of the fenced style ```` ```python ````) — two answer formats mixed
in one pair.  Step 2 then places `\n` and ` has` (the function name with the leading space it would have after `def`)
— a header line beginning with a space, hence `IndentationError: unexpected indent` on the first line.  Everything
after that is largely correct code (the test_fail / runtime rows of S = 16 show the same bodies).  At S = 16 the first
position is placed alone and the format is consistent (`def triangle_are…`).

So the "indentation collapse" is **not** an indentation-dependency failure inside the code body: it is the independent
placement of the first two tokens of the answer, i.e. the first-forward incoherence measured in B9 (§3 there: 6.8
duplicate/incoherent adjacent tokens per all-mask block), landing on the one position where it is fatal for Python.
Student E made it systematic because its MBPP teacher states taught a pair preference (`def` first) that conflicts with
the HumanEval prompt format ("Reply with only the complete function in a ```python block"); the baseline and student A
(§3.1 below) show how often the same first pair breaks without that training.  The S = 8 and S = 16 outputs of E
diverge at token 7 on average (first 32 tokens identical in 4 % of the problems): the decisive difference is made in
the first block.

Token-category rates of the generated code (S = 8 vs 16) are the same within 1 point (indent 14.5 / 14.2 %, newline
9.7 / 8.9 %, bracket 7.3 / 7.6 %), so the student does not under- or over-produce structural tokens; the pair types
placed together at S = 8 are dominated by other+other (7 191), indent+other (1 837), newline+indent (1 272),
other+newline (1 128) — the newline→indent pair, the one that would break indentation inside the body, is placed
jointly 1 272 times and is responsible for none of the 133 errors.

### 2.1 Results (dev2, 150 fresh items; held-out hard-state pair agreement on 7 unseen training prompts)

| arm | hard-state pair agreement with the teacher | dev2 S = 8 | dev2 S = 16 |
|---|---|---|---|
| A  baseline | 2.9 % | 70.0 % | 90.0 % |
| **B  masked CE only, same states, teacher-trajectory targets** | **51.4 %** | **83.3 %** | **90.0 %** |
| C  teacher-label CE (student D) | 48.6 % | 82.0 % | – |
| D  conditional KL only | 43.8 % | 75.3 % | 81.3 % (−13 items: the KL alone damages S = 16) |
| E  CE + KL (student A) | 47.6 % | 81.3 % | 87.3 % |
| F  ordinary data, same budget (continuation of the lr1e-4 adapter, mask-only loss) | – | 74.7 % | 82.0 % (−12 items) |
| B-large: arm B on all 65 k states, 2 000 steps (zeta) | 37.9 % | 80.7 % | 86.0 % (−6 items; HumanEval S = 8 8.5 % = baseline) |

**Gate 1 answered: the gain is the teacher data, not the pair objective.**  The plain masked cross-entropy on the
same 12 000 on-policy S = 8 states — targets = the block the teacher-followed trajectory ended with — matches or beats
every pair-term variant (83.3 vs 81.3–82.0 %), reproduces the teacher's hard pairs *better* (51 vs 44–49 %), and does
**not** lose anything at S = 16 (90.0 vs 87.3 % for the KL student).  The KL to the teacher's conditional is the
weakest single ingredient (75.3 %); the "conditional distribution distillation" of B10 is therefore not a success of
distribution matching but of **training on the student's own denoising states with joint-resolved pseudo-labels** —
what makes the targets informative is that they come from a trajectory in which every pair was resolved by the
chain-rule teacher, i.e. they are high-quality pseudo-labels *in the state distribution the sampler actually visits*.
The same budget of ordinary training (arm F) gives +4.7 at S = 8 but costs 12 items at S = 16 — continuing the
mask-only adapter on gold data at lr 1e-4 after its own schedule simply over-trains it; arm B on the same number of
updates improves S = 8 by 13.3 and leaves S = 16 unchanged.  Scaling arm B to all 65 k states (2 000 steps) gives
80.7 % at S = 8 — again no gain from more data (cf. students C and E in B10): the usable information is in the first
~10 k on-policy states; what is missing is not quantity but states the student does not yet produce (first-forward
and code-format states, §3–§5).

### 3.1 The same analysis for student A (GSM8K-only teacher states) — the general S = 8 code failure

| student A | pass | test_fail | runtime_error | timeout | incomplete | SyntaxError | IndentationError |
|---|---|---|---|---|---|---|---|
| S = 8 | 17 (10.4 %) | 23 | 8 | 2 | 8 | **80** | **26** |
| S = 16 | 90 (54.9 %) | 54 | 14 | 1 | 2 | 3 | 0 |

Here the 106 syntax failures are **spread over every denoising step (11–25 per step) and every position in the
block**, not concentrated: the error tokens are "other" (69), newline (29), bracket (11), quote (5).  Typical first
errors: `mean = sum(numbers) len(numbers)` (a dropped `/`), `if not == string:` (a dropped operand),
`result = []\n()\n    i = 0` (a stray `()` line), `delimeter: int) List[int]:` (a dropped `->`) — single missing or
duplicated tokens at the seam between the two tokens a step places, each plausible alone and wrong together: the
pairwise incoherence of pairwise_joint_diffusion.md, which in prose costs a paraphrase and in Python costs the program.
S = 8 and S = 16 outputs agree on the first 32 tokens in 89 % of the problems and diverge on average at token 105 — the
first block is fine for A (unlike E), the body is where the pairs break.  Both students' failures are therefore the
same mechanism at different places: **independent placement of a dependent pair**; E's MBPP states merely moved the
most frequent break to the first pair of the answer.

### 3.2 Baseline (undistilled adapter) — same picture

| baseline | pass | test_fail | runtime_error | incomplete | SyntaxError | IndentationError |
|---|---|---|---|---|---|---|
| S = 8 | 14 (8.5 %) | 12 | 8 | 5 | **85** | **40** |
| S = 16 | 83 (50.6 %) | 65 | 9 | 3 | 4 | 0 |

Error tokens at S = 8 are again spread over all steps (10–22 per step) and categories (other 65, **newline 53**, bracket
9); the baseline breaks more often at newlines than student A (53 vs 29) — the newline+indent seam and the
statement-end+newline seam are the Python-specific places where two independently placed tokens most often disagree.
S = 8 and S = 16 agree on the first 32 tokens in 90 % of problems and diverge at token 110 on average.

**Gate 2 answered.**  The syntax collapse is located: (a) for student E, one position — the first pair of the answer,
placed by the all-mask first forward, mixing two answer formats (132 / 133 errors); (b) for the baseline and student A,
distributed single-token drops/duplications at the seam between the two tokens of a step (operators, `->`, stray
`()`), with newline seams over-represented in the baseline.  Both are the independent-pair placement failure; neither
is a model-level inability to indent (S = 16 has 0–1 IndentationErrors on the same prompts).  The structural
dependency measurements (§4) quantify how much more often code pairs than prose pairs cross such a seam.

## 4. Structural dependency analysis (brief §4; bench/diag_code_structure.py --refs, teacher-forced, left-to-right S = 8)

Reference = the base model's own greedy answer; blocks of 16 decoded from the reference prefix; "pair" = the two
positions a step places; "crossing a boundary" = at least one of the two reference tokens is a newline, indent,
colon or bracket.  HumanEval (64 prompts, 3 528 pairs) vs GSM8K dev (50 prompts, 2 368 pairs), undistilled adapter.

| | HumanEval: token acc | GSM8K: token acc |
|---|---|---|
| all tokens | 92.7 % | 53.3 % |
| newline | 92.9 % | 58.4 % |
| indent (whitespace-only token) | 91.9 % | 56.4 % |
| colon | 96.4 % | 46.0 % |
| bracket | 89.9 % | 58.4 % |
| quote | 90.1 % | 78.6 % |
| token right after a newline | 92.0 % | 53.1 % |
| **pair crossing a syntactic boundary** | **90.0 %** (n = 1 862) | 48.7 % (n = 1 022) |
| **pair not crossing one** | **92.3 %** (n = 1 666) | 47.0 % (n = 1 346) |
| lowest pair types | newline+indent 86.8 %, bracket+other 87.2 %, indent+other 89.2 % | other+colon 38.6 %, other+newline 43.7 % |
| S = 16 (one token per forward), all tokens | 95.1 % | – |

Reading: (1) code is far more *predictable* than prose under teacher forcing (92.7 vs 53.3 %: a Python continuation
is nearly determined by its header and context; a GSM8K sentence has many valid wordings), so the exact-match
pair accuracy on prose mostly measures paraphrase, not failure.  (2) Structural seams cost code pairs 2.3 points
(90.0 vs 92.3 %), the newline→indent pair 5.5 points — a real but modest structural dependency; the colon is the
*easiest* code token (96.4 %).  (3) The reason code collapses at S = 8 while prose survives is therefore not that
code pairs are much harder, but that **every pair error in code is fatal and the errors compound**: with ~8 pairs
per block and 8–13 % pair error, a 16-token block is syntactically intact with probability ≈ 0.9^8 ≈ 0.43, and a
150-token function needs ~10 blocks.  At S = 16 the pair seam disappears (token accuracy 95.1 %) and 50–55 % of the
programs pass.  (4) The model *has* learned the structural dependencies (newline→indent 86.8 %, indent→statement
89.2 % when placed jointly; 92–95 % when placed sequentially); what it cannot do is apply them to two positions in
one forward — the same statement as for arithmetic, with a harsher loss function.

**Consequence for §5.**  A Python-specific "structure-aware masking" (jointly masking newline/indent/bracket) would
target a 2–5 point seam penalty that is not where the programs are lost; the lost programs come from the general
pair-seam error rate (8–13 %) that the ablation of §2 has already shown how to lower — masked CE on the student's own
S = 8 states with joint-resolved targets (arm B) — and from the first-forward format decision (E).  The minimal
training to run is therefore arm B with *code states in the evaluation's prompt format* added, not a Python rule.

### 4.1 What the MBPP states did to student E (teacher-forced HumanEval, same blocks as §4)

| token accuracy, HumanEval reference blocks | baseline S = 8 | **student E S = 8** | baseline S = 16 | **student E S = 16** |
|---|---|---|---|---|
| all "other" tokens | 92.9 % | 81.0 % | 95.1 % | 81.0 % |
| colon | 96.4 % | **17.9 %** | 96.4 % | **16.7 %** |
| bracket | 89.9 % | 74.7 % | 94.5 % | 77.1 % |
| newline | 92.9 % | 82.2 % | 94.2 % | 75.2 % |
| indent | 91.9 % | 90.4 % | 94.9 % | 90.8 % |
| pair crossing a boundary / plain | 90.0 / 92.3 % | 77.1 / 80.0 % | – | – |

Student E lost code prediction *in general*, at S = 16 as much as at S = 8 — the colon (the end of a `def` line or a
block opener) falls from 96 to 17 %: the MBPP teacher states (headers without type hints, two-space indentation in the
MBPP references, no code fence) taught a different code dialect, and the student now predicts that dialect inside
HumanEval blocks.  This is a *distribution shift of the training states*, not a parallel-decoding effect — it would
be visible with any training recipe on those states — and it is what the function-format code states of §5 are
designed to avoid.  Indentation tokens themselves are unaffected (90–91 %): the "IndentationError" label in §3 was
never about indentation.

## 5. Structural consistency training (brief §5) — the one minimal experiment

Chosen from §2–§4: not a Python rule and not a new loss, but the recipe the ablation identified (arm B: masked CE on
the student's own S = 8 states with joint-resolved targets) applied to the states that are missing — code in the
*evaluation's prompt format*.  Teacher states are generated on 427 MBPP problems rendered as function-completion
prompts (header + docstring + two doctest lines inside a ```python fence; bench/prompts_mbpp_fnformat.json; HumanEval
itself is never used), with the chain-rule joint teacher following its own choices (distill/gen_pairs.py), and arm B
is trained on 12 k GSM8K + these code states for 1 500 steps.  Measured: HumanEval S = 8 / S = 16 (syntax validity
and IndentationError rate from the classifier of §3), dev2 S = 8.  Results in §5.1 (running: GPU 7).

### 5.1 Results

| student (masked CE only, 1 500 steps, hard ×4) | states | HumanEval S = 8 | HumanEval S = 16 | GSM8K dev2 S = 8 |
|---|---|---|---|---|
| undistilled adapter | – | 8.5 % | 50.6 % | 70.0 % |
| arm B (12 k GSM8K states) | 12 k | 8.5 % (B-large) | – | 83.3 % |
| student E (GSM8K + MBPP-format states, CE + KL) | 76 k | 0.0 % | 40.9 % | 80.0 % |
| **arm B + function-format code states** | 12 k + 24.7 k | **19.5 %** (32 / 164) | 47.0 % (−6 items) | 76.0 % (−11 items vs arm B) |
| reference: B8 joint teacher at S = 8 (22 forwards) | – | 23.2 % | – | 89.3 % |

Syntax classification of the B + code student's S = 8 programs: __BCSYN__

Reading: the same recipe that lifted GSM8K lifts code once the states are in the evaluation's format — HumanEval S = 8
8.5 → 19.5 %, i.e. 84 % of the way to the sequential teacher's 23.2 % — with no Python-specific rule or loss; the
MBPP-dialect failure of student E (0 %) is confirmed to have been a data-format problem.  The cost: one LoRA now serves
two state distributions and GSM8K S = 8 falls from 83.3 to 76.0 % on dev2 (code states are 2/3 of the mixture) and
HumanEval S = 16 loses 6 items.  **Gate 3: met for syntax validity and S = 8 code generation (8.5 → 19.5 %), not met
for the 30 % target and not without a GSM8K cost at this mixture.**  Target-wise the ceiling is the teacher's 23 %: at
S = 8 the pairwise teacher does not solve code, so a better teacher (first-forward / longer sequential resolution per
step), not a better student, is the next lever for code.

## 6. Monotonic unmasking vs global refinement (brief §6)

Every number above is left-to-right monotonic unmasking (K/S tokens committed per forward, never revisited).  To test
whether the new training also gives the model a capability a *global* sampler can use, the arm-B student is run with
the non-monotonic confidence re-masking sampler of B9 (any token may return to [MASK]; visible tokens are kept, masks
are filled; exactly S forwards) and compared with the same sampler on the undistilled adapter (B9 §4: 32 / 62 / 74 %
at S = 4 / 8 / 16 on dev-50 vs 32 / 76 / 92 % monotonic).  The simple Jacobi and soft-token samplers are not repeated.

| dev-50, GSM8K | S = 4 | S = 8 | S = 16 |
|---|---|---|---|
| monotonic left-to-right, undistilled (B9 §4) | 32 % | 76 % | 92 % |
| **monotonic, arm-B student** | **56 %** | **82 %** | 92 % |
| non-monotonic confidence re-masking, undistilled (B9 §4) | 32 % | 62 % | 74 % |
| non-monotonic confidence re-masking, arm-B student | 28 % | 62 % | 62 % |

The capability the student acquired is specific to the states it was trained on: monotonic left-to-right S = 8
(and, by transfer, S = 4: 32 → 56 %).  Under a sampler that revisits tokens and places them in confidence order it
gives nothing at S = 4/8 and loses 12 points at S = 16.  **Gate 4 (non-autoregressive capability): not met** — the
training improves the parallel *left-to-right* decoder, not the model's ability to update several arbitrary positions
jointly; a global sampler needs its own on-policy states in the training set (the same recipe, different sampler), which
is the natural next experiment and was not run here (brief §6 forbids repeating the B9 samplers without a matching
noise process; the matching data recipe is now known).

## 7. GSM8K · HumanEval · MBPP (brief §7)

| decoder, K = 16, left-to-right, no AR verification | forwards / block | GSM8K dev2 | GSM8K test | HumanEval | MBPP (sanitized) |
|---|---|---|---|---|---|
| undistilled adapter, S = 16 | 16 | 90.0 | 84.8 | 50.6 | 50.6 (216 / 427) |
| undistilled adapter, S = 8 | 8 | 70.0 | 72.9 | 8.5 | 1.9 (8 / 427) |
| B8 joint teacher, S = 8 | 22 | 89.3 | 80.7 | 23.2 | – |
| B10 student A (CE + KL), S = 8 | 8 | 81.3 | 77.1 | 10.4 | 4.2 (18 / 427) |
| B10 student A, S = 16 | 16 | 87.3 | – | 54.9 | 52.9 (226 / 427) |
| B10 student A' (EOS pairs excluded), S = 8 | 8 | 82.7 | 77.3 | – | – |
| B10 student E (GSM8K + MBPP states), S = 8 | 8 | 80.0 | 78.4 | 0.0 | (trained on MBPP) |
| **ablation arm B (masked CE only, 12 k states), S = 8** | 8 | **83.3** | **77.8** (+184 / −119 vs baseline, p = 0.0002) | – | – |
| arm B, S = 16 | 16 | 90.0 | – | – | – |
| arm B + function-format code states, S = 8 / 16 | 8 / 16 | 76.0 | – | **19.5** / 47.0 | (MBPP-derived prompts in training) |

Syntax validity / IndentationError rate at S = 8 (HumanEval, 164): baseline 39 / 24 %, student A 35 / 16 %, student E
19 / 81 %; at S = 16 all three are ≥ 97 % syntactically valid with 0–1 IndentationErrors.

## 8. Speed and memory

Unchanged from B10 (global_consistency_distillation.md §8): every student is a LoRA merged into the same weights, so
tokens/forward (1.71 at S = 8), tok/s (68–78 on an A100 HF path, 105 on zeta), latency (2.4–2.7 s per GSM8K problem)
and peak VRAM (7.9 GB inference; 20.7–21 GB training with the frozen teacher resident) are those of the baseline
sampler.  Training of each ablation arm: 25 min (12 k states, one A100 or zeta); teacher-state generation 9 s per prompt.
This phase: 10 LoRA runs ≈ 4.5 GPU-hours, diagnostics and evaluations ≈ 9 GPU-hours.

## 9. Failure causes (recorded)

1. **The B10 "conditional distribution distillation" was mis-attributed.**  Its gain comes from training on the
   student's own S = 8 states with joint-resolved pseudo-labels; the KL to the teacher's conditional is the weakest
   ingredient and costs accuracy at S = 16.  The honest name of the technique is *on-policy pseudo-label SFT with a
   sequential joint teacher*.
2. More of the same data (65 k states) does not help and starts to hurt S = 16 (86.0 %); the states that matter are
   the ones the student does not yet produce correctly, not more of the ones it does.
3. Student E's code collapse: MBPP-format teacher states taught a different code dialect (colon accuracy 96 → 17 %
   teacher-forced, independent of S) and made the first pair of the answer mix two formats.
4. The ordinary-data arm (F) over-trains the adapter (S = 16 −12 items) — continuing a converged LoRA on its own data
   is not a neutral control; the on-policy states are what make the extra updates useful.
5. The monotonic-sampler gains do not transfer to a non-monotonic sampler (§6).
6. dev2 (150 items) overstated every S = 8 gain by 3–9 points relative to the test set (teacher 89.3 → 80.7, A 81.3 →
   77.1, E 80.0 → 78.4, A' 82.7 → 77.3): decisions were made on dev2, the quoted numbers are test.

## 10. Go / No-Go and the next architecture

| gate | verdict |
|---|---|
| 1 cause of the B10 gain | **identified**: teacher data (on-policy states + joint-resolved labels), not the pair/KL objective; same-budget ordinary data gives a third of the gain and hurts S = 16 |
| 2 cause of the syntax collapse | **identified**: independent placement of a dependent pair — concentrated on the first pair of the answer for E (format mixing from MBPP states), distributed over seams for A and the baseline; not an indentation deficit |
| 3 training improves syntax validity and free generation | **partly**: HumanEval S = 8 8.5 → 19.5 % with code states in the evaluation format (84 % of the teacher ceiling), GSM8K S = 8 70.0 → 83.3 % (arm B); the two-domain mixture trades 7 GSM8K points for the code gain; 30 % / 80 %-on-test targets not reached |
| 4 non-autoregressive capability | **no**: gains are specific to the monotonic left-to-right sampler |

Next structure, from the measurements: keep the backbone and the one-LoRA, eight-forward budget; change *what states
the student is trained on*, because that is the only lever that moved anything: (i) on-policy states of the sampler
that will be used at inference — including a non-monotonic one if global refinement is the goal — with the chain-rule
teacher resolving each step's placements; (ii) the all-mask first forward as a first-class training state (the format
decision of the whole answer is made there; it is where E broke and where B9 measured 6.8 incoherent adjacent tokens
per block), with the teacher resolving the first 2–4 tokens sequentially; (iii) code states only in the evaluation's
prompt format and dialect; (iv) stop at ~10 k states per new state type — more of the same state does not help.  A
new loss is not indicated by any measurement in this phase; a Python-specific masking rule even less.
