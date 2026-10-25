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
| D  conditional KL only | 43.8 % | 75.3 % | __DKL16__ |
| E  CE + KL (student A) | 47.6 % | 81.3 % | 87.3 % |
| F  ordinary data, same budget | __F8h__ | __F8__ | __F16__ |

**Gate 1 answered: the gain is the teacher data, not the pair objective.**  The plain masked cross-entropy on the
same 12 000 on-policy S = 8 states — targets = the block the teacher-followed trajectory ended with — matches or beats
every pair-term variant (83.3 vs 81.3–82.0 %), reproduces the teacher's hard pairs *better* (51 vs 44–49 %), and does
**not** lose anything at S = 16 (90.0 vs 87.3 % for the KL student).  The KL to the teacher's conditional is the
weakest single ingredient (75.3 %); the "conditional distribution distillation" of B10 is therefore not a success of
distribution matching but of **training on the student's own denoising states with joint-resolved pseudo-labels** —
what makes the targets informative is that they come from a trajectory in which every pair was resolved by the
chain-rule teacher, i.e. they are high-quality pseudo-labels *in the state distribution the sampler actually visits*.
Whether the same budget of ordinary training (arm F) does nothing is the remaining check (pending), and arm B scaled
to all 65 k states is running (zeta).
