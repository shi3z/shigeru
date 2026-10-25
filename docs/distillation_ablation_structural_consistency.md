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
