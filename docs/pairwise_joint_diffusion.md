# Phase B8 — Pairwise Joint Denoising & Trajectory Distillation (K = 16, 8 forwards per block)

Brief: "SHIGERU Phase B8 — Pairwise Joint Denoising & Trajectory Distillation" (2026-10-25).  Hypothesis under test:
the S = 8 gap (72.9 % GSM8K test vs 84.8 % at S = 16, left-to-right, same adapter) comes from placing two tokens per
forward by *independent* argmax, i.e. from ignoring the dependency between the two positions.  Constraints kept: no
AR verification, no draft model, 8 forwards per block for the practical rules (extra forwards are counted and
reported wherever a research ceiling uses them), frozen 4B, no checkpoint modified, nothing from B2/B3/B8/B7 repeated.

## 1. Audit — what the earlier phases already say about this hypothesis

- At S = 8, left-to-right order places positions (i, i+1) together from one forward; confidence order places the two
  most confident masked positions, which are adjacent 35–45 % of the time.  Both reach 70.0 % on the fresh dev2 set
  and 72.9 / 63.4 % on test (trust_aware_diffusion.md §3.3, §7).
- Spending one forward per token (S = 16, left-to-right) — i.e. the exact chain-rule joint of every pair — gives
  90.0 % dev2 / 84.8 % test.  The gap S = 8 → S = 16 is therefore an upper bound on what *any* pairwise joint
  decision can recover within the 8-forward budget: the joint of (i, i+1) with one extra forward per pair IS S = 16.
- The A-series dependency diagnostic (dependency_aware_refinement.md): within a block, dependency is distance-1 and
  left-sided; the B-series §4 experiment (trust_aware_diffusion.md) found that forbidding adjacent pairs does not help
  the B-series, i.e. adjacent pairs are not obviously the ones that fail.
- Re-masking cannot repair what independent placement breaks (oracle 30 % repair, trust_aware_diffusion.md §3.2),
  so if the pair decision is the problem it has to be fixed at placement time — which is what this phase tests.
- Data: dev items 0–49 (e5 base's own greedy answers as reference, results/pure/dev_refs_e5.json) for the
  teacher-forced diagnostics; bench/prompts_gsm8k_dev2.json (150 fresh items) for decisions; GSM8K test for the
  final setting only.

## 2. Pairwise dependency and joint re-ranking without training (brief §2, §9)

Code: engine/bdiff/pairwise.py (`rerank_pair`), bench/diag_pairwise.py, sampler option `pair=` in
engine/bdiff/sampler2.py, `--pair/--pair-k` in bench/eval_trust.py.
Rules compared at every S = 8 step that places two tokens (i, j), candidates = top-k per position (k = 2 → 4 pairs,
k = 4 → 16 pairs):

| rule | score of pair (a, b) | extra forwards per step | status |
|---|---|---|---|
| A independent argmax | log p(a) + log p(b) (product of marginals — identical to independent argmax, verified: "indep_check" row) | 0 | existing sampler |
| B/C joint re-ranking, chain rule forward | log p(a) + log p(b \| a placed at i) | k | research ceiling |
| B/C joint re-ranking, symmetric | ½[log p(a) + log p(b \| a) + log p(b) + log p(a \| b)] | 2k | research ceiling |
| oracle | the reference pair when it is among the k × k candidates | – | ceiling of any re-ranker |
| D/E pairwise compatibility score / dependency-aware pair selection without extra forwards | learned head on hidden states (§3) | 0 | only if Gate 1 passes |
