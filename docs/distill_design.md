# SHIGERU — DeepSeek-distilled causal diffusion LM: design and plan

Date: 2026-10-20. Status: Phase A pilot running, Phase B script ready, Phase C design from `docs/wedlm_analysis.md`.
Everything marked **measured** was run here; everything else is a plan or an estimate.

## 1. What the existing repository already provides (analysis)

| Component | File(s) | State |
|---|---|---|
| Own masked-diffusion engine for LLaDA-8B (exact vs HF, max diff 0.0) | `engine/llada.py`, `engine/sampler.py`, `engine/graphs.py` | measured: 86 % GSM8K-50 at 157 tok/s effective; CUDA-graph path 145 tok/s effective, 82 % |
| OpenAI-compatible server | `engine/server.py` | works, single worker |
| Block-causal Qwen2.5 port (Fast-dLLM v2 class) | `engine/qwen_block.py` | untested WIP, no sampler |
| Benchmarks (speed, GSM8K-50, HumanEval/MBPP, samples) for ours / LLaDA / Fast-dLLM v1,v2 / LLaDA2.1-mini / Qwen3-8B | `bench/` | Phase 1/2/5 chains, results in `docs/phase1.md` |
| Long-context design (1M) and benchmark plan | `docs/long_context_design.md`, `docs/long_context_bench_plan.md` | design only |
| GPU policy and fleet pause/restore | `docs/gpu_policy.md`, `scripts/shigeru_run.py` | procedure confirmed by the fleet operator, **never exercised yet** |

Negative result recorded today: LLaDA2.1-mini's remote code needs `transformers.masking_utils.create_bidirectional_mask`
(transformers ≥ 4.58), while LLaDA-8B's remote code breaks on transformers 5.x. The benchmark venv stays on 4.57.6; the
LLaDA2.1-mini runs failed with `ImportError` and will need their own venv. Training uses a separate `.venv-train`
(torch 2.6.0+cu124, transformers 5.20, peft 0.21, trl 1.16, datasets 5.1) for the same reason.

## 2. Teacher: verified facts (read-only probes, 2026-10-20 08:39 JST)

* The brief names `http://tsuginosuke:8188`. `tsuginosuke` is **this host**. Port 8188 is the H3 fleet's ComfyUI backend
  (`127.0.0.1:8188`, `/system_stats` 200, `/v1/models` 404). The DeepSeek V4.1 Flash OpenAI-compatible server is
  **port 8000** (`run_128k_4slots.sh` in the dsv41 repo; process started 06:57 today). All teacher traffic goes to
  `http://127.0.0.1:8000`; the fleet backend is never touched.
* `/v1/models`: `deepseek-v4.1-flash`, `deepseek-v4.1-flash-abliterated` (same loaded weights) plus alias ids.
  Served checkpoint (from the process command line): `/mnt/raid6/models/DeepSeek-V4.1-Flash-1017-abliterated`
  (mrfakename's abliteration of `deepseek-ai/DeepSeek-V4.1-Flash-1017`). Every record stores this provenance in
  `teacher_model`. Earlier A/B on 58 items showed identical answers between the official and the abliterated weights
  (dsv41 `scripts/eval_quick.py`), but the dataset is nonetheless labelled as coming from the served checkpoint.
* Auth: none required (a request without a key returns 200). Max context: 131 072 tokens, 5 concurrent slots
  (server flags). Supported body fields: `max_tokens`, `temperature`, `top_p`, `seed`, `stop`, `presence/frequency_penalty`,
  `response_format` (json_object / json_schema), `thinking: true` (reasoning mode; `reasoning_content` in the message).
* License: DeepSeek V4.1 Flash is **MIT** (`LICENSE` and README of the checkpoint). Using its outputs to train another
  model is permitted. WeDLM (Tencent) is Apache-2.0 with a "not intended for use within the EU" clause; Qwen3-4B/8B are
  Apache-2.0 (HF API, 2026-10-20). No model under a research-only license is used.
* Load discipline: the server serves production users and a GPU lender treats a busy server as "in use"
  (lending of GPUs 2,3 to the fleet is suppressed while we generate). Phase A uses concurrency 2 (of 5 slots) and a
  0.5 s minimum interval; the teacher service is never started, stopped, reconfigured or restarted from this project.

## 3. GPUs: brief vs reality

The brief allows physical GPUs 2–7 and forbids 0,1. **Physical GPUs 0–3 all hold the teacher** (devices `2,3,0,1` in
the server's launch, 75–79 GiB each). GPUs 4, 6, 7 are the H3 fleet (ComfyUI, ~41 GiB each) and GPU 5 is shared
(TTS/ASR of another session, hf_proxy, Blender: ~40 GiB used). Topology (`nvidia-smi topo -m`, recorded 2026-10-20):
pairs (0,1) (2,3) (4,5) (6,7) are NV12; 0–3 are NUMA node 0, 4–7 NUMA node 1; across pairs it is PCIe/NODE and across
NUMA nodes SYS. Consequences:

1. `torchrun --nproc_per_node=6` on 2–7 is impossible without stopping the teacher, which is forbidden. The usable set is
   **physical 4,5,6,7** (`CUDA_VISIBLE_DEVICES=4,5,6,7`, logical 0–3), and 4/6/7 only while the fleet is paused.
2. Pilot runs (LoRA on a 4B model) use **GPU 5 alone** so the fleet does not have to be paused until a multi-GPU run is
   actually needed. The first fleet pause will be logged in `results/fleet_downtime.log` and announced to the operator.
3. `distill/gpu_guard.py` refuses to start without `CUDA_VISIBLE_DEVICES`, refuses 0/1, and after model load checks
   from `nvidia-smi` that the process holds no memory on physical 0/1.

This conflict is for the user to resolve (move the teacher off 2,3, or accept 4 GPUs). Until then every plan below is
sized for 4 × A100 80 GB with GPU 5 partially occupied.

## 4. Separating the three effects

The hypothesis "causal diffusion lowers size/cost at equal quality" is tested against controls so that compression,
distillation and diffusion-ization are not confounded. Same data, same evaluation, same decoding budget:

| Row | Model | Size | Training | Decoding | Isolates |
|---|---|---|---|---|---|
| R0 | Qwen3-8B (as released) | 8B | – | AR | reference |
| R1 | Qwen3-4B (as released) | 4B | – | AR | compression alone |
| R2 | Qwen3-4B + SFT on teacher data | 4B | Phase B | AR | distillation (vs R1) |
| R3 | Qwen3-8B + SFT on teacher data | 8B | Phase B | AR | distillation at 8B |
| R4 | R2 → causal diffusion (Phase C) | 4B | C | diffusion, τ=None (1 token/step) | conversion loss (vs R2) |
| R5 | R4 | 4B | C | diffusion, parallel | speed/quality of parallel decoding (vs R4) |
| R6 | R5 + step distillation (Phase D) | 4B | D | few-step | few-step gain |
| R7 | R3 → causal diffusion | 8B | C | diffusion | the key question: R5/R6 vs R0 |

Evaluation set is fixed: GSM8K test (`bench/prompts_gsm8k50.json`, full 1319 later), HumanEval, MBPP-sanitized,
`bench/prompts.json` (EN/JA samples), plus the JA/long-context sets from `docs/long_context_bench_plan.md`. None of
these texts may appear in training data (`distill/curate.py` drops any item sharing a 13-gram with them).

## 5. Phase A — teacher data

Pipeline: `distill/gen_teacher.py <yaml>` → `data/distill/<name>.jsonl` (resumable, rate-limited, seeded) →
`distill/curate.py` → `<name>_{train,val,test}.jsonl`, `<name>_rejected.jsonl`, `<name>_stats.json`.

| Category | Prompt source | Verification | Kept when |
|---|---|---|---|
| math | GSM8K **train** split (MIT; test split is the eval set) | final `#### n` vs gold, numeric | `verified_correct` |
| code | teacher writes problem + reference solution + tests; tests the reference fails are dropped (≥3 must remain); a second independent call solves the problem | solution run against the tests in a subprocess sandbox (`python -I`, clean env, 20 s) | `verified_pass` |
| json | templated request with required keys | parser + required keys | `verified_pass` |
| en_general, ja_general | self-instruct from topic lists × seeds | teacher-as-judge rubric 1–5 (separate call) | score ≥ 4 |
| reasoning | self-instruct puzzle with author's answer | judge with the author's answer as reference | score ≥ 4 |

Record fields: `id, task_category, source, instruction, context, response, teacher_model, generation_parameters
(temperature 0.6, top_p 0.95, max_tokens, thinking=false, seed), verification_status, verification_detail, quality_score,
usage, latency_s, finish_reason, created_at, git_commit` (+ `gold`, `tests`, `entry_point` where applicable).
Dedup: exact normalized instruction + 8-gram Jaccard > 0.7. Split 90/5/5 seeded. Truncated (`finish_reason=length`)
responses are rejected.

**Measured pilot (12-item smoke, 2026-10-20):** 26 teacher calls, 165 s at concurrency 2, 7.5k prompt / 7.7k completion
tokens; math 2/2 correct, code 1/2 (the failure was a wrong expected value written by the spec call → fixed by validating
tests against the reference solution), json 1/2 (truncated at 800 tokens → limit raised to 1500), judged items 5,5,5,4,2,5.
The 230-item pilot (`distill/configs/phaseA_small.yaml`) is running; results go to `results/phaseA_small.log`.

Scaling plan (estimate, not measured): ~25 s per item at concurrency 2 → ~7k items/day without disturbing production;
a 20k-item set is ~3 days of background generation. Thinking-mode traces are a separate option (`thinking: true`), to be
compared against visible-answer distillation on R2 before being used.

## 6. Phase B — AR SFT distillation (`distill/train_sft.py`)

LoRA (r 32, α 64, all linear projections) on Qwen3-4B first, chat template with thinking disabled, completion-only loss,
bf16, gradient checkpointing, `max_len` 2048, cosine schedule, checkpoints every 50 steps with automatic resume.
Memory estimate for the 4B pilot: 8 GiB weights + ~10–14 GiB activations at batch 4 × 2048 with checkpointing ≈ 20–24 GiB
(fits beside GPU 5's other tenants). Full fine-tuning of the 4B model (needed for Phase C: a new mask row and the
attention behaviour change) is ~8 GiB weights + 16 GiB fp32 master/Adam states ×2 ≈ 64 GiB per replica → needs
FSDP over 4 GPUs or 8-bit Adam; decided after the LoRA pilot. Every run writes `runs/<name>/run.json` (config, commit,
GPU config, peak VRAM, time, log history) and the merged bf16 weights for benchmarking with `bench/run_ar.py`.

## 7. Phase C — causal diffusion conversion (WeDLM-centric; details in `docs/wedlm_analysis.md`)

Mechanism to implement first, in this order, each step validated before the next:
1. **Data representation**: `[x0 clean | xt]`, xt masked per 32-token block with γ ~ U(0,1], observed-first reordering,
   both halves sharing the original position ids (port of `finetune/src/batch.py`, `masking.py`).
2. **Mask**: x0 causal; xt block k attends to x0 blocks < k and to xt rows of the same block at smaller physical index.
   On A100 this is a FlexAttention `mask_mod` or a dense SDPA mask at ≤ 4K; MagiAttention (Hopper) is not needed.
3. **Loss**: 1/γ-weighted MLM cross-entropy on masked xt rows + next-token AR loss on x0 rows (same forward).
4. **Mask token**: a new embedding row (Qwen3 vocab has free ids above 151668); full fine-tune, not LoRA.
5. **Decoder** in our engine: sliding window W=16 over a prefix-reusable KV cache; each step = causal prefill of the window
   (filled first, masks last) on top of the cache; commit the leftmost filled run; fill masks where
   entropy + λ·distance < τ; refill. Prefix KV is reused unchanged; uncommitted window tokens are re-encoded each step.
6. **Exactness gate**: with τ=None (one token per step) the converted model must reproduce the AR student's greedy
   output on the eval prompts before parallel decoding is evaluated (row R4 vs R2).
Only then A/B the Fast-dLLM v2 style bidirectional-within-block variant that `engine/qwen_block.py` implements.
WeDLM reports 100B CPT tokens on 7B/8B; our budget is 1–2B tokens on the 4B model (estimate: ~1 day on 4 × A100 at
~10k tok/s), so results will be reported as "partial conversion" unless they reach the AR student's quality.

## 8. Phases D–H (plan)

D: step distillation (consistency/progressive) from the converted model at 32/16/8/4/2 steps per 16/32/64/128 block;
E: engine (Triton/FlashAttention-compatible causal kernels, CUDA graphs, paged prefix KV, continuous batching, TP over 2–4
GPUs); F: the table of §4 with measured tok/s, TTFT, VRAM and quality; G: 32K → 1M per `docs/long_context_design.md`;
H: `engine/server.py` extended to the new model family. Success criteria from the brief: 500 tok/s intermediate,
1000 tok/s stretch (single A100, batch 1, measured); negative results are kept in `docs/`.

## 9. Automation conventions

YAML configs under `distill/configs/`; every run records seed, git commit, `CUDA_VISIBLE_DEVICES`, physical GPUs used
and peak VRAM; JSONL logs and `run.json` per run; CSV/Markdown tables via `bench/summarize.py`; long jobs are launched
detached (`setsid`) with logs under `results/`; checkpoints resume automatically. Fleet pause/restore only through
`scripts/shigeru_run.py --pause-fleet` (operator-confirmed procedure), only for runs that need GPUs 4/6/7, with the
downtime logged. No `killall`, no indiscriminate `pkill`, no GPU resets, nothing on physical GPUs 0/1, no changes to
the teacher service.
