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

**Decision (relayed 2026-10-20 by the donkeyhover-35 session from the user, latest wording "4,6,7 でやる"): experiments
run on physical GPUs 4, 6, 7; GPU 5 is left to its other tenants; the teacher stays on 0–3 and is not restarted.** So every plan below is sized for 4 × A100 80 GB
3 × A100 80 GB (`CUDA_VISIBLE_DEVICES=4,6,7`, `torchrun --nproc_per_node=3`); the TTS/ASR and hf_proxy services on GPU 5
are not touched. `distill/run_467.sh` is the launcher (pause → Phase B pilot → evaluation → clean speed baseline → restore). Using 4/6/7 requires the H3 fleet to be stopped. **The user decided (directly, 2026-10-20) to stop the fleet
themselves ("停止は自分でやるので待て")**: this project does not pause or restore the fleet; `distill/run_467.sh`
refuses to start while GPUs 4/6/7 hold memory and waits for the user's go-ahead. `scripts/shigeru_run.py --pause-fleet`
stays available but unused.

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

### 7.1 The conversion trainer (`distill/train_cdlm.py`, steps 1–4 wired into a training loop)

Full fine-tune (not LoRA: the new mask row and the attention-behaviour change need every weight), plain torch loop, no HF
Trainer, same conventions as `train_sft.py` (YAML config, GPU guard, `run.json` with git commit / `CUDA_VISIBLE_DEVICES` /
physical GPUs / peak VRAM / full log history, automatic resume from the latest `checkpoint-*` under `out_dir`).

```
# single GPU
CUDA_VISIBLE_DEVICES=5 .venv-train/bin/python -m distill.train_cdlm distill/configs/phaseC_pilot_4b.yaml
# DDP, one process per GPU (NCCL; optimizer states replicated per rank — no ZeRO)
CUDA_VISIBLE_DEVICES=4,5,6,7 .venv-train/bin/torchrun --nproc_per_node=4 -m distill.train_cdlm distill/configs/phaseC_pilot_4b.yaml
# CPU smoke test (tiny random Qwen3, kill/resume, 2-process gloo DDP), ~25 s
CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -m distill.tests.test_train_cdlm
```

* **Data**: JSONL `instruction`/`context`/`response` → prompt via the chat template with `enable_thinking=False` (as Phase
  B) + response + eos, truncated to `max_len` *logical* tokens (prompt kept whole; prompt-only examples dropped and
  counted); `engine.cdlm.data.build_example` per sample with a fresh γ draw every epoch (prompt tokens never masked,
  `block` 32, γ ~ U(`gamma.min`, `gamma.max`]); `collate` right-pads with the dense `[N,1,T,T]` mask. The dual stream
  doubles the sequence: `max_len` 1024 → 2048 physical rows.
* **Model**: `Qwen3ForCausalLM` bf16, `attn_implementation: sdpa` with the dict 4-D mask of `model_glue`, gradient
  checkpointing (non-reentrant), `add_mask_token` (`<|mask|>` = 151669 for Qwen3: a free padded row, no resize; the
  tokenizer gets the token as an additional special token + `mask_token`, and `final/` saves both plus
  `config.cdlm_mask_token_id` / `cdlm_block`).
* **Loss**: `total = (mlm + ar_weight·ar) / (1 + ar_weight)`, `ar_weight` 1.0 (WeDLM); `loss_mlm` and `loss_ar` are logged
  separately every `logging_steps` to `out_dir/log.jsonl` with lr, grad norm, tokens/s (logical), rows/s (physical),
  masked-row count, micro-batch size and peak VRAM.
* **Optimisation**: AdamW (fused on CUDA; weight decay off for norms/biases), cosine with warmup to `min_lr_ratio`,
  `grad_accum` micro-batches per step (step loss = mean of micro-batch losses, as accelerate), `grad_clip`.
  `master_fp32: false` (default): bf16 weights and bf16 Adam states — cheapest, but bf16 has ~3 significant digits so an
  update of relative size < 1e-3 (lr 1e-5 on O(1e-2) weights) is largely rounded away; if the loss stalls, `master_fp32:
  true` keeps fp32 weights + Adam states and runs the forward under bf16 autocast (`final/` is still written in bf16).
* **Checkpoints**: `checkpoint-<step>/state.pt` = model + optimizer + scheduler + step + micro-batch size + RNG + OOM log,
  every `save_steps`, keeping `save_total_limit` (each one ≈ 3× the bf16 weights); resume re-seeds the sampler with
  `seed + epoch` and skips the micro-batches already consumed, so the data order continues exactly.
* **OOM policy**: a CUDA OOM inside a micro-batch drops that optimizer step's partial gradients, logs the event, advances
  the step counter (schedule and resume stay consistent) and halves the micro-batch size for the rest of the run (loader
  batches are forwarded in chunks, so the effective batch is unchanged). Under DDP an OOM is fatal (the other ranks would
  hang in the all-reduce).
* **Memory, Qwen3-4B, `max_len` 1024 (2048 rows), batch 2** (4.02 B params, 36 layers, hidden 2560, vocab 151 936):
  bf16 weights 8 GiB + bf16 grads 8 GiB + bf16 Adam m/v 16 GiB = 32 GiB of state (`master_fp32`: 16 + 16 + 32 = 64 GiB).
  Activations with checkpointing: 36 boundary tensors × [2, 2048, 2560] bf16 ≈ 0.75 GiB, one layer's recompute ≤ 1.5 GiB
  (SDPA with an explicit mask may fall back to the math kernel: [2, 32, 2048, 2048] scores ≈ 0.5 GiB bf16 + fp32 softmax),
  gathered logits only for the scored rows (≈ 1.5 × response tokens per example ≈ 2.7 k rows × 151 936 × ~10 B for
  bf16 logits + fp32 CE + grad) ≈ 4 GiB. Total ≈ 40 GiB in bf16 (fits GPU 5 beside its tenants) or ≈ 72 GiB with fp32
  master (needs an otherwise empty A100, or DDP does not help — optimizer states are replicated; ZeRO/FSDP or 8-bit Adam
  would). Throughput is unmeasured (CPU-only so far).
* **Config** `distill/configs/phaseC_pilot_4b.yaml` (base `/mnt/raid6/models/hf/Qwen3-4B` or the Phase B merged dir,
  `data/distill/phaseA_small_train.jsonl`, block 32, `max_len` 1024, batch 2 × accum 8, lr 1e-5, 5 % warmup, 2 epochs,
  `runs/phaseC_pilot_4b`) — written, not launched.
* **Test** `distill/tests/test_train_cdlm.py`: tiny random Qwen3 (hidden 32, 2 layers, char-level tokenizer with a
  Qwen-like chat template) saved to a temp dir, 6-example JSONL; (1) 4 steps with a checkpoint every 2, hard exit right
  after `checkpoint-2` (`--exit-after-step`), restart → the log continues at step 3, `run.json` records the resume,
  `final/` loads with `AutoModelForCausalLM`, mask id in `config.cdlm_mask_token_id` and in the tokenizer, weights
  changed and equal to the last checkpoint; (2) epoch tail steps and `master_fp32`; (3) the OOM branch in-process;
  (4) `torchrun --nproc_per_node=2` on CPU with gloo.
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

## 10. Measured so far (2026-10-20)

| Item | Result | Conditions |
|---|---|---|
| R1 Qwen3-4B released, GSM8K-50 greedy | **90 %** (45/50) | `bench/run_ar.py`, 384 new tokens, shared GPU 5 |
| R0 Qwen3-8B released, GSM8K-50 greedy | 84 % | same harness, earlier Phase 1 run |
| R0 Qwen3-8B HumanEval pass@1 | 85.4 % | `bench/run_code_eval.py` |
| Qwen3-4B AR speed | 30 tok/s | **not a valid speed measurement**: GPU 5 was at 99 % utilization from other tenants; to be redone on an idle GPU |
| Phase B LoRA dry run, Qwen3-4B | 3 steps, 9.2 GiB peak, merged weights load in the bench venv | GPU 5 |
| Phase C trainer GPU smoke, Qwen3-0.6B | 4 steps, 8.6 GiB peak, 1.3–2.2k logical tok/s (contended GPU); MLM loss 20.0 → 11.0, AR loss 2.7 → 1.6 | GPU 5, batch 2 × accum 2, max_len 1024 (2048 rows) |
| Phase A pilot (230 items, 50 min, 466 calls, 0 errors) | 205 kept after curation (185/10/10); math 59/60 verified, code 34/40 pass, reasoning 17/30 judged ≥4 | teacher at concurrency 2, 13 s/item mean |
| **Phase B pilot, Qwen3-4B LoRA** (185 items, 2 epochs, 24 steps, 124 s, 12.3 GiB) | train loss 1.40, eval loss 1.51; **GSM8K-50 92 %**, HumanEval **76.2 %** (released 4B: 90 % / 81.7 %); 52.6 tok/s | GPU 7, idle, config repaired |
| **Phase B pilot, Qwen3-8B LoRA** (155 s, 19.3 GiB) | train loss 1.25, eval loss 1.35; **GSM8K-50 94 %**, HumanEval **83.5 %** (released 8B: 84 % / 85.4 %); 50.4 tok/s | GPU 7, idle, config repaired |
| Released Qwen3-4B / Qwen3-8B, idle GPU, HF greedy batch 1 | **53.5 / 50.3 tok/s**, GSM8K-50 90 % / 84 %, HumanEval 81.7 % / 85.4 % | GPUs 4, 7 |
| **Phase C shakedown**: full fine-tune of the Phase B 4B student into the dual-stream causal-diffusion form (185 items × 4 epochs = 46 steps, 6 s/step, 35.3 GiB, 1.25k logical tok/s) | MLM loss 20 → 3.3 (final step), AR loss ≈ 0.8–1.2 | GPU 7 |
| Phase C shakedown decoded with `engine/cdlm/decoder.py` (10 GSM8K items, window 16) | AR rows: 80 %, 50.6 tok/s · τ=None (one mask/step): 60 %, 43.5 tok/s, **0/10 identical to AR** · τ=0.4: 60 %, 1.00 tok/step · τ=1.0: 40 %, 1.04 tok/step; several runs never emit EOS (384-token cap) | `bench/run_cdlm.py`, GPU 7 |

Reading of the Phase C shakedown (negative result, as expected for ~0.6 M training tokens against WeDLM's 100 B): the AR
rows keep the student's quality, but the mask predictor is far from the AR distribution (exactness gate fails 10/10), its
confidence never clears the τ thresholds, so no parallelism is obtained (1.0 token per step) and stop tokens are
unreliable. The conversion recipe is mechanically working end to end (data → mask → loss → decoder with prefix KV reuse);
what is missing is training tokens. The next conversion run uses the medium set (~2 300 items, ~1 M response tokens)
and, if still insufficient, a larger teacher set; the per-step cost (6 s for 16 examples of 2 048 rows on one A100)
puts 1 B tokens at ~5 GPU-days, so a 3-GPU DDP run is the practical path.

Speed observation that shapes the project: at batch 1 with HF decoding, 4B and 8B run at the same ~50 tok/s (kernel-launch
bound), so halving the model does not buy speed by itself. The speed target (500–1000 tok/s) has to come from parallel
decoding plus the engine (CUDA graphs, fused kernels), not from size.

The 50-item GSM8K subset cannot separate 4B from 8B (90 % vs 84 % is within noise at n = 50); Phase F uses the full
1319-item test set. `engine/cdlm/` (data, mask, loss, HF glue, streaming decoder, trainer) is CPU-tested: 14 tests pass
(`engine.cdlm.tests.test_cdlm` 6, `engine.cdlm.tests.test_decoder` 4, `distill.tests.test_train_cdlm` 4).

**Correction (10:05):** the Phase B pilot rows above that were evaluated in the benchmark venv (GSM8K-50 90 %/90 %,
50 tok/s, HumanEval 24 % for the 4B pilot) were run with the **wrong RoPE base**: transformers 5 writes
`rope_parameters`, transformers 4.57 ignores it and falls back to θ = 10 000 instead of 1 000 000. Symptom: doubled
closing brackets in code; the same checkpoint generates correctly under transformers 5. `distill/fix_config.py` repairs the
configs (now applied automatically by both trainers); the affected result files are quarantined in
`results/invalid_rope/` and the pilots are being re-evaluated (`bench/phaseF_pilots_redo.sh`). The `bench/run_cdlm.py`
numbers were produced in the training venv and are unaffected.

### 10.1 Phase F, AR rows on the full GSM8K test set (1 319 items, greedy, thinking off, 384 new tokens, idle A100)

| Row | Model | GSM8K (1 319) | HumanEval pass@1 | tok/s (HF greedy, batch 1) |
|---|---|---|---|---|
| R1 | Qwen3-4B released | 86.1 % | 81.7 % | 53 |
| R0 | Qwen3-8B released | 86.4 % | 85.4 % | 50 |
| R2 | Qwen3-4B + LoRA SFT on 185 teacher items | **89.7 %** | 76.2 % | 53 |
| R3 | Qwen3-8B + LoRA SFT on 185 teacher items | **91.7 %** | 83.5 % | 51 |

Distillation moves GSM8K by +3.6 / +5.3 points with only 185 verified items (59 of them GSM8K-train math), and the
distilled 4B already exceeds the released 8B on GSM8K. HumanEval drops slightly (−5.5 / −1.9); the pilot set has
34 code items, so this is re-checked with the medium set. The 50-item subset's "4B 90 % vs 8B 84 %" was noise: on the
full set both released models are at 86 %.

### 10.2 Phase E timing on the shakedown checkpoint (idle GPU 7, bf16, 10 GSM8K prompts, window 16)

| Decoder | τ=None tok/s | τ=0.4 tok/s | tokens/step |
|---|---|---|---|
| HF greedy generate (AR rows) | 53.2 | – | 1.0 |
| HF DynamicCache decoder (`engine/cdlm/decoder.py`) | 44.8 | 45.5 | 0.99 |
| own engine, plain (`fast_decoder.py`) | **49.1** | 48.9 | 0.99 |
| own engine, static shapes | 33.6 | 33.5 | 0.99 |
| own engine, CUDA graphs | 38.7 | 38.7 | 0.99 |

Reading: with the model still decoding one token per step, the own engine is only ~9 % faster than the HF-cache decoder
and ~8 % slower than plain AR generate (the window re-encodes 16 rows per step). The static/graph paths are **slower**
because the static step attends over the whole 4 096-slot cache every step; they need cache-length buckets (as in
`engine/graphs.py`) before graphs can pay off. The engine's real lever is tokens/step, which is 1.0 until the
conversion training (next run) produces a confident mask predictor. Also to check: τ=0.4 accuracy differs between the HF
decoder (0.6) and the own engine (0.3) on these 10 prompts — bf16 differences flip threshold decisions; a bf16 GPU
token-identity test is pending (CPU float32 identity holds).

### 10.3 Phase C stage 1 (continued pretraining conversion) — measured 2026-10-20 17:16–17:40

Run: `distill/configs/phaseC_cpt_4b_3gpu.yaml`, Phase B 4B student → dual-stream causal-diffusion objective on 60 M
logical tokens of FineWeb-Edu + FineWeb-2 JA (58 594 sequences of 1 024, 2 442 steps, 3 × A100 DDP, 3 h 51 min,
4.6 k logical tok/s aggregate, 43.5 GiB per rank, no OOM). MLM loss 12.4 → 3.7–3.9 (plateau from step ~500), AR loss
2.3–2.6 throughout (raw text).

Diffusion decoding of the result (`bench/run_cdlm.py`, HF-cache decoder, 20 GSM8K prompts, window 16, idle GPU 4):

| Decoding | accuracy (20) | tokens/step | tok/s |
|---|---|---|---|
| AR rows (HF greedy) | 90 % | 1.00 | 54.2 |
| τ=None (one mask per step) | **80 %** (shakedown: 60 %) | 0.99 | 47.0 |
| τ=0.4, λ=0.02 | 10 % | 1.14 | 54.4 |
| τ=0.7, λ=0.02 | 0 % | 1.23 | 60.4 |
| τ=1.0, λ=0.02 | 0 % | 1.34 | 64.1 |
| τ=0.4, λ=0.1 | 20 % | 1.12 | 47.4 |
| τ=0.4, λ=0.3 | 55 % | 1.07 | 49.9 |

Reading: 60 M tokens of CPT bring the mask predictor close to the AR rows at one token per step (80 % vs 90 %), but
parallel filling still breaks answers: the typical failure is a multi-token number filled as independent masks
("#### 10" for 100), i.e. the conditional-independence problem that WeDLM's distance penalty and far more training
address. A larger λ trades the parallelism back for accuracy. The exactness gate (τ=None identical to AR) is still
0/20. Next: stage 2 on the medium teacher set (`phaseC_sft_4b.yaml`, queued on GPU 7), then more CPT tokens if the
budget allows; decoding-side: confidence rules that forbid filling adjacent masks in one step unless both clear a
stricter threshold.

### 10.4 Idle-GPU re-measurement of the LLaDA-8B engines (`bench/phase2_idle.sh`, GPU 7, 256 tokens, 10 prompts)

| Engine / configuration | effective tok/s | raw tok/s |
|---|---|---|
| LLaDA official sampler, 256 steps | 19.9 | 24.1 |
| our engine, fixed 256 steps, no cache (parity) | 19.4 | 24.3 |
| Fast-dLLM v1 dual cache, thr 0.9 | 47.9 | 69.2 |
| Fast-dLLM v2 (Qwen2.5-7B block diffusion), thr 0.9 | 79.2 | – |
| **our engine, thr 0.9 + prefix/block cache** | **74.8** | **106.7** |
| our engine, thr 0.9 + cache + CUDA graphs | 59.2 | 84.5 |
| our engine, thr 0.9 + cache + CUDA graphs + EOS stop | 60.2 | 86.9 |
| Qwen3-8B HF greedy (AR) | 52.3 | – |

Correction to the morning's log: the CUDA-graph path is **slower** than the plain cached path on an idle GPU (it
attends over the full 4 096-slot cache every step, like the cdlm static path); the earlier "171 raw / 145 effective"
graph figure does not reproduce and is withdrawn. Accuracy of the graph+EOS configuration is unaffected (GSM8K-50 88 %).

**Fill-selection rules on the CPT checkpoint (same 20 prompts, HF decoder, `engine/cdlm` options added 18:10):**

| Rule | accuracy | tokens/step |
|---|---|---|
| τ=0.4 / 0.7, `no_adjacent` | 25 % / 30 % | 1.01 / 1.04 |
| τ=0.4 / 0.7, `fill_mode=leftmost_run` | 30 % / 20 % | 1.18 / 1.29 |
| margin ≥ 0.5 (no entropy rule) | 10 % | 1.69 |
| margin ≥ 0.5 + `no_adjacent` | 35 % | 1.21 |
| τ=None (frontier mask only, reference) | 80 % | 0.99 |

Reading: even at 1.01 tokens/step (`no_adjacent`, i.e. ~1 % of fills ahead of the frontier) accuracy collapses from 80 % to
25 %, so adjacency is not the main cause: **any token filled ahead of the frontier is unreliable at this training
budget and is never revised**, and one wrong committed token derails the answer. Parallelism therefore has to come with
verification: use the ahead-of-frontier fills as *drafts* and accept them only where the AR rows of the same forward
agree (AR-exact speculative decoding with the diffusion head as the drafter; the same mechanism WeDLM's analysis §11
lists as a free proposal). This keeps the AR student's quality by construction and turns the mask predictor's
accuracy into speed. Implementation queued in both decoders with CPU exactness tests; measured next on the CPT
checkpoint and on the stage-2 checkpoint.

**Verified (speculative) decoding on the CPT checkpoint (18:45, same 20 prompts, HF decoder, GPU 4):**

| Decoding | accuracy | tokens/step | draft acceptance | tok/s |
|---|---|---|---|---|
| AR greedy (reference) | 90 % | 1.00 | – | 50.9 |
| verify, τ=0.4, W=16 | **90 % (identical outputs)** | 1.19 | 74 % | 50.3 |
| verify, τ=1.0, W=16 | 90 % | 1.27 | 43 % | 55.4 |
| verify, τ=1.0, W=32 | 90 % | 1.27 | 41 % | 56.3 |
| verify, τ=1.0, W=16, **own engine** (`--engine fast`) | 90 % | 1.27 | 39 % | **60.0** |

The guarantee holds on the real checkpoint (outputs identical to AR greedy, so quality is the AR student's by
construction), and the diffusion head now contributes speed instead of errors: 1.27 tokens per step. End-to-end that is
only ~1.1× because the HF-cache decoder re-encodes W+1 rows per step; the own engine and a better drafter (stage 2,
CPT continuation) are the two levers, and the ceiling is (W+1)/2 tokens per step with a perfect drafter.

### 10.5 Phase C stage 2 (teacher-data fine-tune of the CPT model) — measured 2026-10-20 19:43–20:00

Run: `distill/configs/phaseC_sft_4b.yaml`: `runs/phaseC_cpt_4b/final` → 1 714 medium teacher items × 3 epochs
(321 steps, 35 min, 35.5 GiB, GPU 7; first attempt died when the root filesystem filled up mid-checkpoint — see the
disk guard in `train_cdlm.py`). MLM loss 2.7, AR loss 1.2 on teacher data.

| Decoding (20 GSM8K prompts, W=16, idle GPU 7) | accuracy | tokens/step | acceptance | tok/s |
|---|---|---|---|---|
| AR greedy (reference) | 100 % | 1.00 | – | 50.6 |
| τ=None (frontier only) | 80 % | 0.99 | – | 43.7 |
| unverified τ=0.4 / 0.7 / 1.0 | 75 % / 60 % / 45 % | 1.29 / 1.42 / 1.59 | – | 55.8 / 60.9 / 69.6 |
| **verified** τ=1.0, HF-cache decoder | 95 % | 1.46 | 40 % | 63.0 |
| **verified** τ=0.7, own engine | 95 % | 1.42 | 58 % | 65.5 |
| **verified** τ=1.0, own engine | **95 %** | **1.50** | 46 % | **68.4** |

Stage 2 is the biggest single gain so far: unverified parallel decoding went from 10 % (CPT only) to 75 % at
τ=0.4, and verified decoding from 1.27 to 1.50 tokens/step, 68.4 tok/s vs 50.6 for AR greedy (**1.35×**, 4B model,
batch 1, one A100). On the "identical to AR" guarantee in bf16: 12/20 outputs are byte-identical; the 8 others diverge at
near-tied tokens (e.g. "Determine apple pie sales" vs "Determine the number of boxes"), i.e. bf16 argmax flips between
the window forward and HF's single-row forward, not drafting errors (the CPU float32 identity tests pass). The honest
statement is "AR-equivalent up to bf16 ties".

Next levers, in order of expected gain: (1) more conversion data — CPT continuation (60 M tokens, queued on GPUs 4/6)
and a larger teacher set; (2) engine: the own engine already gives +9 % over the HF-cache path and has cache-length
buckets and CUDA graphs still to do; (3) window/threshold tuning per task; (4) 8B conversion for the R7 row.

### 10.6 Phase B on the medium teacher set (1 714 train items, 2 epochs, LoRA r 32)

| Row | Model | GSM8K (1 319) | HumanEval | note |
|---|---|---|---|---|
| R2 (pilot) | Qwen3-4B + 185 items | 89.7 % | 76.2 % | |
| R2 (medium) | Qwen3-4B + 1 714 items | 88.2 % | 72.0 % | GSM8K-50: 92 % |
| R1 | Qwen3-4B released | 86.1 % | 81.7 % | |
| R3 (pilot) | Qwen3-8B + 185 items | 91.7 % | 83.5 % | |
| R3 (medium) | Qwen3-8B + 1 714 items | 91.1 % | 78.7 % | GSM8K-50: 94 % |
| R0 | Qwen3-8B released | 86.4 % | 85.4 % | |

Negative result (both sizes): 9× more teacher data (same recipe, lr 1e-4, 2 epochs) did not improve the 4B student over the 185-item
pilot and cost 4 more HumanEval points. Candidate causes, to be tested with the 8B numbers in hand: the LoRA recipe
over-fits style with more epochs' worth of steps (216 steps vs 24), the teacher's code answers (full programs from prose
problems) are a format mismatch with HumanEval (completion of a signature + docstring), and category balance
(math 678 of 1 904). Ablation 1 (lr 5e-5, 1 epoch, 108 steps): GSM8K-50 92 %, HumanEval **71.3 %** — the gentler recipe does not recover
code, so the cause is the data, not the number of steps. Ablation 2 (queued on GPU 7): same recipe without the 325 code
items, to test whether the teacher's code answers themselves cause the regression.

### 10.7 Headline row R5 on the full GSM8K test set (1 319 items, stage-2 4B model, own engine, idle GPU 7, bf16)

| Decoding | GSM8K (1 319) | tokens/step | draft acceptance | tok/s |
|---|---|---|---|---|
| AR greedy (same weights, HF generate) | 90.4 % | 1.00 | – | 52.1 |
| **verified diffusion decoding, τ=1.0, W=16** | **90.1 %** | **1.41** | 42 % | **66.2** |

Reference rows on the same set: released Qwen3-4B 86.1 %, released Qwen3-8B 86.4 % (both at ~50 tok/s), Phase B
LoRA students 89.7 % (4B) / 91.7 % (8B). So **Shigeru-4B-Diffusion at 90.1 % and 66 tok/s beats the released Qwen3-8B
(86.4 %, 50 tok/s) on both axes**, with the quality coming from distillation and the speed (1.27×) from verified
parallel decoding. The 0.3-point gap to its own AR decoding is bf16 tie-breaking (identical on CPU float32), not
drafting error. The speed target of the brief (500 tok/s) is far away: tokens/step is the lever (ceiling (W+1)/2 = 8.5
with a perfect drafter) and the engine still lacks CUDA graphs with cache-length buckets.
