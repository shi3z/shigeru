# Open Mercury — Phase-0 research: five diffusion-LLM repositories and the Mercury paper

Date: 2026-10-20. Read-only survey of the shallow clones under
`/mnt/ssdraid/project/open-mercury/third_party/` (nothing installed, no model run, repos untouched).
Line numbers refer to the checked-out revisions listed in §0. Anything I could not confirm from the
repo, the paper text, or a Hugging Face file fetched today is marked **[unverified]**.

---

## 0. Revisions, licenses, and the local machine

| Repo | Remote | HEAD | License (file) |
|---|---|---|---|
| LLaDA2.X | github.com/inclusionAI/LLaDA2.X | `a798329` "Update README for LLaDA2.2 release" | Apache-2.0 |
| dInfer | github.com/inclusionAI/dInfer | `1ffeb96` (v0.2 line, `__version__ = "0.1"` in `python/dinfer/__init__.py:12`) | Apache-2.0 |
| Fast-dLLM | github.com/NVlabs/Fast-dLLM | `a9b81e4` "Update Fast-dDrive HF repo path" | Apache-2.0 (NVIDIA headers, e.g. `v1/llada/generate.py:1-16`) |
| DiffusionLLM | github.com/lamm-mit/DiffusionLLM | `f66371d` "Add paired graph-canvas diffusion training" | Apache-2.0 (+ `NOTICE` crediting ZHZisZZ/dllm) |
| dllm | github.com/ZHZisZZ/dllm | `ca17675` "Refactor GRPO pipeline…" | Apache-2.0 |

Local environment snapshot (`nvidia-smi`, `python3 -c import …`):

* 8× NVIDIA A100 80GB PCIe. At survey time GPUs 0–3 had ~75–79 GB in use, GPUs 4/6/7 ~41 GB, GPU 5 ~19 GB (i.e. only GPU 5 had >60 GB free). Pick the GPU at run time; do not assume GPU 0.
* `nvcc` on PATH is CUDA 11.8; `/usr/local/cuda-12.{1,3,4,8,9}` also exist (set `CUDA_HOME=/usr/local/cuda-12.4` for anything that compiles).
* System Python 3.10.12: torch 2.6.0+cu124, triton 3.2.0, flash-attn 2.8.3.post1, transformers **5.5.0.dev0**, accelerate 1.6.0, lm_eval 0.4.8, vllm **0.8.3**, **no sglang**.
* Consequence: every repo below pins a *different* transformers (4.49 / 4.53.1 / 4.57 / ≥5.13) and dInfer wants vllm 0.10.2 + sglang 0.5.3.post1. Each must get its own venv; none can be `pip install -e .`'d into the system Python without breaking the DeepSeek runtime that lives there.

---

## 1. LLaDA2.X (inclusionAI) — model-release repo, no code

**Purpose.** Release hub for Ant Group's LLaDA2.0 / 2.1 / 2.2 Mixture-of-Experts diffusion LMs (16B "mini" and 100B "flash"), converted from the autoregressive Ling-2.0 models by a Warmup–Stable–Decay block-size schedule and then post-trained (SFT/DPO, later RL and Levenshtein editing). The repo contains only `README.md`, the three tech-report PDFs, and two figures; all inference code lives in dInfer/SGLang and the HF remote-code `modeling_llada2_moe.py`.

**License.** Apache-2.0 (`LICENSE`, README "License"). HF model cards for `inclusionAI/LLaDA2.0-mini` also say `license: apache-2.0`; no additional model-usage restriction found. (Upstream Ling-2.0 base license not checked **[unverified]**.)

**Checkpoints** (README table + HF `config.json` fetched today):

| HF id | Total / active params | Arch | bf16 size | 1×A100 80GB? |
|---|---|---|---|---|
| `inclusionAI/LLaDA2.0-mini` | 16B / **1.4B active** (model card) | MoE, 20 layers, d=2048, 256 experts top-8, vocab 157,184, 32k ctx | 30.3 GiB (7 shards) | **Yes** (weights 30 GiB) |
| `inclusionAI/LLaDA2.1-mini` | same shape | same (`config.json` identical to 2.0-mini) | 30.3 GiB | Yes |
| `inclusionAI/LLaDA2.0-mini-CAP` | same, "Confidence-Aware Parallel" fine-tune | same | (exists, HTTP 200) | Yes |
| `inclusionAI/LLaDA2.0-flash`, `2.1-flash`, `2.2-flash` | 100B / active count not stated in the files I read **[unverified; the report figure compares it with Qwen3-30B-A3B]** | MoE, 32 layers, d=4096, 256 experts top-8; 2.2-flash has 128k ctx and `block_size: 32` in config | ~190 GiB est. from 100B×2B | **No** (dInfer recipes use TP=4; "quant versions" exist per dInfer news 2025-12-10) |
| `*-flash-CAP` | as flash | | | No |

**Inference techniques (from the tech reports, since the repo has no code).**
* Block diffusion with block size 32, threshold decoding at 0.95, temperature 0 (`tech_report.pdf` §"All LLaDA2.0 models … block size of 32 … threshold of 0.95"); ablation: threshold 0.85 gives peak 3.31 tokens-per-forward vs 2.55 at 0.95; block 16/32/64 → 32 is the sweet spot.
* KV-cache reuse across blocks is explicitly a design goal of the "Decay" phase ("KV-cache reuse and fast variable-length generation").
* CAP = confidence-aware-parallel post-training to raise predictive confidence so more tokens pass the threshold per step.
* LLaDA2.1 adds Token-to-Token editing with two thresholds (τ_mask, τ_edit) in a "draft-and-edit" loop; LLaDA2.2 adds KEEP/SUBSTITUTE/DELETE/INSERT edit tokens, 128k context, and "block routing" for MoE.
* Serving: "custom inference engine based on dInfer and SGLang" (README), and for 2.1 "a customized version of SGLang" with radix caching and batching for block-diffusion LLMs.

**Install / run.** None in this repo. The HF model card's plain-transformers path is:
```python
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cuda:0").to(torch.bfloat16).eval()
generated_tokens = model.generate(inputs=input_ids, eos_early_stop=True, gen_length=512, block_length=32, steps=32, temperature=0.0)
```
I read the remote `modeling_llada2_moe.py` (1,403 lines) from HF: `generate()` at L1233 builds a block-causal attention mask (L1310-1316) and re-runs the full window every denoising step with **no KV cache**, and the MoE block is a Python loop over the 256 experts (`for i, expert in enumerate(self.experts)` at L317). `_supports_flash_attn_2 = False`, `_supports_sdpa = True` (L667-668). So the plain-transformers path *works* on one A100 but is slow; the fast path is dInfer (§2).

**Reported speed/quality.** LLaDA2.0 report: flash-CAP 535 TPS vs flash 383 TPS vs AR Ling-flash-2.0 256 / Qwen3-30B-A3B 237 TPS, "served with dInfer", AR via SGLang; GPU for that figure not stated in the lines I extracted **[unverified; the dInfer README's matching table is 8×H20]**. LLaDA2.1: 892 TPS HumanEval+ / 801 BigCodeBench / 663 LiveCodeBench for the 100B flash (quantized), "LLaDA2.1-mini reaches 1586.93 peak TPS" (customized SGLang; GPU not captured **[unverified]**). README's main-bench figure is an image; mini quality numbers not in text.

---

## 2. dInfer (inclusionAI) — the production inference framework for LLaDA models

**Purpose.** Modular dLLM inference engine (model / diffusion-iteration manager / decoder / KV-cache manager) that runs LLaDA, LLaDA-1.5, LLaDA-MoE and LLaDA2 with tensor/expert parallelism, torch.compile and CUDA graphs, on top of vLLM layers (LLaDA/LLaDA-MoE) or SGLang (LLaDA2). It is the engine behind the 1,100-TPS LLaDA-MoE and 580-TPS LLaDA2-flash numbers.

**License.** Apache-2.0 (`LICENSE`, `setup.py:36`). Python files carry NVIDIA Apache headers where ported from Fast-dLLM (`python/dinfer/model/__init__.py:1-16`).

**Checkpoints** (README "Supported Models" + HF today):

| HF id | Params | Impl. file | bf16 | 1×A100? |
|---|---|---|---|---|
| `inclusionAI/LLaDA2.0-mini`, `-mini-preview` | 16B MoE / 1.4B act. | `python/dinfer/model/modeling_llada2_moe.py` | 30.3 GiB | Yes (weights); recipes use TP=4 |
| `inclusionAI/LLaDA2.0-flash`, `-flash-preview` | 100B MoE | same | ~190 GiB | No |
| `inclusionAI/LLaDA-MoE-7B-A1B-{Base,Instruct}` | 7B / 1B active, 64 experts top-8, 16 layers | `modeling_fused_olmoe.py` (needs `tools/transfer.py` FusedMoE conversion) | 13.7 GiB | Yes |
| `GSAI-ML/LLaDA-8B-{Base,Instruct}`, `GSAI-ML/LLaDA-1.5` | 8B dense, 32 layers d=4096, vocab 126,464 | `modeling_llada.py` | 14.9 GiB | Yes |

**Techniques and where they live.**
* Confidence-threshold parallel decoding: `python/dinfer/decoding/parallel_strategy.py` — `get_transfer_index_threshold` L354, `ThresholdParallelDecoder` L388 (`decode` L404, batched `batch_decode` L427), `CreditThresholdParallelDecoder` L461 (credit fusion across iterations, `_apply_credit_fusion` L479), `HierarchyDecoder` L591 (high/low threshold hierarchy, `get_transfer_index_hierarchy_fast_v2` L17, remask variant L93), `FixedParallelDecoder` L549, dynamic `get_transfer_index_dynamic` L270.
* Block-wise diffusion loop and iteration managers: `decoding/generate_uniform.py` — `BlockWiseDiffusionLLM` L498 (semi-AR blocks, `early_stop`, `maximum_unroll`, `expected_tpf` L519), `BlockDiffusionLLM` L983 (true block-diffusion models = LLaDA2; `naive_batching_generate` L1044, `dynamic_batching_generate` L1092 = batching), `BlockDiffusionRunner` L124 (`prefill` L145, `decode` L198), `IterSmoothDiffusionLLM` L623 / `IterationSmooth` L558 (iteration smoothing: feeds soft embeddings weighted by `cont_weight`), `VicinityCacheDiffusionLLM` L713 (vicinity/"dual"-style cache refresh with `prefix_look`/`after_look`/`warmup_steps`), `BlockWiseDiffusionLLMWithSP` L835 (sequence parallel).
* KV cache / dual cache: `decoding/utils.py` — `KVCache` L281 (vLLM- or SGLang-backed, `update` with `replace_position` L347), `DiffusionKVCacheManager` L383 (`cache_type` prefix/dual, `require_update` L402, `range_update` L449), `BlockDiffusionPrefixCacheManager` L499, `KVCacheFactory` L534; `decoding/generate_cache.py` `generate_with_cache` L68, `generate_with_prefixcache_update` L204; Fast-dLLM baseline port in `decoding/generate_fastdllm.py`.
* CUDA graphs + torch.compile: `decoding/diffusion_runner.py` — `ModelRunner` L127 (`enable_cuda_graph`, `init_device_graphs` L168), `CudaGraphRunner` L267 (`capture` L316, `capture_one_batch_size` L428 keyed by (bs, phase, length, cache_length), `replay` L518), `set_torch_compile_config` L68, `patch_model` L81; this file imports SGLang's graph machinery (L7-34). The vLLM-backend path instead does `model.forward = torch.compile(..., mode='reduce-overhead')` (`benchmarks/benchmark.py:98`), i.e. inductor CUDA graphs.
* Triton kernels: not written in-repo; the MoE uses vLLM's Triton `FusedMoE` (`model/modeling_llada2_moe.py:64`, `modeling_fused_olmoe.py:64`) and `configs/` holds fused-MoE tuning JSONs **only for `NVIDIA_H20-3e`** — on A100 vLLM falls back to its default config (works, untuned). Attention: `ATTENTION_CLASSES = {eager, flash_attention_2, sdpa}` (`modeling_llada2_moe.py:1094-1098`); flash-attn 2 is sm80-compatible.
* Serving: `decoding/serving.py` `DiffusionLLMServing` L531 (multi-process TP/DP/EP servers; SGLang path `_sglang_llada2_server_process` L289), `SamplingParams` L166 (threshold 0.9, low_threshold 0.6, cache 'dual', cont_weight 0.3 defaults). `into_sglang/algorithm.py` `dInferDiffusionAlgorithm` L26 plugs the decoders into `sglang.srt.diffusion` **[whether that module exists in the pinned sglang 0.5.3.post1 is unverified; the LLaDA2.0 report cites sglang issue #12766 and the 2025-12-19 LMSYS "Day-0 support for LLaDA 2.0" post]**.

**Install / run (verbatim).**
```bash
git clone https://github.com/inclusionAI/dInfer.git && cd dInfer && pip install .
pip install vllm==0.10.2          # LLaDA / LLaDA-MoE backend
pip install sglang==0.5.3.post1   # LLaDA2 backend
# LLaDA-MoE only: convert to FusedMoE
python -m tools.transfer --input /path/to/LLaDA-MoE-7B-A1B-Instruct --output /path/to/LLaDA-MoE-7B-A1B-Instruct-fused
```
Single-sample speed benchmark (README shows 4 GPUs; `--gpu 0` with `process_args` at `benchmarks/benchmark.py:154-173` disables TP for one GPU):
```bash
python benchmarks/benchmark.py --model_name GSAI-ML/LLaDA-8B-Instruct --model_type llada \
  --gen_len 2048 --block_length 32 --gpu 0,1,2,3 --use_tp --parallel_decoding threshold --threshold 0.9 --cache prefix
```
LLaDA2 dataset benchmark (SGLang backend, prints TPS; needs a prompt JSON with `details[].prompt`):
```bash
python benchmarks/benchmark_dataset_sglang.py --model_name inclusionAI/LLaDA2.0-mini --dataset dataset_path \
  --gen_len 2048 --block_length 32 --gpu 0,1,2,3 --output_dir runs/llada2_mini --use_tp \
  --parallel_decoding threshold --threshold 0.9 --cache prefix --use_bd
```
lm-eval (speed+accuracy): `evaluations/eval_llada_mini.sh` (tasks `gsm8k_llada_mini`, `mbpp_sanitized_llada_mini`, `tp_size=4`, `use_compile=True`, `use_bd=True`) and `evaluations/README.md` (`eval_dinfer.py --model dInfer_eval --model_args model_path=…,cache=dual,use_compile=True,use_cudagraph=True,…`).

Versions: `setup.py` requires only scipy/tqdm (`python_requires>=3.6`); real deps are the commented `vllm==0.10.2` / `sglang==0.5.3.post1`. Every model file hard-imports vLLM (`modeling_llada.py:57`, `modeling_llada2_moe.py:64-77`), so vLLM 0.10.2 is mandatory even for one GPU; local vLLM is 0.8.3 → separate venv. sglang 0.5.3.post1's torch requirement vs local torch 2.6 **[unverified]**.

A100 issues: nothing Hopper-only found (flash-attn 2, SDPA, Triton fused-MoE all run on sm80), but all tuned MoE configs are H20; `modeling_llada.py:284` wraps SDPA in `@torch.compile()`; LLaDA2 prompts use the Ling chat format `<role>SYSTEM</role>detailed thinking off<|role_end|>…` (`benchmarks/benchmark.py:43,130`); `tests/test_bd.py` hard-codes internal `/mnt/infra/...` checkpoint paths, so tests are not runnable as-is.

**Reported speed/quality.** README: LLaDA-MoE >1,100 TPS at bs=1 on HumanEval, 800+ avg over six benchmarks, on **one node of 8×H800** (TP); "10× faster than Fast-dLLM, 2–3× faster than Qwen2.5-3B on vLLM" (paper abstract, same). LLaDA2-flash-CAP on **8×H20**, threshold 0.95, gen 1000: bs=1 avg 580.7 TPS (HumanEval 753, GSM8K 592, IFEval 223), bs=32 avg 1,966 TPS. No single-A100 numbers anywhere.

---

## 3. Fast-dLLM (NVlabs) — v1 training-free acceleration, v2 block-diffusion from Qwen2.5

**Purpose.** v1: training-free KV cache (prefix and "dual" cache) plus confidence-aware parallel decoding for LLaDA and Dream, with lm-eval harness integration (ICLR 2026). v2: a recipe + checkpoints that turn Qwen2.5-Instruct into a block-diffusion LM with ~1B fine-tuning tokens, with hierarchical (block + sub-block) KV caching and batched parallel decoding. (`fast_dvlm/` and `fast_ddrive/` are vision/driving variants, out of scope; `third_party/` vendors LMFlow and a full SGLang tree for training.)

**License.** Apache-2.0 (`LICENSE`, README). `Efficient-Large-Model/Fast_dLLM_v2_7B` card: "Released under Apache 2.0, following the base Qwen2.5 license". `GSAI-ML/LLaDA-8B-Instruct` card: `license: mit`. Dream license not checked **[unverified]**.

**Checkpoints.**

| Path | HF id | Params | bf16 | 1×A100? |
|---|---|---|---|---|
| v1 | `GSAI-ML/LLaDA-8B-Instruct` (also `GSAI-ML/LLaDA-1.5`) | 8B dense | 14.9 GiB | Yes |
| v1 | `Dream-org/Dream-v0-Base-7B` / `-Instruct-7B` | 7B dense (Qwen2.5-7B-shaped: 28 layers d=3584) | 14.2 GiB | Yes |
| v2 | `Efficient-Large-Model/Fast_dLLM_v2_7B` | 7B dense, Qwen2.5-7B-Instruct derived (`model_type: Fast_dLLM_Qwen`, mask id 151665) | 14.2 GiB | Yes |
| v2 | `Efficient-Large-Model/Fast_dLLM_v2_1.5B` | 1.5B (config fetched: d=1536, 28 layers) | single file | Yes |

**Techniques.**
* v1 LLaDA (`v1/llada/generate.py`): `generate` L86 (baseline, block-wise semi-AR with `block_length`), `generate_with_prefix_cache` L132 (prefix KV cache refreshed once per block), `generate_with_dual_cache` L211 (prefix+suffix cache; current block re-written in place via a boolean `replace_position` L243-272), `get_transfer_index` L298 (confidence from float64 softmax L320; **threshold mode** L335-337 `mask_index & (confidence >= threshold)`; otherwise top-k per step from `get_num_transfer_tokens` L61), `get_transfer_index_dynamic` L379 (factor-based "parallel factor" rule), `add_gumbel_noise` L27 (float64 Gumbel). Model `v1/llada/model/modeling_llada.py` uses `flash_attn_func` when importable (L614-619, L677-678) else SDPA. No CUDA graphs, no Triton, no torch.compile; cache modes assume batch size 1.
* v1 Dream (`v1/dream/model/generation_utils_block.py`): `diffusion_generate` L319 (`threshold` L373, `dual_cache` L375), `_sample` L387 (prefix cache `use_cache=True` L451, dual cache + `replace_position` L488, `alg='confidence_threshold'` L495-519), entropy/margin confidence in `sample_tokens` L80.
* v2 (`v2/generation_functions.py`): `Fast_dLLM_QwenForCausalLM.batch_sample` L17 — block-level AR over `block_size` (32) with sub-blocks of `small_block_size` (8), threshold unmasking `unmask_idx = (x1_p > threshold)` L118, block-level KV cache (`update_past_key_values=True` L81), optional sub-block cache `use_block_cache` L29/L101-108, **batching** with per-sample finish flags and KV-cache row pruning (L154-155); `mdm_sample_with_visualization` L171. The HF remote `modeling.py` (fetched; 866 lines) implements the block-diffusion mask with **FlexAttention** compiled by `@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")` (`fused_flex_attention` L40-42, `block_diff_mask` L44, `eval_mask` L424, SDPA fallback L237), block cache write at L216-222, and `generate(threshold=1, small_block_size=8, block_size=32)` L653. Token-shift parameterization (README "Architecture").
* Eval: v1 `llada/eval_llada.py` (lm-eval model `llada_dist`, `show_speed`), v2 `eval.py` (`fast_dllm_v2`, bf16, `batch_size 32` in `eval_script.sh`).

**Install / run (verbatim).**
```bash
# v1
cd v1 && pip install -r requirements.txt       # transformers==4.49.0 lm_eval==0.4.8 accelerate==0.34.2 … gradio
python llada/chat.py --gen_length 128 --steps 128 --block_size 32
# v1 GSM8K, prefix cache + parallel (v1/llada/eval.md)
export HF_ALLOW_CODE_EVAL=1; export HF_DATASETS_TRUST_REMOTE_CODE=true
accelerate launch eval_llada.py --tasks gsm8k --num_fewshot 5 --confirm_run_unsafe_code --model llada_dist \
 --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=8,block_length=32,use_cache=True,threshold=0.9,show_speed=True
# (dual cache: add dual_cache=True; Dream: see v1/dream/eval.md, model_args alg=confidence_threshold,threshold=0.9,use_cache=true)

# v2
cd v2 && pip install -e .                      # requirements.txt: transformers==4.53.1, torch>=2.0.1, deepspeed>=0.14.4, …; Python 3.10 advised
python run_chatbot.py                          # model.generate(block_size=32, max_new_tokens=2048, small_block_size=8, threshold=0.9)
accelerate launch eval.py --tasks gsm8k --batch_size 32 --num_fewshot 0 --model fast_dllm_v2 \
 --model_args model_path=Efficient-Large-Model/Fast_dLLM_v2_7B,threshold=0.9
```
Minimal v2 path with no repo at all (HF card): `pip install transformers torch numpy`, `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`, `model.generate(..., max_new_tokens=512, small_block_size=8, threshold=0.9)`.

Versions / A100: v1 pins transformers 4.49.0 (old; needs own venv), flash-attn optional. v2 pins transformers 4.53.1; the remote code imports `transformers.modeling_layers.GradientCheckpointingLayer` and `integrations.use_kernel_forward_from_hub` (≥4.52) and uses `torch.nn.attention.flex_attention` (needs torch ≥2.5 → local 2.6 OK; FlexAttention/Triton runs on sm80). v2's `requirements.txt` pulls DeepSpeed (training only) which needs `CUDA_HOME`/nvcc at import — set to cuda-12.4. Nothing Hopper-only; no flash-attn-3. Behaviour under transformers 5.x **[unverified]**.

**Reported speed/quality** (v1 paper §"Experimental Setup": "All experiments are conducted on an NVIDIA A100 80GB GPU", batch size 1, no serving framework):

| Model / task (gen len) | Baseline acc / tok/s | +Cache | +Parallel | +Cache+Parallel |
|---|---|---|---|---|
| LLaDA-8B-Instruct GSM8K 5-shot (256) | 79.3 / 6.73 | 79.5 / 21.23 | 79.2 / 16.53 | 78.5 / **54.4 (8.1×)** |
| same (512) | 77.5 / 3.23 | 77.0 / 10.43 | 77.6 / 18.63 | 77.2 / 35.3 (11.0×) |
| LLaDA HumanEval 0-shot (256) | 41.5 / 30.5 | 42.7 / 40.7 | 43.9 / 101.5 | 43.3 / 114.1 (3.7×) |
| Dream-v0-Base-7B GSM8K (256) | 75.0 / 9.1 | 74.3 / 32.5 | 74.2 / 14.2 | 74.8 / 48.2 (5.3×) |

(v1 README table; Dream row from the dllm reproduction table, "Official" column.) v2: "2.54× higher throughput than Qwen2.5-7B-Instruct" (paper Fig. 5 measured on A100 and H100 across batch sizes; absolute tok/s are only in figures **[unverified]**); quality 7B: HumanEval 63.4, MBPP 63.0, GSM8K 83.7, MATH 61.6, IFEval 61.4, MMLU 66.6, GPQA 31.9 (avg 60.3); 1.5B avg 45.0.

---

## 4. DiffusionLLM (lamm-mit, "classroom-diffusion-llm" 0.4.0) — AR→MDLM conversion + a feature-rich sampler

**Purpose.** A compact, teachable pipeline to convert a small Qwen2/2.5, Qwen3 or Llama checkpoint into a masked-diffusion LM (bidirectional attention, `<|diffusion_mask|>` token), train it (legacy MDLM / v2 / block-shift / hybrid / graph-edit objectives, LoRA or full, DDP/ZeRO/FSDP2/Slurm), and sample with a sophisticated CPU-friendly sampler (commit policies, true remasking, CFG, GIF export). It is a rewrite informed by `ZHZisZZ/dllm` (see `NOTICE`, `docs/upstream-review.md`).

**License.** Apache-2.0; `NOTICE` credits dllm (Apache-2.0).

**Checkpoints.** No official large model. Trained artifacts referenced in the README exist on HF (HTTP 200 today): `lamm-mit/qwen2.5-1.5b-diffusion-ultrachat` (Qwen2.5-1.5B-Instruct → UltraChat SFT, 3 epochs, one RTX 6000 Ada) and `lamm-mit/qwen2.5-1.5b-diffusion-chatmix-1024-2m-block-shift-v3`. Both 1.5B dense → trivially fit one A100. README itself: "Short runs do not reproduce published diffusion-LLM quality."

**Techniques** (`src/diffusion_llm/sampling.py`, 1,356 lines): `MaskedDiffusionSampler` L208 — `sample` L736 (block-wise left-to-right when `block_size < max_new_tokens`; `block_size == max_new_tokens` = fully parallel), commit policies max-prob / margin / entropy / position / UNCODE (`_raw_scores` L169, `_information_weights` L286), fixed-count or **confidence-threshold** commit schedule with `--max-commit`, **genuine remasking** (`_select_remasks` L594, `_remask_scope` L572; policies confidence / rescore / random, acceptance `improve`), classifier-free guidance (`_guided_forward` L492, `_unconditional_inputs` L398), NFE budget (`_iteration_budgets` L711), EOS stability. **No KV cache** (`use_cache: False` at L446), no CUDA graphs, no Triton, no fused kernels; attention is stock HF (`modeling.py` subclasses `Qwen2Model`/`Qwen3Model`/`LlamaModel` L237/322/416; block-causal masks in `attention.py`). Batching only via `evaluate --batch-size`.

**Install / run (verbatim).**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh; git clone https://github.com/lamm-mit/DiffusionLLM.git; cd DiffusionLLM
uv sync --python 3.12; uv run diffusion-llm doctor
uv run diffusion-llm generate --model lamm-mit/qwen2.5-1.5b-diffusion-ultrachat \
  --prompt "What are some common mistakes that leaders make, and how can they be avoided?" --chat-template \
  --max-new-tokens 192 --steps 96 --block-size 32 --temperature 0.2 --device cuda:0
uv run --no-sync diffusion-llm evaluate --model "$MODEL" --dataset lamm-mit/diffusion-chat-mixture-1024 --dataset-config chatmix_2m \
  --split validation --num-samples 32 --batch-size 2 --max-new-tokens 512 --steps 512 --block-size 8 --temperature 0 --dtype bfloat16 --device cuda:0 --output artifacts/heldout.jsonl
```
Versions: Python ≥3.10 (README uses 3.12), `torch>=2.6`, **`transformers>=5.13,<6`** ("attention implementation is version-sensitive") — local transformers 5.5.0.dev0 is too old, so this needs its own uv venv. No A100-specific issues; no custom kernels at all.

**Reported speed/quality.** None quantitative (no tok/s, no benchmark scores); the sampler prints NFE/commits/remasks/elapsed to stderr. Training examples on RTX 6000 Ada, DGX Spark, 8×A100/H100.

---

## 5. dllm (ZHZisZZ) — unified training/inference/eval library covering LLaDA, LLaDA2/2.1, Dream, A2D, Fast-dLLM

**Purpose.** A `transformers`-Trainer-based library that unifies MDLM/BD3LM/EditFlow/GRPO training and lm-eval-harness evaluation for open dLLMs, with faithful ports of the official samplers (LLaDA, LLaDA2.0/2.1, Dream) and of Fast-dLLM v1 ("identical hyperparameters yield identical outputs"). It also ships Tiny-A2D (Qwen3-0.6B → diffusion) recipes and checkpoints.

**License.** Apache-2.0.

**Checkpoints it runs** (examples/*/README): `GSAI-ML/LLaDA-8B-{Base,Instruct}` (8B dense), `inclusionAI/LLaDA-MoE-7B-A1B-*` (set `model_type: lladamoe`), `inclusionAI/LLaDA2.0-mini` / `LLaDA2.1-mini` (16B MoE, 1.4B active, 30.3 GiB), `Dream-org/Dream-v0-{Base,Instruct}-7B`, `dllm-hub/Qwen3-0.6B-diffusion-{mdlm,bd3lm}-v0.1` (0.6B, exists), BERT-Chat models. All but LLaDA2-flash fit one A100 in bf16.

**Techniques.**
* `dllm/core/samplers/mdlm.py` `MDLMSampler` L35: block-wise masked denoising (L130-135), `remasking` = `low_confidence` | `random` (L197-207) — note this is LLaDA's "which masks to reveal" rule, not true remasking; `infill` L241.
* `dllm/core/samplers/bd3lm.py` `BD3LMSampler` L155: block-causal attention mask + position ids (`_prepare_for_sampling` L16-82), per-block diffusion steps, **prefix KV cache** (`use_cache=True` L299) and CFG (L304-313).
* `dllm/pipelines/fastdllm/llada/sampler.py` `FastdLLMLLaDASampler` L180: `use_cache` none/prefix/dual (L171-172, L296-300; dual needs `replace_position` support in `models/modeling_llada.py` L824-861), confidence **threshold** (`_get_transfer_index` L32, L111-112) or `factor`, per-block quotas; model uses flash-attn when present (L715-720, 782-792) else SDPA. Batch>1 only with equal prompt lengths in cache modes (L302-310).
* `dllm/pipelines/llada2/sampler.py` `LLaDA2Sampler` L96: block-causal mask (L147-154), threshold 0.95 default (L90, L210), even transfer schedule, **no KV cache**; `llada21/sampler.py` adds the T2T edit loop.
* Eval: `dllm/pipelines/{llada,dream,fastdllm/*,a2d,bert}/eval.py` lm-eval adapters, `accelerate launch --num_processes N` data parallel. No CUDA graphs, no Triton, no torch.compile.

**Install / run (verbatim).**
```bash
conda create -n dllm python=3.10 -y && conda activate dllm
conda install cuda=12.4 -c nvidia
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e .                                   # transformers==4.57.0 accelerate==1.11.0 deepspeed==0.18.0 peft==0.17.1 datasets==4.2.0 …; optional flash-attn==2.8.3, vllm==0.8.5.post1
git submodule update --init --recursive && pip install -e "lm-evaluation-harness[ifeval,math]"
# single-GPU Fast-dLLM sampling (prints NFE, seconds, tokens/s)
python examples/fastdllm/llada/sample.py --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" --use_cache prefix --threshold 0.9
# single-GPU evaluation sweep (baseline / parallel / prefix / dual / combos), GSM8K + others
bash examples/fastdllm/llada/eval.sh --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" --instruct True --num_gpu 1
accelerate launch --num_processes 1 dllm/pipelines/fastdllm/llada/eval.py --tasks "gsm8k" --num_fewshot 5 --model "fastdllm_llada" --apply_chat_template \
  --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,use_cache=prefix,threshold=0.9,max_new_tokens=256,steps=256,block_size=32,suppress_tokens=[],begin_suppress_tokens=[]"
python -u examples/llada2/sample.py --model_name_or_path "inclusionAI/LLaDA2.0-mini"
```
Versions: the pins (torch 2.6.0 cu124, flash-attn 2.8.3, Python 3.10) match this machine's torch/triton/flash-attn exactly; only `transformers==4.57.0` differs from the system 5.5.0.dev0 → use a venv. The shallow clone has an empty `lm-evaluation-harness/` submodule (must init). No sm80 issues found.

**Reported speed/quality** (`examples/fastdllm/README.md`, "Reproduced" rows, GPU not stated **[unverified]**): LLaDA-8B-Instruct GSM8K len 256: 78.0 / 8.1 tok/s baseline → 78.0 / 52.8 tok/s (6.5×) with cache+parallel; HumanEval 38.4/18.8 → 36.0/68.1; MBPP 36.4/9.3 → 37.8/44.7. Dream-v0-Base-7B GSM8K 75.4/9.0 → 74.2/42.2. LLaDA-8B-Instruct full reproduction (examples/llada/README): MMLU 69.8, GSM8K 74.7, HumanEval 47.0, MBPP 40.0 (official 65.5/69.4/49.4/41.0).

---

## 6. Side-by-side

| | Fast-dLLM v1 | Fast-dLLM v2 | dInfer | dllm | DiffusionLLM |
|---|---|---|---|---|---|
| Block-wise / semi-AR | yes (`generate.py:86`) | block diffusion (model) | yes + true BD (`BlockDiffusionLLM`) | yes | yes |
| Confidence threshold | yes (L335) | yes (`batch_sample` L118) | yes + credit + hierarchy | yes (Fast-dLLM port, LLaDA2) | yes (+entropy/UNCODE) |
| KV cache | prefix + dual | block + sub-block | prefix/dual + vicinity, SGLang paged | prefix/dual (FastdLLM), prefix (BD3LM) | none |
| Dynamic step scheduling | factor rule (L379) | threshold only | `expected_tpf`, early stop, credit | factor/threshold | NFE budget, max-commit |
| Remasking (true) | no | no | hierarchy remask variant | no | yes (confidence/rescore) |
| CUDA graphs | no | FlexAttention compile (no cudagraphs) | yes (SGLang runner / inductor reduce-overhead) | no | no |
| Triton kernels | no | FlexAttention (Triton) | vLLM FusedMoE (Triton) | no | no |
| Batching | bs=1 in cache modes | yes (finish flags) | yes (dynamic) | equal-length only | eval batch |
| Stack | HF 4.49 | HF 4.53.1 | vLLM 0.10.2 / SGLang 0.5.3.post1 | HF 4.57 | HF ≥5.13 |
| Reference GPU for speed | **A100 80GB** | A100/H100 (figures) | 8×H800, 8×H20 | unstated | none |

---

## 7. arXiv 2506.17298 — "Mercury: Ultra-Fast Language Models Based on Diffusion" (Inception Labs, 17 Jun 2025, 15 pp., CC BY-NC-SA 4.0)

Fetched via `export.arxiv.org/api` and `arxiv.org/html/2506.17298v1` (both reachable with `-4`).

**Disclosed.**
* Models: Mercury Coder **Mini** and **Small**; commercial, closed weights; API at platform.inceptionlabs.ai (OpenAI-compatible), playground at chat.inceptionlabs.ai.
* Formulation: discrete diffusion with a forward Markov noising process q(z_t|z_{t-1}) over token sequences and a learned reverse p_θ; trained by a weighted denoising objective L(x) = −E_t[γ(t)·E_{z_t∼q} log p_θ(x|z_t)]; generation p_θ(z_{t-1}|z_t) = Σ_x q(z_{t-1}|z_t,x) p_θ(x|z_t). "Our methods extend [28]" where [28] = Lou, Meng & Ermon, *Discrete diffusion language modeling by estimating the ratios of the data distribution* (SEDD).
* Architecture: "based on a Transformer architecture"; the choice is said to be orthogonal to diffusion. Context 32,768 tokens out of the box, 128k with extension.
* Training: "on the order of trillions of tokens"; web crawls + curated real and synthetic proprietary data; all development on a large H100 cluster; fine-tuning/alignment via RLHF or DPO with the AR loss replaced by the denoising loss.
* Inference: supports zero-/few-shot and chain-of-thought prompting; "proprietary inference engine" with "dynamically batched sampling and paging" that "automatically navigate[s] the speed/quality trade-off", plus "a set of custom kernels for parallel inference workloads". Coarse-to-fine parallel generation is credited with higher arithmetic intensity on GPUs.
* Speed (Artificial Analysis, API endpoint hosted Feb 2025, ~1,000-token prompts/outputs, output tokens/s from first output token): Mini **1,109 tok/s**, Small **737 tok/s** on NVIDIA H100; 3–10× faster than speed-optimized frontier APIs (e.g. Gemini 2.0 Flash Lite 201, Qwen2.5-Coder-7B 195). Copilot Arena: Mini 25 ms latency (rank 1), Elo tied 2nd.
* Quality (Table 1, pass@1): Mini HumanEval 88.0 / MBPP 77.1 / EvalPlus 78.6 / MultiPL-E 74.1 / LiveCodeBench 17.0 / BigCodeBench 42.0; Small 90.0 / 76.6 / 80.4 / 76.2 / 25.0 / 45.5. FIM: both beat Codestral 2501. Small > Mini on every benchmark ("scaling").

**Not disclosed.** Parameter counts of Mini/Small; depth/width/vocab/tokenizer; the noise schedule γ(t) and the forward-process family (masking vs. uniform); number of denoising steps or tokens-per-step; whether decoding is block-wise/semi-autoregressive and the block size; the unmasking/acceptance rule (confidence threshold etc.); KV-caching strategy; batch size, number of GPUs and TP degree behind the 1,109 tok/s figure; training token count and data mixture; any evaluation harness settings; weights or code. The H100 numbers are end-to-end API throughput measured by a third party, not a kernel benchmark, so they are **not directly comparable** to the single-GPU bs=1 numbers in §3–§5.

---

## 8. Recommendation

**Phase-1 baseline: run Fast-dLLM v1 on `GSAI-ML/LLaDA-8B-Instruct` on one A100** (MIT-licensed 8B dense checkpoint, 14.9 GiB bf16), using `dllm`'s faithful port (`examples/fastdllm/llada/sample.py`, `examples/fastdllm/llada/eval.sh --num_gpu 1`) or the NVlabs `v1/` code itself. Reasons:
1. It is the only system whose published tokens/s were measured on **exactly our hardware** (one A100 80GB, bs=1): GSM8K-256 6.7 → 54.4 tok/s at 78.5% accuracy. That gives a pass/fail check that our environment is sane before we write any kernel.
2. Pure PyTorch + HF (transformers 4.57 / torch 2.6 cu124 / flash-attn 2.8.3 — the dllm pins match the libraries already on this box), no vLLM/SGLang, single GPU, lm-eval wired in with `show_speed`.
3. It isolates the four knobs Open Mercury must re-implement — prefix cache, dual cache, confidence-threshold parallel decoding, and the factor/dynamic step rule — each measurable in isolation (baseline / +cache / +parallel / +both) with one script.

Immediately after (Phase-1b, same venv family): **`Efficient-Large-Model/Fast_dLLM_v2_7B`** via its HF remote code (`model.generate(block_size=32, small_block_size=8, threshold=0.9)`). It is the closest open analogue to Mercury's likely design (block diffusion from an AR model, hierarchical KV cache, batched decoding, FlexAttention) and has an *exact* AR twin for fairness. Phase-2 ceiling: `inclusionAI/LLaDA2.0-mini` (16B-A1.4B MoE, 30 GiB) through dInfer+SGLang for the 500+ TPS class — but only after a dedicated venv with vllm 0.10.2 / sglang 0.5.3.post1 is proven on A100, since all its recipes and MoE tuning configs assume 4×GPU / H20.

**Fair autoregressive comparison.** For LLaDA-8B-Instruct: **`Qwen/Qwen3-8B`** (dense, 36 layers, d=4096, 151,936 vocab — same size class), served two ways: HF `generate` in the same venv (apples-to-apples with the dLLM code path) and vLLM 0.8.3 (already installed) as the "optimized AR" reference, both bf16, bs=1, same prompts and `max_new_tokens`. For Fast-dLLM v2 7B the fair pair is its own base `Qwen/Qwen2.5-7B-Instruct` (identical weights lineage) plus Qwen3-8B. For LLaDA2.0-mini (1.4B active / 16B total) the report's own peer is Ling-mini-2.0; among Qwen3, `Qwen3-30B-A3B` (MoE, 3B active) is the architectural match and `Qwen3-1.7B` the active-compute match — report both.

**What I could not verify** (collected): active parameter count of LLaDA2-flash; GPU behind the LLaDA2.0 535-TPS figure and the LLaDA2.1 TPS numbers; whether sglang 0.5.3.post1 contains `sglang.srt.diffusion` and its torch pin; Fast-dLLM v2 absolute tok/s (figure only) and its behaviour on transformers 5.x; dllm's reproduction GPU; Dream's license; Ling-2.0 upstream license terms.
