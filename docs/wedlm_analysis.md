# WeDLM (Tencent WeChat AI) — code-level analysis for SHIGERU Phase C

Date: 2026-10-20. Read-only survey of a shallow clone at
`/mnt/ssdraid/project/open-mercury/third_party/WeDLM` (HEAD `8d3f66b`, "Merge pull request #26 … fix/bare-except",
2026-03-03). Nothing was installed and no model was run. The paper text was extracted from `paper/wedlm.pdf`
(arXiv 2512.22737). Hugging Face facts come from `config.json` / `tokenizer_config.json` / `README.md` / `LICENSE.txt`
fetched today from `tencent/WeDLM-{7B,8B}-{Base,Instruct}`. Everything I could not confirm from code, paper or a
fetched file is marked **[unverified]**. Line numbers are for the checked-out revision.

Comparison target: Fast-dLLM v2 remote code at `/mnt/ssd/models/hf/Fast_dLLM_v2_7B/modeling.py` (866 lines,
`Efficient-Large-Model/Fast_dLLM_v2_7B` commit `0661abf`, see `SOURCE.txt`).

---

## 0. One-paragraph summary

WeDLM keeps the model a **completely standard causal Transformer** (Qwen2.5/Qwen3 weights, unmodified lower-triangular
mask, RoPE). Mask recovery works because, before every forward, the tokens of the active span are **physically reordered
so that observed tokens come first and `[MASK]` tokens last**, while each token keeps its **original logical position id**
for RoPE. Under a causal mask every `[MASK]` row then sees every observed token, and masks see earlier masks (a directed,
left-to-right factorization among masks). Training adds a **dual stream**: the physical input is `[x0 (clean), xt (block-wise
masked + reordered)]` with the *same* position ids for both halves and a block-structured 2-D mask so that `xt` blocks
condition on the *clean* history of earlier blocks only. Loss = 1/γ-weighted MLM cross-entropy on the masked `xt` rows +
standard next-token AR loss on the `x0` rows (same forward). Inference is a **streaming sliding window** of W slots over a
paged, prefix-reusable KV cache: each step is a small causal "prefill" of the window (filled tokens first, masks last) on
top of the cached prefix; the leftmost contiguous run of filled tokens is committed (its K/V are already in the right slots,
nothing is recomputed), masks are filled by an entropy + distance-penalty rule, and the window is refilled with new masks.

---

## 1. License and model availability

### 1.1 Code license
* `LICENSE` (283 lines) is Apache-2.0 text wrapped in Tencent's "License Terms of WeDLM" with an extra clause that
  **prevails on conflict**: `LICENSE:3` and `LICENSE:15` — "WeDLM IS NOT INTENDED FOR USE WITHIN THE EUROPEAN UNION".
  `pyproject.toml:10` declares `license = {text = "Apache-2.0"}`; `Readme.md` badge says Apache 2.0. The same clause
  is in the HF `LICENSE.txt` of every checkpoint (fetched: "WeDLM-8B-Instruct IS NOT INTENDED FOR USE WITHIN THE EUROPEAN
  UNION"). HF model cards nonetheless say `license: apache-2.0`. No MAU cap, no commercial restriction, no field-of-use
  restriction found (`grep -i commercial|military|prohibit` only hits the standard Apache liability paragraph).
* Third-party components: engine is a fork of nano-vllm (`Readme.md:361`); fine-tuning code is Tencent's own.

### 1.2 Checkpoints (HF collection `tencent/wedlm`, all ungated, `custom_code`)

| HF id | Base (HF card / Readme) | Arch (fetched `config.json`) | Context (Readme / config) | Files |
|---|---|---|---|---|
| `tencent/WeDLM-7B-Base` | Qwen2.5-7B | 28 layers, d=3584, ffn 18944, 28 Q / 4 KV heads, head_dim 128, `attention_bias: true`, `qk_norm: false`, θ=1e6 | Readme 32k; `max_position_embeddings: 16384`; `sliding_window: 131072` field present | 4 safetensors shards, bf16 |
| `tencent/WeDLM-7B-Instruct` | WeDLM-7B | same | same | same |
| `tencent/WeDLM-8B-Base` | Qwen3-8B (card: `base_model: Qwen/Qwen3-8B`) | 36 layers, d=4096, ffn 12288, 32 Q / 8 KV heads, head_dim 128, `attention_bias: false`, `qk_norm: true`, θ=1e6, vocab 151936 | Readme 32k; `max_position_embeddings: 16384` | 4 shards, bf16 |
| `tencent/WeDLM-8B-Instruct` ⭐ | WeDLM-8B | same | same | same |

Readme model zoo: `Readme.md:241-248`. Shard sizes were not downloaded; ~15-16 GB bf16 per model is an **estimate**.
`architectures: ["WeDLMForCausalLM"]`, `model_type: "wedlm"`, `auto_map` to `modeling_wedlm.py` (the HF copy is
byte-identical to `hf_compat/modeling_wedlm.py` except the copyright header — verified by `diff`).

**Mask token.** `config.json` has `"mask_token_id": null` in all four checkpoints. The engine falls back to **151665**
(`wedlm/engine/model_runner.py:126-128`), and the trainer hard-codes `MASK_TOKEN_ID = 151665`
(`finetune/src/trainer.py:25`). In the WeDLM tokenizers the added tokens stop at `<|file_sep|>` = 151664
(`added_tokens.json`, `tokenizer_config.json` fetched; `151665` is absent), so 151665 is a *nameless* embedding row inside
`vocab_size` 151936. Note: in the original Qwen3-8B tokenizer 151665 = `<tool_response>`, 151667 = `<think>`; WeDLM-8B
**drops the Qwen3 think/tool tokens** (its chat template is the Qwen2.5 one) — thinking mode does not exist in WeDLM.
Fast-dLLM v2 uses the same id and *does* register it as `|<MASK>|` (151665) in its `added_tokens.json`.

### 1.3 Published vs. described only
Published in the repo: the inference engine (`wedlm/`), an HF-compatible modeling file (`hf_compat/`, standard causal
forward + a naive block decoder, **no training mask**), an SFT reference trainer (`finetune/`), evaluation scripts, the paper.
Described only / not published: the continued-pretraining pipeline and data (100B tokens, `paper §6.1`), the internal SFT
data, the vLLM integration the paper claims ("Our WeDLM models are also served via vLLM", paper §6.2) — not in the repo,
the project page's visualizations, the Docker image `aiweiliu/wedlm:v3` (Readme only). `finetune/README.md:8`: the
fine-tuning code "is **not** the internal training framework we used … provided as a functional reference".

---

## 2. Training-example layout, masking schedule, auxiliary tokens

All in `finetune/src/`. Entry: `trainer.py:178-210` (`train_step`) → `batch.py:33` (`build_wedlm_batch`) →
`model.py:54` (`wedlm_forward`) → `loss.py`.

### 2.1 Raw sample → packed sequence (`data.py`)
* Chat template applied, tokenized, truncated (`data.py:178-228`). Labels = ids with the prompt part set to `-100`
  (`data.py:224`), i.e. **prompt tokens are never masked and never scored**.
* **Auxiliary tokens:** `num_learnable_im_end` (default 8 in `config.py:27`, **0** in `configs/example.yaml:14`) appends
  `num_learnable_im_end - 1` extra `<|im_end|>` tokens after each sample (`data.py:191-220`) so the model learns to fill the
  tail of a window with EOS. Not described in the paper **[purpose inferred from code]**.
* Samples are packed into `max_seq_length × per_device_train_batch_size` tokens; `cum_seqlens` marks sample boundaries
  (`data.py:230-287`).

### 2.2 Dual-stream batch (`batch.py:33-131`)
For every packed sample of length `L` (positions `0..L-1`, `batch.py:64`), blocks of `B = block_size = 32`
(`config.py:23`, `example.yaml:10`), `nblk = ceil(L/B)` (`batch.py:65`):

```
physical input  = [ x0 : L clean tokens | xt : L tokens, block-wise masked and reordered ]   (batch.py:98)
position ids    = [ 0..L-1               | per block: observed positions asc, then masked positions asc ]  (batch.py:100)
loss mask       = [ all False            | per block: [False]*n_obs + [True]*n_mask ]   (batch.py:86-89, 101)
p_mask          = [ 0                    | per block: [0]*n_obs + [γ_k]*n_mask ]   (masking.py:60-63, batch.py:102)
targets         = original ids in the same (reordered) order   (batch.py:99)
```

Per block (`batch.py:71-91`): maskable = positions whose label ≠ -100 (`batch.py:75`); `n_to_mask = round(n_maskable·γ_k)`
with a random permutation (`masking.py:23-39`); `reorder_block` (`masking.py:43-65`) builds
`[tokens[~mask], [MASK]*n_mask]`, `[positions[~mask], positions[mask]]` — logical positions preserved, physical order
changed. The packed sequence length becomes `2L` (`batch.py:107`).

### 2.3 Noise schedule
`masking.py:8-19`: **γ_k ~ Uniform(0,1]** independently per block (`mask_per_block=True`, `clamp_min(eps)`); the
alternative `mask_per_block=False` draws one γ per sequence and emits a warning that it "does not match the paper"
(`config.py:88-89`). No timestep embedding, no schedule over training, no clipping.

### 2.4 Losses (`loss.py`, `trainer.py:198-210`)
* MLM: `F.cross_entropy(logits[masked], targets[masked])`, weighted by `1/(p_mask+eps)` normalized to sum 1
  (`loss.py:38-45`; "uniform" = plain mean). **No token shift**: the `[MASK]` row predicts its own position.
* AR: on the `x0` half only (`trainer.py:212-229` slices `logits[pst:pst+L]`), standard shift-by-one next-token CE over
  labels ≠ -100 (`loss.py:71-81`).
* `total = (mlm + w·ar) / (1 + w)`, `w = ar_loss_weight = 1.0` (`trainer.py:207-208`, `config.py:30-31`).
* Optimizer AdamW lr 3e-6 cosine, warmup 0.1, wd 0.01, grad clip 1.0, full fine-tune (no LoRA), bf16, DeepSpeed ZeRO-2
  optional (`config.py:41-46`, `trainer.py:158-171`).

---

## 3. Mask recovery under causal attention: the exact masks

### 3.1 Training mask (dense backend, `masking.py:69-90`)
For a sample of length L the `(2L × 2L)` boolean mask (True = may attend) is, with `x0_q = q < L`, `xt_q = q ≥ L`,
`blk = (idx − L)//B` for `xt` and `idx//B` for `x0`:

```python
intra_block_causal = (blk_q == blk_k) & (xt_q == xt_k) & (k_idx <= q_idx)   # masking.py:86
xt_to_x0_prev      = (blk_k <  blk_q) & xt_q & x0_k                         # masking.py:87
x0_cross_block     = (blk_k <  blk_q) & x0_q & x0_k                         # masking.py:88
mask = intra_block_causal | xt_to_x0_prev | x0_cross_block                  # masking.py:90
```

So:
* **x0 (memory) query** → x0 keys at any earlier physical index (lines 86 + 88 together = plain causal over `x0`). Never
  sees `xt`. ⇒ the `x0` half is exactly the base AR model; this is what the AR loss is computed on.
* **xt query in block k** → (a) all `x0` keys of blocks `< k` (clean history; never the same block, which would leak the
  masked answers), (b) `xt` keys of the *same* block with physical index ≤ its own. Because the block is reordered
  `[observed…, masked…]`, every `[MASK]` row sees all observed tokens of its block and all masks before it; an observed
  row sees only the observed tokens before it. **xt never attends to earlier xt blocks** (`blk_k < blk_q` requires `x0_k`).
* Leakage: a masked position `m` can reach its own answer only through the `x0` row at logical position `m`, which lives in
  the same block of `x0` and is excluded by `blk_k < blk_q`; later blocks of `x0` are excluded for the same reason; later
  masks in its own block are excluded by `k_idx <= q_idx`. Observed tokens of the same block are the only "future-position"
  information a mask sees — and they are the conditioning set by design.
* Both halves get the **same RoPE position ids** (`batch.py:100`, `model.py:66-67`), so an `xt` query attends `x0` keys by
  logical relative offset exactly as a normal causal prefix would at inference.

The Magi backend (`masking.py:94-164`, `attention.py:34-78`) expresses the same pattern as a list of `(q_range, k_range,
attn_type)` with `attn_type 1` = causal (xt intra-block, x0) and `0` = full (xt → earlier x0 blocks). The dense backend
materializes the mask as additive `0/-inf` and calls `F.scaled_dot_product_attention(..., is_causal=False)`
(`attention.py:114-128`). The whole forward is a hand-rolled loop over Qwen layers that bypasses HF attention
(`model.py:19-84`; the model is loaded with `attn_implementation="eager"`, `trainer.py:83`).

### 3.2 Inference mask
A **plain causal mask**, nothing else:
* Prefill: `flash_attn_varlen_func(..., causal=True, block_table=...)` (`wedlm/layers/attention.py:97-108`).
* Decode: `flash_attn_varlen_func(q, k_cache, v_cache, cu_seqlens_q=cumsum(window sizes), cu_seqlens_k=cumsum(prefix_len +
  window size), block_table, causal=True)` (`attention.py:117-139`). The queries are the reordered window
  `[filled…, masks…]`; the keys are the cached prefix followed by the window's own K/V, which were just written into the
  cache in physical (reordered) order (`store_kvcache`, `attention.py:25-61, 90`). FlashAttention's `causal=True` with
  `seqlen_q < seqlen_k` aligns the triangle to the bottom-right, so query row j sees prefix + window rows `0..j`.
* What a query may attend to: filled window token at physical row j → cached prefix + filled tokens with smaller window
  index; `[MASK]` row → prefix + **all** filled window tokens + masks with smaller logical position. Future information
  cannot leak because masks are physically last and are ordered by ascending logical position (`wedlm_decoder.py:236-241`).

### 3.3 HF-compat reference (`hf_compat/modeling_wedlm.py`)
`WeDLMModel.forward` uses `create_causal_mask` (`:508`) and `is_causal=True` (`:264`) — a standard Qwen2/3 model with
`position_ids` passed through. `_generate_one_block` (`:607-731`) reorders with `_efficient_reorder_sequence` (`:593-605`:
`cat(tokens[~mask], tokens[mask])`, same for position ids), runs the **whole** sequence with `use_cache=False` (`:680`) every
step, takes the rows after the non-mask count as mask logits, and fills by `max_prob >= confidence_threshold` (`:702`) else
argmax. It is a block decoder without streaming or caching — for reference only (`Readme.md:322-323`).

---

## 4. Topological reordering: what, when, how positions are preserved

**What is reordered:** only the *active span* — in training, each 32-token block of the `xt` half; at inference, the W-slot
window. The committed prefix and the `x0` half are never reordered.

**When:** training — once per batch at batch construction (`batch.py:79`); inference — at the start of **every decode step**
(`wedlm_decoder.py:211-250`), because the set of filled slots changes each step.

**How positions are preserved:** the token list and the position list are permuted *together*; the model receives an
explicit `positions` tensor that indexes the RoPE table.

Training (`finetune/src/masking.py:43-65`):
```python
xt = torch.where(mask_indices, mask_token_id, tokens)
unmasked = ~mask_indices
xt_reordered   = torch.cat([xt[unmasked], xt[mask_indices]])
orig_reordered = torch.cat([tokens[unmasked], tokens[mask_indices]])
pos_reordered  = torch.cat([positions[unmasked], positions[mask_indices]])
```
and `model.py:66-67`: `position_ids = batch.logical_positions.unsqueeze(0); cos, sin = base_model.rotary_emb(h, position_ids)`.

Inference (`wedlm/engine/wedlm_decoder.py:231-245`):
```python
non_mask_idx = [i for i, flag in enumerate(window_mask_flags) if not flag]
mask_idx     = [i for i, flag in enumerate(window_mask_flags) if flag]
order = non_mask_idx + mask_idx
ordered_tokens    = [window_tokens[i] for i in order]
ordered_positions = [state.current_seq_len + i for i in order]
```
`WeDLMAttention.forward` applies RoPE from the explicit `positions` (`wedlm/models/wedlm.py:106`,
`wedlm/layers/rotary_embedding.py:59`: `cos_sin = self.cos_sin_cache[positions]`). The engine's RoPE asserts
`rope_scaling is None` (`rotary_embedding.py:74`) — no YaRN path in the engine.

---

## 5. KV cache

* **Prefix cache reused unchanged.** Prefill is a normal causal prefill into a paged cache (`model_runner.py:290-360`,
  nano-vllm style hash-based prefix sharing in `block_manager.py:71-99`). Decode never rewrites prefix slots.
* **Where the window's K/V go.** `_compute_slot_mapping` (`wedlm_decoder.py:138-165`) maps physical row `k` of the
  reordered window to the slot of logical position `current_seq_len + k`. Because the leftmost contiguous *filled* run is
  identical in window order and in reordered order, its K/V land in exactly their final logical slots during the same
  forward. Rows further right (filled-after-a-gap tokens and masks) occupy slots that will be **overwritten next step**.
* **Commit = pointer bump.** After the forward, `prune_count` = index of the first mask in window order
  (`wedlm_decoder.py:439-442`); `_process_pruned_tokens` advances `current_seq_len` by that many
  (`wedlm_decoder.py:357-387`). No K/V copy, no recomputation — the committed K/V were computed with RoPE at their logical
  position attending only to the cached prefix and earlier committed-run tokens, i.e. exactly what an AR model would have
  computed. (Paper Alg. 1 line 8 "Extend (K,V) with (K_W[0:n], V_W[0:n])" is implemented as "they are already there".)
* **What is recomputed.** Every token that is in the window but not yet in the committed run is re-encoded every step
  (its K/V depend on which earlier slots are filled). A token is therefore forwarded ≥ 2 times: once as `[MASK]` (predicted)
  and once as a filled token at the front (where its K/V become final and it is committed) — the refill happens *before* the
  forward (`prepare_decode_inputs`, `wedlm_decoder.py:269-274`), so one forward both commits the run and predicts new masks.
  Hence `tokens_per_forward` printed by the engine (`llm_engine.py:414-422`) is the paper's p_cache numerator/denominator.
* **Capacity.** The scheduler pre-allocates `current_len + 2·W` slots per sequence (`block_manager.py:133-151`), and the CUDA
  graph buckets are captured for `2·W` tokens per sequence (`model_runner.py:549, 575`), because the window can grow to
  `W + prefix_len ≤ 2W` between refill and prune. Paged block size defaults to **4096 tokens** and must be a multiple of
  256 (`wedlm/config.py:44, 76-77`) — a FlashAttention paged-KV requirement.

---

## 6. Decoding algorithm (the exact loop)

Engine entry: `LLMEngine.step` (`llm_engine.py:276-325`) → `Scheduler.schedule` (`scheduler.py:48-99`: prefill waiting
seqs first, otherwise one decode step for all running seqs) → `ModelRunner.run` (`model_runner.py:499-524`) →
`_wedlm_decode_one_step` (`model_runner.py:448-495`).

Per decode step, per sequence (`wedlm_decoder.py`):
1. **Init** (first step): window = `[MASK] × min(W, kv_budget)` (`:118-126`), `current_seq_len = len(prompt)`.
2. **Refill** (`_refill_window_masks`, `:167-189`): `prefix_len` = number of leading filled slots; append `min(prefix_len,
   kv_budget)` new masks at the end. Window is now `W + prefix_len` slots.
3. **Reorder + forward**: inputs `[filled…, masks…]` with logical positions (`:211-250`); one causal forward over the
   window with the paged prefix (`model_runner.py:479-489`; CUDA graph if ≤ 512 tokens, `model_runner.py:381`).
4. **Commit** (`process_decode_outputs`, `:391-478`): `prune_count` = index of first mask; those tokens are appended to the
   output, `current_seq_len += prune_count`; if a stop token is among them the sequence finishes and tokens after the stop
   token are dropped (`:357-387`). Reaching `max_tokens` also finishes.
5. **Predict** (`Sampler.process_mask_positions`, `sampler.py:224-273`): logits of the mask rows only
   (`wedlm_decoder.py:454`); entropy per row (`sampler.py:49-61`); token sample per row with top-k → top-p → temperature
   (`sampler.py:123-167`, greedy if T=0); **position selection** (`sampler.py:169-222`):
   ```python
   adjusted_entropy = entropy + (mask_pos - first_mask_pos) * pos_penalty_factor      # sampler.py:207-212
   fill = (adjusted_entropy < entropy_threshold).nonzero()  or  [argmin(adjusted_entropy)]  # :215-222
   ```
   Selected rows are written into the window and un-flagged (`wedlm_decoder.py:469-474`). At least one token is filled per
   step, so the number of steps is ≤ `max_tokens` and is not fixed in advance.
6. **No remasking**: a filled slot is never reverted (there is no code path that sets a flag back to True).
7. Loop until the window is empty (finished); the scheduler appends committed tokens and frees blocks (`scheduler.py:108-158`).

Defaults: `wedlm_window_size` W = 16 (`wedlm/config.py:48`, `example.py:30`, `evaluation_base.sh:8`);
`wedlm_entropy_threshold` τ = 0.4, `wedlm_pos_penalty_factor` λ = 0.02, `temperature` 0.2 (`sampling_params.py:40-50`;
`example.py` uses τ=0.6, T=0); multiple-choice evals use τ=None (one token per step), W=4, λ=1 (`evaluation/demo.sh`).
Paper quality tables use a unified "one token per step, lowest entropy" scheme with W=6, λ=0.10, T=0.1, 512 max tokens
(paper §6.2); recommended τ ∈ [0.3, 0.6] (paper §6.4).

---

## 7. Conversion recipe from an AR model, and reported results

**Recipe (paper §6.1, `Readme.md:73`):** initialize from Qwen2.5-7B / Qwen3-8B; continued pretraining on **100B tokens**
with the dual-stream objective, B = 32, lr 3e-6 → 3e-7 cosine, MagiAttention for the irregular mask, **auxiliary AR loss**
on the same sequences; then SFT with lr 3e-6 cosine. Data volume for SFT is stated inconsistently: "10B tokens for SFT"
(§1) vs. "10K internal instruction-response pairs" (§6.1) **[unresolved in the paper]**. No step distillation, no
consistency/self-distillation, no LoRA, no curriculum over block size or mask ratio. The released `finetune/` reproduces
only the SFT-style stage (same objective) on Alpaca-format chat data.

**Ablations (paper §6.5, Fig. 5):** block size B ∈ {4, 8, 32} gives the same average (70.45-71.23) → train with large B and
decode with any smaller W; causal intra-block attention beats bidirectional intra-block; **small bases degrade**
(Qwen3-0.6B −3.9, Qwen2.5-1.5B −0.6, Qwen2.5-7B +3.6 average points vs. the AR base) — the paper recommends "7B+ scale".
Dropping λ from 0.05 to 0.01 costs 2.6 accuracy points for 3 % speed (Fig. 4b).

**Quality (paper Tables 1-2, `Readme.md:213-235`):** base WeDLM-8B avg 74.72 vs Qwen3-8B 72.61 (GSM8K 90.20 vs 85.97, MATH
53.60 vs 50.80, HumanEval 75.00 vs 68.90, MMLU 75.46 vs 74.03); instruct WeDLM-8B-Instruct avg 77.53 vs Qwen3-8B-Instruct
75.12 (MATH 64.80 vs 69.60 is the loss; HumanEval 80.49 vs 71.95). WeDLM-7B-Instruct avg 72.78 vs Qwen2.5-7B-Inst 71.09.

**Speed (paper §6.4, Fig. 5a, Readme table at :195-200):** GSM8K with WeDLM-8B-Instruct 272-445 tok/s; conservative
(τ=0.2, λ=0.01) 92.3 % accuracy at **1.97×** vLLM Qwen3-8B; aggressive (τ=0.9, λ=0.01) **3.2×** at > 79 %. Streaming vs.
block-wise decoding 423 vs 221 tok/s (1.9×) at τ=0.9. Case studies: 1673 tok/s counting, 745 tok/s math derivation,
198 tok/s open-ended prose. Readme claims 3-6× on math, 2-3× code, 1.5-2× open QA, "up to 10×" counting.
**GPU not stated in the paper for the speed tables [unverified]**; the Readme example (689 tok/s, 218 tokens) is on an
NVIDIA **H20** (`Readme.md:139`). Evaluation scripts assume 8 GPUs, `max_model_len` 4096, batch 16 per worker
(`evaluation/evaluation_base.sh:4-9`); the vLLM AR baseline is run by `evaluation/ar_baseline_script.sh`.

---

## 8. Dependencies, pins, Hopper-only parts

`requirements.txt` (reference pins, "DO NOT pip install -r"): `torch==2.8.0+cu129`, `transformers==4.56.1`,
`triton==3.4.0`, `flash-attn==2.7.4.post1`, `xxhash 3.6.0`, `numpy 2.3.4`, `safetensors 0.6.2`, `huggingface_hub 0.35.0`,
`flask 3.1.1`. `install.sh:7-9` installs torch 2.8.0+cu129 then flash-attn 2.7.4.post1 with `--no-build-isolation`.
`pyproject.toml:63-75` floors: `torch>=2.1`, `transformers>=4.40`, `triton>=3.0`, `flash-attn>=2.5` (optional extra).
`Readme.md:256`: Python 3.9+, PyTorch 2.1+, CUDA 11.8+. Fine-tune extras: `accelerate deepspeed pyyaml datasets`
(`finetune/README.md:21`); `magi_attention` optional (`finetune/src/attention.py:12`).

What the code actually needs:
* Engine: `flash_attn_varlen_func` with `block_table` (paged KV; FA ≥ 2.5) and `causal=True` — **Ampere works**;
  a 20-line Triton `store_kvcache_kernel` (`attention.py:25-61`); `@torch.compile` on RoPE (`rotary_embedding.py:52`);
  CUDA graphs; `torch.distributed` NCCL for TP ≤ 8 (`config.py:79`).
* HF-compat modeling imports `transformers.masking_utils.create_causal_mask`, `utils.generic.check_model_inputs`
  (`modeling_wedlm.py:28-31`) → needs transformers ≳ 4.53; whether it runs on the 5.x dev install on this box is
  **[unverified]**.
* Fine-tune `dense` backend: SDPA with an explicit additive `(2L)²` mask (`attention.py:114-128`) → any GPU, but O(L²)
  memory for the mask and no flash kernel; `magi` backend: **MagiAttention's flex-flash-attention kernels are Hopper-only
  according to the MagiAttention project [not verified here]** — so the paper's training setup does not transfer to A100.
* Nothing in the *model* is Hopper-specific; the local stack (torch 2.6 cu124, flash-attn 2.8.3, triton 3.2 per
  `docs/research.md`) satisfies the floors; the pinned `torch 2.8.0+cu129` is only the authors' environment.

---

## 9. Differences vs. Fast-dLLM v2 (block diffusion, complementary mask, bidirectional within block)

| Aspect | WeDLM (`finetune/src/*`, `wedlm/engine/*`) | Fast-dLLM v2 (`/mnt/ssd/models/hf/Fast_dLLM_v2_7B/modeling.py`) |
|---|---|---|
| Attention inside the active block | **Causal** over the reordered order (`masking.py:86`) | **Bidirectional** (`block_diff_mask` "block_diagonal": same block ∧ same stream, no index constraint, `:74`) |
| Attention mask at inference | plain `causal=True` FlashAttention (`attention.py:106,138`) | block-causal `block_q >= block_kv` (`eval_block_diff_mask`, `:89-94, :424-432`), SDPA with explicit mask (`:237-250`) |
| Dual stream layout | `[x0 clean, xt masked]` (`batch.py:98`); x0 token-causal; xt block k → x0 blocks < k only | `[xt noisy, x0 clean]` (`:598`); x0 **block**-causal incl. its own block (`:84`); xt block k → x0 blocks < k (`:77-81`) |
| Reordering / positions | observed-first reorder, logical position ids explicit (`masking.py:56-58`) | no reorder; both halves share position ids by applying RoPE to each half separately with the same cos/sin (`:197-207`, `cache_position` of length n/2, `:472-475`) |
| Masking schedule | γ ~ U(0,1] per 32-block, exact count `round(n·γ)` (`masking.py:8-39`) | per-32-block `t ~ U(0,1)`, `p = (1-ε)t + ε`, Bernoulli per token (`:586-593`) |
| Complementary mask | none (each token masked with prob γ of its block) | yes: a second copy with `~mask_indices` concatenated on the batch dim (`:600-613`), so every token is scored once per pair |
| Prediction row / token shift | `[MASK]` row predicts its own position; no shift (`loss.py:38-40`) | **token shift**: HF causal-LM loss shifts by one (`:641`) and generation rolls logits right by one (`:775, :788, :797`) — position i is predicted from row i−1 |
| Loss weighting | 1/γ normalized (`loss.py:43-45`) + AR loss on x0 (`trainer.py:205-208`) | plain mean CE over masked xt rows only (`:633-641`); x0 half is context only, **no AR loss** |
| Mask token | 151665, unnamed row (engine fallback) | 151665 = `|<MASK>|` registered in tokenizer |
| KV cache | paged prefix cache; committed tokens' K/V are final at commit time; uncommitted window re-encoded each step | block cache written only when a block has **no masks left** (`:729-747`, `update_past_key_values=True`), i.e. the whole block is re-encoded once at the end ("stop-and-wait"); optional sub-block cache for the in-progress block (`:211-222`, `replace_position`) |
| Decoding unit | sliding window W (16) with continuous refill; commit any leftmost run | fixed 32-block; inside it, 8-token small blocks decoded left-to-right (`:749-815`); block start token comes from the previous block's last row (`:698-700, :745-746`) |
| Token selection | entropy + distance penalty, threshold τ, ≥1 per step (`sampler.py:169-222`) | top-p-renormalized probability of the sampled token > `threshold` (default 0.9 in the card; code default 1 ⇒ 1 token/step), argmax forced in (`:805-815`) |
| Remasking | none | none |
| Training compute reported | 100B CPT + SFT (7B/8B) | ~1B tokens fine-tune (model card) |
| Kernels needed for training | SDPA dense mask or MagiAttention (Hopper) | `torch.nn.attention.flex_attention` with `create_block_mask` (`:40-42, :434-439`), torch ≥ 2.5; runs on A100 |

Consequences for us: WeDLM's committed prefix is *exact* AR K/V (no "frozen prefix approximation" as in our LLaDA/Fast-dLLM
v1 engine, cf. `docs/long_context_design.md` §1), and its per-step forward is a small causal prefill that any AR engine
(including our Phase-2 paged cache) can run. Fast-dLLM v2 gets more context per prediction (bidirectional block) but pays
one re-encode per block and cannot commit early; the paper's ablation says the causal variant is also *more* accurate for
AR-initialized models (Fig. 5c), though that is their measurement, not ours.

---

## 10. Caveats I could not verify
* Actual safetensors sizes, the dtype of the mask-row embedding, and whether the released weights were trained with
  `num_learnable_im_end` > 0.
* The GPU used for the paper's speed numbers; the "3-6×" Readme figure versus the paper's 1.97-3.2× (GSM8K).
* MagiAttention's Hopper-only restriction (from memory of its project page, not checked today).
* Whether `hf_compat/modeling_wedlm.py` runs under transformers 5.x.
* The SFT data volume (10B tokens vs 10K pairs).
* The claim that WeDLM runs inside vLLM (no code in the repo).

---

## 11. Recommendation for SHIGERU Phase C (distilled Qwen3-4B/8B → causal diffusion)

**Implement WeDLM's mechanism first (topological reordering + dual-stream causal training + streaming window decode),
not Fast-dLLM v2's bidirectional block.** Reasons, all from the code above: (1) zero architecture change — the model stays
a Qwen3 causal LM, so the distilled checkpoint's weights, our SDPA/flash kernels, the Phase-1/2 paged prefix cache and the
long-context design (`docs/long_context_design.md`) are reused unchanged, and the committed prefix is exact rather than
approximate; (2) training needs only a batch builder (~130 lines, `batch.py` + `masking.py`) and a 2-D mask — no new
kernels, no Hopper; (3) the AR auxiliary loss is free (same forward, x0 half) and directly protects the distilled
capabilities, which matters because the paper shows small bases (0.6B/1.5B) can lose points — 4B is untested by anyone, so
we must measure; (4) decoding is ~300 lines on top of a standard engine (`wedlm_decoder.py` + `sampler.py`) and gives
tokens-per-forward (p_cache) as a direct, honest speed metric.

Concretely, what must be built and trained:
1. **Mask token**: add one embedding row (Qwen3 vocab 151936 has free rows above 151668; pick e.g. 151669 and name it
   `<|mask|>` in the tokenizer rather than clobbering `<tool_response>` = 151665, so Qwen3 think/tool tokens survive).
   Initialize it from the mean embedding; it is trained with everything else.
2. **Batch builder**: port `finetune/src/batch.py` + `masking.py` (per-block γ ~ U(0,1], B = 32, observed-first reorder,
   shared position ids, loss mask, 1/γ weights, labels −100 on prompts). Keep `mask_per_block=True`.
3. **Attention mask**: on A100 use `torch.nn.attention.flex_attention` with a `mask_mod` reproducing `masking.py:86-90`
   (the way Fast-dLLM v2 does it at `modeling.py:434-439`) — avoids the dense `(2L)²` mask and the Hopper-only Magi kernel.
   Validate the mask with the three leakage checks in §3.1 (mask row must not see x0 of its own/later blocks, nor later masks).
4. **Loss**: MLM(1/γ) on masked xt rows + next-token AR CE on x0 rows, equal weight (WeDLM default). Full fine-tune, lr
   3e-6 cosine, bf16, ZeRO-2 on GPUs 4-7 per `docs/gpu_policy.md`. Append a few `<|im_end|>` tokens after answers
   (`num_learnable_im_end` idea) so window tails learn to emit EOS.
5. **Data/compute**: WeDLM used 100B tokens for 7-8B; Fast-dLLM v2 used ~1B. Start with 1-2B tokens on the 4B model
   (the B-insensitivity result means we can train B = 32 and decode W ≤ 32), measure (a) AR loss stays at the distilled
   baseline, (b) MLM loss, (c) GSM8K-50/HumanEval at τ = None (1 token/step — must match the AR base within noise) before
   turning on parallel decoding; scale tokens only if (c) fails.
6. **Decoder**: streaming window in our engine: per step forward `[filled…, masks…]` with logical positions, write K/V at
   `current_len + row`, commit the leftmost filled run by bumping `current_len`, entropy + λ·distance selection with τ, refill.
   Report tokens/forward and tokens/s against our HF Qwen3 baseline in `docs/phase1.md` style.
7. **Phase C-2 (after the above works)**: A/B against a Fast-dLLM-v2-style bidirectional-block variant at equal training
   tokens, and try using the discarded next-token logits of the filled window rows as a free speculative proposal — both are
   cheap once the WeDLM path exists.
