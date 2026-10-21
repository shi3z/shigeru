"""Dual-stream training examples (port of third_party/WeDLM/finetune/src/batch.py + masking.py).

One example of L tokens (prompt_len prompt tokens followed by the response) becomes a physical sequence of 2L rows:

    input_ids    = [ x0: the L clean tokens | xt: per block of B tokens, [observed tokens asc, [MASK] * n_mask] ]
    position_ids = [ 0 .. L-1              | per block: observed positions asc, then masked positions asc      ]
    mlm_labels   = [ -100 ...              | per block: -100 * n_obs, true ids of the masked tokens            ]
    mlm_weights  = [ 0 ...                 | per block: 0 * n_obs, (1/gamma_k) * n_mask                         ]
    ar_labels    = [ next token (tokens[i+1]) on x0 rows, -100 where the *target* is a prompt token (optional)  ]
                   [ -100 on every xt row ]

Per block k, gamma_k ~ U(0, 1] (independent per block, `uniform_gamma`; pass your own `gamma_sampler(nblk, rng)` for
curricula or tests) and n_mask = round(n_maskable * gamma_k) positions are masked among the block's response tokens —
prompt tokens are never masked and never scored.  WHICH n_mask positions is `mask_mode`:
    "random" (default, WeDLM): drawn without replacement (uniform random subset of the block's response tokens);
    "suffix": the LAST n_mask maskable positions of the block (a contiguous masked suffix; the first
              n_maskable - n_mask response tokens of the block stay observed).  This is the situation the streaming
              decoder (decoder.py) actually poses: a window after the committed frontier whose rows are masks with
              observed context only on their left, at distance 1..W from the frontier.  With a random subset a mask
              almost always has observed tokens on its right inside the block, which inference never provides, and
              a mask at distance >= 2 from the last observed token is rare (docs/distill_design.md section 10.9 / 11.1:
              the trained drafter is reliable only 1-2 positions past the frontier).  n_mask is the same round(n * gamma_k)
              draw as in "random" mode (not a separate k ~ U{1..B}), so the 1/gamma_k weighting and the count statistics
              are unchanged; only the choice of positions differs;
    "mixed": per EXAMPLE one extra Bernoulli(1/2) draw from `rng` picks "random" or "suffix" for all of its blocks.
The effective mode of an example ("random" or "suffix") is returned as `mask_mode`.  In "random" mode nothing is drawn
beyond gamma and the subsets, so its rng stream — and every example — is byte-identical to what it was before mask_mode
existed.  Observed-first reordering changes only the physical order; the RoPE position of every token is its original one
(position_ids), shared with the x0 half.  In "suffix" mode the observed-first order of a block coincides with the original
order (observed prefix, then the masks in ascending position), exactly the window layout of the decoder.
The [MASK] row predicts its *own* position (no token shift), the x0 rows predict the next token (shift folded into
ar_labels so both losses are plain row-wise cross-entropy, see loss.py).

Mask token id: for Qwen3 the tokenizer uses ids <= 151668 while the embedding matrix has 151936 rows (padded), so any
id in [151669, 151935] is a free row and needs no resize (docs/wedlm_analysis.md section 11 suggests 151669 and naming it
<|mask|>; `tokenizer.add_special_tokens({"additional_special_tokens": ["<|mask|>"]})` hands out exactly 151669).  Do not
reuse an existing special token (Fast-dLLM v2 clobbers <tool_response> = 151665).  model_glue.add_mask_token() resizes the
embedding/lm_head when the chosen id is beyond the current matrix and initialises the row from the mean embedding.
"""
import math
import torch

from .mask import batch_mask

IGNORE = -100
MASK_MODES = ("random", "suffix", "mixed")
__all__ = ["IGNORE", "MASK_MODES", "uniform_gamma", "build_example", "collate"]


def uniform_gamma(nblk, rng=None, eps=1e-8):
    """gamma_k ~ U(0, 1] per block (WeDLM `sample_block_mask_ratios`, mask_per_block=True)."""
    return torch.rand(nblk, generator=rng).clamp_min(eps)


def build_example(tokens, prompt_len, mask_id, block=32, gamma_sampler=None, rng=None, ar_on_prompt=False,
                  mask_mode="random"):
    """Build one dual-stream example (dict of 1-D CPU tensors over the 2L physical rows plus scalars).

    tokens: 1-D int sequence (prompt + response), prompt_len: number of leading prompt tokens, mask_id: [MASK] id,
    block: B, gamma_sampler(nblk, rng) -> float tensor [nblk] in (0, 1], rng: torch.Generator for reproducibility,
    ar_on_prompt: also score next-token prediction of prompt tokens on the x0 rows (WeDLM does not),
    mask_mode: "random" (default) | "suffix" | "mixed" — which round(n * gamma_k) positions of a block are masked (module
    docstring); labels, weights, position ids and the attention mask are built the same way in every mode.
    """
    if mask_mode not in MASK_MODES:
        raise ValueError(f"mask_mode={mask_mode!r}, expected one of {MASK_MODES}")
    tokens = torch.as_tensor(tokens, dtype=torch.long).flatten()
    L = tokens.numel()
    assert L > 0 and 0 <= prompt_len <= L, (L, prompt_len)
    nblk = math.ceil(L / block)
    gamma = (gamma_sampler or uniform_gamma)(nblk, rng).to(torch.float32)
    assert gamma.shape == (nblk,) and bool((gamma > 0).all()) and bool((gamma <= 1).all()), gamma
    if mask_mode == "mixed":                                    # one extra draw per example, only in this mode
        mask_mode = "suffix" if bool(torch.rand(1, generator=rng) < 0.5) else "random"
    positions = torch.arange(L)

    xt_ids, xt_pos, xt_lab, xt_w = [], [], [], []
    for b in range(nblk):
        s, e = b * block, min(L, (b + 1) * block)
        ids, pos = tokens[s:e], positions[s:e]
        g = float(gamma[b])
        cand = (pos >= prompt_len).nonzero().flatten()           # maskable = response tokens of this block
        masked = torch.zeros(e - s, dtype=torch.bool)
        n_mask = int(round(cand.numel() * g))
        if n_mask and mask_mode == "suffix":
            masked[cand[-n_mask:]] = True                        # the last n_mask response positions of the block
        elif n_mask:
            masked[cand[torch.randperm(cand.numel(), generator=rng)[:n_mask]]] = True
        obs, msk = (~masked).nonzero().flatten(), masked.nonzero().flatten()   # each ascending -> observed first
        order = torch.cat([obs, msk])
        row_ids = ids[order].clone()
        row_ids[obs.numel():] = mask_id
        lab = torch.full((e - s,), IGNORE, dtype=torch.long)
        lab[obs.numel():] = ids[msk]
        w = torch.zeros(e - s)
        w[obs.numel():] = 1.0 / g
        xt_ids.append(row_ids); xt_pos.append(pos[order]); xt_lab.append(lab); xt_w.append(w)

    ar_labels = torch.full((2 * L,), IGNORE, dtype=torch.long)
    target_ok = torch.ones(L - 1, dtype=torch.bool) if ar_on_prompt else (positions[1:] >= prompt_len)
    ar_labels[: L - 1][target_ok] = tokens[1:][target_ok]
    return dict(
        input_ids=torch.cat([tokens, *xt_ids]),
        position_ids=torch.cat([positions, *xt_pos]),
        mlm_labels=torch.cat([torch.full((L,), IGNORE, dtype=torch.long), *xt_lab]),
        mlm_weights=torch.cat([torch.zeros(L), *xt_w]),
        ar_labels=ar_labels,
        is_xt=torch.arange(2 * L) >= L,
        seq_len=L, prompt_len=prompt_len, block=block, gamma=gamma, mask_mode=mask_mode,
    )


def collate(examples, pad_id, with_mask=True):
    """Right-pad examples to a [N, T] batch (T = max 2L).  Pad rows: input pad_id, position 0, labels -100, weight 0.

    Returns input_ids, position_ids, mlm_labels, mlm_weights, ar_labels [N, T]; seq_lens [N] (logical L); block (int);
    attention_mask [N, 1, T, T] bool from mask.batch_mask when with_mask (dense regime, fine up to a few K rows).
    """
    assert len({e["block"] for e in examples}) == 1, "all examples must share the block size"
    N, T = len(examples), max(e["input_ids"].numel() for e in examples)
    fill = dict(input_ids=pad_id, position_ids=0, mlm_labels=IGNORE, mlm_weights=0.0, ar_labels=IGNORE)
    out = {}
    for key, val in fill.items():
        dtype = examples[0][key].dtype
        buf = torch.full((N, T), val, dtype=dtype)
        for i, e in enumerate(examples):
            buf[i, : e[key].numel()] = e[key]
        out[key] = buf
    out["seq_lens"] = torch.tensor([e["seq_len"] for e in examples])
    out["block"] = examples[0]["block"]
    if with_mask:
        out["attention_mask"] = batch_mask(out["seq_lens"], out["block"], T)
    return out
