"""The dual-stream attention mask (port of the rule in third_party/WeDLM/finetune/src/masking.py:69-90).

Physical layout of one example of logical length L with block size B:  rows [0, L) are x0 (clean tokens, original
order), rows [L, 2L) are xt (block-wise masked, each block reordered observed-first).  Row r belongs to block
blk(r) = r // B for x0 and (r - L) // B for xt.  A query q may attend key k iff

    intra_block_causal : blk(q) == blk(k) and both rows are in the same stream and k <= q   (physical index!)
    previous_x0_block  : blk(k) <  blk(q) and k is an x0 row

Consequences (the leakage argument, cf. docs/wedlm_analysis.md section 3.1):
  * x0 rows see only x0 rows at smaller-or-equal physical index: the x0 half *is* the plain causal LM, so the AR loss
    on it trains exactly the base model and never sees [MASK] tokens or xt rows.
  * an xt row in block j sees the clean x0 tokens of blocks < j (the committed prefix, same RoPE positions as at
    inference) and the xt rows of its own block with smaller physical index.  Because each block is reordered
    [observed..., masked...], a [MASK] row sees every observed token of its block plus the masks before it, and an
    observed row sees only the observed tokens before it.
  * a masked logical position p can only recover its answer from the x0 row at p, which lives in x0 block blk(p) ==
    blk(q) and is therefore excluded by `blk(k) < blk(q)`; later x0 blocks are excluded by the same test; later
    masks of the same block by `k <= q`; earlier xt blocks are never visible at all (an xt key needs blk(k) == blk(q)).
  * the relation is transitive (everything a visible row sees is itself visible), so even through many layers no
    information from a hidden row can reach a query — which is what tests/test_cdlm.py checks by perturbation.

Padded rows (physical index >= 2L inside a padded batch) attend only themselves, so no softmax row is empty (SDPA would
produce NaN); they are never scored and no real row can see them.
"""
import torch

__all__ = ["allowed", "dense_mask", "batch_mask", "make_mask_mod", "to_additive"]


def allowed(q_idx, kv_idx, seq_len, block):
    """The one rule, on broadcastable integer tensors (q_idx, kv_idx physical rows; seq_len L; block B)."""
    L = seq_len
    xt_q, xt_k = q_idx >= L, kv_idx >= L
    blk_q = torch.where(xt_q, q_idx - L, q_idx) // block
    blk_k = torch.where(xt_k, kv_idx - L, kv_idx) // block
    valid = (q_idx < 2 * L) & (kv_idx < 2 * L)
    intra_block_causal = (blk_q == blk_k) & (xt_q == xt_k) & (kv_idx <= q_idx)
    previous_x0_block = (blk_k < blk_q) & ~xt_k
    return (valid & (intra_block_causal | previous_x0_block)) | (q_idx == kv_idx)


def dense_mask(seq_len, block, total=None, device=None):
    """[T, T] bool (True = may attend) for one example; T = total or 2*seq_len (padded rows: diagonal only)."""
    T = 2 * seq_len if total is None else total
    idx = torch.arange(T, device=device)
    return allowed(idx[:, None], idx[None, :], seq_len, block)


def batch_mask(seq_lens, block, total=None, device=None):
    """[N, 1, T, T] bool for a padded batch; seq_lens is the logical length L of each example (physical 2L)."""
    seq_lens = [int(x) for x in seq_lens]
    T = 2 * max(seq_lens) if total is None else total
    return torch.stack([dense_mask(L, block, T, device) for L in seq_lens])[:, None]


def make_mask_mod(seq_lens, block):
    """FlexAttention `mask_mod(b, h, q_idx, kv_idx) -> bool` with the same semantics as `dense_mask`.

    `seq_lens` is a LongTensor [N] of logical lengths (on the attention device).  Use with
    torch.nn.attention.flex_attention.create_block_mask(mask_mod, N, None, T, T) and attn_implementation="flex_attention"
    (not exercised here; the tests only materialize it with Python loops against `dense_mask`).
    """
    seq_lens = torch.as_tensor(seq_lens)

    def mask_mod(b, h, q_idx, kv_idx):
        return allowed(q_idx, kv_idx, seq_lens[b], block)

    return mask_mod


def to_additive(mask_bool, dtype):
    """bool mask -> additive float mask (0 where allowed, dtype-min elsewhere) as the eager attention path adds it."""
    return torch.zeros(mask_bool.shape, dtype=dtype, device=mask_bool.device).masked_fill(
        ~mask_bool, torch.finfo(dtype).min)
