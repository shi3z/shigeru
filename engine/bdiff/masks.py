"""Block-bidirectional attention masks (brief §3).

Single sequence [prefix (P rows, clean) | block (K rows, noisy)]:
  prefix -> prefix : causal        prefix -> block : BLOCKED  (prefix hidden states never depend on the block, so the
                                                               prefix K/V can be cached and reused across denoising steps)
  block  -> prefix : allowed       block  -> block : bidirectional (every block row sees every block row incl. itself)
`train_mask(P, K)` returns the [P+K, P+K] bool matrix for a training forward of the whole sequence; `block_mask(P, K)`
returns the [K, P+K] rows of the block only (for an inference forward of the K block rows on top of a P-row cache).
A visible block token attends itself, so a correction target on a visible row is only learnable if the training
corruption includes replaced (wrong) visible tokens — the model cannot tell a correct from a wrong visible token by
copying (brief §3.1).  Masked rows carry [MASK] as input, so no gold token of a masked position is visible anywhere.
"""
import torch

__all__ = ["train_mask", "block_mask"]


def train_mask(P, K, device=None):
    L = P + K
    m = torch.zeros(L, L, dtype=torch.bool, device=device)
    m[:P, :P] = torch.tril(torch.ones(P, P, dtype=torch.bool, device=device))
    m[P:, :P] = True
    m[P:, P:] = True
    return m


def block_mask(P, K, device=None):
    m = torch.zeros(K, P + K, dtype=torch.bool, device=device)
    m[:, :P] = True
    m[:, P:] = True
    return m
