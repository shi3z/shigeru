"""Forward corruption q(x_t | x_0, t) and the training corruption mixture (brief §2, §6.1).

Absorbing-mask process: at time t in (0, 1] each block token is independently replaced by [MASK] with probability
gamma(t); gamma_linear(t) = t, gamma_cosine(t) = 1 - cos(pi t / 2) (slow start, LLaDA/MaskGIT-style), and "uniform"
samples the mask RATIO directly (equivalent to linear in t but with the ratio, not t, drawn uniformly).  n = round(K *
gamma) positions are masked (at least 1 for t > 0), chosen uniformly, so the loss weight 1/gamma is the standard
absorbing-diffusion ELBO weight (LLaDA eq. 3 with the per-block ratio).
Additional corruption of the VISIBLE positions (self-correction training, §6): with probability p_replace per block,
a fraction r of the visible positions gets a wrong token — "random" (uniform over the vocabulary), "similar" (the
second-best token of a reference distribution if given, else a token sampled from the top-20 of the prefix-conditioned
first forward of the same block when available), or "model" (a token the A-series decoder placed there in a recorded
on-policy state, distill/gen_onpolicy.py).  The targets of corrupted visible positions are the clean tokens; correct
visible positions keep their token as target (identity), so the x0-prediction is trained on every block position.
"""
import math

import torch

__all__ = ["SCHEDULES", "gamma", "corrupt_block"]

SCHEDULES = ("linear", "cosine", "uniform")


def gamma(t, schedule="linear"):
    if schedule == "cosine":
        return 1.0 - math.cos(math.pi * t / 2)
    return t


def corrupt_block(x0, t, mask_id, rng, schedule="linear", p_replace=0.0, r_replace=0.25, replace_pool=None, vocab=None,
                  state=None):
    """x0: list[int] of K clean tokens.  Returns (xt list[int], is_mask list[bool], is_corrupt list[bool], g).
    state: optional on-policy state (list of K ints, -1 = mask) that REPLACES the random process for this block."""
    K = len(x0)
    if state is not None:
        xt = [x0[i] if False else (mask_id if state[i] == -1 else int(state[i])) for i in range(K)]
        is_mask = [s == -1 for s in state]
        is_corrupt = [(not is_mask[i]) and xt[i] != x0[i] for i in range(K)]
        g = max(sum(is_mask) / K, 1e-3)
        return xt, is_mask, is_corrupt, g
    g = gamma(t, schedule)
    n = max(1, int(round(K * g))) if t > 0 else 0
    perm = torch.randperm(K, generator=rng).tolist()
    masked = set(perm[:n])
    xt = [mask_id if i in masked else x0[i] for i in range(K)]
    is_mask = [i in masked for i in range(K)]
    is_corrupt = [False] * K
    if p_replace > 0 and float(torch.rand(1, generator=rng)) < p_replace:
        visible = [i for i in range(K) if i not in masked]
        n_rep = int(round(r_replace * len(visible)))
        for i in visible[: n_rep] if n_rep else []:
            if replace_pool is not None and replace_pool.get(i):
                cand = replace_pool[i]
                tok = cand[int(torch.randint(len(cand), (1,), generator=rng))]
            else:
                tok = int(torch.randint(vocab or 151643, (1,), generator=rng))
            if tok != x0[i]:
                xt[i] = tok
                is_corrupt[i] = True
    return xt, is_mask, is_corrupt, max(g, 1e-3)
