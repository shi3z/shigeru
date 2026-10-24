"""Reverse process for the B-series (brief §8).

Absorbing-mask reverse diffusion ("monotonic unmasking"): start from K masks at t = 1; at step s of S the target mask
ratio is gamma(t_s) with t_s = 1 - s/S; the network predicts x0 for every masked position; the positions to unmask
(enough to reach the target number of masks) are chosen by rule — "confidence" (highest max-prob first), "random", or
"left" (position order) — and receive their argmax (or a temperature sample); unmasked tokens are never changed.  This
is the standard approximation of p(x_{t-dt} | x_t) for absorbing diffusion (LLaDA / MaskGIT): once revealed, a token
is treated as part of x_0.
Self-correcting variant ("remask", a heuristic on top of the forward process, not derived from it): after each step,
visible tokens whose current x0-prediction differs from their token (the network predicts a different token at a
visible position) are re-masked, up to a budget per step; the schedule still drives the number of masks down.  With
the visible-correction loss of the trainer the network's prediction at a visible position is a trained estimate of
the clean token, so this uses the same network call and no AR logits.
Forward count: one network call per step (S calls per block) + the prefix cache per block; tokens/forward = K/S.
"""
import math

import torch

from .corrupt import gamma

__all__ = ["sample_block", "ORDERS"]

ORDERS = ("confidence", "random", "left")


@torch.no_grad()
def sample_block(model, cache, P, K, steps, mask_id, schedule="linear", order="confidence", temperature=0.0,
                 remask=False, remask_budget=2, generator=None, stop_ids=(), trace=False):
    dev = model.device
    x = torch.full((1, K), mask_id, dtype=torch.long, device=dev)
    tr = []
    forwards = 0
    n_remasked = 0
    for s in range(1, steps + 1):
        t_prev = 1.0 - (s - 1) / steps
        t_now = 1.0 - s / steps
        g_prev = gamma(t_prev, schedule)
        is_mask = x[0] == mask_id
        n_mask = int(is_mask.sum())
        if n_mask == 0 and not remask:
            break
        logits = model.denoise(cache, P, x, max(g_prev, 1e-3)).float()
        forwards += 1
        logits[:, mask_id] = float("-inf")
        if temperature > 0:
            probs = torch.softmax(logits / temperature, -1)
            pred = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            conf = torch.softmax(logits, -1).gather(-1, pred[:, None]).squeeze(-1)
        else:
            probs = torch.softmax(logits, -1)
            conf, pred = probs.max(-1)
        target_masks = int(round(K * gamma(t_now, schedule))) if s < steps else 0
        n_unmask = max(0, n_mask - target_masks)
        masked_idx = is_mask.nonzero().flatten()
        if n_unmask > 0 and len(masked_idx):
            if order == "confidence":
                sel = masked_idx[conf[masked_idx].argsort(descending=True)[:n_unmask]]
            elif order == "random":
                sel = masked_idx[torch.randperm(len(masked_idx), generator=generator)[:n_unmask].to(dev)]
            else:
                sel = masked_idx[:n_unmask]
            x[0, sel] = pred[sel]
        remasked = []
        if remask and s < steps:
            vis = (~is_mask).nonzero().flatten()
            diff = vis[pred[vis] != x[0, vis]]
            if len(diff):
                diff = diff[conf[diff].argsort(descending=True)[:remask_budget]]
                x[0, diff] = mask_id
                remasked = diff.tolist()
                n_remasked += len(remasked)
        if trace:
            tr.append(dict(step=s, tokens=x[0].tolist(), pred=pred.tolist(), conf=[round(c, 3) for c in conf.tolist()], remasked=remasked))
    # anything still masked (remask variant) gets its last prediction
    still = (x[0] == mask_id).nonzero().flatten()
    if len(still):
        logits = model.denoise(cache, P, x, 1e-3).float(); forwards += 1
        logits[:, mask_id] = float("-inf")
        x[0, still] = logits.argmax(-1)[still]
    return x[0].tolist(), dict(forwards=forwards, remasked=n_remasked, trace=tr if trace else None)
