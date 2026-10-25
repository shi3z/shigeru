"""Phase B9-C — global refinement samplers for the block-bidirectional model (docs/global_block_denoising.md §4).

All of them keep all K positions "alive" for the whole process and spend exactly S network calls per block (the
caller adds the prefix pass).  None reads a reference; none calls the AR path.

  C1 monotonic       the existing sampler (engine/bdiff/sampler2.py) — comparison baseline, not here.
  C2 sync            full-block synchronous refinement: step 1 fills all K positions with the argmax of the all-mask
                     forward; every later step re-forwards the fully visible block and replaces every position whose
                     argmax differs from its token (Jacobi).  Measures self-copying directly: the fraction of positions
                     that change per step.
  C2r remask         non-monotonic confidence re-masking (MaskGIT-style with revisiting): at step s every position whose
                     confidence under the current forward is below the schedule's quantile is set back to [MASK]
                     (even if it was placed earlier), the rest keep their argmax; the number of masks follows the
                     schedule (K*(1 - s/S)), so the last step fills everything.  A token is never final before step S.
  C3 soft            distribution-level refinement: masked positions are fed the probability-weighted mixture of token
                     embeddings (soft token) of the previous step's distribution instead of the [MASK] embedding; no
                     position is ever committed before the last step (argmax of the final distribution).
  C4 anneal          C3 with a temperature schedule on the soft distributions (sharpening over steps) and a mixing
                     weight alpha_s between the soft embedding and the [MASK] embedding (alpha from 0 to 1), so the
                     model sees inputs close to its training distribution early and committed-like inputs late.
  C3/C4 are *out of the training distribution* of the model (it never saw soft inputs); a failure without training
  is recorded as such, not as a disproof.
Returns (tokens, info) with info["forwards"], per-step change counts, entropies and (if gold is given) correct->wrong /
wrong->correct counts of the per-step argmax block against the reference.
"""
import math

import torch

from .corrupt import gamma

__all__ = ["global_sample_block", "GLOBAL_MODES"]

GLOBAL_MODES = ("sync", "remask", "soft", "anneal")


@torch.no_grad()
def _forward(model, cache, P, x=None, emb=None, g=1.0):
    if emb is None:
        return model.denoise(cache, P, x[None], g).float()
    return model.denoise_embeds(cache, P, emb[None], g).float()


@torch.no_grad()
def global_sample_block(model, cache, P, K, steps, mask_id, mode="remask", schedule="linear", temperature_end=0.2,
                        temperature_start=1.0, alpha_end=1.0, gold=None, trace=False):
    dev = model.device
    E = model.lm.get_input_embeddings().weight          # [V, H]
    e_mask = E[mask_id]
    x = torch.full((K,), mask_id, dtype=torch.long, device=dev)
    probs = None
    forwards = 0
    per_step = []
    tr = []
    gt = torch.tensor(gold, device=dev) if gold is not None else None
    prev_guess = None
    for s in range(1, steps + 1):
        last = s == steps
        frac = s / steps
        if mode in ("sync", "remask"):
            g = max(float((x == mask_id).sum()) / K, 1e-3)
            logits = _forward(model, cache, P, x=x, g=g); forwards += 1
        else:
            if probs is None:
                emb = e_mask[None].expand(K, -1).clone()
            else:
                T = temperature_start + (temperature_end - temperature_start) * ((s - 1) / max(1, steps - 1)) if mode == "anneal" else 1.0
                q = torch.softmax(probs.log().clamp_min(-30) / T, -1) if T != 1.0 else probs
                soft = q.to(E.dtype) @ E                                   # [K, H]
                alpha = alpha_end * ((s - 1) / max(1, steps - 1)) if mode == "anneal" else 1.0
                emb = alpha * soft + (1 - alpha) * e_mask[None]
            logits = _forward(model, cache, P, emb=emb.to(E.dtype), g=1.0 - (s - 1) / steps); forwards += 1
        logits[:, mask_id] = float("-inf")
        new_probs = torch.softmax(logits, -1)
        conf, pred = new_probs.max(-1)
        ent = -(new_probs.clamp_min(1e-12) * new_probs.clamp_min(1e-12).log()).sum(-1)
        guess = pred.clone()                                            # the block the step would output now
        changed = 0
        if mode == "sync":
            if s == 1:
                x = pred.clone(); changed = K
            else:
                diff = pred != x
                changed = int(diff.sum()); x = torch.where(diff, pred, x)
            guess = x.clone()
        elif mode == "remask":
            x = pred.clone()
            if not last:
                n_mask = int(round(K * gamma(1.0 - frac, schedule)))
                if n_mask > 0:
                    low = conf.argsort()[:n_mask]
                    x[low] = mask_id
            changed = int((prev_guess != pred).sum()) if prev_guess is not None else K
            guess = pred.clone()
        else:
            probs = new_probs
            changed = int((prev_guess != pred).sum()) if prev_guess is not None else K
            guess = pred.clone()
            if last:
                x = pred.clone()
        row = dict(step=s, changed=changed, masked=int((x == mask_id).sum()), entropy=round(float(ent.mean()), 4), conf=round(float(conf.mean()), 4))
        if gt is not None:
            row["acc"] = round(float((guess == gt).float().mean()), 4)
            if prev_guess is not None:
                row["w2c"] = int(((prev_guess != gt) & (guess == gt)).sum()); row["c2w"] = int(((prev_guess == gt) & (guess != gt)).sum())
        per_step.append(row)
        if trace:
            tr.append(dict(step=s, guess=guess.tolist(), masked=(x == mask_id).tolist()))
        prev_guess = guess
    out = x if mode in ("sync",) else (pred if mode != "remask" else x)
    assert int((out == mask_id).sum()) == 0
    return out.tolist(), dict(forwards=forwards, per_step=per_step, trace=tr if trace else None)
