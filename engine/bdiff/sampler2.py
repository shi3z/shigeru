"""Fixed-budget B-series sampler with selective re-masking and dependency-aware unmasking
(docs/trust_aware_diffusion.md §3, §4, §6).

Exactly S network calls per block (plus the prefix pass the caller does); the S-th call fills every position that is
still masked, so a block of K tokens is always complete after S forwards and no re-masking can add a forward.  The
caller sees the real forward count in info["forwards"].

Unmasking rule (which masked positions receive their prediction at this step; n is set by the schedule):
  confidence   highest max-prob first (current B-series default)
  left         leftmost first
  nonadjacent  highest confidence, skipping a position adjacent (distance 1) to one already chosen this step
  dependency   like nonadjacent, and a position is only eligible if its left neighbour is visible (or it is the
               block's first position) — the A-series dependency diagnostic found the dependency is distance-1 and
               left-sided; falls back to plain confidence order when too few positions are eligible
  adaptive     every masked position with conf >= tau_unmask, at least the schedule's n (and all at the last step)

Re-masking rule (which VISIBLE positions are set back to [MASK] after the unmasking of this step, at most `budget`
per step, never at the last step, and never a position placed in the same step):
  none         monotonic unmasking (current B-series)
  disagree     the model's argmax at the position differs from the placed token, most confident disagreement first
               (the existing "remask" heuristic, now under the fixed budget)
  lowprob      the placed token's probability under the current forward is below tau_remask, lowest first
  stability    the argmax disagreed with the placed token in this AND the previous step (persistent disagreement)
  dependency   disagreement, and a neighbour (distance 1) of the position was placed or re-masked at the previous
               step — i.e. the context the token was placed in has changed since
  trust        a logistic trust score P(correct | features) (engine/bdiff/trust.py, weights fitted on teacher-forced
               trajectories) below tau_remask, least trusted first
A re-masked position is predicted again at the next forward; the schedule's mask count is computed over all masks, so
re-masked positions are unmasked again as the schedule drives the mask count down (by the last step at the latest).
"""
import torch

from .corrupt import gamma
from .trust import step_features, FEATURES

__all__ = ["sample_block_fixed", "UNMASK", "REMASK"]

UNMASK = ("confidence", "left", "nonadjacent", "dependency", "adaptive")
REMASK = ("none", "disagree", "lowprob", "stability", "dependency", "trust")


def _select_unmask(masked_idx, conf, x_vis, n, rule, K, tau):
    if n <= 0 or len(masked_idx) == 0:
        return masked_idx[:0]
    if rule == "left":
        return masked_idx[:n]
    order = masked_idx[conf[masked_idx].argsort(descending=True)]
    if rule == "confidence":
        return order[:n]
    if rule == "adaptive":
        hi = order[conf[order] >= tau]
        return hi if len(hi) >= n else order[:n]
    chosen = []
    for i in order.tolist():
        if any(abs(i - j) == 1 for j in chosen):
            continue
        if rule == "dependency" and i > 0 and not x_vis[i - 1]:
            continue
        chosen.append(i)
        if len(chosen) == n:
            break
    if len(chosen) < n:                         # fall back to confidence order for the remainder
        for i in order.tolist():
            if i not in chosen:
                chosen.append(i)
            if len(chosen) == n:
                break
    return torch.tensor(chosen, dtype=masked_idx.dtype, device=masked_idx.device)


@torch.no_grad()
def sample_block_fixed(model, cache, P, K, steps, mask_id, schedule="linear", unmask="confidence", remask="none",
                       remask_budget=2, remask_start=2, tau_unmask=0.9, tau_remask=0.5, trust=None, trace=False,
                       early_stop=True, feature_log=None, gold=None):
    """Returns (tokens list[int], info).  feature_log (list) receives per-step dicts with the feature matrix, the
    current tokens, predictions and (if gold is given) correctness labels — used by the trust diagnostic."""
    dev = model.device
    x = torch.full((K,), mask_id, dtype=torch.long, device=dev)
    state = dict(prev_pred=None, placed_step=torch.full((K,), -1, dtype=torch.long, device=dev),
                 conf_at=torch.zeros(K, device=dev), changes=torch.zeros(K, dtype=torch.long, device=dev),
                 nb_changed=torch.zeros(K, dtype=torch.bool, device=dev), step=0)
    forwards = 0
    n_remasked = 0
    per_step = []
    tr = []
    for s in range(1, steps + 1):
        state["step"] = s
        is_mask = x == mask_id
        n_mask = int(is_mask.sum())
        if n_mask == 0 and remask == "none" and early_stop:
            break
        g = max(n_mask / K, 1e-3)
        logits = model.denoise(cache, P, x[None], g).float()
        forwards += 1
        logits[:, mask_id] = float("-inf")
        probs = torch.softmax(logits, -1)
        feat, pred, conf = step_features(probs, x, mask_id, state)
        if state["prev_pred"] is not None:
            state["changes"] += (pred != state["prev_pred"]).long()
        if feature_log is not None:
            row = dict(step=s, tokens=x.tolist(), pred=pred.tolist(), feat=feat.cpu(), is_mask=is_mask.cpu())
            if gold is not None:
                gt = torch.tensor(gold, device=dev)
                row["tok_correct"] = (x == gt).cpu()           # visible: placed token right?
                row["pred_correct"] = (pred == gt).cpu()       # prediction right?
            feature_log.append(row)
        last = s == steps
        # ---- unmask
        target_masks = 0 if last else int(round(K * gamma(1.0 - s / steps, schedule)))
        n_unmask = max(0, n_mask - target_masks)
        masked_idx = is_mask.nonzero().flatten()
        x_vis = ~is_mask
        sel = _select_unmask(masked_idx, conf, x_vis, n_unmask, unmask, K, tau_unmask) if not last else masked_idx
        changed = torch.zeros(K, dtype=torch.bool, device=dev)
        if len(sel):
            x[sel] = pred[sel]
            state["placed_step"][sel] = s
            state["conf_at"][sel] = conf[sel]
            changed[sel] = True
        # ---- remask (visible positions placed at an earlier step)
        remasked = []
        if remask != "none" and not last and s >= remask_start:
            cand = (x_vis & (state["placed_step"] < s)).nonzero().flatten()
            if len(cand):
                p_tok = feat[cand, FEATURES.index("p_tok")]
                agree = feat[cand, FEATURES.index("agree")] > 0.5
                if remask == "disagree":
                    keep = ~agree; sc = -conf[cand]                        # most confident disagreement first
                elif remask == "lowprob":
                    keep = p_tok < tau_remask; sc = p_tok
                elif remask == "stability":
                    prev_dis = state.get("prev_disagree")
                    keep = (~agree) & (prev_dis[cand] if prev_dis is not None else torch.zeros_like(agree)); sc = -conf[cand]
                elif remask == "dependency":
                    keep = (~agree) & state["nb_changed"][cand]; sc = -conf[cand]
                elif remask == "trust":
                    t = trust(feat[cand]).to(dev)
                    keep = t < tau_remask; sc = t
                else:
                    raise ValueError(remask)
                c = cand[keep]; score = sc[keep]
                if len(c):
                    c = c[score.argsort()[:remask_budget]]
                    x[c] = mask_id
                    state["placed_step"][c] = -1
                    changed[c] = True
                    remasked = c.tolist()
                    n_remasked += len(remasked)
        # ---- running state for the next step
        state["prev_disagree"] = (pred != x) & (x != mask_id)
        nb = torch.zeros(K, dtype=torch.bool, device=dev)
        nb[1:] |= changed[:-1]
        nb[:-1] |= changed[1:]
        state["nb_changed"] = nb
        state["prev_pred"] = pred
        per_step.append(dict(step=s, fixed=int((x != mask_id).sum()), masked=int((x == mask_id).sum()), unmasked=len(sel), remasked=len(remasked)))
        if trace:
            tr.append(dict(step=s, tokens=x.tolist(), pred=pred.tolist(), conf=[round(c, 3) for c in conf.tolist()], remasked=remasked))
    assert int((x == mask_id).sum()) == 0
    return x.tolist(), dict(forwards=forwards, remasked=n_remasked, per_step=per_step, trace=tr if trace else None)
