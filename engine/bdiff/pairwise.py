"""Pairwise joint denoising for the two positions a fixed-budget step places (docs/pairwise_joint_diffusion.md §2).

At S = 8 the sampler places two tokens per forward from ONE set of marginals p(x_i | context), p(x_j | context); the
independent argmax ignores the dependency between the two.  Given top-k candidates per position (k = 2 -> 4 pairs,
k = 4 -> 16 pairs) the pair can be re-ranked by a joint score:

  independent    log p(a) + log p(b)                       (product of marginals == independent argmax; control)
  joint_fwd      log p(a) + log p(b | a placed at i)        chain rule under the model: ONE extra forward per
                                                           candidate a (k forwards), the other position still masked
  joint_both     0.5 * [joint_fwd + log p(b) + log p(a | b placed at j)]   symmetric, 2k extra forwards
  oracle         the pair equal to the reference (diagnostic only; needs gold)

The extra forwards are counted by the caller; these rules are the research ceiling of pair re-ranking.  A rule that
needs no extra forward must come from a learned compatibility score (§3); `pair_scores_independent` is the control
that any such score has to beat.
"""
import torch

__all__ = ["rerank_pair", "PAIR_RULES"]

PAIR_RULES = ("none", "joint_fwd", "joint_both")


@torch.no_grad()
def rerank_pair(model, cache, P, x, probs, i, j, mask_id, k=2, rule="joint_fwd"):
    """x [K] current block tokens (i and j masked), probs [K, V] of the current forward.  Returns
    (tok_i, tok_j, extra_forwards, info) where info has the candidate lists and the k*k score matrices."""
    ci = probs[i].topk(k).indices
    cj = probs[j].topk(k).indices
    lp_i = probs[i][ci].clamp_min(1e-12).log()
    lp_j = probs[j][cj].clamp_min(1e-12).log()
    indep = lp_i[:, None] + lp_j[None, :]
    extra = 0
    K = x.shape[0]
    g = max(int((x == mask_id).sum()) / K, 1e-3)
    # p(b | a): place a at i, read position j
    cond_j = torch.zeros(k, k, device=probs.device)
    for ai in range(k):
        x2 = x.clone(); x2[i] = ci[ai]
        lg = model.denoise(cache, P, x2[None], g).float(); extra += 1
        lg[:, mask_id] = float("-inf")
        pj = torch.softmax(lg[j], -1)
        cond_j[ai] = pj[cj].clamp_min(1e-12).log()
    fwd = lp_i[:, None] + cond_j
    score = fwd
    cond_i = None
    if rule == "joint_both":
        cond_i = torch.zeros(k, k, device=probs.device)
        for bj in range(k):
            x2 = x.clone(); x2[j] = cj[bj]
            lg = model.denoise(cache, P, x2[None], g).float(); extra += 1
            lg[:, mask_id] = float("-inf")
            pi = torch.softmax(lg[i], -1)
            cond_i[:, bj] = pi[ci].clamp_min(1e-12).log()
        bwd = lp_j[None, :] + cond_i
        score = 0.5 * (fwd + bwd)
    best = int(score.argmax())
    ai, bj = divmod(best, k)
    info = dict(cand_i=ci.tolist(), cand_j=cj.tolist(), indep=indep.cpu(), fwd=fwd.cpu(), bwd=(lp_j[None, :] + cond_i).cpu() if cond_i is not None else None,
                score=score.cpu(), cond_j=cond_j.cpu())
    return int(ci[ai]), int(cj[bj]), extra, info
