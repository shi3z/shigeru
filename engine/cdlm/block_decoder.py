"""Pure blockwise masked-diffusion decoder (no AR verification, no speculative drafts) on top of FastQwen3.

docs/pure_diffusion_research.md.  The generation is a sequence of blocks of K positions after the committed prefix.
A block starts as K [MASK] rows and is denoised in place over several steps; when every position holds a token the
block is committed (appended to the output, its K/V written to the cache in natural order) and the next block starts.

Why the existing weights support this (audit, docs/pure_diffusion_research.md section 1): the training forward of
engine/cdlm/data.py puts every block in observed-first order [observed tokens asc | masks asc] under a causal mask, so a
[MASK] row attends the clean prefix (earlier blocks), ALL observed tokens of its own block — on its left and on its
right in logical position, RoPE keeps the original positions — and the masks before it.  That is exactly the context
this decoder gives a mask row at every step: tokens filled anywhere in the block are observed, the rest are masks.
Mask rows never see later masks, which carries no information anyway (a [MASK] embedding at a known position), so the
"bidirectional attention inside the block" of a MaskGIT/LLaDA-style sampler is realised here as observed-first causal
attention, with no change to the architecture or the weights.  In `mask_mode="suffix"` training (the final stage-2
checkpoints) a mask never had an observed token on its right, so right-context conditioning is out of distribution
for those checkpoints — a measured question, not an assumption (section 2 of the research doc).

One denoising step on a block with `filled` (set of slots holding tokens) and the rest masks:
  1. forward rows [filled slots asc | mask slots asc] with position ids cur + slot on top of the cache (cache_len =
     cur, the committed prefix); the causal window mask of FastQwen3 gives each mask row the prefix, all filled rows
     and the masks before it.  Only the mask rows' logits are read (logits of a filled row predict the NEXT token —
     they are the AR head and are NOT used: this decoder never verifies).
  2. per mask row: greedy token (or a temperature sample), confidence = max softmax prob (float32), entropy, margin.
  3. choose which masks to fill now (`order` x `schedule`), see below; write the tokens; record their confidence.
  4. optional remasking of previously filled tokens (`remask`), so that an early guess can be revised later: the
     remasked slots become masks again and are re-predicted at the next step.
  5. a block is complete when no slot is a mask (and the remask rule yields none); the step that completes it is the
     last denoising step of the block.
Commit: the block's K/V in the cache were written in observed-first order, with masks in context for some rows, so
they are not the K/V of a causal LM over the final tokens.  The block is therefore re-encoded ONCE in natural order
(`commit forward`, K rows, write=True) — merged into the first forward of the next block when `merge_commit=True`
(rows [block b tokens asc | block b+1 masks], one launch of 2K rows), else a separate forward.  Both are counted:
stats["forwards"] is every model launch, stats["steps"] is the number of denoising steps (mask predictions).
The streaming decoder (decoder.py) avoids this re-encoding by committing only the leftmost run; the price there is
that a token filled ahead of the frontier is never revised and later masks never see tokens on their right.

Selection (`order`): which masks are candidates to fill first
  "confidence"  descending max prob (MaskGIT / LLaDA low-confidence remasking's dual)
  "entropy"     ascending entropy of the row's distribution
  "margin"      descending top1 - top2 probability
  "random"      a random permutation (torch.Generator `generator`), the no-information baseline
  "leftmost"    ascending position (the diagnostic "one token per step from the left" when n=1 per step)
Schedule (`steps`, `schedule`): how many of the remaining masks to fill at denoising step t of a block of K
  steps=S, schedule="linear"  : K/S per step (rounded so that S steps complete the block)   [MaskGIT linear]
  steps=S, schedule="cosine"  : MaskGIT cosine — few tokens first, many at the end
  steps=S, schedule="geometric": many first, few at the end (the reverse of cosine)
  steps=None, tau=t           : adaptive — every mask whose confidence >= tau, at least one (`min_fill`) per step
  n_t is always clipped to the remaining masks; every step fills >= 1 so a block of K takes <= K steps.
Remasking (`remask`, applied in step 4 after the fills of this step; a slot filled THIS step is never remasked)
  None                        : never (tokens are final once filled)
  ("lowconf", frac, until)    : after each step t < until (fraction of the schedule), re-mask the ceil(frac * n_filled)
                                filled tokens with the lowest recorded confidence, where the recorded confidence is
                                re-estimated each step from the *current* prediction of their row is NOT available
                                (a filled row has no own-token logits), so it is the confidence at fill time;
  ("tau", c, until)           : re-mask filled tokens whose fill-time confidence < c, same `until` cut-off.
  `until` < 1 guarantees termination: the last (1-until) fraction of the planned steps never remask, and after the
  planned steps are exhausted the schedule fills everything left.  stats["remasked"] counts re-masked tokens.
EOS: when a stop id is filled, every slot to its right is dropped at commit (they are re-encoded as nothing) and the
generation ends with finish_reason "stop"; the budget `max_new_tokens` ends it with "length" (the last block is
shorter).  The model is never asked anything by its AR head, so the output is not the AR output.

Returns (new_token_ids, stats): steps (denoising steps), forwards (launches incl. prefill chunks and commit
forwards), window_rows (rows forwarded in denoising steps), generated, tokens_per_step = generated / steps,
tokens_per_forward, blocks, steps_per_block (list), filled_per_step ({n: count}), remasked, finish_reason,
filled_positions (per step, absolute positions filled), block_size, prompt_len, cache_len, and `trace` when
trace=True (per step: block, step index, slots filled, their confidences, slots remasked).
"""
import math

import torch

from .decoder import entropy_from_logits, margin_from_logits

__all__ = ["ORDERS", "SCHEDULES", "plan_fills", "block_generate"]

ORDERS = ("confidence", "entropy", "margin", "random", "leftmost")
SCHEDULES = ("linear", "cosine", "geometric")


def plan_fills(K, steps, schedule="linear"):
    """Number of tokens to fill at each of `steps` denoising steps of a block of K masks (sums to K, each >= 1).
    steps > K is clipped to K.  cosine: MaskGIT's gamma(t) = cos(pi/2 * t) mask ratio, i.e. few fills first.
    geometric: the reverse of cosine (many first).  linear: as even as possible."""
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule={schedule!r}, expected one of {SCHEDULES}")
    S = max(1, min(int(steps), K))
    if schedule == "linear":
        base, extra = divmod(K, S)
        plan = [base + (1 if i < extra else 0) for i in range(S)]
    else:
        # cumulative number of UNMASKED tokens after step t of S: K * (1 - cos(pi/2 * t/S)) (cosine)
        cum = [K * (1.0 - math.cos(math.pi / 2 * t / S)) for t in range(S + 1)]
        plan = [int(round(cum[t + 1])) - int(round(cum[t])) for t in range(S)]
        if schedule == "geometric":
            plan = plan[::-1]
        # every step must fill >= 1: steal from the largest steps
        for i in range(S):
            while plan[i] < 1:
                j = max(range(S), key=lambda k: plan[k])
                plan[j] -= 1
                plan[i] += 1
    assert sum(plan) == K and all(p >= 1 for p in plan), (K, steps, schedule, plan)
    return plan


def _rank(conf, ent, margin, order, generator, n_masks, device):
    """Indices of the mask rows (ascending slot order) sorted by the selection rule, best first."""
    if order == "confidence":
        return torch.argsort(conf, descending=True)
    if order == "entropy":
        return torch.argsort(ent)
    if order == "margin":
        return torch.argsort(margin, descending=True)
    if order == "random":
        return torch.randperm(n_masks, generator=generator).to(device)
    if order == "leftmost":
        return torch.arange(n_masks, device=device)
    raise ValueError(f"order={order!r}, expected one of {ORDERS}")


@torch.no_grad()
def block_generate(model, tok, prompt_ids, max_new_tokens, block=32, steps=8, schedule="linear", order="confidence",
                   tau=None, min_fill=1, remask=None, mask_id=None, stop_ids=None, greedy=True, temperature=1.0,
                   generator=None, static=False, use_graph=False, merge_commit=True, prefill_chunk=512, trace=False):
    """Decode one sequence with blockwise masked diffusion (module docstring).  model: FastQwen3; tok only defaults
    stop_ids to [eos].  steps=None switches to the adaptive schedule (tau required).  static/use_graph: the bucketed /
    CUDA-graph forward of FastQwen3.run (a block of K rows needs K <= the largest bucket; 2K with merge_commit)."""
    if order not in ORDERS:
        raise ValueError(f"order={order!r}, expected one of {ORDERS}")
    if steps is None and tau is None:
        raise ValueError("steps=None (adaptive) needs tau")
    kind, val, until = None, 0.0, 0.0
    if remask is not None:
        kind, val, until = remask
        if kind not in ("lowconf", "tau") or not (0.0 <= float(until) <= 1.0):
            raise ValueError(f"remask={remask!r}")
    if mask_id is None:
        mask_id = model.cfg.mask_id
        if mask_id is None:
            raise ValueError("mask_id not given and the checkpoint has no cdlm_mask_token_id")
    if stop_ids is None:
        stop_ids = [tok.eos_token_id] if tok is not None and tok.eos_token_id is not None else []
    stop_ids = set(int(s) for s in stop_ids)
    dev = model.device
    prompt = torch.as_tensor(prompt_ids, dtype=torch.long, device=dev).flatten()
    prompt_len = int(prompt.numel())
    assert prompt_len > 0 and block >= 1 and max_new_tokens >= 0
    assert prompt_len + max_new_tokens + block <= model.kv.max_len, "cache too small for prompt + budget + block"

    _, cache_len = model.prefill(prompt, 0, chunk=prefill_chunk)
    forwards = -(-prompt_len // prefill_chunk)
    assert cache_len == prompt_len
    cur = prompt_len
    out = []
    n_steps = window_rows = remasked_total = 0
    filled_hist, filled_positions, steps_per_block, tr = {}, [], [], []
    finish = "length" if max_new_tokens == 0 else None
    pending_commit = None                        # tokens of the previous block, to be re-encoded (merge_commit)

    def run(ids, pos, cl, rows):
        return model.run(torch.tensor(ids, dtype=torch.long), torch.tensor(pos, dtype=torch.long), cl, rows,
                         static=static, use_graph=use_graph)

    while finish is None:
        K = min(block, max_new_tokens - len(out))
        assert K >= 1
        slots = [mask_id] * K
        is_mask = [True] * K
        conf_at_fill = [0.0] * K
        plan = plan_fills(K, steps, schedule) if steps is not None else None
        n_planned = len(plan) if plan else K
        t = 0
        while any(is_mask):
            filled_idx = [i for i in range(K) if not is_mask[i]]
            mask_idx = [i for i in range(K) if is_mask[i]]
            order_idx = filled_idx + mask_idx
            ids = [slots[i] for i in order_idx]
            pos = [cur + i for i in order_idx]
            if pending_commit is not None:       # merged commit forward of the previous block (natural order)
                ids = pending_commit + ids
                pos = list(range(cur - len(pending_commit), cur)) + pos
                cl = cur - len(pending_commit)
            else:
                cl = cur
            n_lead = len(ids) - len(mask_idx)
            rows = list(range(n_lead, len(ids)))
            logits = run(ids, pos, cl, rows)     # [n_masks, V] float32
            pending_commit = None
            forwards += 1
            n_steps += 1
            window_rows += len(ids)
            probs = torch.softmax(logits.float() / (temperature if not greedy else 1.0), dim=-1)
            if greedy:
                toks = logits.argmax(-1)
            else:
                toks = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            conf = probs.gather(-1, toks[:, None]).squeeze(-1)
            ent = entropy_from_logits(logits)
            margin = margin_from_logits(logits)
            rank = _rank(conf, ent, margin, order, generator, len(mask_idx), logits.device)
            if plan is not None:
                n_now = plan[t] if t < len(plan) else len(mask_idx)
                n_now = min(max(1, n_now), len(mask_idx))
                sel = rank[:n_now].tolist()
            else:
                ok = (conf >= float(tau))
                sel = [k for k in rank.tolist() if bool(ok[k])]
                if len(sel) < min_fill:
                    sel = rank[:min(min_fill, len(mask_idx))].tolist()
            toks_l, conf_l = toks.tolist(), conf.tolist()
            filled_now = []
            for k in sel:
                s = mask_idx[k]
                slots[s] = int(toks_l[k])
                is_mask[s] = False
                conf_at_fill[s] = float(conf_l[k])
                filled_now.append(s)
            # remasking of earlier fills (never the ones from this step)
            rem = []
            if remask is not None and t + 1 < n_planned and (t + 1) < until * n_planned:
                cands = [i for i in range(K) if not is_mask[i] and i not in filled_now]
                if kind == "lowconf" and cands:
                    n_rem = int(math.ceil(float(val) * len(cands)))
                    rem = sorted(cands, key=lambda i: conf_at_fill[i])[:n_rem]
                elif kind == "tau":
                    rem = [i for i in cands if conf_at_fill[i] < float(val)]
                for i in rem:
                    slots[i] = mask_id
                    is_mask[i] = True
                remasked_total += len(rem)
            filled_hist[len(sel)] = filled_hist.get(len(sel), 0) + 1
            filled_positions.append([cur + s for s in sorted(filled_now)])
            if trace:
                tr.append(dict(block=len(steps_per_block), step=t, filled=sorted(filled_now),
                               conf=[round(conf_at_fill[s], 4) for s in sorted(filled_now)], remasked=sorted(rem),
                               n_masks_before=len(mask_idx)))
            t += 1
        steps_per_block.append(t)
        # commit: cut at the first stop id, append, schedule the natural-order re-encoding
        keep = K
        for j in range(K):
            if slots[j] in stop_ids:
                keep, finish = j + 1, "stop"
                break
        committed = slots[:keep]
        out += committed
        cur += keep
        if len(out) >= max_new_tokens:
            finish = finish or "length"
        if finish is not None or not merge_commit:
            run(committed, list(range(cur - keep, cur)), cur - keep, [])      # re-encode the block, write its K/V
            forwards += 1
        else:
            pending_commit = committed
    assert len(out) <= max_new_tokens and cur == prompt_len + len(out)
    stats = dict(steps=n_steps, forwards=forwards, window_rows=window_rows, generated=len(out),
                 tokens_per_step=(len(out) / n_steps if n_steps else 0.0),
                 tokens_per_forward=(len(out) / forwards if forwards else 0.0), blocks=len(steps_per_block),
                 steps_per_block=steps_per_block, filled_per_step=dict(sorted(filled_hist.items())),
                 remasked=remasked_total, finish_reason=finish, filled_positions=filled_positions, block_size=block,
                 prompt_len=prompt_len, cache_len=cur, verify=False)
    if trace:
        stats["trace"] = tr
    return out, stats
