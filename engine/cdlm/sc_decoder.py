"""Self-correcting blockwise masked-diffusion decoder (SHIGERU Phase D, docs/pure_diffusion_research.md section 6).

Same block structure, forward and commit as block_decoder.py (K masks after the committed prefix, observed-first rows
[observed asc | masks asc] on top of the cache, one natural-order re-encoding per committed block, no AR logits ever
read), but a token placed in the block is TENTATIVE: it can be re-masked and re-predicted in a later step, and it is
only final when the block is committed.  Nothing is fixed by construction; what limits the revisions is a budget.

Why re-masking is the only self-check available here: a filled row's own logits are the next-token (AR) head of the
model, so the confidence of the token *at* a filled position cannot be read from that row without consulting the AR
head.  Re-masking the position and reading the mask row instead asks the model the same question it was trained on —
"what is at p given the observed tokens around it" — now with more observed context than when the token was first
placed (later fills, including tokens on its right).  A re-prediction that repeats the token is a confirmation; one
that differs is a correction.  Both are logged per position.

Per block (K slots; status per slot: MASK / TENTATIVE / CONFIRMED; per slot: token, conf (max prob at its last
prediction), ent (entropy at its last prediction), agree (consecutive identical re-predictions), filled_at):
  step t = 0, 1, ...:
    1. choose R = the TENTATIVE slots to re-evaluate now (`remask` strategy, below); they are forwarded as masks.
    2. forward rows [CONFIRMED + TENTATIVE\\R asc | MASK ∪ R asc]; read the mask rows only.
    3. slots in R: new token / conf / ent.  Same token -> agree += 1 (CONFIRMED when agree >= confirm_k); a different
       token -> correction: the token is replaced, agree = 0, stays TENTATIVE.  (`stats["corrections"]`,
       `stats["reeval"]`.)  A re-evaluated slot whose new token differs is NOT compared against ground truth here —
       the diagnostic bench/diag_selfcorrect.py does that with teacher-forced blocks.
    4. slots in MASK: fill as block_decoder does (anchor_leftmost + confidence >= tau, or a fixed plan of n_t fills per
       step, confidence order); new tokens are TENTATIVE with agree = 0 (confirm_k = 0 makes them CONFIRMED at once,
       which with remask="none" is exactly block_decoder.block_generate).
    5. stop when every slot is CONFIRMED, or when t + 1 >= max_steps (everything TENTATIVE is then accepted), or when
       the adaptive stop rule fires (`stop`: "confirmed" (default) | "conf": no MASK left and every TENTATIVE slot has
       conf >= stop_tau | "stable": no MASK left and no correction happened in this step).
  Remask strategies (`remask`, applied to the TENTATIVE slots at the start of a step; the slots filled in the previous
  step are eligible — that is the point):
    "none"                : never (control group)
    "all"                 : every TENTATIVE slot, every step (full re-evaluation until confirmed)
    "lowconf:<tau>"       : TENTATIVE slots with conf < tau (confidence at their last prediction)
    "entropy:<h>"         : TENTATIVE slots with ent > h nats
    "random:<p>"          : each TENTATIVE slot with probability p (torch.Generator)
    "scheduled:<f0>"      : the ceil(f0 * (1 - t / max_steps) * n_tentative) lowest-confidence TENTATIVE slots
    "drop:<d>"            : confidence-drop — TENTATIVE slots whose conf fell by more than d since their first
                            prediction, plus every slot with agree == 0 that has never been re-evaluated (so each
                            placement is checked at least once); the "drop" is measured on re-evaluations.
  `max_remask_frac` caps |R| at that fraction of K per step (lowest confidence first), so a step never spends most of
  its rows re-checking.  `min_reeval_steps` forces the block to run at least that many steps after the last fill
  before accepting (gives every placement at least one re-evaluation when the strategy selects it).
Stats (added to block_decoder's): reeval (slots re-evaluated), corrections (re-evaluations that changed the token),
confirmed_by_agree, accepted_tentative (tentative slots accepted at max_steps / stop rule), per-block steps, and with
trace=True per step: filled, reevaluated, corrected (slot, old, new), confirmed.
"""
import math

import torch

from .block_decoder import plan_fills, _rank
from .decoder import entropy_from_logits, margin_from_logits

__all__ = ["REMASK", "STOP_RULES", "parse_remask", "denoise_block", "sc_generate"]

REMASK = ("none", "all", "lowconf", "entropy", "random", "scheduled", "drop")
STOP_RULES = ("confirmed", "conf", "stable")
MASK, TENT, CONF = 0, 1, 2


def parse_remask(spec):
    """'none' | 'all' | 'lowconf:0.9' | 'entropy:1.0' | 'random:0.3' | 'scheduled:0.5' | 'drop:0.1' -> (kind, value)."""
    if spec is None:
        return ("none", None)
    if isinstance(spec, tuple):
        return spec
    kind, _, val = str(spec).partition(":")
    if kind not in REMASK:
        raise ValueError(f"remask={spec!r}, expected one of {REMASK}")
    if kind in ("none", "all"):
        return (kind, None)
    if val == "":
        raise ValueError(f"remask={spec!r} needs a value")
    return (kind, float(val))


def _choose_reeval(status, conf, ent, first_conf, agree, reevals, kind, val, t, max_steps, K, max_remask_frac,
                   generator):
    tent = [i for i in range(K) if status[i] == TENT]
    if not tent or kind == "none":
        return []
    if kind == "all":
        R = tent
    elif kind == "lowconf":
        R = [i for i in tent if conf[i] < val]
    elif kind == "entropy":
        R = [i for i in tent if ent[i] > val]
    elif kind == "random":
        draws = torch.rand(len(tent), generator=generator).tolist()
        R = [i for i, d in zip(tent, draws) if d < val]
    elif kind == "scheduled":
        n = int(math.ceil(val * max(0.0, 1.0 - t / max(1, max_steps)) * len(tent)))
        R = sorted(tent, key=lambda i: conf[i])[:n]
    elif kind == "drop":
        R = [i for i in tent if (reevals[i] == 0 and agree[i] == 0) or (first_conf[i] - conf[i] > val)]
    else:
        raise ValueError(kind)
    cap = int(math.floor(max_remask_frac * K)) if max_remask_frac is not None else K
    if len(R) > cap:
        R = sorted(R, key=lambda i: conf[i])[:cap]
    return sorted(R)


@torch.no_grad()
def denoise_block(fwd, cur, K, mask_id, *, steps=None, schedule="linear", order="confidence", tau=0.99, min_fill=1,
                  anchor_leftmost=True, remask="none", confirm_k=1, max_steps=None, max_remask_frac=0.5,
                  min_reeval_steps=0, stop="confirmed", stop_tau=0.99, greedy=True, temperature=1.0, generator=None,
                  trace=False, prefix_rows=None):
    """Denoise ONE block of K slots at logical positions cur..cur+K-1 and return (tokens[K], info).

    fwd(ids: list[int], pos: list[int], rows: list[int]) -> float32 logits [len(rows), V] over the rows of a forward of
    `ids` at positions `pos` on top of the committed prefix (the caller binds the cache state and the write policy).
    prefix_rows: optional (ids, pos) of rows to prepend (the previous block's natural-order re-encoding, merged into
    this block's first forward; the caller's cache_len must then be the start of those rows).
    steps=None: adaptive (anchor + confidence >= tau); steps=S: fixed plan of n_t fills per step (plan_fills), the
    anchor is added on top when anchor_leftmost.  max_steps defaults to K + 1 (adaptive) or S + 2 (fixed).
    """
    kind, val = parse_remask(remask)
    if stop not in STOP_RULES:
        raise ValueError(f"stop={stop!r}, expected one of {STOP_RULES}")
    plan = plan_fills(K, steps, schedule) if steps is not None else None
    if max_steps is None:
        max_steps = (len(plan) + 2) if plan else K + 1
    max_steps = max(1, int(max_steps))
    slots = [mask_id] * K
    status = [MASK] * K
    conf = [0.0] * K
    first_conf = [0.0] * K
    ent = [0.0] * K
    agree = [0] * K
    reevals = [0] * K
    filled_at = [-1] * K
    tr = []
    n_reeval = n_corr = n_confirm = 0
    forwards = 0
    last_fill_step = -1
    t = 0
    while True:
        R = _choose_reeval(status, conf, ent, first_conf, agree, reevals, kind, val, t, max_steps, K, max_remask_frac,
                           generator)
        for i in R:
            status[i] = MASK                                   # forwarded as a mask this step (token kept aside)
        obs_idx = [i for i in range(K) if status[i] != MASK]
        msk_idx = [i for i in range(K) if status[i] == MASK]
        order_idx = obs_idx + msk_idx
        ids = [slots[i] if status[i] != MASK else mask_id for i in order_idx]
        pos = [cur + i for i in order_idx]
        n_lead = len(obs_idx)
        if prefix_rows is not None:
            pids, ppos = prefix_rows
            ids, pos = list(pids) + ids, list(ppos) + pos
            n_lead += len(pids)
            prefix_rows = None
        logits = fwd(ids, pos, list(range(n_lead, len(ids))))
        forwards += 1
        probs = torch.softmax(logits.float() / (temperature if not greedy else 1.0), dim=-1)
        toks = logits.argmax(-1) if greedy else torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        pconf = probs.gather(-1, toks[:, None]).squeeze(-1)
        pent = entropy_from_logits(logits)
        pmargin = margin_from_logits(logits)
        toks_l, conf_l, ent_l = toks.tolist(), pconf.tolist(), pent.tolist()
        row_of = {s: k for k, s in enumerate(msk_idx)}
        # 3. re-evaluated slots
        corrected, confirmed_now = [], []
        for i in R:
            k = row_of[i]
            new, c, h = int(toks_l[k]), float(conf_l[k]), float(ent_l[k])
            n_reeval += 1
            reevals[i] += 1
            if new == slots[i]:
                agree[i] += 1
            else:
                corrected.append((i, slots[i], new))
                n_corr += 1
                slots[i] = new
                agree[i] = 0
            conf[i], ent[i] = c, h
            if agree[i] >= confirm_k:
                status[i] = CONF
                confirmed_now.append(i)
                n_confirm += 1
            else:
                status[i] = TENT
        # 4. fresh masks
        Rset = set(R)
        fresh = [i for i in msk_idx if i not in Rset]
        filled_now = []
        if fresh:
            fk = [row_of[i] for i in fresh]
            fconf = pconf[fk]
            fent = pent[fk]
            fmargin = pmargin[fk]
            rank = _rank(fconf, fent, fmargin, order, generator, len(fresh), logits.device).tolist()
            if plan is not None:
                n_now = plan[t] if t < len(plan) else len(fresh)
                sel = rank[:min(max(1, n_now), len(fresh))]
            else:
                ok = (fconf >= float(tau)).tolist()
                sel = [r for r in rank if ok[r]]
                if len(sel) < min_fill:
                    sel = rank[:min(min_fill, len(fresh))]
            if anchor_leftmost and 0 not in sel:
                sel = [0] + sel                                  # fresh[0] is the leftmost remaining mask
            for r in sel:
                i = fresh[r]
                k = fk[r]
                slots[i] = int(toks_l[k])
                conf[i] = first_conf[i] = float(conf_l[k])
                ent[i] = float(ent_l[k])
                agree[i] = 0
                filled_at[i] = t
                status[i] = CONF if confirm_k <= 0 else TENT
                if confirm_k <= 0:
                    n_confirm += 1
                filled_now.append(i)
            if filled_now:
                last_fill_step = t
        if trace:
            tr.append(dict(step=t, filled=sorted(filled_now), reevaluated=list(R), corrected=corrected,
                           confirmed=sorted(confirmed_now), tokens=list(slots), status=list(status),
                           conf=[round(c, 4) for c in conf]))
        t += 1
        # 5. stop rules
        n_mask = sum(1 for s in status if s == MASK)
        n_tent = sum(1 for s in status if s == TENT)
        if n_mask == 0 and n_tent == 0:
            break
        if t >= max_steps:
            break
        if n_mask == 0 and t - 1 - last_fill_step >= min_reeval_steps:
            if stop == "conf" and all(conf[i] >= stop_tau for i in range(K) if status[i] == TENT):
                break
            if stop == "stable" and not corrected and t - 1 > last_fill_step:
                break
    accepted_tent = sum(1 for s in status if s == TENT)
    info = dict(steps=t, forwards=forwards, reeval=n_reeval, corrections=n_corr, confirmed_by_agree=n_confirm,
                accepted_tentative=accepted_tent, trace=tr if trace else None)
    return list(slots), info


@torch.no_grad()
def sc_generate(model, tok, prompt_ids, max_new_tokens, block=32, mask_id=None, stop_ids=None, static=False,
                use_graph=False, merge_commit=True, prefill_chunk=512, trace=False, **kw):
    """Generate with the self-correcting block denoiser (denoise_block kwargs via **kw).  Returns (ids, stats) in the
    format of block_decoder.block_generate plus the self-correction counters."""
    if mask_id is None:
        mask_id = model.cfg.mask_id
    if stop_ids is None:
        stop_ids = [tok.eos_token_id] if tok is not None and tok.eos_token_id is not None else []
    stop_ids = set(int(s) for s in stop_ids)
    dev = model.device
    prompt = torch.as_tensor(prompt_ids, dtype=torch.long, device=dev).flatten()
    prompt_len = int(prompt.numel())
    assert prompt_len > 0 and block >= 1 and max_new_tokens >= 0
    assert prompt_len + max_new_tokens + block <= model.kv.max_len
    _, cache_len = model.prefill(prompt, 0, chunk=prefill_chunk)
    forwards = -(-prompt_len // prefill_chunk)
    cur = prompt_len
    out = []
    n_steps = reeval = corr = confirmed = acc_tent = 0
    steps_per_block, traces = [], []
    finish = "length" if max_new_tokens == 0 else None
    pending = None

    while finish is None:
        K = min(block, max_new_tokens - len(out))
        state = {"cl": cur if pending is None else cur - len(pending)}     # first forward may carry the merged commit rows
        prefix_rows = (pending, list(range(cur - len(pending), cur))) if pending is not None else None

        def fwd(ids, pos, rows, _state=state, _cur=cur):
            out = model.run(torch.tensor(ids, dtype=torch.long), torch.tensor(pos, dtype=torch.long), _state["cl"], rows,
                            static=static, use_graph=use_graph)
            _state["cl"] = _cur                 # the previous block's rows are in the cache now: later steps start at cur
            return out

        toks, info = denoise_block(fwd, cur, K, mask_id, trace=trace, prefix_rows=prefix_rows, **kw)
        pending = None
        n_steps += info["steps"]
        forwards += info["forwards"]
        reeval += info["reeval"]
        corr += info["corrections"]
        confirmed += info["confirmed_by_agree"]
        acc_tent += info["accepted_tentative"]
        steps_per_block.append(info["steps"])
        if trace:
            traces.append(dict(block=len(steps_per_block) - 1, start=cur, steps=info["trace"]))
        keep = K
        for j in range(K):
            if toks[j] in stop_ids:
                keep, finish = j + 1, "stop"
                break
        committed = toks[:keep]
        out += committed
        cur += keep
        if len(out) >= max_new_tokens:
            finish = finish or "length"
        if finish is not None or not merge_commit:
            model.run(torch.tensor(committed, dtype=torch.long), torch.tensor(list(range(cur - keep, cur)), dtype=torch.long),
                      cur - keep, [], static=static, use_graph=use_graph)
            forwards += 1
        else:
            pending = committed
    stats = dict(steps=n_steps, forwards=forwards, generated=len(out), tokens_per_step=(len(out) / n_steps if n_steps else 0.0),
                 tokens_per_forward=(len(out) / forwards if forwards else 0.0), blocks=len(steps_per_block),
                 steps_per_block=steps_per_block, reeval=reeval, corrections=corr, confirmed_by_agree=confirmed,
                 accepted_tentative=acc_tent, remasked=reeval, finish_reason=finish, block_size=block, prompt_len=prompt_len,
                 cache_len=cur, verify=False)
    if trace:
        stats["trace"] = traces
    return out, stats
