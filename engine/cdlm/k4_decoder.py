"""Small-block (K = 4 by default) pure diffusion decoder with leave-one-out refinement — SHIGERU Phase E
(docs/four_token_diffusion.md) and the K-scaling failure analysis (docs/k_scaling_failure_analysis.md).
No AR logits are read anywhere; no draft model.

Forward types (both on the committed prefix, cache_len = cur; FastQwen3.forward / run / run_masked)
  standard  : rows [observed asc | masks asc] under the causal window mask (prefix + rows before you).  Used for the
              first forward of a block (merged with the natural-order re-encoding of the previous block as in
              block_decoder.py, so the commit costs no extra launch), for the later groups of a split initialisation
              and for the second step of "conf2".
  probe     : leave-one-out refinement in ONE forward of K groups x K rows.  Group i = [placed t_j for j != i, ascending,
              causal among themselves and over the prefix | probe mask m_i at position cur+i, attending the prefix, the
              group's K-1 placed rows and itself].  Groups never see each other.  This is exactly the training layout of
              engine/cdlm/data.py for the block with position i masked and the others observed, so m_i is the model's
              trained conditional p(x_i | prefix, the other K-1 tokens of the block) with observed context on BOTH
              sides.  A single shared set of placed rows would leak t_i into m_i through the causal encoding of
              t_{j>i}, which is why the groups are separate (tests/test_k4.py).  Probe forwards never write K/V that
              anything attends as committed context (plain path: write=False; static path: slots overwritten by the
              next natural-order re-encoding).  Cost: K^2 rows (16 / 36 / 64 for K = 4 / 6 / 8).
Strategies (`strategy`)
  "s1"          predict [M]*K once, place all K                                                          (brief A)
  "conf2"       place the ceil(K/2) most confident (anchor: the leftmost always), then the rest           (brief B)
  "refine"      place all K, then one probe round                                                       (brief C)
  "refineN"     probe rounds until no position changes, a cycle is detected, or max_rounds                (brief D/E)
  "conf2refine" conf2 followed by one probe round
  "seq"         leftmost one per forward (= AR; the diagnostic ceiling)
Adoption rule of a probe round (`adopt`; the K-scaling brief's update policies E1-E6)
  "always"      Jacobi: every position whose probe argmax differs is replaced
  "conf"        confidence-gated: replaced only if the probe's max prob exceeds the position's current confidence
  "nonadjacent" of the changed positions keep a set with no two adjacent (highest probe confidence first)
  "one"         a single change per round (the most confident), Gauss-Seidel style
  "two"         at most two changes per round (the two most confident)
  "oddeven"     adopt the changes at even positions, re-probe, adopt the changes at odd positions (2 forwards / round)
  "damped"      the probe distribution is averaged with the previous probe distribution of the same position
                (probability level, never token ids); the argmax of the mixture is adopted if it differs
Initialisation (`init`): None = one all-mask forward places every position; a list of group sizes, e.g. [3, 3] or
  [2, 2, 2], places the groups sequentially (forward g sees the groups < g as observed; forwards are all counted).
  `first` in the returned info is always the prediction of the first forward for every position.
Oracle seeds (`fixed`: {pos: token}, diagnostics only): observed in every forward, never changed, never counted.
EOS (`eos_policy`, `stop_ids`): "current" (a stop id anywhere in the final block cuts the block there);
  "suppress_first" (stop ids cannot be predicted in the FIRST forward; probes may introduce them later);
  "verify" (if the final block contains a stop id and the loop did not end in a quiet round, one extra probe round
  must leave the stop position unchanged, else refinement continues).
Oscillation: the block state after each round is hashed; a state seen before ends the loop (`outcome` = "cycle",
  `cycle_period`), no change ends it as "converged", exhausting max_rounds as "budget".
"""
import time as _time

import torch

from .decoder import entropy_from_logits

__all__ = ["STRATEGIES", "ADOPT", "EOS_POLICIES", "k4_denoise_block", "k4_generate", "probe_key_mask", "probe_rows",
           "probe_win_mask"]

STRATEGIES = ("s1", "conf2", "refine", "refineN", "conf2refine", "seq")
ADOPT = ("always", "conf", "nonadjacent", "one", "two", "oddeven", "damped")
EOS_POLICIES = ("current", "suppress_first", "verify")


def probe_rows(tokens, cur, mask_id):
    """(ids, pos, probe_row_indices) of the K-group leave-one-out layout for the placed `tokens`."""
    K = len(tokens)
    ids, pos, rows = [], [], []
    for i in range(K):
        for j in range(K):
            if j != i:
                ids.append(tokens[j]); pos.append(cur + j)
        ids.append(mask_id); pos.append(cur + i)
        rows.append(len(ids) - 1)
    return ids, pos, rows


def probe_win_mask(K, device=None):
    """bool [K*K, K*K]: the window-internal part of probe_key_mask (group-causal), for FastQwen3.run_masked."""
    T = K * K
    m = torch.zeros(T, T, dtype=torch.bool, device=device)
    for i in range(K):
        for r in range(K):
            m[i * K + r, i * K: i * K + r + 1] = True
    return m


def probe_key_mask(cache_len, K, device=None):
    """bool [K*K, cache_len + K*K]: group i occupies rows i*K .. i*K+K-1 (K-1 placed rows then the probe); every row
    sees the prefix and, within its group, the rows at or before it (causal), nothing of other groups."""
    T = K * K
    m = torch.zeros(T, cache_len + T, dtype=torch.bool, device=device)
    m[:, :cache_len] = True
    for i in range(K):
        base = cache_len + i * K
        for r in range(K):
            m[i * K + r, base: base + r + 1] = True
    return m


def _predict(logits, mask_id, suppress=None, return_probs=False):
    """argmax / max prob / entropy per row; the [MASK] token itself is never emitted; `suppress` ids are excluded."""
    lg = logits.float().clone()
    lg[:, mask_id] = float("-inf")
    if suppress:
        for s in suppress:
            lg[:, s] = float("-inf")
    probs = torch.softmax(lg, dim=-1)
    conf, tok = probs.max(-1)
    if return_probs:
        return tok.tolist(), conf.tolist(), entropy_from_logits(lg).tolist(), probs
    return tok.tolist(), conf.tolist(), entropy_from_logits(lg).tolist()


@torch.no_grad()
def k4_denoise_block(fwd_std, fwd_probe, cur, K, mask_id, strategy="refine", adopt="always", max_rounds=3,
                     anchor_leftmost=True, prefix_rows=None, trace=False, fixed=None, init=None, stop_ids=None,
                     eos_policy="current"):
    """Denoise one block of K positions cur..cur+K-1 (module docstring).
    fwd_std(ids, pos, rows) -> logits [len(rows), V]  (standard causal window forward, may write the cache)
    fwd_probe(ids, pos, K, rows) -> logits             (key mask probe_key_mask(cur, K); must not expose its K/V)
    Returns (tokens[K], info): forwards, probes, corrections, first / first_conf (first-forward prediction of every
    position), final_conf, rounds, outcome ("converged" | "cycle" | "budget" | "none"), cycle_period, eos_verified,
    changes (list of (round, pos, old, new)), trace."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy={strategy!r}, expected one of {STRATEGIES}")
    if adopt not in ADOPT:
        raise ValueError(f"adopt={adopt!r}, expected one of {ADOPT}")
    if eos_policy not in EOS_POLICIES:
        raise ValueError(f"eos_policy={eos_policy!r}")
    fixed = dict(fixed or {})
    stop_ids = set(stop_ids or [])
    tr, changes = [], []
    forwards = corrections = probes = 0
    tokens = [mask_id] * K
    conf = [0.0] * K
    for p_, t_ in fixed.items():
        tokens[p_], conf[p_] = t_, 1.0
    prev_probs = [None] * K                                  # for adopt == "damped"

    def std(obs_slots, mask_slots, first_call, suppress=None):
        nonlocal forwards
        ids = [tokens[s] for s in obs_slots] + [mask_id] * len(mask_slots)
        pos = [cur + s for s in obs_slots] + [cur + s for s in mask_slots]
        n_lead = len(obs_slots)
        if first_call and prefix_rows is not None:
            pids, ppos = prefix_rows
            ids, pos, n_lead = list(pids) + ids, list(ppos) + pos, n_lead + len(pids)
        forwards += 1
        return fwd_std(ids, pos, list(range(n_lead, len(ids))))

    def probe():
        nonlocal forwards, probes
        ids, pos, rows = probe_rows(tokens, cur, mask_id)
        forwards += 1
        probes += 1
        return fwd_probe(ids, pos, K, rows)

    free = [i for i in range(K) if i not in fixed]
    suppress_first = sorted(stop_ids) if eos_policy == "suppress_first" else None
    # ---- first forward: observed = fixed seeds, masks = the rest (the all-mask forward when nothing is fixed)
    logits = std(sorted(fixed), free, True, suppress_first)
    f_tok, f_conf, _ = _predict(logits, mask_id, suppress_first)
    first = list(tokens)
    first_conf = list(conf)
    for k, i in enumerate(free):
        first[i], first_conf[i] = f_tok[k], f_conf[k]
    if trace:
        tr.append(dict(kind="all_mask", pred=list(first), conf=[round(c, 4) for c in first_conf]))

    def place(idx):
        for i in idx:
            tokens[i], conf[i] = first[i], first_conf[i]

    def place_groups():
        """init: None -> place all free positions from the first forward; list -> sequential groups."""
        if not init:
            place(free)
            return
        groups, s = [], 0
        for g in init:
            groups.append([i for i in range(s, min(K, s + g)) if i not in fixed])
            s += g
        groups.append([i for i in range(s, K) if i not in fixed])     # anything the sizes did not cover
        groups = [g for g in groups if g]
        place(groups[0])
        for grp in groups[1:]:
            obs = sorted(i for i in range(K) if tokens[i] != mask_id)
            masks = [i for i in range(K) if tokens[i] == mask_id]
            lg = std(obs, masks, False)
            t2, c2, _ = _predict(lg, mask_id)
            for k, i in enumerate(masks):
                if i in grp:
                    tokens[i], conf[i] = t2[k], c2[k]
            if trace:
                tr.append(dict(kind="init_group", placed=obs, group=grp, pred=list(t2), conf=[round(c, 4) for c in c2]))

    rounds = 0
    outcome = "none"
    cycle_period = 0
    eos_verified = 0
    seen = {}

    def adopt_changes(cand, new, nconf, rnd):
        nonlocal corrections
        cand = [i for i in cand if i not in fixed]
        if adopt in ("one", "two") and cand:
            cand = sorted(sorted(cand, key=lambda i: -nconf[i])[: (1 if adopt == "one" else 2)])
        elif adopt == "nonadjacent" and cand:
            keep = []
            for i in sorted(cand, key=lambda i: -nconf[i]):
                if all(abs(i - j) > 1 for j in keep):
                    keep.append(i)
            cand = sorted(keep)
        changed = []
        for i in cand:
            changed.append((i, tokens[i], new[i]))
            changes.append((rnd, i, tokens[i], new[i]))
            tokens[i] = new[i]
        for i in range(K):
            if i not in fixed and new[i] == tokens[i]:
                conf[i] = nconf[i]
        corrections += len(changed)
        return changed

    def refine_round(rnd, parity=None):
        """One probe forward and its adoption; parity selects even/odd positions for 'oddeven'."""
        lg = probe()
        if adopt == "damped":
            new, nconf, _, probs = _predict(lg, mask_id, return_probs=True)
            mixed_new, mixed_conf = list(new), list(nconf)
            for i in range(K):
                if prev_probs[i] is not None:
                    pm = 0.5 * (prev_probs[i] + probs[i])
                    c_, t_ = pm.max(-1)
                    mixed_new[i], mixed_conf[i] = int(t_), float(c_)
                prev_probs[i] = probs[i].clone()
            new, nconf = mixed_new, mixed_conf
        else:
            new, nconf, _ = _predict(lg, mask_id)
        cand = [i for i in range(K) if new[i] != tokens[i] and (adopt != "conf" or nconf[i] > conf[i])]
        if parity is not None:
            cand = [i for i in cand if i % 2 == parity]
        changed = adopt_changes(cand, new, nconf, rnd)
        if trace:
            tr.append(dict(kind="probe", pred=list(new), conf=[round(c, 4) for c in nconf], changed=changed, parity=parity))
        return changed

    def refine_loop(budget):
        nonlocal rounds, outcome, cycle_period
        seen[tuple(tokens)] = rounds
        for _ in range(budget):
            rounds += 1
            if adopt == "oddeven":
                ch = refine_round(rounds, 0) + refine_round(rounds, 1)
            else:
                ch = refine_round(rounds)
            if not ch:
                outcome = "converged"
                return
            key = tuple(tokens)
            if key in seen:
                outcome, cycle_period = "cycle", rounds - seen[key]
                return
            seen[key] = rounds
        outcome = "budget"

    if strategy == "s1":
        place_groups()
    elif strategy in ("conf2", "conf2refine"):
        n1 = -(-len(free) // 2)
        order = sorted(free, key=lambda i: -first_conf[i])
        sel = order[:n1]
        if anchor_leftmost and free and free[0] not in sel:
            sel = [free[0]] + sel[:-1]
        place(sel)
        rest = [i for i in free if i not in sel]
        if rest:
            obs = sorted(i for i in range(K) if tokens[i] != mask_id)
            lg = std(obs, rest, False)
            t2, c2, _ = _predict(lg, mask_id)
            for k, i in enumerate(rest):
                tokens[i], conf[i] = t2[k], c2[k]
            if trace:
                tr.append(dict(kind="std2", placed=obs, pred=list(t2), conf=[round(c, 4) for c in c2]))
        if strategy == "conf2refine":
            refine_loop(1)
    elif strategy == "refine":
        place_groups()
        refine_loop(1)
    elif strategy == "refineN":
        place_groups()
        refine_loop(max_rounds)
        if eos_policy == "verify" and stop_ids & set(tokens) and outcome != "converged":
            before = list(tokens)
            ch = refine_round(rounds + 1)
            rounds += 1
            eos_verified += 1
            if ch and any(before[i] in stop_ids for i, _, _ in ch):
                refine_loop(max_rounds)                      # the stop was revised: keep refining
    elif strategy == "seq":
        if free:
            place([free[0]])
        for i in free[1:]:
            obs = sorted(j for j in range(K) if tokens[j] != mask_id)
            masks = [j for j in range(K) if tokens[j] == mask_id]
            lg = std(obs, masks, False)
            t2, c2, _ = _predict(lg, mask_id)
            k = masks.index(i)
            tokens[i], conf[i] = t2[k], c2[k]
    assert all(t != mask_id for t in tokens), tokens
    info = dict(forwards=forwards, corrections=corrections, probes=probes, first=first, first_conf=first_conf,
                final_conf=list(conf), rounds=rounds, outcome=outcome, cycle_period=cycle_period, eos_verified=eos_verified,
                changes=changes, trace=tr if trace else None)
    return tokens, info


@torch.no_grad()
def k4_generate(model, tok, prompt_ids, max_new_tokens, block=4, strategy="refine", adopt="always", max_rounds=3,
                anchor_leftmost=True, mask_id=None, stop_ids=None, prefill_chunk=512, trace=False, static=False,
                use_graph=False, init=None, eos_policy="current", timing=False):
    """Generate with the small-block decoder on a FastQwen3.  static/use_graph select the bucketed / CUDA-graph path for
    both the standard forwards (model.run) and the probe forwards (model.run_masked; K*K rows must fit the largest
    bucket).  timing=True records wall-clock per phase (prefill / first forward / probes / other) with device syncs.
    Returns (ids, stats)."""
    if mask_id is None:
        mask_id = model.cfg.mask_id
    if stop_ids is None:
        stop_ids = [tok.eos_token_id] if tok is not None and tok.eos_token_id is not None else []
    stop_ids = set(int(s) for s in stop_ids)
    dev = model.device
    prompt = torch.as_tensor(prompt_ids, dtype=torch.long, device=dev).flatten()
    prompt_len = int(prompt.numel())
    assert prompt_len + max_new_tokens + block * block <= model.kv.max_len
    if static or use_graph:
        from .fast_model import BUCKETS
        assert block * block <= BUCKETS[-1] and 2 * block <= BUCKETS[-1], f"static path: K={block} probe needs K*K <= {BUCKETS[-1]} rows"
    win = probe_win_mask(block, device=dev)

    def now():
        if timing and dev.type == "cuda":
            torch.cuda.synchronize()
        return _time.perf_counter()

    t0 = now()
    _, cache_len = model.prefill(prompt, 0, chunk=prefill_chunk)
    tim = dict(prefill=now() - t0, first=0.0, probe=0.0, other=0.0)
    n_prefill = -(-prompt_len // prefill_chunk)
    cur = prompt_len
    out = []
    decode_forwards = corrections = probes = 0
    outcomes = dict(converged=0, cycle=0, budget=0, none=0)
    rounds_total = 0
    blocks = []
    finish = "length" if max_new_tokens == 0 else None
    pending = None
    while finish is None:
        K = min(block, max_new_tokens - len(out))
        state = {"cl": cur if pending is None else cur - len(pending), "n": 0}
        prefix_rows = (pending, list(range(cur - len(pending), cur))) if pending is not None else None

        def fwd_std(ids, pos, rows, _state=state, _cur=cur):
            t = now()
            lg = model.run(torch.tensor(ids), torch.tensor(pos), _state["cl"], rows, static=static, use_graph=use_graph)
            _state["cl"] = _cur
            tim["first" if _state["n"] == 0 else "other"] += now() - t
            _state["n"] += 1
            return lg

        def fwd_probe(ids, pos, Kp, rows, _cur=cur):
            t = now()
            if static or use_graph:
                wm = win if Kp == block else probe_win_mask(Kp, device=dev)
                lg = model.run_masked(torch.tensor(ids), torch.tensor(pos), _cur, wm, rows, static=static, use_graph=use_graph)
            else:
                km = probe_key_mask(_cur, Kp, device=dev)
                lg = model.forward(torch.tensor(ids, device=dev), torch.tensor(pos, device=dev), _cur, write=False, rows=rows,
                                   key_mask=km)
            tim["probe"] += now() - t
            return lg

        toks, info = k4_denoise_block(fwd_std, fwd_probe, cur, K, mask_id, strategy=strategy, adopt=adopt,
                                      max_rounds=max_rounds, anchor_leftmost=anchor_leftmost, prefix_rows=prefix_rows,
                                      trace=trace, init=init, stop_ids=stop_ids, eos_policy=eos_policy)
        pending = None
        decode_forwards += info["forwards"]
        corrections += info["corrections"]
        probes += info["probes"]
        outcomes[info["outcome"]] += 1
        rounds_total += info["rounds"]
        if trace:
            blocks.append(dict(start=cur, tokens=toks, first=info["first"], outcome=info["outcome"], rounds=info["rounds"],
                               changes=info["changes"], trace=info["trace"]))
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
        if finish is not None:
            t = now()
            model.run(torch.tensor(committed), torch.tensor(list(range(cur - keep, cur))), cur - keep, [], static=static,
                      use_graph=use_graph)
            tim["other"] += now() - t
            decode_forwards += 1
        else:
            pending = committed
    n_blocks = -(-len(out) // block) if out else 0
    stats = dict(steps=decode_forwards, forwards=decode_forwards + n_prefill, generated=len(out),
                 tokens_per_step=(len(out) / decode_forwards if decode_forwards else 0.0),
                 tokens_per_forward=(len(out) / decode_forwards if decode_forwards else 0.0), blocks=n_blocks,
                 corrections=corrections, probes=probes, remasked=probes * block, finish_reason=finish, block_size=block,
                 strategy=strategy, adopt=adopt, init=init, eos_policy=eos_policy, outcomes=outcomes,
                 rounds_per_block=(rounds_total / n_blocks if n_blocks else 0.0), prompt_len=prompt_len, cache_len=cur,
                 verify=False)
    if timing:
        stats["timing_s"] = {k: round(v, 4) for k, v in tim.items()}
    if trace:
        stats["trace"] = blocks
    return out, stats
