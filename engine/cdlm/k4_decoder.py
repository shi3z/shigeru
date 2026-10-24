"""Small-block (K = 4 by default) pure diffusion decoder with leave-one-out refinement — SHIGERU Phase E
(docs/four_token_diffusion.md).  No AR logits are read anywhere; no draft model.

The question this decoder answers (brief §0): can a block of 4 future tokens be generated in TWO denoising forwards —
predict all four in parallel from [M M M M], then re-evaluate — at near-AR accuracy, i.e. ~2 tokens per forward?

Forward types (both on the committed prefix, cache_len = cur; FastQwen3.forward / run)
  standard  : rows [observed asc | masks asc] under the causal window mask (prefix + rows before you).  Used for the
              first forward of a block (all masks; merged with the natural-order re-encoding of the previous block as
              in block_decoder.py, so the commit costs no extra launch) and for the second step of "conf2".
  probe     : leave-one-out refinement in ONE forward of K groups x K rows.  Group i = [placed t_j for j != i, ascending,
              causal among themselves and over the prefix | probe mask m_i at position cur+i, attending the prefix, the
              group's K-1 placed rows and itself].  Groups never see each other.  This is exactly the training layout of
              engine/cdlm/data.py for the block with position i masked and the others observed (observed-first, causal
              among observed, the mask sees all observed), so m_i is the model's trained conditional
              p(x_i | prefix, the other K-1 tokens of the block) with observed context on BOTH sides.  A single shared
              set of placed rows would leak t_i into m_i through the causal encoding of t_{j>i}, which is why the groups
              are separate (tests/test_k4.py checks that changing t_i does not move m_i).  Implemented with
              FastQwen3.forward(key_mask=...) and write=False: nothing of the probe enters the cache; the block's final
              tokens are re-encoded in natural order by the next block's merged forward (or a final forward), exactly
              as block_decoder does, so the cache always holds a causal encoding.  Cost: K^2 rows (16 for K=4).
Strategies (`strategy`; S = forwards per block, the merged commit included in the first)
  "s1"          S=1  predict [M M M M] once, place all K                             (brief A)
  "conf2"       S=2  place the ceil(K/2) most confident (anchor: the leftmost always), then the rest   (brief B)
  "refine"      S=2  place all K, then one leave-one-out probe; a position whose probe argmax differs is replaced when
                     `adopt` allows: "always"; "conf" (only if the probe confidence exceeds the placed confidence);
                     "nonadjacent" (of the changed positions keep a set with no two adjacent, highest probe confidence
                     first — adjacent simultaneous changes were conditioned on each other's OLD token); "one" (only the
                     single most confident change per round, Gauss-Seidel style)                                   (brief C)
  "refineN"     S<=1+max_rounds  probe rounds until no position changes (adaptive, brief E) or max_rounds (brief D)
  "conf2refine" S=3  conf2 followed by one probe round
  "seq"         S=K  leftmost one per forward (the AR conditional at every step; the diagnostic ceiling, not a design)
Returns per block the first-forward prediction of every position (argmax + conf) and the final tokens, so the
diagnostic can score first-step vs second-step accuracy by distance (bench/diag_k4.py).  Stats: steps (= decode
forwards), forwards (incl. prefill chunks), tokens_per_forward (generated / decode forwards), blocks, corrections
(probe changes), probes, repetition and EOS statistics are computed by the benchmark from the text.
"""
import torch

from .decoder import entropy_from_logits

__all__ = ["STRATEGIES", "ADOPT", "k4_denoise_block", "k4_generate", "probe_key_mask", "probe_rows"]

STRATEGIES = ("s1", "conf2", "refine", "refineN", "conf2refine", "seq")
ADOPT = ("always", "conf", "nonadjacent", "one")


def probe_rows(tokens, cur, mask_id):
    """(ids, pos, probe_row_indices) of the K-group leave-one-out layout for the placed `tokens` (module docstring)."""
    K = len(tokens)
    ids, pos, rows = [], [], []
    for i in range(K):
        for j in range(K):
            if j != i:
                ids.append(tokens[j]); pos.append(cur + j)
        ids.append(mask_id); pos.append(cur + i)
        rows.append(len(ids) - 1)
    return ids, pos, rows


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


def _predict(logits, mask_id):
    """argmax / max prob / entropy per row; the [MASK] token itself is never emitted."""
    lg = logits.float().clone()
    lg[:, mask_id] = float("-inf")
    probs = torch.softmax(lg, dim=-1)
    conf, tok = probs.max(-1)
    return tok.tolist(), conf.tolist(), entropy_from_logits(lg).tolist()


@torch.no_grad()
def k4_denoise_block(fwd_std, fwd_probe, cur, K, mask_id, strategy="refine", adopt="always", max_rounds=3,
                     anchor_leftmost=True, prefix_rows=None, trace=False):
    """Denoise one block of K positions cur..cur+K-1.
    fwd_std(ids, pos, rows) -> logits [len(rows), V]  (standard causal window forward, may write the cache)
    fwd_probe(ids, pos, K, rows) -> logits             (key mask probe_key_mask(cur, K); must NOT write the cache)
    prefix_rows: (ids, pos) of the previous block to re-encode in the first forward (merged commit).
    Returns (tokens[K], info) with info: forwards, corrections, probes, first (argmax of the all-mask forward per
    position), first_conf, trace (per forward: kind, placed tokens, confs)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy={strategy!r}, expected one of {STRATEGIES}")
    if adopt not in ADOPT:
        raise ValueError(f"adopt={adopt!r}")
    tr = []
    forwards = corrections = probes = 0

    def std(obs_tokens, obs_slots, mask_slots, first_call):
        nonlocal forwards
        ids = list(obs_tokens) + [mask_id] * len(mask_slots)
        pos = [cur + s for s in obs_slots] + [cur + s for s in mask_slots]
        n_lead = len(obs_slots)
        if first_call and prefix_rows is not None:
            pids, ppos = prefix_rows
            ids, pos, n_lead = list(pids) + ids, list(ppos) + pos, n_lead + len(pids)
        forwards += 1
        return fwd_std(ids, pos, list(range(n_lead, len(ids))))

    def probe(tokens):
        nonlocal forwards, probes
        ids, pos, rows = probe_rows(tokens, cur, mask_id)
        forwards += 1
        probes += 1
        return fwd_probe(ids, pos, K, rows)

    # forward 1: all masks (merged commit of the previous block)
    logits = std([], [], list(range(K)), True)
    first, first_conf, _ = _predict(logits, mask_id)
    tokens = [mask_id] * K
    conf = [0.0] * K
    if trace:
        tr.append(dict(kind="all_mask", pred=list(first), conf=[round(c, 4) for c in first_conf]))

    def place_all():
        for i in range(K):
            tokens[i], conf[i] = first[i], first_conf[i]

    def refine_round():
        nonlocal corrections
        lg = probe(tokens)
        new, nconf, _ = _predict(lg, mask_id)
        cand = [i for i in range(K) if new[i] != tokens[i] and (adopt != "conf" or nconf[i] > conf[i])]
        if adopt == "one" and cand:
            cand = [max(cand, key=lambda i: nconf[i])]
        elif adopt == "nonadjacent" and cand:
            keep = []
            for i in sorted(cand, key=lambda i: -nconf[i]):
                if all(abs(i - j) > 1 for j in keep):
                    keep.append(i)
            cand = sorted(keep)
        changed = []
        for i in cand:
            changed.append((i, tokens[i], new[i]))
            tokens[i] = new[i]
        for i in range(K):
            if new[i] == tokens[i]:
                conf[i] = nconf[i]
        corrections += len(changed)
        if trace:
            tr.append(dict(kind="probe", pred=list(new), conf=[round(c, 4) for c in nconf], changed=changed))
        return changed

    if strategy == "s1":
        place_all()
    elif strategy in ("conf2", "conf2refine"):
        n1 = -(-K // 2)
        order = sorted(range(K), key=lambda i: -first_conf[i])
        sel = order[:n1]
        if anchor_leftmost and 0 not in sel:
            sel = [0] + sel[:-1]
        for i in sel:
            tokens[i], conf[i] = first[i], first_conf[i]
        rest = [i for i in range(K) if i not in sel]
        if rest:
            obs = sorted(sel)
            lg = std([tokens[i] for i in obs], obs, rest, False)
            t2, c2, _ = _predict(lg, mask_id)
            for k, i in enumerate(rest):
                tokens[i], conf[i] = t2[k], c2[k]
            if trace:
                tr.append(dict(kind="std2", placed=obs, pred=list(t2), conf=[round(c, 4) for c in c2]))
        if strategy == "conf2refine":
            refine_round()
    elif strategy == "refine":
        place_all()
        refine_round()
    elif strategy == "refineN":
        place_all()
        for _ in range(max_rounds):
            if not refine_round():
                break
    elif strategy == "seq":
        for i in range(K):
            if i == 0:
                tokens[0], conf[0] = first[0], first_conf[0]
                continue
            obs = list(range(i))
            lg = std([tokens[j] for j in obs], obs, list(range(i, K)), False)
            t2, c2, _ = _predict(lg, mask_id)
            tokens[i], conf[i] = t2[0], c2[0]
    assert all(t != mask_id for t in tokens)
    info = dict(forwards=forwards, corrections=corrections, probes=probes, first=first, first_conf=first_conf,
                final_conf=list(conf), trace=tr if trace else None)
    return tokens, info


@torch.no_grad()
def k4_generate(model, tok, prompt_ids, max_new_tokens, block=4, strategy="refine", adopt="always", max_rounds=3,
                anchor_leftmost=True, mask_id=None, stop_ids=None, prefill_chunk=512, trace=False):
    """Generate with the small-block decoder on a FastQwen3 (plain forward path; the probe forward needs a custom key
    mask, which the static/graph path does not take yet).  Returns (ids, stats)."""
    if mask_id is None:
        mask_id = model.cfg.mask_id
    if stop_ids is None:
        stop_ids = [tok.eos_token_id] if tok is not None and tok.eos_token_id is not None else []
    stop_ids = set(int(s) for s in stop_ids)
    dev = model.device
    prompt = torch.as_tensor(prompt_ids, dtype=torch.long, device=dev).flatten()
    prompt_len = int(prompt.numel())
    assert prompt_len + max_new_tokens + block * block <= model.kv.max_len
    _, cache_len = model.prefill(prompt, 0, chunk=prefill_chunk)
    n_prefill = -(-prompt_len // prefill_chunk)
    cur = prompt_len
    out = []
    decode_forwards = corrections = probes = 0
    blocks = []
    finish = "length" if max_new_tokens == 0 else None
    pending = None
    while finish is None:
        K = min(block, max_new_tokens - len(out))
        state = {"cl": cur if pending is None else cur - len(pending)}
        prefix_rows = (pending, list(range(cur - len(pending), cur))) if pending is not None else None

        def fwd_std(ids, pos, rows, _state=state, _cur=cur):
            lg = model.forward(torch.tensor(ids, device=dev), torch.tensor(pos, device=dev), _state["cl"], write=True, rows=rows)
            _state["cl"] = _cur
            return lg

        def fwd_probe(ids, pos, Kp, rows, _cur=cur):
            km = probe_key_mask(_cur, Kp, device=dev)
            return model.forward(torch.tensor(ids, device=dev), torch.tensor(pos, device=dev), _cur, write=False, rows=rows,
                                 key_mask=km)

        toks, info = k4_denoise_block(fwd_std, fwd_probe, cur, K, mask_id, strategy=strategy, adopt=adopt,
                                      max_rounds=max_rounds, anchor_leftmost=anchor_leftmost, prefix_rows=prefix_rows,
                                      trace=trace)
        pending = None
        decode_forwards += info["forwards"]
        corrections += info["corrections"]
        probes += info["probes"]
        if trace:
            blocks.append(dict(start=cur, tokens=toks, first=info["first"], trace=info["trace"]))
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
            model.forward(torch.tensor(committed, device=dev), torch.tensor(list(range(cur - keep, cur)), device=dev),
                          cur - keep, write=True, rows=[])
            decode_forwards += 1
        else:
            pending = committed
    stats = dict(steps=decode_forwards, forwards=decode_forwards + n_prefill, generated=len(out),
                 tokens_per_step=(len(out) / decode_forwards if decode_forwards else 0.0),
                 tokens_per_forward=(len(out) / decode_forwards if decode_forwards else 0.0), blocks=-(-len(out) // block),
                 corrections=corrections, probes=probes, remasked=probes * block, finish_reason=finish, block_size=block,
                 strategy=strategy, prompt_len=prompt_len, cache_len=cur, verify=False)
    if trace:
        stats["trace"] = blocks
    return out, stats
