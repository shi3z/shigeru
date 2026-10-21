"""The streaming-window decoder of engine/cdlm/decoder.py on top of fast_model.FastQwen3 (no HF, no DynamicCache).

Same algorithm, same arguments, same (new_ids, stats) — token-for-token identical to decoder.generate on the same
weights (tests/test_fast.py):
  prefill   : FastQwen3.prefill writes the prompt K/V at slots 0..P-1; cache_len = P.
  per step  : refill masks (budget-capped) -> forward the window observed-first [filled..., masks...] with their
              ORIGINAL position ids (cache_len + slot) at cache slots cache_len.. -> commit the leftmost filled run by
              cache_len += kept (no crop: the rows after the run sit beyond cache_len and are overwritten next step)
              -> predict the mask rows (entropy + lam*distance < tau; tau=None = leftmost mask only; predict="ar"
              gives the leftmost mask the next-token logits of the preceding row; the optional fill-selection rules
              margin_tau / no_adjacent / max_fill / fill_mode are decoder.select_positions itself, so they are the
              same function in both decoders) -> window = window[m:].
Logits are gathered only for the rows that are used: the mask rows and, for predict="ar", the last committed row.
`static=True` runs the bucketed static-shape path of fast_model (and `use_graph=True` replays CUDA graphs when the
model is on CUDA; on CPU it falls back to the eager static path).  stats carries `cache_len` instead of a DynamicCache.
verify=True runs decoder.generate_verified (the verified / speculative loop is one function shared by both decoders;
decoder.py module docstring "Verified (speculative) decoding") on this model: forward = model.run over all window
rows (pending token + W slots, so W <= 31 on the static path), commit = cache_len.
"""
import torch

from .decoder import check_fill_options, entropy_from_logits, generate_verified, margin_from_logits, select_positions

__all__ = ["generate"]


@torch.no_grad()
def generate(model, tok, prompt_ids, max_new_tokens, window=16, tau=None, lam=0.02, mask_id=None, stop_ids=None,
             greedy=True, temperature=1.0, predict="mask", generator=None, static=False, use_graph=False,
             prefill_chunk=512, margin_tau=None, no_adjacent=False, max_fill=None, fill_mode="all", verify=False):
    """Decode one sequence with a FastQwen3.  Arguments as decoder.generate (mask_id defaults to the checkpoint's
    cdlm_mask_token_id; tok only defaults stop_ids to [tok.eos_token_id]; margin_tau / no_adjacent / max_fill /
    fill_mode are the fill-selection rules of decoder.py, all off by default; verify=True is verified decoding,
    exactly greedy AR output, with the extra stats of decoder.generate_verified).  Returns (new_token_ids, stats) with
    stats = steps, forwards, window_rows, generated, tokens_per_step, filled_per_step, committed_per_step,
    filled_positions, finish_reason, prompt_len, cache_len (slots [0, cache_len) of model.kv hold [prompt | output])."""
    if predict not in ("mask", "ar"):
        raise ValueError(f"predict={predict!r}")
    no_adjacent = check_fill_options(margin_tau, no_adjacent, max_fill, fill_mode)
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
    assert prompt_len > 0 and window >= 1 and max_new_tokens >= 0
    assert prompt_len + max_new_tokens + window <= model.kv.max_len, "cache too small for prompt + budget + window"

    last_logits, cache_len = model.prefill(prompt, 0, chunk=prefill_chunk)
    assert cache_len == prompt_len

    def sample(logits):
        if greedy:
            return logits.argmax(-1)
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        return torch.multinomial(probs, 1, generator=generator).squeeze(-1)

    if verify:
        state = {"cache_len": cache_len}

        def forward(ids, pos, cl):
            assert cl == state["cache_len"]
            return model.run(ids, pos, cl, list(range(int(ids.numel()))), static=static, use_graph=use_graph)

        def commit(n):
            state["cache_len"] = n

        # ids/pos are built on the CPU: model.run stages them to the device itself (one copy per step)
        out, stats = generate_verified(forward, commit, last_logits, prompt_len, max_new_tokens, window, tau, lam,
                                       mask_id, stop_ids, sample, predict, margin_tau, no_adjacent, max_fill,
                                       fill_mode, torch.device("cpu"))
        assert state["cache_len"] == prompt_len + len(out)
        stats.update(prompt_len=prompt_len, cache_len=state["cache_len"])
        return out, stats

    cur = prompt_len
    out = []
    slots = [mask_id] * min(window, max_new_tokens)                  # window slot i <-> logical position cur + i
    is_mask = [True] * len(slots)
    steps = window_rows = 0
    filled_hist, commit_hist, filled_positions = {}, {}, []
    finish = "length" if not slots else None

    while slots and finish is None:
        # 1. refill within the budget
        m = next((i for i, f in enumerate(is_mask) if f), len(slots))
        room = max_new_tokens - (cur - prompt_len) - len(slots)
        add = min(m, room)
        if add > 0:
            slots += [mask_id] * add
            is_mask += [True] * add
        # 2. forward [filled..., masks...] with original positions at cache slots cache_len.. (ids/pos are CPU
        #    tensors: model.run moves them in one copy, or stages them for the static buffers)
        filled_idx = [i for i, f in enumerate(is_mask) if not f]
        mask_idx = [i for i, f in enumerate(is_mask) if f]
        order = filled_idx + mask_idx
        ids = torch.tensor([slots[i] for i in order], dtype=torch.long)
        pos = torch.tensor([cur + i for i in order], dtype=torch.long)
        T = len(order)
        assert cache_len == cur
        n_commit = mask_idx[0] if mask_idx else len(slots)
        assert n_commit == m and filled_idx[:n_commit] == list(range(n_commit))
        keep = n_commit
        for j in range(n_commit):
            if slots[j] in stop_ids:
                keep, finish = j + 1, "stop"
                break
        if cur - prompt_len + keep >= max_new_tokens:
            keep = max_new_tokens - (cur - prompt_len)
            finish = finish or "length"
        n_filled = len(filled_idx)
        want_last = predict == "ar" and keep > 0
        rows = ([keep - 1] if want_last else []) + list(range(n_filled, T))
        logits = model.run(ids, pos, cache_len, rows, static=static, use_graph=use_graph)
        steps += 1
        window_rows += T
        # 3. commit the leftmost filled run: its K/V at slots cur..cur+keep-1 are final
        out += slots[:keep]
        cur += keep
        cache_len = cur
        if want_last:
            last_logits = logits[0].clone()
        commit_hist[keep] = commit_hist.get(keep, 0) + 1
        if finish is not None:
            break
        slots, is_mask = slots[n_commit:], is_mask[n_commit:]
        if not slots:
            finish = "length"
            break
        # 4. predict the mask rows
        mask_logits = logits[1:] if want_last else logits
        mask_slots = [i - n_commit for i in mask_idx]
        if predict == "ar":
            mask_logits = torch.cat([last_logits[None], mask_logits[1:]], 0)
        ent = entropy_from_logits(mask_logits)
        margin = margin_from_logits(mask_logits) if margin_tau is not None else None
        toks = sample(mask_logits)
        sel = select_positions(ent, mask_slots, tau, lam, margin, margin_tau, no_adjacent, max_fill, fill_mode)
        toks = toks.tolist() if sel else []
        for k in sel:
            slots[mask_slots[k]] = int(toks[k])
            is_mask[mask_slots[k]] = False
        filled_hist[len(sel)] = filled_hist.get(len(sel), 0) + 1
        filled_positions.append([cur + mask_slots[k] for k in sel])

    assert len(out) <= max_new_tokens and cache_len == prompt_len + len(out)
    stats = dict(steps=steps, forwards=steps + 1, window_rows=window_rows, generated=len(out),
                 tokens_per_step=(len(out) / steps if steps else 0.0),
                 filled_per_step=dict(sorted(filled_hist.items())), committed_per_step=dict(sorted(commit_hist.items())),
                 filled_positions=filled_positions, finish_reason=finish, prompt_len=prompt_len, cache_len=cache_len)
    return out, stats
