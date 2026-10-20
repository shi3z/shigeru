"""WeDLM-style streaming-window decoder on top of a HF Qwen3ForCausalLM with a DynamicCache (transformers 5.20).

Port of the loop in third_party/WeDLM/wedlm/engine/wedlm_decoder.py + sampler.py as analysed in docs/wedlm_analysis.md
sections 5-6 (nothing is copied; the paged-cache/slot-mapping machinery is replaced by HF's DynamicCache).  Single
sequence, no batching.

The loop (one sequence)
  prefill    : plain causal forward of the prompt into the cache; cur = len(prompt).
  init       : window = [MASK] x min(W, budget); window slot i has logical position cur + i.
  per step   : 1. refill: m = index of the first mask (= number of leading filled slots); append min(m, room) masks.
               2. forward the window rows in observed-first order [filled..., masks...] with their ORIGINAL position ids
                  (cur + slot) and a plain causal mask over [cache | window rows] (window_attention_mask), on top of the
                  cache.  The cache now holds [prefix | all window rows].
               3. commit: the leftmost contiguous filled run (slots 0..m-1) is appended to the output; cur += m; the cache
                  is cropped so that it holds exactly [prefix | committed run] (DynamicCache.crop(-(T - m)); crop removes
                  from the end, and the run is at the front of the window, so no K/V are copied or recomputed).  A stop
                  token inside the run, or reaching max_new_tokens, finishes the sequence (tokens after the stop are
                  dropped, the cache is cropped to what was kept).
               4. predict: logits of the mask rows only; entropy per row; token = argmax (greedy) or a sample;
                  adjusted = entropy + lam * (slot - first_mask_slot); fill the rows with adjusted < tau (at least one:
                  argmin if none passes); tau=None fills exactly one token per step, the leftmost mask (the reference
                  takes argmin(adjusted) in that case, which coincides with the leftmost mask whenever lam >= ln(V)).
               5. window = window[m:]; loop while the window is non-empty.

Why the committed K/V are valid without recomputation (docs/wedlm_analysis.md section 5)
  The committed run occupies window slots 0..m-1 and, because filled rows go first and keep their slot order, also the
  physical rows 0..m-1 of the forward.  Physical row i < m therefore (a) carries the token that is committed at logical
  position cur + i, (b) is rotated with position id cur + i, and (c) attends, under the causal mask over
  [cache | window], exactly the cached prefix (positions < cur, already final) plus physical rows < i, which are the
  committed tokens cur..cur+i-1.  That is precisely the context of a plain causal LM at position cur + i, so the K/V
  written by this forward for those rows are bit-for-bit what a fresh causal forward of [prefix | run] would produce
  (test_decoder.test_cache_matches_full_forward checks this at 1e-5 in float32, ~1e-15 in float64).  Rows >= m
  (filled-after-a-gap tokens and masks) attended a context that still contains [MASK] rows, so their K/V are discarded
  by the crop and those tokens are re-encoded next step - the reference does the same by overwriting their slots.

Predictor (`predict`)
  "mask" (default, WeDLM): a mask row at logical position p predicts the token at p from its own logits.
  "ar": the LEFTMOST mask takes the next-token logits of the row just before it (the last row of the committed run, or
        the last cached token when the window starts with a mask); the other mask rows keep their own logits.  With
        tau=None this is exactly greedy autoregressive decoding of the same model, whatever its weights, which makes it
        the mechanical exactness check of the cache/position/mask plumbing (test_decoder.test_tau_none_equals_hf_greedy).
        The design doc's exactness gate (docs/distill_design.md section 7 step 6) is the *trained-model* statement that
        "mask" with tau=None reproduces the AR student; it cannot hold for untrained weights and is not a test here.
"""
import torch
from transformers import DynamicCache

from .model_glue import model_attention_mask

__all__ = ["generate", "window_attention_mask", "entropy_from_logits", "select_positions"]


def window_attention_mask(model, past_len, n_rows, device=None):
    """Causal mask over [cache (past_len) | window (n_rows)] for the n_rows query rows, as the dict Qwen3Model.forward
    hands to the layers untouched (bool for sdpa, additive for eager; see model_glue.model_attention_mask)."""
    q = torch.arange(n_rows, device=device)[:, None] + past_len
    k = torch.arange(past_len + n_rows, device=device)[None, :]
    return model_attention_mask(model, (k <= q)[None, None])


def entropy_from_logits(logits):
    """Shannon entropy (nats) of softmax(logits) per row, computed in float32 like the reference's Categorical."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)


def select_positions(entropy, mask_slots, tau, lam):
    """Indices (into mask_slots / the mask rows) to fill this step.  tau=None: the leftmost mask only.  Otherwise all
    rows with entropy + lam*(slot - first slot) < tau, or the argmin of that quantity if none passes (reference
    sampler.py:169-222)."""
    if tau is None:
        return [0]
    d = torch.tensor([s - mask_slots[0] for s in mask_slots], dtype=entropy.dtype, device=entropy.device)
    adjusted = entropy + lam * d
    cand = (adjusted < tau).nonzero(as_tuple=True)[0]
    if cand.numel() > 0:
        return cand.tolist()
    return [int(adjusted.argmin())]


def _crop_to(cache, keep_total):
    """Shrink the DynamicCache to its first keep_total positions (crop takes a negative 'tokens to remove')."""
    n = cache.get_seq_length() - keep_total
    if n > 0:
        cache.crop(-n)
    assert cache.get_seq_length() == keep_total


@torch.no_grad()
def generate(model, tok, prompt_ids, max_new_tokens, window=16, tau=None, lam=0.02, mask_id=None, stop_ids=None,
             greedy=True, temperature=1.0, predict="mask", generator=None):
    """Decode one sequence.  Returns (new_token_ids: list[int], stats: dict).

    model: Qwen3ForCausalLM (sdpa or eager) with a mask row (model_glue.add_mask_token).  tok: tokenizer, used only to
    default stop_ids to [tok.eos_token_id] (may be None).  prompt_ids: list/1-D tensor of ids.  window: W.  tau: entropy
    threshold or None (one token per step).  lam: distance penalty.  mask_id: default model.config.cdlm_mask_token_id.
    stop_ids: ids that end generation (the stop token is kept in the output).  greedy: argmax, else sample at
    `temperature` (torch.Generator `generator`).  predict: "mask" | "ar" (see module docstring).

    stats: steps (decode forwards), forwards (steps + 1 prefill), window_rows (rows forwarded in decode steps),
    generated, tokens_per_step, filled_per_step / committed_per_step ({count: n_steps}), finish_reason
    ("stop" | "length"), cache (the DynamicCache holding [prompt | output], reusable for a continuation).
    """
    if predict not in ("mask", "ar"):
        raise ValueError(f"predict={predict!r}")
    if mask_id is None:
        mask_id = getattr(model.config, "cdlm_mask_token_id", None)
        if mask_id is None:
            raise ValueError("mask_id not given and model.config.cdlm_mask_token_id unset (run add_mask_token)")
    if stop_ids is None:
        stop_ids = [tok.eos_token_id] if tok is not None and tok.eos_token_id is not None else []
    stop_ids = set(int(s) for s in stop_ids)
    dev = model.get_input_embeddings().weight.device
    prompt = torch.as_tensor(prompt_ids, dtype=torch.long, device=dev).flatten()
    prompt_len = int(prompt.numel())
    assert prompt_len > 0 and window >= 1 and max_new_tokens >= 0

    cache = DynamicCache(config=model.config)
    out_pf = model.model(input_ids=prompt[None], past_key_values=cache, use_cache=True)
    last_logits = model.lm_head(out_pf.last_hidden_state[0, -1])     # next-token logits of the last cached token
    assert cache.get_seq_length() == prompt_len

    cur = prompt_len
    out = []
    slots = [mask_id] * min(window, max_new_tokens)                  # window slot i <-> logical position cur + i
    is_mask = [True] * len(slots)
    steps = window_rows = 0
    filled_hist, commit_hist = {}, {}
    finish = "length" if not slots else None

    def sample(logits):
        if greedy:
            return logits.argmax(-1)
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        return torch.multinomial(probs, 1, generator=generator).squeeze(-1)

    while slots and finish is None:
        # 1. refill: as many new masks as there are leading filled slots, within the max_new_tokens budget
        m = next((i for i, f in enumerate(is_mask) if f), len(slots))
        room = max_new_tokens - (cur - prompt_len) - len(slots)
        add = min(m, room)
        if add > 0:
            slots += [mask_id] * add
            is_mask += [True] * add
        # 2. forward [filled..., masks...] with original positions on top of the cache
        filled_idx = [i for i, f in enumerate(is_mask) if not f]
        mask_idx = [i for i, f in enumerate(is_mask) if f]
        order = filled_idx + mask_idx
        ids = torch.tensor([slots[i] for i in order], dtype=torch.long, device=dev)
        pos = torch.tensor([cur + i for i in order], dtype=torch.long, device=dev)
        T = len(order)
        past = cache.get_seq_length()
        assert past == cur
        h = model.model(input_ids=ids[None], position_ids=pos[None],
                        attention_mask=window_attention_mask(model, past, T, dev),
                        past_key_values=cache, use_cache=True).last_hidden_state[0]
        logits = model.lm_head(h)
        steps += 1
        window_rows += T
        # 3. commit the leftmost filled run = window slots 0..m-1 = physical rows 0..m-1 (their K/V are final)
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
        out += slots[:keep]
        cur += keep
        _crop_to(cache, cur)                                           # drop every row that is not committed
        if keep > 0:
            last_logits = logits[keep - 1]
        commit_hist[keep] = commit_hist.get(keep, 0) + 1
        if finish is not None:
            break
        slots, is_mask = slots[n_commit:], is_mask[n_commit:]
        if not slots:                                                  # only when max_new_tokens == 0 (budget exhausted)
            finish = "length"
            break
        # 4. predict the mask rows (physical rows n_filled.. are the masks, in slot order)
        n_filled = len(filled_idx)
        mask_logits = logits[n_filled:]
        mask_slots = [i - n_commit for i in mask_idx]
        if predict == "ar":
            # leftmost mask at logical cur + mask_slots[0]; the row before it is the last committed token (whose
            # next-token logits are last_logits after the update above) - exact AR for the leftmost position
            mask_logits = torch.cat([last_logits[None], mask_logits[1:]], 0)
        ent = entropy_from_logits(mask_logits)
        toks = sample(mask_logits)
        sel = select_positions(ent, mask_slots, tau, lam)
        for k in sel:
            slots[mask_slots[k]] = int(toks[k])
            is_mask[mask_slots[k]] = False
        filled_hist[len(sel)] = filled_hist.get(len(sel), 0) + 1

    assert len(out) <= max_new_tokens and cache.get_seq_length() == prompt_len + len(out)
    stats = dict(steps=steps, forwards=steps + 1, window_rows=window_rows, generated=len(out),
                 tokens_per_step=(len(out) / steps if steps else 0.0),
                 filled_per_step=dict(sorted(filled_hist.items())), committed_per_step=dict(sorted(commit_hist.items())),
                 finish_reason=finish, cache=cache)
    return out, stats
