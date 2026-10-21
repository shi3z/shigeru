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
                  Optional fill-selection rules (all off by default, `select_positions`): margin_tau, no_adjacent,
                  max_fill, fill_mode="leftmost_run" - see "Fill-selection rules" below.
               5. window = window[m:]; loop while the window is non-empty.

Fill-selection rules (`select_positions`; identical in fast_decoder.py, which imports it)
  Motivation (docs/distill_design.md section 10.3): masks filled in the same step are conditionally independent given
  the context, so a multi-token number is split ("#### 10" for "#### 100") as soon as two adjacent masks both clear
  the threshold.  The rules below restrict *which* of the passing masks are filled; none of them changes the forward,
  the cache or the commit logic, and with all of them off the selection is exactly the reference rule above.
  Candidate set C (step 1):
    tau given, margin_tau None : C = {k : entropy_k + lam*d_k < tau}            (reference rule)
    tau given, margin_tau given: C = {k : entropy_k + lam*d_k < tau  and  margin_k >= margin_tau}   (both must hold)
    tau None,  margin_tau given: C = {k : margin_k >= margin_tau}                 (margin replaces the entropy rule)
    tau None,  margin_tau None : C = {0}                                          (leftmost mask, the exact-AR step)
    margin_k = p_top1 - p_top2 of softmax(logits_k) (float32, no temperature, same logits as the entropy).  The
    mandatory >= 1 fill when C is empty: argmin(entropy + lam*d) when tau is given (reference fallback), else the
    leftmost mask (index 0, whose left context is fully committed).
  Post-filters (step 2), applied in this order; each keeps the leftmost elements, so the mandatory fill survives:
    fill_mode="leftmost_run": keep the run 0, 1, 2, ... of mask indices that are all in C and occupy consecutive
        logical slots, anchored at the leftmost mask (slot 0 of the window after the commit = logical position cur,
        whose left context is entirely committed); if the leftmost mask is not in C it is filled alone (the tau=None
        step).  Everything right of the first failing mask waits, so the window never holds filled-after-a-gap rows
        and every filled token is committed in the next step.  fill_mode="all" (default) keeps C as is.
    no_adjacent=True: of every maximal run of candidates at consecutive logical slots keep only the leftmost
        (slots 0,1,2,3,5,6,8 -> 0,5,8); two masks adjacent in logical position are never filled in the same step.
    max_fill=n: keep the n leftmost remaining candidates.
  The absolute logical positions filled in each step are recorded in stats["filled_positions"] (list per step).

Verified (speculative) decoding (`verify=True`; the loop is `generate_verified`, imported by fast_decoder.py)
  Motivation (docs/distill_design.md section 10.3, "Fill-selection rules on the CPT checkpoint"): on the converted
  checkpoint every token filled ahead of the frontier is unreliable and is never revised, so parallel filling collapses
  the accuracy even at 1.01 tokens/step.  With verify=True the fills are DRAFTS that the AR logits of the same forward
  verify, and the output is exactly the greedy output of the AR rows (HF generate(do_sample=False)) whatever the
  drafts are - the mask head's accuracy only buys speed.
  State: the committed output `out` (cur = prompt_len + len(out)); the window of W slots (slot i <-> logical position
  cur + i, each a draft or a mask; W rows are forwarded per step, the budget aside); `pending`: the last committed
  token while its K/V are not in the cache yet (the cache then holds cur - 1 rows).
  per step : 1. refill the window to W masks (budget-capped; fresh masks are forwarded now and draftable after step 3).
             2. forward [pending | drafts (slot order) | masks (slot order)] with original positions (pending at
                cur - 1).  The pending row's logits are the exact next-token logits of position cur; the row of a
                draft at slot i of the LEADING contiguous drafted run attended exactly prefix + pending + drafts
                0..i-1, so its logits are the exact AR logits of position cur+i+1 whenever drafts 0..i are the
                greedy tokens.
             3. verify: walk j = 0, 1, ...: g = argmax of the AR logits of position cur+j (the pending row for j = 0,
                or the prefill logits on the first step; the row of the draft at slot j-1 otherwise); if slot j holds
                a draft equal to g it is accepted and the walk continues, else it stops.  The accepted drafts 0..j-1
                are committed (cache cropped/advanced to cur+j: their K/V are final by the argument below, they are
                physical rows 1..j behind the pending row) and g is committed as the new pending: it is appended to
                `out` and counted now, its K/V are written by the next forward, which carries it as the first observed
                row.  A stop token or the budget inside the accepted run ends the sequence without a correction; when
                the correction g is a stop token or exhausts the budget, one extra one-row forward writes its K/V
                (stats completion_forward, counted in `forwards`) so that the cache holds [prompt | output] exactly as
                in the plain mode.
             4. window = slots j+1..: if slot j was a draft (mismatch) every draft to its right is re-masked (they
                were conditioned on the wrong token); if slot j was a mask the drafts to its right stay drafts.
             5. draft: the masks that were forwarded as masks this step (a re-masked row has no mask logits until the
                next forward) are filled with the same rule as the plain mode (select_positions with tau / lam /
                margin_tau / no_adjacent / max_fill / fill_mode) but WITHOUT the mandatory >= 1 fill (min_fill=0):
                a step may draft nothing, because progress is guaranteed by the correction.  predict="ar": the
                leftmost draftable mask takes the AR logits of the draft just before it when that draft is kept (the
                mask-stop case); the mask right after the correction keeps its own logits (the corrected token's
                next-token logits do not exist before the next forward, where they do the verification).
  Every step commits >= 1 token (the correction, or the accepted run when it hits a stop id / the budget), so
  tokens_per_step >= 1; the ceiling is W tokens in one step (W-1 accepted drafts + the correction) and (W+1)/2 on
  average (a fully accepted step consumes the whole window, so the next step starts from W fresh masks and commits
  only the correction).  W=1 is plain greedy decoding (one correction per step; the mask row is never used).
  Guarantee: by induction `out` is a prefix of the greedy AR output: every committed token is the argmax of logits
  computed on the exact committed prefix, and a draft is only ever committed after it was found equal to that argmax
  (test_decoder.test_verify_equals_hf_greedy: every tau, W, predict, with/without stop ids, every fill rule;
  test_fast.test_fast_decoder_matches_hf_decoder_verify: both decoders token- and stats-identical).  greedy=False
  only randomises the drafts; the output stays the greedy one.
  stats (both decoders): steps, forwards (= steps + 1 prefill + completion_forward), window_rows (incl. the pending
  row), tokens_per_step, filled_per_step (drafts per step; 0 possible), committed_per_step (>= 1 per step),
  accepted_per_step ({accepted drafts: n_steps}), drafted, accepted, draft_acceptance_rate (accepted / drafted; None
  without drafts), filled_positions, finish_reason, verify=True, completion_forward.

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

__all__ = ["generate", "generate_verified", "verify_drafts", "window_attention_mask", "entropy_from_logits",
           "margin_from_logits", "select_positions", "FILL_MODES"]

FILL_MODES = ("all", "leftmost_run")


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


def margin_from_logits(logits):
    """Top-1 minus top-2 probability of softmax(logits) per row (float32, no temperature)."""
    top2 = torch.softmax(logits.float(), dim=-1).topk(2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def check_fill_options(margin_tau=None, no_adjacent=False, max_fill=None, fill_mode="all"):
    """Validate the fill-selection options (shared by both decoders)."""
    if fill_mode not in FILL_MODES:
        raise ValueError(f"fill_mode={fill_mode!r}, expected one of {FILL_MODES}")
    if max_fill is not None and int(max_fill) < 1:
        raise ValueError(f"max_fill={max_fill!r} must be >= 1 or None")
    if margin_tau is not None and not (0.0 <= float(margin_tau) <= 1.0):
        raise ValueError(f"margin_tau={margin_tau!r} must be in [0, 1] or None")
    return bool(no_adjacent)


def select_positions(entropy, mask_slots, tau, lam, margin=None, margin_tau=None, no_adjacent=False, max_fill=None,
                     fill_mode="all", min_fill=1):
    """Indices (into mask_slots / the mask rows, ascending) to fill this step.

    Reference rule (all options off): tau=None -> the leftmost mask only; otherwise all rows with
    entropy + lam*(slot - first slot) < tau, or the argmin of that quantity if none passes (sampler.py:169-222).
    Options (module docstring, "Fill-selection rules"): margin_tau needs `margin` (margin_from_logits of the same
    rows) and is ANDed with the entropy rule when tau is given, replaces it when tau is None (fallback: index 0);
    then fill_mode="leftmost_run", no_adjacent and max_fill thin the candidates in that order, always keeping >= 1.
    min_fill=0 (verified decoding) drops the two fallbacks (empty candidate set, empty leftmost run) and returns []
    instead; the unconditional tau=None / margin_tau=None rule (the leftmost mask) is not a fallback and stays.
    """
    n = len(mask_slots)
    if margin_tau is not None and margin is None:
        raise ValueError("margin_tau needs the margin per mask row")
    if tau is None:
        cand = list(range(n))
        fallback = [0]
        ok = (margin >= margin_tau).tolist() if margin_tau is not None else None
    else:
        d = torch.tensor([s - mask_slots[0] for s in mask_slots], dtype=entropy.dtype, device=entropy.device)
        adjusted = entropy + lam * d
        # the pass flags, the argmin and the margin flags are read back from the device in ONE sync
        parts = [(adjusted < tau).long(), adjusted.argmin().view(1)]
        if margin_tau is not None:
            parts.append((margin >= margin_tau).long())
        flat = torch.cat(parts).tolist()
        cand = [k for k in range(n) if flat[k]]
        fallback = [int(flat[n])]
        ok = flat[n + 1:] if margin_tau is not None else None
    if min_fill < 1:
        fallback = []
    if margin_tau is not None:
        cand = [k for k in cand if ok[k]]
    elif tau is None:
        cand = [0]                                                     # reference: exactly the leftmost mask
    if not cand:
        cand = fallback
    if fill_mode == "leftmost_run":
        run = []
        for k in cand:
            if k != len(run) or (k > 0 and mask_slots[k] != mask_slots[k - 1] + 1):
                break
            run.append(k)
        cand = run or ([] if min_fill < 1 else [0])
    if no_adjacent:
        kept = []
        for i, k in enumerate(cand):
            if i > 0 and mask_slots[k] == mask_slots[cand[i - 1]] + 1:
                continue                     # continues a run of adjacent candidates: only its leftmost is filled
            kept.append(k)
        cand = kept
    if max_fill is not None:
        cand = cand[: int(max_fill)]
    return cand


def verify_drafts(ar_logits, slots, is_mask):
    """The verification walk of verified decoding.  ar_logits[j] = next-token logits of logical position (frontier + j):
    [0] the pending row (or the prefill logits), [1 + i] the row of the draft at window slot i (valid for the leading
    contiguous drafted run, which is all the walk ever reads).  Returns (j, g): slots 0..j-1 are the accepted drafts
    (each equal to the greedy argmax given the tokens before it) and g is the greedy token of slot j (the correction,
    or the token of the mask at j, or the token after the window when every slot was a draft and all were accepted).
    ar_logits may also be the greedy tokens themselves (a list of ints or a 1-D long tensor): the callers compute the
    argmax of all rows in one kernel and one device sync instead of one per walked row (same walk, same result).
    """
    if torch.is_tensor(ar_logits):
        greedy = ar_logits.argmax(-1).tolist() if ar_logits.dim() == 2 else ar_logits.tolist()
    elif ar_logits and torch.is_tensor(ar_logits[0]):
        greedy = torch.stack(list(ar_logits)).argmax(-1).tolist()
    else:
        greedy = list(ar_logits)
    j = 0
    while True:
        g = int(greedy[j])
        if j < len(slots) and not is_mask[j] and slots[j] == g:
            j += 1
            continue
        return j, g


@torch.no_grad()
def generate_verified(forward, commit, last_logits, prompt_len, max_new_tokens, window, tau, lam, mask_id, stop_ids,
                      sample, predict, margin_tau, no_adjacent, max_fill, fill_mode, device):
    """The verify=True loop (module docstring, "Verified (speculative) decoding"), shared by both decoders.

    forward(ids[T], pos[T], cache_len) -> logits [T, V] of all T window rows, forwarded on top of the first cache_len
    cached rows with the causal window mask (and written to the cache at cache_len..); commit(n) keeps exactly the
    first n cached rows (DynamicCache crop / cache_len = n).  last_logits: next-token logits of the last prompt token.
    sample(logits [n, V]) -> draft tokens [n] (argmax or a sample; the verification itself is always argmax).
    device: where the ids/pos tensors handed to forward are built (the own engine passes "cpu": its run() stages them
    itself, so no per-step host->device tensor is created here).
    Returns (new_token_ids, stats) without the cache entry (the caller adds cache / cache_len).
    """
    cur = prompt_len                            # prompt_len + len(out); the cache holds cur - (pending is not None) rows
    out, pending = [], None
    slots, is_mask = [], []                     # window slot i <-> logical position cur + i
    steps = window_rows = drafted = accepted = 0
    filled_hist, commit_hist, acc_hist, filled_positions = {}, {}, {}, []
    completion = False
    finish = "length" if max_new_tokens == 0 else None
    while finish is None:
        # 1. refill the window to W within the budget
        add = min(window - len(slots), max_new_tokens - len(out) - len(slots))
        if add > 0:
            slots += [mask_id] * add
            is_mask += [True] * add
        assert slots and len(slots) <= window and len(out) + len(slots) <= max_new_tokens
        # 2. forward [pending | drafts... | masks...] with original positions
        filled_idx = [i for i, f in enumerate(is_mask) if not f]
        mask_idx = [i for i, f in enumerate(is_mask) if f]
        n_pend = int(pending is not None)
        order = filled_idx + mask_idx
        ids = torch.tensor(([pending] if n_pend else []) + [slots[i] for i in order], dtype=torch.long, device=device)
        pos = torch.tensor(([cur - 1] if n_pend else []) + [cur + i for i in order], dtype=torch.long, device=device)
        T = int(ids.numel())
        logits = forward(ids, pos, cur - n_pend)
        steps += 1
        window_rows += T
        # 3. verify the leading drafted run (physical rows n_pend.. are the drafts in slot order): the greedy tokens of
        #    all AR rows in one argmax + one sync (the walk only reads the leading contiguous run of them)
        n_filled = len(filled_idx)
        ar_tok = logits[: n_pend + n_filled].argmax(-1).tolist()
        if not n_pend:
            ar_tok = [int(last_logits.argmax())] + ar_tok
        j, g = verify_drafts(ar_tok, slots, is_mask)
        # 4. commit the accepted drafts (their K/V are final), then the correction g as the new pending token
        keep = j
        for i in range(j):
            if slots[i] in stop_ids:
                keep, finish = i + 1, "stop"
                break
        if len(out) + keep >= max_new_tokens:
            keep = max_new_tokens - len(out)
            finish = finish or "length"
        out += slots[:keep]
        cur += keep
        commit(cur)                             # cache = [prefix | pending | accepted run]
        pending = None
        accepted += keep
        acc_hist[keep] = acc_hist.get(keep, 0) + 1
        n_commit = keep
        if finish is None:                      # len(out) < max_new_tokens: the correction fits
            out.append(g)
            cur += 1
            pending = g
            n_commit += 1
            if g in stop_ids:
                finish = "stop"
            elif len(out) >= max_new_tokens:
                finish = "length"
        commit_hist[n_commit] = commit_hist.get(n_commit, 0) + 1
        if finish is not None:
            if pending is not None:             # write its K/V so that the cache holds [prompt | output]
                forward(torch.tensor([pending], dtype=torch.long, device=device),
                        torch.tensor([cur - 1], dtype=torch.long, device=device), cur - 1)
                commit(cur)
                completion = True
            break
        # 5. window = slots j+1..; a mismatch (slot j was a draft) re-masks every draft to its right
        mismatch = j < len(slots) and not is_mask[j]
        drop = j + 1
        new_slots, new_mask = slots[drop:], is_mask[drop:]
        if mismatch:
            new_slots, new_mask = [mask_id] * len(new_slots), [True] * len(new_mask)
        # 6. draft: the rows forwarded as masks this step and still in the window
        draftable = [s for s in mask_idx if s >= drop]
        sel_pos = []
        if draftable:
            # draftable is a suffix of mask_idx (ascending), so its rows are the trailing mask rows: a slice, no gather
            first = n_pend + n_filled + (len(mask_idx) - len(draftable))
            mask_logits = logits[first:first + len(draftable)]
            mask_slots = [s - drop for s in draftable]
            prev = draftable[0] - 1
            if predict == "ar" and not mismatch and prev >= drop and not is_mask[prev]:
                frow = {s: n_pend + k for k, s in enumerate(filled_idx)}
                mask_logits = torch.cat([logits[frow[prev]][None], mask_logits[1:]], 0)
            ent = entropy_from_logits(mask_logits)
            margin = margin_from_logits(mask_logits) if margin_tau is not None else None
            toks = sample(mask_logits)
            sel = select_positions(ent, mask_slots, tau, lam, margin, margin_tau, no_adjacent, max_fill, fill_mode,
                                   min_fill=0)
            toks = toks.tolist() if sel else []
            for k in sel:
                new_slots[mask_slots[k]] = int(toks[k])
                new_mask[mask_slots[k]] = False
            sel_pos = [cur + mask_slots[k] for k in sel]
        slots, is_mask = new_slots, new_mask
        drafted += len(sel_pos)
        filled_hist[len(sel_pos)] = filled_hist.get(len(sel_pos), 0) + 1
        filled_positions.append(sel_pos)
    assert len(out) <= max_new_tokens and cur == prompt_len + len(out)
    stats = dict(steps=steps, forwards=steps + 1 + int(completion), window_rows=window_rows, generated=len(out),
                 tokens_per_step=(len(out) / steps if steps else 0.0),
                 filled_per_step=dict(sorted(filled_hist.items())), committed_per_step=dict(sorted(commit_hist.items())),
                 filled_positions=filled_positions, finish_reason=finish, verify=True,
                 accepted_per_step=dict(sorted(acc_hist.items())), drafted=drafted, accepted=accepted,
                 draft_acceptance_rate=(accepted / drafted if drafted else None), completion_forward=completion)
    return out, stats


def _crop_to(cache, keep_total):
    """Shrink the DynamicCache to its first keep_total positions (crop takes a negative 'tokens to remove')."""
    n = cache.get_seq_length() - keep_total
    if n > 0:
        cache.crop(-n)
    assert cache.get_seq_length() == keep_total


@torch.no_grad()
def generate(model, tok, prompt_ids, max_new_tokens, window=16, tau=None, lam=0.02, mask_id=None, stop_ids=None,
             greedy=True, temperature=1.0, predict="mask", generator=None, margin_tau=None, no_adjacent=False,
             max_fill=None, fill_mode="all", verify=False):
    """Decode one sequence.  Returns (new_token_ids: list[int], stats: dict).

    model: Qwen3ForCausalLM (sdpa or eager) with a mask row (model_glue.add_mask_token).  tok: tokenizer, used only to
    default stop_ids to [tok.eos_token_id] (may be None).  prompt_ids: list/1-D tensor of ids.  window: W.  tau: entropy
    threshold or None (one token per step).  lam: distance penalty.  mask_id: default model.config.cdlm_mask_token_id.
    stop_ids: ids that end generation (the stop token is kept in the output).  greedy: argmax, else sample at
    `temperature` (torch.Generator `generator`).  predict: "mask" | "ar" (see module docstring).
    margin_tau / no_adjacent / max_fill / fill_mode: fill-selection rules, all off by default (module docstring).
    verify: verified (speculative) decoding - the fills are drafts checked against the AR logits, the output is exactly
    HF greedy generate() (module docstring, "Verified (speculative) decoding"; extra stats listed there).

    stats: steps (decode forwards), forwards (steps + 1 prefill), window_rows (rows forwarded in decode steps),
    generated, tokens_per_step, filled_per_step / committed_per_step ({count: n_steps}), filled_positions (per step,
    the absolute logical positions filled, ascending), finish_reason ("stop" | "length"), cache (the DynamicCache
    holding [prompt | output], reusable for a continuation).
    """
    if predict not in ("mask", "ar"):
        raise ValueError(f"predict={predict!r}")
    no_adjacent = check_fill_options(margin_tau, no_adjacent, max_fill, fill_mode)
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

    def sample(logits):
        if greedy:
            return logits.argmax(-1)
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        return torch.multinomial(probs, 1, generator=generator).squeeze(-1)

    if verify:
        def forward(ids, pos, cache_len):
            assert cache.get_seq_length() == cache_len
            h = model.model(input_ids=ids[None], position_ids=pos[None],
                            attention_mask=window_attention_mask(model, cache_len, int(ids.numel()), dev),
                            past_key_values=cache, use_cache=True).last_hidden_state[0]
            return model.lm_head(h)

        out, stats = generate_verified(forward, lambda n: _crop_to(cache, n), last_logits, prompt_len, max_new_tokens,
                                       window, tau, lam, mask_id, stop_ids, sample, predict, margin_tau, no_adjacent,
                                       max_fill, fill_mode, dev)
        assert cache.get_seq_length() == prompt_len + len(out)
        stats["cache"] = cache
        return out, stats

    cur = prompt_len
    out = []
    slots = [mask_id] * min(window, max_new_tokens)                  # window slot i <-> logical position cur + i
    is_mask = [True] * len(slots)
    steps = window_rows = 0
    filled_hist, commit_hist, filled_positions = {}, {}, []
    finish = "length" if not slots else None

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
        margin = margin_from_logits(mask_logits) if margin_tau is not None else None
        toks = sample(mask_logits)
        sel = select_positions(ent, mask_slots, tau, lam, margin, margin_tau, no_adjacent, max_fill, fill_mode)
        toks = toks.tolist() if sel else []
        for k in sel:
            slots[mask_slots[k]] = int(toks[k])
            is_mask[mask_slots[k]] = False
        filled_hist[len(sel)] = filled_hist.get(len(sel), 0) + 1
        filled_positions.append([cur + mask_slots[k] for k in sel])

    assert len(out) <= max_new_tokens and cache.get_seq_length() == prompt_len + len(out)
    stats = dict(steps=steps, forwards=steps + 1, window_rows=window_rows, generated=len(out),
                 tokens_per_step=(len(out) / steps if steps else 0.0),
                 filled_per_step=dict(sorted(filled_hist.items())), committed_per_step=dict(sorted(commit_hist.items())),
                 filled_positions=filled_positions, finish_reason=finish, cache=cache)
    return out, stats
