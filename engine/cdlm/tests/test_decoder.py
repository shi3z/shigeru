"""Plain-assert tests for engine.cdlm.decoder (CPU only, tiny random Qwen3, no downloads).

Run from the repo root:  CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -m engine.cdlm.tests.test_decoder
"""
import copy
import os
import sys
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

from engine.cdlm import add_mask_token
from engine.cdlm import decoder as decoder_mod
from engine.cdlm.decoder import generate, margin_from_logits, select_positions

VOCAB = 97
BASE_CFG = Qwen3Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                       tie_word_embeddings=False, initializer_range=0.2)


def tiny_model(impl, seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    m = Qwen3ForCausalLM(copy.deepcopy(BASE_CFG)).to(dtype).eval()
    m.set_attn_implementation(impl)
    mask_id = add_mask_token(m)["mask_id"]                      # row VOCAB; mean-initialised by add_mask_token ...
    with torch.no_grad():                                       # ... then made random so nothing relies on that init
        g = torch.Generator().manual_seed(seed + 100)
        m.get_input_embeddings().weight[mask_id] = torch.randn(m.config.hidden_size, generator=g, dtype=dtype)
        m.get_output_embeddings().weight[mask_id] = torch.randn(m.config.hidden_size, generator=g, dtype=dtype)
    return m, mask_id


def hf_greedy(m, prompt, n, stop_ids):
    kw = dict(max_new_tokens=n, min_new_tokens=0, do_sample=False, pad_token_id=0, num_beams=1)
    if stop_ids:
        kw["eos_token_id"] = list(stop_ids)
    g = m.generate(prompt[None], **kw)[0]
    return g[prompt.numel():].tolist()


def naive_mask_greedy(m, prompt, n, stop_ids, mask_id):
    """Uncached reference for predict='mask', tau=None: forward [prompt | out | MASK] from scratch each step and take
    the MASK row's argmax (the decoder's leftmost mask sees exactly this context)."""
    out = []
    for _ in range(n):
        ids = torch.tensor(prompt.tolist() + out + [mask_id])
        t = int(m(input_ids=ids[None], use_cache=False).logits[0, -1].argmax())
        out.append(t)
        if t in stop_ids:
            break
    return out


def random_prompts(g, k=6):
    return [torch.randint(0, VOCAB, (int(torch.randint(2, 24, (1,), generator=g)),), generator=g) for _ in range(k)]


def check_stats(stats, out, max_new):
    assert stats["forwards"] == stats["steps"] + 1 and stats["generated"] == len(out) <= max_new
    assert sum(stats["committed_per_step"].values()) == stats["steps"]
    assert sum(k * v for k, v in stats["committed_per_step"].items()) == len(out)
    assert stats["finish_reason"] in ("stop", "length")
    assert stats["cache"].get_seq_length() == stats["cache"].layers[0].keys.shape[-2]
    check_filled_positions(stats)


def check_filled_positions(stats):
    """filled_positions: one list per fill step (every step but the last, which only commits), lengths matching the
    filled_per_step histogram, each position filled once, ascending within a step."""
    fp = stats["filled_positions"]
    assert len(fp) == max(stats["steps"] - 1, 0), (len(fp), stats["steps"])
    assert sorted(len(x) for x in fp) == sorted(k for k, v in stats["filled_per_step"].items() for _ in range(v))
    flat = [q for x in fp for q in x]
    assert len(flat) == len(set(flat)) and all(x == sorted(x) and len(x) >= 1 for x in fp)


def test_tau_none_equals_hf_greedy():
    """(1) predict='ar', tau=None: output == HF greedy generate(), exactly, for random prompts / windows / impls."""
    g = torch.Generator().manual_seed(1)
    n_checked = 0
    for impl in ("sdpa", "eager"):
        m, mask_id = tiny_model(impl)
        for prompt in random_prompts(g):
            for stop_ids in ([], [int(torch.randint(0, VOCAB, (1,), generator=g))]):
                ref = hf_greedy(m, prompt, 30, stop_ids)
                for W in (1, 3, 16):
                    out, st = generate(m, None, prompt, 30, window=W, tau=None, mask_id=mask_id, stop_ids=stop_ids,
                                       predict="ar")
                    assert out == ref, (impl, W, stop_ids, out, ref)
                    check_stats(st, out, 30)
                    assert set(st["filled_per_step"]) <= {1} and set(st["committed_per_step"]) <= {0, 1}
                    # one token per step: step 1 commits nothing, every later step commits exactly one token
                    assert st["committed_per_step"].get(0, 0) == 1 and st["steps"] == len(out) + 1, st
                    n_checked += 1
    assert n_checked == 72, n_checked


def test_tau_none_mask_predictor_matches_naive():
    """(1b) predict='mask' (faithful WeDLM), tau=None: output == uncached naive loop predicting from the MASK row."""
    g = torch.Generator().manual_seed(2)
    for impl in ("sdpa", "eager"):
        m, mask_id = tiny_model(impl, seed=1)
        for prompt in random_prompts(g, 4):
            stop_ids = [int(torch.randint(0, VOCAB, (1,), generator=g))]
            ref = naive_mask_greedy(m, prompt, 20, stop_ids, mask_id)
            for W in (1, 4, 16):
                out, st = generate(m, None, prompt, 20, window=W, tau=None, mask_id=mask_id, stop_ids=stop_ids)
                assert out == ref, (impl, W, out, ref)
                check_stats(st, out, 20)
        # the two predictors genuinely differ on an untrained model (otherwise the test above would be vacuous)
        prompt = random_prompts(g, 1)[0]
        a, _ = generate(m, None, prompt, 20, window=4, tau=None, mask_id=mask_id, stop_ids=[], predict="ar")
        b, _ = generate(m, None, prompt, 20, window=4, tau=None, mask_id=mask_id, stop_ids=[], predict="mask")
        assert a == hf_greedy(m, prompt, 20, []) and a != b


def test_cache_matches_full_forward():
    """(2) after generate(), the cache's K/V for [prompt | output] equal a fresh full causal forward of that sequence
    (float32: < 1e-5; float64 sdpa: < 1e-9; float64 eager: < 1e-4 because HF eager runs the softmax in float32,
    modeling_qwen3.py:194, the same caveat as test_cdlm.test_tau_none_equivalence), for every tau / predictor / window."""
    g = torch.Generator().manual_seed(3)
    worst = {}
    for dtype, impl, tol in ((torch.float32, "sdpa", 1e-5), (torch.float32, "eager", 1e-5),
                             (torch.float64, "sdpa", 1e-9), (torch.float64, "eager", 1e-4)):
        m, mask_id = tiny_model(impl, seed=2, dtype=dtype)
        for prompt in random_prompts(g, 3):
            for tau, predict, W in ((None, "ar", 16), (None, "mask", 4), (0.5, "mask", 16), (2.0, "mask", 8),
                                    (1e9, "mask", 16), (1.0, "ar", 16)):
                out, st = generate(m, None, prompt, 25, window=W, tau=tau, lam=0.02, mask_id=mask_id,
                                   stop_ids=[3], predict=predict)
                check_stats(st, out, 25)
                cache = st["cache"]
                full = torch.tensor(prompt.tolist() + out)
                ref = DynamicCache(config=m.config)
                m.model(input_ids=full[None], past_key_values=ref, use_cache=True)
                assert cache.get_seq_length() == ref.get_seq_length() == full.numel()
                for li in range(m.config.num_hidden_layers):
                    for a, b in ((cache.layers[li].keys, ref.layers[li].keys),
                                 (cache.layers[li].values, ref.layers[li].values)):
                        assert a.shape == b.shape, (a.shape, b.shape)
                        d = float((a - b).abs().max())
                        worst[(dtype, impl)] = max(worst.get((dtype, impl), 0.0), d)
                        assert d < tol, (dtype, impl, tau, predict, W, li, d)
    print("  max |cache - full forward|: " + ", ".join(f"{str(k[0])[6:]}/{k[1]} {v:.2e}" for k, v in worst.items()))


def test_tau_large_terminates_and_respects_budget():
    """(3) tau huge: every mask is filled each step (W tokens per step after the first), the loop terminates, the output
    length is exactly max_new_tokens (no stop) or ends at the first stop token, also when max_new_tokens % W != 0."""
    g = torch.Generator().manual_seed(4)
    m, mask_id = tiny_model("sdpa", seed=3)
    for prompt in random_prompts(g, 4):
        for W, n in ((16, 37), (16, 16), (4, 50), (1, 7), (8, 3)):
            out, st = generate(m, None, prompt, n, window=W, tau=1e9, mask_id=mask_id, stop_ids=[])
            check_stats(st, out, n)
            assert len(out) == n and st["finish_reason"] == "length", (W, n, len(out))
            # every step fills all masks: fills are min(W, remaining budget); commits are W except the last partial one
            assert all(k <= W for k in st["filled_per_step"]) and st["filled_per_step"].get(min(W, n), 0) >= 1
            assert st["steps"] == -(-n // W) + 1, (W, n, st["steps"])         # ceil(n/W) fill steps + 1 commit step
            assert st["committed_per_step"].get(0, 0) == 1 and st["tokens_per_step"] > 0
            # with a stop token the output stops there and the cache matches
            stop = out[min(len(out) - 1, 5)]
            out2, st2 = generate(m, None, prompt, n, window=W, tau=1e9, mask_id=mask_id, stop_ids=[stop])
            assert out2 == out[:out.index(stop) + 1] and st2["finish_reason"] == "stop", (W, n)
            assert st2["cache"].get_seq_length() == prompt.numel() + len(out2)
    # sampling path runs and respects the budget too
    out, st = generate(m, None, random_prompts(g, 1)[0], 23, window=16, tau=0.5, mask_id=mask_id, stop_ids=[],
                       greedy=False, generator=torch.Generator().manual_seed(0))
    assert len(out) == 23 and st["finish_reason"] == "length"
    # max_new_tokens=0 is a no-op
    out, st = generate(m, None, random_prompts(g, 1)[0], 0, window=16, tau=None, mask_id=mask_id, stop_ids=[])
    assert out == [] and st["steps"] == 0 and st["finish_reason"] == "length"


# ----------------------------------------------------------------------------------------------------------------------
# Fill-selection rules (margin_tau, no_adjacent, max_fill, fill_mode) — docs/distill_design.md section 10.3

def reference_select_positions(entropy, mask_slots, tau, lam, *_, **__):
    """Frozen copy of select_positions as it was before the fill-selection options (the reference rule); extra
    arguments are ignored so it can be patched in place of the new function."""
    if tau is None:
        return [0]
    d = torch.tensor([s - mask_slots[0] for s in mask_slots], dtype=entropy.dtype, device=entropy.device)
    adjusted = entropy + lam * d
    cand = (adjusted < tau).nonzero(as_tuple=True)[0]
    if cand.numel() > 0:
        return cand.tolist()
    return [int(adjusted.argmin())]


def documented_select(entropy, mask_slots, tau, lam, margin, margin_tau, no_adjacent, max_fill, fill_mode):
    """Pure-Python transcription of the documented semantics (decoder.py, 'Fill-selection rules'), written
    independently of select_positions: candidate set, fallback, then leftmost_run -> no_adjacent -> max_fill."""
    n = len(mask_slots)
    ent = [float(e) for e in entropy]
    adj = [ent[k] + lam * (mask_slots[k] - mask_slots[0]) for k in range(n)]
    if tau is None and margin_tau is None:
        cand = [0]
    else:
        cand = [k for k in range(n) if (tau is None or adj[k] < tau) and (margin_tau is None or float(margin[k]) >= margin_tau)]
        if not cand:
            cand = [min(range(n), key=lambda k: adj[k])] if tau is not None else [0]
    if fill_mode == "leftmost_run":
        run = []
        for k in range(n):
            if k not in cand or (k > 0 and mask_slots[k] != mask_slots[k - 1] + 1):
                break
            run.append(k)
        cand = run or [0]
    if no_adjacent:
        runs = []                                                    # maximal runs of candidates at consecutive slots
        for k in cand:
            if runs and mask_slots[k] == mask_slots[runs[-1][-1]] + 1:
                runs[-1].append(k)
            else:
                runs.append([k])
        cand = [r[0] for r in runs]
    if max_fill is not None:
        cand = cand[:max_fill]
    return cand


OPTION_SETS = [dict(), dict(no_adjacent=True), dict(max_fill=1), dict(max_fill=3), dict(margin_tau=0.3),
               dict(margin_tau=0.7), dict(fill_mode="leftmost_run"), dict(no_adjacent=True, max_fill=2),
               dict(margin_tau=0.2, fill_mode="leftmost_run"), dict(margin_tau=0.1, no_adjacent=True, max_fill=4),
               dict(no_adjacent=True, fill_mode="leftmost_run")]


def random_mask_slots(g, n):
    """n ascending slots with random gaps (filled-after-a-gap rows sit between masks in fill_mode='all')."""
    gaps = torch.randint(0, 3, (n,), generator=g).tolist()
    gaps[0] = 0
    slots, cur = [], 0
    for gp in gaps:
        cur += gp
        slots.append(cur)
        cur += 1
    return slots


def test_fill_options_default_to_reference():
    """(a) with every option off, select_positions is the frozen reference rule on random inputs (incl. gapped mask
    slots) and generate() gives identical tokens / stats with the reference rule patched in; every option combination
    equals the documented semantics transcribed independently; invalid options raise."""
    g = torch.Generator().manual_seed(11)
    n_unit = 0
    for _ in range(300):
        n = int(torch.randint(1, 17, (1,), generator=g))
        slots = random_mask_slots(g, n)
        ent = torch.rand(n, generator=g, dtype=torch.float32) * 4.0
        margin = torch.rand(n, generator=g, dtype=torch.float32)
        for tau in (None, 0.5, 2.0, 1e9):
            for lam in (0.0, 0.02, 0.3):
                assert select_positions(ent, slots, tau, lam) == reference_select_positions(ent, slots, tau, lam)
                assert select_positions(ent, slots, tau, lam, margin) == reference_select_positions(ent, slots, tau, lam)
                for opts in OPTION_SETS:
                    o = dict(margin=margin, margin_tau=None, no_adjacent=False, max_fill=None, fill_mode="all")
                    o.update(opts)
                    got = select_positions(ent, slots, tau, lam, **o)
                    want = documented_select(ent, slots, tau, lam, **o)
                    assert got == want, (tau, lam, opts, slots, got, want)
                    assert len(got) >= 1 and got == sorted(got) and all(0 <= k < n for k in got)
                    n_unit += 1
    # generate(): default arguments == the frozen reference rule patched into the decoder module
    m, mask_id = tiny_model("sdpa", seed=5)
    n_gen = 0
    for prompt in random_prompts(g, 3):
        for W in (4, 16):
            for tau in (None, 0.5, 2.5, 1e9):
                for predict in ("mask", "ar"):
                    out, st = generate(m, None, prompt, 30, window=W, tau=tau, mask_id=mask_id, stop_ids=[5], predict=predict)
                    check_stats(st, out, 30)
                    decoder_mod.select_positions = reference_select_positions
                    try:
                        ref, rs = generate(m, None, prompt, 30, window=W, tau=tau, mask_id=mask_id, stop_ids=[5], predict=predict)
                    finally:
                        decoder_mod.select_positions = select_positions
                    assert out == ref and {k: v for k, v in st.items() if k != "cache"} == {k: v for k, v in rs.items() if k != "cache"}
                    n_gen += 1
    for bad in (dict(fill_mode="rightmost"), dict(max_fill=0), dict(margin_tau=1.5)):
        try:
            generate(m, None, prompt, 5, window=4, mask_id=mask_id, stop_ids=[], **bad)
            raise RuntimeError(f"expected ValueError for {bad}")
        except ValueError:
            pass
    try:
        select_positions(torch.zeros(3), [0, 1, 2], None, 0.02, None, 0.5)
        raise RuntimeError("expected ValueError: margin_tau without margin")
    except ValueError:
        pass
    print(f"  {n_unit} selection cases == documented semantics, {n_gen} generate() cases == frozen reference rule")


def test_no_adjacent_never_fills_adjacent():
    """(b) no_adjacent: in no step are two filled positions adjacent (from stats['filled_positions']), for tau in
    {2.5, 3.2, 1e9}, margin-only, W in {4, 16}; non-vacuous (some steps fill >= 2); tau=1e9 degenerates to one fill per
    step (all masks pass and are contiguous, so every run collapses to its leftmost)."""
    g = torch.Generator().manual_seed(12)
    m, mask_id = tiny_model("sdpa", seed=6)
    multi = adjacent_without_rule = 0
    for prompt in random_prompts(g, 4):
        P = prompt.numel()
        for W in (4, 16):
            for kw in (dict(tau=2.5), dict(tau=3.2), dict(tau=1e9), dict(tau=None, margin_tau=0.3), dict(tau=2.5, margin_tau=0.2)):
                out, st = generate(m, None, prompt, 40, window=W, mask_id=mask_id, stop_ids=[], no_adjacent=True, **kw)
                check_stats(st, out, 40)
                assert len(out) == 40
                for pos in st["filled_positions"]:
                    assert all(b - a >= 2 for a, b in zip(pos, pos[1:])), (W, kw, pos)
                    assert all(P <= q < P + 40 for q in pos)
                    multi += len(pos) >= 2
                if kw.get("tau") == 1e9:
                    assert st["filled_per_step"] == {1: 40}, st["filled_per_step"]
                # the same runs without the rule do fill adjacent masks (the rule is not vacuous)
                _, st0 = generate(m, None, prompt, 40, window=W, mask_id=mask_id, stop_ids=[], **kw)
                adjacent_without_rule += sum(b - a == 1 for pos in st0["filled_positions"] for a, b in zip(pos, pos[1:]))
    assert multi > 0 and adjacent_without_rule > 0, (multi, adjacent_without_rule)
    print(f"  {multi} steps filled >= 2 masks, none adjacent (without the rule: {adjacent_without_rule} adjacent pairs)")


def test_max_fill_respected():
    """(c) max_fill caps the fills per step; with tau=1e9 every step fills exactly min(max_fill, W) (budget aside)
    and steps == ceil(n / min(max_fill, W)) + 1; stacks with no_adjacent and margin_tau."""
    g = torch.Generator().manual_seed(13)
    m, mask_id = tiny_model("sdpa", seed=7)
    for prompt in random_prompts(g, 3):
        for W in (4, 16):
            for mf in (1, 2, 3, 5):
                for kw in (dict(tau=1e9), dict(tau=2.5), dict(tau=None, margin_tau=0.3), dict(tau=2.5, no_adjacent=True)):
                    out, st = generate(m, None, prompt, 40, window=W, mask_id=mask_id, stop_ids=[], max_fill=mf, **kw)
                    check_stats(st, out, 40)
                    assert len(out) == 40 and max(st["filled_per_step"]) <= mf, (W, mf, kw, st["filled_per_step"])
                    if kw.get("tau") == 1e9:
                        k = min(mf, W)
                        assert st["steps"] == -(-40 // k) + 1 and st["filled_per_step"].get(k, 0) == 40 // k, (W, mf, st)
    # max_fill=1 with tau=1e9 is one token per step but NOT the tau=None step: it fills the lowest adjusted mask
    out1, st1 = generate(m, None, prompt, 20, window=8, tau=1e9, mask_id=mask_id, stop_ids=[], max_fill=1)
    assert st1["filled_per_step"] == {1: 20} and all(len(x) == 1 for x in st1["filled_positions"])


def test_margin_tau_and_leftmost_run():
    """margin_tau: every filled mask has top1-top2 margin >= margin_tau unless it is the mandatory fallback (tau=None:
    the leftmost mask; tau given: argmin of entropy + lam*d), and with tau it is ANDed with the entropy rule (spied
    through the module's select_positions).  leftmost_run: the positions filled in a step are exactly
    cur, cur+1, ..., so every fill is committed in the next step (commit histogram == fill histogram + {0: 1}), the
    window never carries filled-after-a-gap rows, and the rule does speed up decoding over tau=None on this model."""
    g = torch.Generator().manual_seed(14)
    m, mask_id = tiny_model("sdpa", seed=8)
    calls = []

    def spy(ent, mask_slots, tau, lam, margin=None, margin_tau=None, no_adjacent=False, max_fill=None, fill_mode="all"):
        sel = select_positions(ent, mask_slots, tau, lam, margin, margin_tau, no_adjacent, max_fill, fill_mode)
        calls.append((ent.clone(), list(mask_slots), tau, lam, None if margin is None else margin.clone(), margin_tau, sel))
        return sel

    decoder_mod.select_positions = spy
    try:
        n_margin = n_fallback = 0
        for prompt in random_prompts(g, 4):
            P = prompt.numel()
            for W in (4, 16):
                for tau in (None, 0.5, 2.5):
                    for mt in (0.1, 0.5, 0.9):
                        calls.clear()
                        out, st = generate(m, None, prompt, 30, window=W, tau=tau, mask_id=mask_id, stop_ids=[], margin_tau=mt)
                        check_stats(st, out, 30)
                        assert len(calls) == st["steps"] - 1
                        for ent, slots, _, lam, margin, _, sel in calls:
                            assert margin is not None and margin.shape == ent.shape
                            adj = ent + lam * torch.tensor([s - slots[0] for s in slots], dtype=ent.dtype)
                            passing = [k for k in range(len(slots)) if margin[k] >= mt and (tau is None or adj[k] < tau)]
                            if passing:
                                assert sel == passing; n_margin += len(sel)
                            else:
                                assert sel == ([0] if tau is None else [int(adj.argmin())]); n_fallback += 1
                        # the margin is the documented quantity: recompute it from a fresh forward of the first step
                        ids = torch.tensor(prompt.tolist() + [mask_id] * min(W, 30))
                        lg = m(input_ids=ids[None], use_cache=False).logits[0, P:]
                        assert torch.allclose(calls[0][4], margin_from_logits(lg), atol=1e-6)
                        # leftmost_run on top: contiguous from the leftmost mask, committed next step
                        out2, st2 = generate(m, None, prompt, 30, window=W, tau=tau, mask_id=mask_id, stop_ids=[],
                                             margin_tau=mt, fill_mode="leftmost_run")
                        check_stats(st2, out2, 30)
                        cur = P
                        for pos in st2["filled_positions"]:
                            assert pos == list(range(cur, cur + len(pos))), (W, tau, mt, cur, pos)
                            cur += len(pos)
                        assert cur == P + 30 and len(out2) == 30
                        fh = dict(st2["filled_per_step"]); fh[0] = 1
                        assert st2["committed_per_step"] == dict(sorted(fh.items())), (st2["filled_per_step"], st2["committed_per_step"])
        assert n_margin > 0 and n_fallback > 0, (n_margin, n_fallback)
    finally:
        decoder_mod.select_positions = select_positions
    # leftmost_run with the plain entropy rule: same contiguity, and tau=1e9 is exactly W per step (no gaps to wait on)
    for prompt in random_prompts(g, 3):
        P = prompt.numel()
        for W in (4, 16):
            for tau in (0.5, 2.5, 1e9):
                out, st = generate(m, None, prompt, 40, window=W, tau=tau, mask_id=mask_id, stop_ids=[], fill_mode="leftmost_run")
                check_stats(st, out, 40)
                cur = P
                for pos in st["filled_positions"]:
                    assert pos == list(range(cur, cur + len(pos))); cur += len(pos)
                assert cur == P + 40
                if tau == 1e9:
                    assert st["steps"] == -(-40 // W) + 1 and st["filled_per_step"].get(min(W, 40), 0) >= 1
                fh = dict(st["filled_per_step"]); fh[0] = 1
                assert st["committed_per_step"] == dict(sorted(fh.items()))
    # a stop token inside a run: the run is committed up to the stop, the rest dropped, the cache matches
    stop = out[5]
    out2, st2 = generate(m, None, prompt, 40, window=16, tau=1e9, mask_id=mask_id, stop_ids=[stop], fill_mode="leftmost_run")
    assert out2 == out[:out.index(stop) + 1] and st2["finish_reason"] == "stop"
    assert st2["cache"].get_seq_length() == P + len(out2)


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1; print(f"PASS {t.__name__}")
        except Exception:
            failed += 1; print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
