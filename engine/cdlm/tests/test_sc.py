"""Plain-assert tests for engine.cdlm.sc_decoder (CPU only, tiny random Qwen3 from test_fast's fixture).

Run from the repo root:  CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -m engine.cdlm.tests.test_sc
"""
import os
import sys
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch

from engine.cdlm.block_decoder import block_generate
from engine.cdlm.sc_decoder import REMASK, STOP_RULES, denoise_block, parse_remask, sc_generate
from engine.cdlm.tests.test_fast import pair, random_prompts


def test_parse_remask():
    assert parse_remask("none") == ("none", None) and parse_remask(None) == ("none", None)
    assert parse_remask("lowconf:0.9") == ("lowconf", 0.9) and parse_remask("drop:0.1") == ("drop", 0.1)
    for bad in ("foo", "lowconf"):
        try:
            parse_remask(bad)
            assert False, bad
        except ValueError:
            pass


def test_reduces_to_block_decoder():
    """confirm_k=0 + remask none + the same fill rule == block_generate (tokens and step counts)."""
    for seed in range(3):
        hf, fast, mask_id = pair(seed)
        g = torch.Generator().manual_seed(seed)
        for prompt in random_prompts(g, k=3):
            for kw in (dict(steps=None, tau=0.5, anchor_leftmost=True), dict(steps=4, anchor_leftmost=False, tau=None),
                       dict(steps=None, tau=0.3, anchor_leftmost=False)):
                ref, rs = block_generate(fast, None, prompt, 20, block=8, mask_id=mask_id, stop_ids=[], **kw)
                fast.kv.zero_()
                out, st = sc_generate(fast, None, prompt, 20, block=8, mask_id=mask_id, stop_ids=[], remask="none",
                                      confirm_k=0, **kw)
                fast.kv.zero_()
                assert out == ref, (seed, kw, out, ref)
                assert st["steps"] == rs["steps"] and st["corrections"] == 0 and st["reeval"] == 0, (st, rs)


def test_strategies_run_and_counters():
    hf, fast, mask_id = pair(1)
    g = torch.Generator().manual_seed(3)
    prompt = random_prompts(g, k=1)[0]
    specs = ["none", "all", "lowconf:0.9", "entropy:0.5", "random:0.5", "scheduled:0.5", "drop:0.05"]
    assert {s.split(":")[0] for s in specs} == set(REMASK)
    for spec in specs:
        for stop in STOP_RULES:
            out, st = sc_generate(fast, None, prompt, 16, block=8, mask_id=mask_id, stop_ids=[], remask=spec, confirm_k=1,
                                  steps=None, tau=0.5, anchor_leftmost=True, stop=stop, stop_tau=0.9, trace=True,
                                  generator=torch.Generator().manual_seed(0))
            fast.kv.zero_()
            assert len(out) == 16 and st["blocks"] == 2, (spec, stop, st)
            assert all(s <= 9 for s in st["steps_per_block"]), (spec, st["steps_per_block"])   # max_steps = K + 1
            n_re = sum(len(t["reevaluated"]) for b in st["trace"] for t in b["steps"])
            n_co = sum(len(t["corrected"]) for b in st["trace"] for t in b["steps"])
            assert n_re == st["reeval"] and n_co == st["corrections"] and n_co <= n_re, (spec, st)
            if spec == "none":
                assert st["reeval"] == 0 and st["accepted_tentative"] == 16, st
            # every slot ends as a token, every block trace ends with no MASK
            for b in st["trace"]:
                assert all(s != 0 for s in b["steps"][-1]["status"]), b["steps"][-1]
    # fixed schedule + remask all
    out, st = sc_generate(fast, None, prompt, 16, block=8, mask_id=mask_id, stop_ids=[], remask="all", confirm_k=1,
                          steps=4, tau=None, anchor_leftmost=False)
    assert len(out) == 16 and all(s <= 6 for s in st["steps_per_block"]), st


def test_denoise_block_confirms_and_corrects():
    """With remask='all' and confirm_k=1 every placed token is re-evaluated at least once before the block can end
    (unless the budget runs out), and tokens that change are counted as corrections."""
    hf, fast, mask_id = pair(2)
    g = torch.Generator().manual_seed(5)
    prompt = random_prompts(g, k=1)[0]
    P = len(prompt)
    fast.kv.zero_()
    fast.prefill(torch.tensor(list(prompt)), 0)

    def fwd(ids, pos, rows):
        return fast.forward(torch.tensor(ids), torch.tensor(pos), P, write=False, rows=rows)

    toks, info = denoise_block(fwd, P, 8, mask_id, steps=None, tau=0.0, anchor_leftmost=True, remask="all", confirm_k=1,
                               max_steps=20, max_remask_frac=1.0, trace=True)
    assert len(toks) == 8 and all(t != mask_id for t in toks)
    # tau=0 fills everything at step 0; step 1 re-evaluates all 8; confirmations + corrections == 8 at step 1
    s1 = info["trace"][1]
    assert len(s1["reevaluated"]) == 8 and len(s1["confirmed"]) + len(s1["corrected"]) == 8, s1
    assert info["reeval"] >= 8 and info["corrections"] + info["confirmed_by_agree"] >= 8
    if info["accepted_tentative"] == 0:
        assert info["confirmed_by_agree"] == 8


def test_cache_valid_after_generation():
    for seed in range(2):
        hf, fast, mask_id = pair(seed)
        g = torch.Generator().manual_seed(seed + 7)
        prompt = random_prompts(g, k=2)[0]
        for merge in (True, False):
            out, st = sc_generate(fast, None, prompt, 20, block=8, mask_id=mask_id, stop_ids=[], remask="lowconf:0.9",
                                  confirm_k=1, steps=None, tau=0.5, merge_commit=merge)
            K = [fast.kv.K[i, :, :, : st["cache_len"]].clone() for i in range(len(fast.layers))]
            fast.kv.zero_()
            fast.prefill(torch.tensor(list(prompt) + list(out)), 0)
            for i in range(len(fast.layers)):
                d = (K[i] - fast.kv.K[i, :, :, : st["cache_len"]]).abs().max().item()
                assert d < 1e-4, (seed, merge, i, d)
            fast.kv.zero_()


def test_no_ar_head_dependence():
    hf, fast, mask_id = pair(4)
    g = torch.Generator().manual_seed(11)
    prompt = random_prompts(g, k=1)[0]
    kw = dict(remask="all", confirm_k=1, steps=None, tau=0.5, anchor_leftmost=True)
    ref, _ = sc_generate(fast, None, prompt, 16, block=8, mask_id=mask_id, stop_ids=[], **kw)
    fast.kv.zero_()
    orig = fast.run

    def run(tokens, pos_ids, cache_len, rows, static=False, use_graph=False):
        out = orig(tokens, pos_ids, cache_len, list(range(int(tokens.numel()))), static=static, use_graph=use_graph).clone()
        is_mask = tokens == mask_id
        out[~is_mask] = torch.randn_like(out[~is_mask]) * 10
        return out[torch.as_tensor(list(rows), dtype=torch.long)] if len(rows) else out[:0]

    fast.run = run
    try:
        out, _ = sc_generate(fast, None, prompt, 16, block=8, mask_id=mask_id, stop_ids=[], **kw)
    finally:
        fast.run = orig
    assert out == ref, (out, ref)


def test_stop_ids():
    hf, fast, mask_id = pair(2)
    g = torch.Generator().manual_seed(5)
    prompt = random_prompts(g, k=1)[0]
    out, st = sc_generate(fast, None, prompt, 13, block=8, mask_id=mask_id, stop_ids=list(range(200)), remask="all",
                          confirm_k=1, steps=None, tau=0.5)
    assert len(out) == 1 and st["finish_reason"] == "stop" and st["cache_len"] == len(prompt) + 1, st


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
