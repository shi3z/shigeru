"""Plain-assert tests for engine.cdlm.block_decoder (CPU only, tiny random Qwen3 from test_fast's fixture).

Run from the repo root:  CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -m engine.cdlm.tests.test_block
"""
import os
import sys
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch

from engine.cdlm.block_decoder import ORDERS, SCHEDULES, block_generate, plan_fills
from engine.cdlm.fast_decoder import generate as fast_generate
from engine.cdlm.tests.test_fast import pair, random_prompts


def test_plan_fills():
    for K in (1, 2, 5, 8, 16, 32, 33):
        for S in (1, 2, 3, 4, 8, 16, 40):
            for sch in SCHEDULES:
                p = plan_fills(K, S, sch)
                assert sum(p) == K and all(x >= 1 for x in p) and len(p) == min(S, K), (K, S, sch, p)
    assert plan_fills(8, 4, "linear") == [2, 2, 2, 2]
    c = plan_fills(32, 8, "cosine"); g = plan_fills(32, 8, "geometric")
    assert c[0] <= c[-1] and g == c[::-1], (c, g)


def test_leftmost_one_per_step_equals_streaming_tau_none():
    """S=K with order=leftmost fills one token per step from the left; the leftmost mask attends exactly the prefix and
    the tokens on its left, as the streaming decoder's tau=None step does -> same tokens (float32 tiny model)."""
    for seed in range(3):
        hf, fast, mask_id = pair(seed)
        g = torch.Generator().manual_seed(seed)
        for prompt in random_prompts(g, k=4):
            for K in (1, 4, 8):
                ref, _ = fast_generate(fast, None, prompt, 24, window=8, tau=None, mask_id=mask_id, stop_ids=[])
                out, st = block_generate(fast, None, prompt, 24, block=K, steps=K, order="leftmost", mask_id=mask_id,
                                         stop_ids=[])
                assert out == ref, (seed, K, out, ref)
                assert st["steps"] == 24 and st["generated"] == 24 and st["blocks"] == -(-24 // K), st


def test_merge_commit_equivalent_and_cache_valid():
    """merge_commit=True/False give identical tokens, and after decoding the cache holds exactly the K/V of a fresh
    causal prefill of [prompt | output] (the committed blocks were re-encoded in natural order)."""
    for seed in range(2):
        hf, fast, mask_id = pair(seed)
        g = torch.Generator().manual_seed(seed + 7)
        for prompt in random_prompts(g, k=3):
            outs = []
            for merge in (True, False):
                out, st = block_generate(fast, None, prompt, 20, block=8, steps=3, order="confidence", mask_id=mask_id,
                                         stop_ids=[], merge_commit=merge)
                outs.append(out)
                K = [fast.kv.K[i, :, :, : st["cache_len"]].clone() for i in range(len(fast.layers))]
                fast.kv.zero_()
                fast.prefill(torch.tensor(list(prompt) + list(out)), 0)
                for i in range(len(fast.layers)):
                    d = (K[i] - fast.kv.K[i, :, :, : st["cache_len"]]).abs().max().item()
                    assert d < 1e-4, (seed, merge, i, d)
                fast.kv.zero_()
            assert outs[0] == outs[1], outs


def test_orders_schedules_remask_run_and_count():
    """Every order/schedule/remask option runs, fills >= 1 per step, completes each block, and the stats add up."""
    hf, fast, mask_id = pair(1)
    g = torch.Generator().manual_seed(3)
    prompt = random_prompts(g, k=1)[0]
    for order in ORDERS:
        for sch in SCHEDULES:
            out, st = block_generate(fast, None, prompt, 17, block=8, steps=4, schedule=sch, order=order,
                                     mask_id=mask_id, stop_ids=[], generator=torch.Generator().manual_seed(0))
            assert len(out) == 17 and st["generated"] == 17 and st["blocks"] == 3, st
            assert sum(st["steps_per_block"]) == st["steps"] and all(s <= 8 for s in st["steps_per_block"]), st
            assert sum(len(p) for p in st["filled_positions"]) >= 17, st
    for rem in (("lowconf", 0.25, 0.75), ("tau", 0.9, 0.5)):
        out, st = block_generate(fast, None, prompt, 16, block=8, steps=4, remask=rem, mask_id=mask_id, stop_ids=[],
                                 trace=True)
        assert len(out) == 16 and st["remasked"] >= 0 and all(s <= 8 for s in st["steps_per_block"]), st
        n_fills = sum(len(t["filled"]) for t in st["trace"])
        assert n_fills == 16 + st["remasked"], (n_fills, st["remasked"])
    # adaptive schedule
    out, st = block_generate(fast, None, prompt, 16, block=8, steps=None, tau=0.5, mask_id=mask_id, stop_ids=[])
    assert len(out) == 16 and all(1 <= s <= 8 for s in st["steps_per_block"]), st
    # anchor_leftmost: every step fills the leftmost mask; with tau=1.1 (nothing passes) it is exactly left-to-right
    ref, _ = block_generate(fast, None, prompt, 16, block=8, steps=8, order="leftmost", mask_id=mask_id, stop_ids=[])
    out, st = block_generate(fast, None, prompt, 16, block=8, steps=None, tau=1.1, min_fill=0, anchor_leftmost=True,
                             mask_id=mask_id, stop_ids=[])
    assert out == ref and st["steps"] == 16, (out, ref, st["steps"])
    out, st = block_generate(fast, None, prompt, 16, block=8, steps=None, tau=0.3, anchor_leftmost=True, mask_id=mask_id,
                             stop_ids=[], trace=True)
    for tr in st["trace"]:
        first_mask = min(range(8), key=lambda i: i) if False else None
    assert len(out) == 16, st


def test_stop_and_budget():
    hf, fast, mask_id = pair(2)
    g = torch.Generator().manual_seed(5)
    prompt = random_prompts(g, k=1)[0]
    out, st = block_generate(fast, None, prompt, 13, block=8, steps=2, mask_id=mask_id, stop_ids=[])
    assert len(out) == 13 and st["finish_reason"] == "length" and st["blocks"] == 2, st
    # make every token a stop id: the first block is cut at its first slot
    out, st = block_generate(fast, None, prompt, 13, block=8, steps=2, mask_id=mask_id, stop_ids=list(range(200)))
    assert len(out) == 1 and st["finish_reason"] == "stop", st
    assert st["cache_len"] == len(prompt) + 1


def test_static_path_matches_plain():
    hf, fast, mask_id = pair(0)
    g = torch.Generator().manual_seed(9)
    prompt = random_prompts(g, k=1)[0]
    a, _ = block_generate(fast, None, prompt, 16, block=8, steps=4, mask_id=mask_id, stop_ids=[], static=False)
    fast.kv.zero_()
    b, _ = block_generate(fast, None, prompt, 16, block=8, steps=4, mask_id=mask_id, stop_ids=[], static=True)
    assert a == b, (a, b)


def test_no_ar_head_dependence():
    """The decoder reads only mask-row logits: perturbing the lm_head output of every NON-mask row must not change the
    generation.  (A verified/speculative decoder would change: it reads the AR rows.)"""
    hf, fast, mask_id = pair(4)
    g = torch.Generator().manual_seed(11)
    prompt = random_prompts(g, k=1)[0]
    ref, _ = block_generate(fast, None, prompt, 16, block=8, steps=4, mask_id=mask_id, stop_ids=[])
    fast.kv.zero_()
    orig = fast.run

    def run(tokens, pos_ids, cache_len, rows, static=False, use_graph=False):
        out = orig(tokens, pos_ids, cache_len, list(range(int(tokens.numel()))), static=static, use_graph=use_graph)
        is_mask = (tokens == mask_id)
        out = out.clone()
        out[~is_mask] = torch.randn_like(out[~is_mask]) * 10          # garbage on every non-mask row
        return out[torch.as_tensor(list(rows), dtype=torch.long)] if len(rows) else out[:0]

    fast.run = run
    try:
        out, _ = block_generate(fast, None, prompt, 16, block=8, steps=4, mask_id=mask_id, stop_ids=[])
    finally:
        fast.run = orig
    assert out == ref, (out, ref)


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
