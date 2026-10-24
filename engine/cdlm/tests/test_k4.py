"""Plain-assert tests for engine.cdlm.k4_decoder (CPU only, tiny random Qwen3 from test_fast's fixture).

Run from the repo root:  CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -m engine.cdlm.tests.test_k4
"""
import os
import sys
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch

from engine.cdlm.fast_decoder import generate as fast_generate
from engine.cdlm.k4_decoder import STRATEGIES, k4_generate, probe_key_mask, probe_rows
from engine.cdlm.tests.test_fast import pair, random_prompts


def test_probe_key_mask():
    C, K = 5, 4
    m = probe_key_mask(C, K)
    assert m.shape == (K * K, C + K * K)
    assert bool(m[:, :C].all())                                   # prefix visible to all
    for i in range(K):
        for r in range(K):
            row = m[i * K + r]
            assert row[C + i * K: C + i * K + r + 1].all()         # causal within the group
            assert not row[C + i * K + r + 1: C + (i + 1) * K].any()
            others = torch.cat([row[C: C + i * K], row[C + (i + 1) * K:]])
            assert not others.any()                               # nothing of other groups
    ids, pos, rows = probe_rows([11, 12, 13, 14], 100, 97)
    assert len(ids) == 16 and rows == [3, 7, 11, 15]
    assert ids[:4] == [12, 13, 14, 97] and pos[:4] == [101, 102, 103, 100]
    assert ids[4:8] == [11, 13, 14, 97] and pos[4:8] == [100, 102, 103, 101]
    assert bool(m.any(1).all())                                   # no empty softmax row


def test_probe_does_not_see_own_token():
    """Changing placed token i must not change probe i's logits (it sees every placed token but its own) and must
    change some other probe's logits."""
    hf, fast, mask_id = pair(0)
    g = torch.Generator().manual_seed(1)
    prompt = random_prompts(g, k=1)[0]
    P = len(prompt)
    fast.prefill(torch.tensor(list(prompt)), 0)
    K = 4
    toks = [1, 2, 3, 4]

    def probe(t):
        ids, pos, rows = probe_rows(list(t), P, mask_id)
        return fast.forward(torch.tensor(ids), torch.tensor(pos), P, write=False, rows=rows, key_mask=probe_key_mask(P, K))

    a = probe(toks)
    b = probe([1, 2, 50, 4])
    assert torch.allclose(a[2], b[2], atol=1e-5), (a[2] - b[2]).abs().max()
    assert not torch.allclose(a[0], b[0], atol=1e-5) or not torch.allclose(a[1], b[1], atol=1e-5)
    # and the probe equals the training layout: observed-first forward of the block with position 2 masked
    ids = torch.tensor([1, 2, 4, mask_id]); pos = torch.tensor([P, P + 1, P + 3, P + 2])
    ref = fast.forward(ids, pos, P, write=False, rows=[3])
    assert torch.allclose(a[2], ref[0], atol=1e-4), (a[2] - ref[0]).abs().max()


def test_seq_equals_ar_and_strategies_run():
    for seed in range(2):
        hf, fast, mask_id = pair(seed)
        g = torch.Generator().manual_seed(seed + 3)
        for prompt in random_prompts(g, k=2):
            ref, _ = fast_generate(fast, None, prompt, 16, window=8, tau=None, mask_id=mask_id, stop_ids=[])
            fast.kv.zero_()
            out, st = k4_generate(fast, None, prompt, 16, block=4, strategy="seq", mask_id=mask_id, stop_ids=[])
            fast.kv.zero_()
            if mask_id not in ref:                                   # k4 never emits [MASK]; the streaming decoder may
                assert out == ref, (out, ref)
            assert st["steps"] == 16 + 1, st["steps"]               # K forwards per block + the final re-encode
            for strat in STRATEGIES:
                out, st = k4_generate(fast, None, prompt, 16, block=4, strategy=strat, mask_id=mask_id, stop_ids=[],
                                      trace=True)
                fast.kv.zero_()
                assert len(out) == 16 and st["blocks"] == 4, (strat, st)
                exp = {"s1": 1, "conf2": 2, "refine": 2, "conf2refine": 3, "seq": 4}.get(strat)
                if exp:
                    assert st["steps"] == 4 * exp + 1, (strat, st["steps"])
                if strat in ("s1", "conf2"):
                    assert st["corrections"] == 0 and st["probes"] == 0
                for b in st["trace"]:
                    assert len(b["first"]) == 4 and b["trace"][0]["kind"] == "all_mask"


def test_adopt_rules():
    from engine.cdlm.k4_decoder import ADOPT
    hf, fast, mask_id = pair(3)
    g = torch.Generator().manual_seed(13)
    prompt = random_prompts(g, k=1)[0]
    for adopt in ADOPT:
        out, st = k4_generate(fast, None, prompt, 16, block=4, strategy="refineN", adopt=adopt, max_rounds=4, mask_id=mask_id,
                              stop_ids=[], trace=True)
        fast.kv.zero_()
        assert len(out) == 16, (adopt, st)
        for b in st["trace"]:
            for tr in b["trace"]:
                if tr["kind"] != "probe":
                    continue
                ch = [c[0] for c in tr["changed"]]
                if adopt == "one":
                    assert len(ch) <= 1, tr
                if adopt == "nonadjacent":
                    assert all(abs(x - y) > 1 for x in ch for y in ch if x != y), tr


def test_static_path_matches_plain():
    """run_masked on the static path == the plain probe forward; whole k4 generations agree between paths."""
    from engine.cdlm.k4_decoder import probe_win_mask
    hf, fast, mask_id = pair(5)
    g = torch.Generator().manual_seed(17)
    prompt = random_prompts(g, k=1)[0]
    P = len(prompt)
    fast.prefill(torch.tensor(list(prompt)), 0)
    ids, pos, rows = probe_rows([1, 2, 3, 4], P, mask_id)
    a = fast.forward(torch.tensor(ids), torch.tensor(pos), P, write=False, rows=rows, key_mask=probe_key_mask(P, 4))
    b = fast.run_masked(torch.tensor(ids), torch.tensor(pos), P, probe_win_mask(4), rows, static=True)
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()
    for strat in ("refineN", "conf2refine"):
        fast.kv.zero_()
        x, _ = k4_generate(fast, None, prompt, 16, block=4, strategy=strat, adopt="one", max_rounds=4, mask_id=mask_id, stop_ids=[])
        fast.kv.zero_()
        y, st = k4_generate(fast, None, prompt, 16, block=4, strategy=strat, adopt="one", max_rounds=4, mask_id=mask_id, stop_ids=[],
                            static=True)
        fast.kv.zero_()
        assert x == y, (strat, x, y)


def test_cache_valid_after_generation():
    for strat in ("refine", "conf2", "refineN"):
        hf, fast, mask_id = pair(1)
        g = torch.Generator().manual_seed(9)
        prompt = random_prompts(g, k=1)[0]
        out, st = k4_generate(fast, None, prompt, 12, block=4, strategy=strat, mask_id=mask_id, stop_ids=[])
        K = [fast.kv.K[i, :, :, : st["cache_len"]].clone() for i in range(len(fast.layers))]
        fast.kv.zero_()
        fast.prefill(torch.tensor(list(prompt) + list(out)), 0)
        for i in range(len(fast.layers)):
            d = (K[i] - fast.kv.K[i, :, :, : st["cache_len"]]).abs().max().item()
            assert d < 1e-4, (strat, i, d)


def test_no_ar_head_dependence():
    hf, fast, mask_id = pair(4)
    g = torch.Generator().manual_seed(11)
    prompt = random_prompts(g, k=1)[0]
    ref, _ = k4_generate(fast, None, prompt, 12, block=4, strategy="conf2refine", mask_id=mask_id, stop_ids=[])
    fast.kv.zero_()
    orig = fast.forward

    def forward(tokens, pos_ids, cache_len=0, write=True, rows=None, key_mask=None):
        out = orig(tokens, pos_ids, cache_len, write=write, rows=None, key_mask=key_mask).clone()
        is_mask = tokens == mask_id
        out[~is_mask] = torch.randn_like(out[~is_mask]) * 10
        return out[torch.as_tensor(list(rows), dtype=torch.long)] if rows is not None and len(rows) else (out if rows is None else out[:0])

    fast.forward = forward
    try:
        out, _ = k4_generate(fast, None, prompt, 12, block=4, strategy="conf2refine", mask_id=mask_id, stop_ids=[])
    finally:
        fast.forward = orig
    assert out == ref, (out, ref)


def test_stop_ids_and_other_K():
    hf, fast, mask_id = pair(2)
    g = torch.Generator().manual_seed(5)
    prompt = random_prompts(g, k=1)[0]
    out, st = k4_generate(fast, None, prompt, 13, block=4, strategy="refine", mask_id=mask_id, stop_ids=list(range(200)))
    assert len(out) == 1 and st["finish_reason"] == "stop" and st["cache_len"] == len(prompt) + 1, st
    for K in (2, 8):
        out, st = k4_generate(fast, None, prompt, 16, block=K, strategy="conf2refine", mask_id=mask_id, stop_ids=[])
        assert len(out) == 16 and st["blocks"] == 16 // K, st


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
