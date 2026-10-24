"""Gate 0 tests for the B-series (CPU, tiny Qwen3 from engine/cdlm/tests/test_fast.py's fixture).

Run from the repo root:  CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -m engine.bdiff.tests.test_bdiff
"""
import os
import sys
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch

from engine.bdiff.corrupt import corrupt_block, gamma
from engine.bdiff.masks import block_mask, train_mask
from engine.bdiff.model import BDiffModel
from engine.bdiff.sampler import sample_block
from engine.cdlm.tests.test_fast import hf_model


def make(seed=0, time_cond=True):
    lm, mask_id = hf_model(seed)
    return BDiffModel(lm.eval(), mask_id, time_cond=time_cond).eval(), mask_id


def test_masks():
    m = train_mask(5, 4)
    assert m.shape == (9, 9)
    assert torch.equal(m[:5, :5], torch.tril(torch.ones(5, 5, dtype=torch.bool))) and not m[:5, 5:].any()
    assert m[5:, :].all()
    b = block_mask(5, 4)
    assert b.shape == (4, 9) and b.all()


def test_prefix_independent_of_block_and_block_bidirectional():
    model, mask_id = make(0)
    P, K = 7, 6
    g = torch.Generator().manual_seed(1)
    prefix = torch.randint(0, 90, (1, P), generator=g)
    blk = torch.randint(0, 90, (1, K), generator=g)
    ids = torch.cat([prefix, blk], 1)
    emb = model.lm.get_input_embeddings()(ids)
    mask = train_mask(P, K)[None, None]
    pos = torch.arange(P + K)[None]
    la = model.lm(inputs_embeds=emb, attention_mask=mask, position_ids=pos).logits[0]
    blk2 = blk.clone(); blk2[0, 3] = (blk2[0, 3] + 7) % 90
    emb2 = model.lm.get_input_embeddings()(torch.cat([prefix, blk2], 1))
    lb = model.lm(inputs_embeds=emb2, attention_mask=mask, position_ids=pos).logits[0]
    assert torch.allclose(la[:P], lb[:P], atol=1e-5), "prefix logits changed with the block"
    assert not torch.allclose(la[P + 0], lb[P + 0], atol=1e-5), "block row 0 does not see block row 3 (no bidirectional)"
    assert not torch.allclose(la[P + 5], lb[P + 5], atol=1e-5), "block row 5 does not see block row 3"


def test_masked_rows_carry_no_gold_and_visible_rows_see_themselves():
    """A masked position's input is [MASK]: changing its gold token changes nothing anywhere.  A visible position sees
    its own token (by design; the copy shortcut is why corrupted visible tokens must be in the training mixture)."""
    model, mask_id = make(1)
    P, K = 6, 5
    g = torch.Generator().manual_seed(2)
    prefix = torch.randint(0, 90, (1, P), generator=g)
    x0 = torch.randint(0, 90, (K,), generator=g).tolist()
    xt = list(x0); xt[2] = mask_id
    out_a = model.forward_block(prefix, torch.tensor([xt]), torch.tensor([0.2]))
    x0b = list(x0); x0b[2] = (x0b[2] + 3) % 90            # gold of the masked position changes -> inputs identical
    xtb = list(x0b); xtb[2] = mask_id
    out_b = model.forward_block(prefix, torch.tensor([xtb]), torch.tensor([0.2]))
    assert torch.allclose(out_a, out_b, atol=1e-6)
    xtc = list(xt); xtc[4] = (xtc[4] + 3) % 90             # a visible token changes -> every block row may change
    out_c = model.forward_block(prefix, torch.tensor([xtc]), torch.tensor([0.2]))
    assert not torch.allclose(out_a[0, 4], out_c[0, 4], atol=1e-5) and not torch.allclose(out_a[0, 2], out_c[0, 2], atol=1e-5)


def test_cache_path_equals_full_forward_and_positions():
    model, mask_id = make(2)
    P, K = 8, 6
    g = torch.Generator().manual_seed(3)
    prefix = torch.randint(0, 90, (1, P), generator=g)
    blk = torch.randint(0, 90, (1, K), generator=g); blk[0, 1] = mask_id; blk[0, 4] = mask_id
    full = model.forward_block(prefix, blk, torch.tensor([0.33]))[0]
    cache = model.prefix_cache(prefix)
    assert cache.get_seq_length() == P
    via = model.denoise(cache, P, blk, 0.33)
    assert cache.get_seq_length() == P, "block K/V leaked into the cache"
    assert torch.allclose(full, via, atol=1e-4), (full - via).abs().max()
    # a second denoise call with a different block gives different logits (the cache was not polluted by the first)
    blk2 = blk.clone(); blk2[0, 0] = (blk2[0, 0] + 1) % 90
    via2 = model.denoise(cache, P, blk2, 0.33)
    full2 = model.forward_block(prefix, blk2, torch.tensor([0.33]))[0]
    assert torch.allclose(full2, via2, atol=1e-4)
    # positions: shifting the block by one position (wrong RoPE) changes the logits
    emb = model.lm.get_input_embeddings()(torch.cat([prefix, blk], 1))
    pos_wrong = torch.cat([torch.arange(P), torch.arange(P + 1, P + K + 1)])[None]
    lw = model.lm(inputs_embeds=emb, attention_mask=train_mask(P, K)[None, None], position_ids=pos_wrong).logits[0, P:]
    assert not torch.allclose(lw, full, atol=1e-4)


def test_time_embedding_effect_and_no_time():
    model, mask_id = make(3, time_cond=True)
    with torch.no_grad():
        model.time.emb.weight.normal_()
    P, K = 5, 4
    prefix = torch.randint(0, 90, (1, P)); blk = torch.full((1, K), mask_id)
    a = model.forward_block(prefix, blk, torch.tensor([0.1]))
    b = model.forward_block(prefix, blk, torch.tensor([1.0]))
    assert not torch.allclose(a, b, atol=1e-5)
    m2, _ = make(3, time_cond=False)
    c = m2.forward_block(prefix, blk, torch.tensor([0.1])); d = m2.forward_block(prefix, blk, torch.tensor([1.0]))
    assert torch.allclose(c, d)


def test_corruption_and_sampler():
    rng = torch.Generator().manual_seed(5)
    x0 = list(range(10, 26))
    for sch in ("linear", "cosine", "uniform"):
        for t in (0.0, 0.25, 0.5, 1.0):
            xt, im, ic, g = corrupt_block(x0, t, 97, rng, schedule=sch)
            n = sum(im)
            assert n == (0 if t == 0 else max(1, round(16 * gamma(t, sch)))), (sch, t, n)
            assert all((xt[i] == 97) == im[i] for i in range(16)) and not any(ic)
    xt, im, ic, g = corrupt_block(x0, 0.25, 97, rng, p_replace=1.0, r_replace=0.5, vocab=90)
    assert sum(ic) >= 1 and all(xt[i] != x0[i] for i in range(16) if ic[i]) and not any(im[i] and ic[i] for i in range(16))
    st = [-1, 5, -1, 6] + [-1] * 12
    xt, im, ic, g = corrupt_block(x0, 0.5, 97, rng, state=st)
    assert xt[1] == 5 and im[0] and ic[1] and ic[3] and abs(g - 14 / 16) < 1e-6
    model, mask_id = make(4)
    P, K = 6, 8
    prefix = torch.randint(0, 90, (1, P))
    cache = model.prefix_cache(prefix)
    for order in ("confidence", "random", "left"):
        for S in (1, 4, 8):
            toks, info = sample_block(model, cache, P, K, S, mask_id, order=order, generator=torch.Generator().manual_seed(0))
            assert len(toks) == K and mask_id not in toks and info["forwards"] == S, (order, S, info)
            assert cache.get_seq_length() == P
    toks, info = sample_block(model, cache, P, K, 4, mask_id, remask=True, trace=True)
    assert len(toks) == K and mask_id not in toks and info["forwards"] >= 4


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
