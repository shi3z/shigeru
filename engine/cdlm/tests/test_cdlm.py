"""Plain-assert tests for engine.cdlm (CPU only, tiny random Qwen3, no downloads).

Run from the repo root:  .venv-train/bin/python -m engine.cdlm.tests.test_cdlm
"""
import copy
import os
import sys
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from engine.cdlm import (IGNORE, add_mask_token, ar_loss, batch_mask, build_example, collate, dense_mask, forward_batch,
                         forward_rows, make_mask_mod, mlm_loss, model_attention_mask, total_loss)

VOCAB, B = 97, 8
BASE_CFG = Qwen3Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                       tie_word_embeddings=False, initializer_range=0.2)


def tiny_model(impl, seed=0):
    torch.manual_seed(seed)
    m = Qwen3ForCausalLM(copy.deepcopy(BASE_CFG)).double().eval()
    m.set_attn_implementation(impl)
    info = add_mask_token(m)                       # vocab row VOCAB becomes [MASK]
    assert info == {"mask_id": VOCAB, "old_rows": VOCAB, "new_rows": VOCAB + 1, "added": [VOCAB]}, info
    assert m.config._attn_implementation == impl and m.config.cdlm_mask_token_id == VOCAB
    return m, info["mask_id"]


def all_rows(T):
    return (torch.zeros(T, dtype=torch.long), torch.arange(T))


def hidden_all(m, ex):
    """Last hidden state for every physical row of one example (used by the perturbation tests)."""
    mask = batch_mask([ex["seq_len"]], ex["block"])
    out = m.model(input_ids=ex["input_ids"][None], position_ids=ex["position_ids"][None],
                  attention_mask=model_attention_mask(m, mask), use_cache=False)
    return out.last_hidden_state[0], mask[0, 0]


def test_mask_dense_vs_mask_mod():
    for L in (1, 5, 8, 13, 20):
        T = 2 * L + 3                                       # with padded rows
        d = dense_mask(L, B, T)
        mm = make_mask_mod(torch.tensor([L]), B)
        mat = torch.zeros(T, T, dtype=torch.bool)
        for q in range(T):
            for k in range(T):
                mat[q, k] = bool(mm(torch.tensor(0), torch.tensor(0), torch.tensor(q), torch.tensor(k)))
        assert torch.equal(d, mat), L
        # structural checks straight from the analysis (section 3.1)
        for q in range(2 * L):
            for k in range(2 * L):
                xq, xk = q >= L, k >= L
                bq, bk = (q - L) // B if xq else q // B, (k - L) // B if xk else k // B
                exp = (bq == bk and xq == xk and k <= q) or (bk < bq and not xk)
                assert bool(d[q, k]) == exp, (L, q, k)
        for q in range(2 * L, T):                          # pad rows: diagonal only, unseen by real rows
            assert d[q].sum() == 1 and d[q, q] and not d[:2 * L, q].any()
        assert d.diagonal().all()
    bm = batch_mask([5, 3], B)
    assert bm.shape == (2, 1, 10, 10) and torch.equal(bm[1, 0], dense_mask(3, B, 10))


def test_no_leakage():
    """(a) perturbing any token a query must not see leaves that query's output unchanged (and visible ones change)."""
    g = torch.Generator().manual_seed(3)
    for impl in ("sdpa", "eager"):
        m, mask_id = tiny_model(impl)
        ex = build_example(torch.randint(0, VOCAB, (21,), generator=g), 4, mask_id, block=B, rng=g)
        T = ex["input_ids"].numel()
        ref, mask = hidden_all(m, ex)
        n_hidden = n_seen = 0
        for k in range(T):
            pert = dict(ex)
            pert["input_ids"] = ex["input_ids"].clone()
            pert["input_ids"][k] = (ex["input_ids"][k] + 1) % (VOCAB + 1)
            h, _ = hidden_all(m, pert)
            diff = (h - ref).abs().amax(-1) > 1e-9
            cannot_see = ~mask[:, k]
            assert not (diff & cannot_see).any(), (impl, k, (diff & cannot_see).nonzero().flatten().tolist())
            # rows that may attend k must change (random weights; transitivity means every visible row reacts)
            assert (diff | cannot_see).all(), (impl, k, (~diff & ~cannot_see).nonzero().flatten().tolist())
            n_hidden += int(cannot_see.sum()); n_seen += int((~cannot_see).sum())
        assert n_hidden > 0 and n_seen > 0


def test_single_block_masks_visibility():
    """(b) gamma > 0 only in block k: each MLM row of block k sees exactly x0 positions < k*B plus the observed tokens of
    its block (and the masks before it), never the x0 row of its own position / its own or later blocks."""
    g = torch.Generator().manual_seed(5)
    m, mask_id = tiny_model("sdpa")
    L, kblk = 23, 1
    gam = lambda n, rng: torch.tensor([0.5 if b == kblk else 1e-8 for b in range(n)])   # 1e-8 -> round(n*g) = 0 masks
    ex = build_example(torch.randint(0, VOCAB, (L,), generator=g), 3, mask_id, block=B, gamma_sampler=gam, rng=g)
    ids, pos, lab = ex["input_ids"], ex["position_ids"], ex["mlm_labels"]
    mlm_rows = (lab != IGNORE).nonzero().flatten().tolist()
    assert mlm_rows and all(ids[r] == mask_id and kblk * B <= pos[r] < (kblk + 1) * B for r in mlm_rows)
    assert (ids[L:] == mask_id).sum() == len(mlm_rows) == 4           # round(8 * 0.5)
    mask = dense_mask(L, B)
    ref, _ = hidden_all(m, ex)
    obs_rows = [r for r in range(L + kblk * B, L + (kblk + 1) * B) if ids[r] != mask_id]
    for r in mlm_rows:
        visible = set(mask[r].nonzero().flatten().tolist())
        expect = set(range(kblk * B)) | set(obs_rows) | {q for q in mlm_rows if q <= r}
        assert visible == expect, (r, sorted(visible ^ expect))
        assert L + pos[r] not in visible and int(pos[r]) not in visible   # own position in neither stream
        # information check by perturbation: exactly the x0 prefix and the observed tokens of the block matter
        for k in range(2 * L):
            if ids[k] == mask_id:
                continue
            pert = dict(ex); pert["input_ids"] = ids.clone(); pert["input_ids"][k] = (ids[k] + 1) % VOCAB
            h, _ = hidden_all(m, pert)
            changed = bool((h[r] - ref[r]).abs().max() > 1e-9)
            assert changed == (k in expect), (r, k, changed)


def test_collate_padding():
    """(c) padding is inert: padded batch == per-example forward on real rows; pad rows carry no labels/weights."""
    g = torch.Generator().manual_seed(7)
    m, mask_id = tiny_model("sdpa")
    exs = [build_example(torch.randint(0, VOCAB, (n,), generator=g), p, mask_id, block=B, rng=g)
           for n, p in ((19, 6), (9, 2), (24, 0))]
    b = collate(exs, pad_id=0)
    N, T = b["input_ids"].shape
    assert (N, T) == (3, 48) and b["attention_mask"].shape == (3, 1, T, T) and b["seq_lens"].tolist() == [19, 9, 24]
    for i, e in enumerate(exs):
        n = 2 * e["seq_len"]
        for key in ("input_ids", "position_ids", "mlm_labels", "mlm_weights", "ar_labels"):
            assert torch.equal(b[key][i, :n], e[key]), key
        assert (b["input_ids"][i, n:] == 0).all() and (b["position_ids"][i, n:] == 0).all()
        assert (b["mlm_labels"][i, n:] == IGNORE).all() and (b["ar_labels"][i, n:] == IGNORE).all()
        assert (b["mlm_weights"][i, n:] == 0).all()
        # example semantics: x0 half is the clean sequence; xt is a permutation of it with masks; positions line up
        L = e["seq_len"]
        xt_pos = e["position_ids"][L:]
        assert torch.equal(xt_pos.sort().values, torch.arange(L))
        restored = torch.where(e["mlm_labels"][L:] == IGNORE, e["input_ids"][L:], e["mlm_labels"][L:])
        assert torch.equal(restored[xt_pos.argsort()], e["input_ids"][:L])     # masked rows' labels are the true ids
        assert (xt_pos[e["mlm_labels"][L:] != IGNORE] >= e["prompt_len"]).all()   # prompt never masked
        for blk in range((L + B - 1) // B):                                        # observed-first inside each block
            s, t = blk * B, min(L, (blk + 1) * B)
            is_m = (e["input_ids"][L + s:L + t] == mask_id).long()
            assert (is_m.diff() >= 0).all() and torch.equal(xt_pos[s:t][is_m == 0], xt_pos[s:t][is_m == 0].sort().values)
            assert torch.equal(xt_pos[s:t][is_m == 1], xt_pos[s:t][is_m == 1].sort().values)
            if is_m.sum():
                assert torch.allclose(e["mlm_weights"][L + s:L + t][is_m == 1], 1.0 / e["gamma"][blk])
        # AR labels: next token on x0, -100 where the target is a prompt token, -100 on xt
        exp_ar = torch.full((2 * L,), IGNORE, dtype=torch.long)
        for i2 in range(L - 1):
            if i2 + 1 >= e["prompt_len"]:
                exp_ar[i2] = e["input_ids"][i2 + 1]
        assert torch.equal(e["ar_labels"], exp_ar)
    # batched forward equals per-example forward on the real rows
    out = forward_batch(m, b)
    rows_b = (b["mlm_labels"] != IGNORE).nonzero(as_tuple=True)
    j = 0
    for i, e in enumerate(exs):
        n = 2 * e["seq_len"]
        single = forward_rows(m, e["input_ids"][None], e["position_ids"][None], batch_mask([e["seq_len"]], B), all_rows(n))
        sel = (e["mlm_labels"] != IGNORE).nonzero().flatten()
        assert torch.allclose(out["mlm_logits"][j:j + sel.numel()], single[sel], atol=1e-9)
        j += sel.numel()
    assert j == out["mlm_logits"].shape[0] == rows_b[0].numel()
    e0 = build_example(torch.arange(5), 2, mask_id, block=B, rng=g, ar_on_prompt=True)
    assert e0["ar_labels"][:5].tolist() == [1, 2, 3, 4, IGNORE]


def test_losses():
    """(d) shapes, WeDLM normalisation, 1/gamma weighting actually applied, AR loss == plain CE."""
    torch.manual_seed(11)
    V, R = 13, 10
    logits = torch.randn(R, V, requires_grad=True)
    labels = torch.randint(0, V, (R,)); labels[[0, 4]] = IGNORE
    gamma = torch.tensor([1.0, 0.5, 0.25, 0.1, 1.0, 0.5, 0.5, 0.2, 1.0, 0.05])
    w = 1.0 / gamma; w[labels == IGNORE] = 0
    sel = labels != IGNORE
    ce = F.cross_entropy(logits[sel], labels[sel], reduction="none")
    ref = (ce * w[sel]).sum() / w[sel].sum()
    got = mlm_loss(logits, labels, w)
    assert got.shape == () and torch.allclose(got, ref, atol=1e-6)
    assert torch.allclose(mlm_loss(logits, labels, w, normalize="tokens"), (ce * w[sel]).sum() / sel.sum())
    got.backward(); assert logits.grad is not None and logits.grad[~sel].abs().sum() == 0
    # uniform weights -> plain mean; shifting weight onto the hardest row raises the loss (weights matter)
    assert torch.allclose(mlm_loss(logits, labels, torch.ones(R)), ce.mean())
    hard = sel.nonzero().flatten()[ce.argmax()]
    w2 = torch.ones(R); w2[hard] = 1e3
    assert mlm_loss(logits, labels, w2) > ce.mean() and abs(mlm_loss(logits, labels, w2) - ce.max()) < 0.1
    # batch-shaped inputs are flattened; all-ignored gives a differentiable zero
    assert torch.allclose(mlm_loss(logits.view(2, 5, V), labels.view(2, 5), w.view(2, 5)), ref)
    z = mlm_loss(logits, torch.full((R,), IGNORE), w); assert z == 0 and z.requires_grad
    ar = ar_loss(logits, labels)
    assert torch.allclose(ar, F.cross_entropy(logits, labels, ignore_index=IGNORE)) and ar.shape == ()
    assert torch.allclose(total_loss(torch.tensor(2.0), torch.tensor(4.0)), torch.tensor(3.0))
    assert total_loss(torch.tensor(2.0), torch.tensor(4.0), ar_weight=0.0) == 2.0


def test_tau_none_equivalence():
    """(e) with zero masks the x0 stream reproduces the plain causal forward exactly, for sdpa and eager; and so do the
    (all-observed) xt rows, which see the same prefix through the dual-stream mask."""
    g = torch.Generator().manual_seed(13)
    for impl in ("sdpa", "eager"):
        m, mask_id = tiny_model(impl)
        toks = torch.randint(0, VOCAB, (27,), generator=g)
        L = toks.numel()
        ex = build_example(toks, 5, mask_id, block=B, gamma_sampler=lambda n, rng: torch.full((n,), 1e-8), rng=g)
        assert (ex["mlm_labels"] == IGNORE).all() and torch.equal(ex["input_ids"][L:], toks)
        plain = m(input_ids=toks[None]).logits[0]                   # HF causal path, no custom mask
        dual = forward_rows(m, ex["input_ids"][None], ex["position_ids"][None], batch_mask([L], B), all_rows(2 * L))
        assert torch.allclose(dual[:L], plain, atol=1e-9), (impl, float((dual[:L] - plain).abs().max()))
        # xt rows see the same keys in a different physical order; HF eager runs softmax in float32 (modeling_qwen3.py:194)
        # so its summation-order noise is ~1e-6 even on a float64 model, sdpa stays exact.
        tol = 1e-9 if impl == "sdpa" else 1e-4
        assert torch.allclose(dual[L:], plain, atol=tol), (impl, float((dual[L:] - plain).abs().max()))
        # and greedy next tokens agree row by row
        assert torch.equal(dual[:L].argmax(-1), plain.argmax(-1))


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
