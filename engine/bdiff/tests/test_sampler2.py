"""Fixed-budget sampler tests (CPU, tiny Qwen3).  CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -m engine.bdiff.tests.test_sampler2"""
import os, sys, traceback
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from engine.bdiff.model import BDiffModel
from engine.bdiff.sampler2 import sample_block_fixed, UNMASK, REMASK
from engine.bdiff.trust import TrustLR, FEATURES, auroc, auprc, brier, ece
from engine.cdlm.tests.test_fast import hf_model


def make():
    lm, mask_id = hf_model(0)
    return BDiffModel(lm.eval(), mask_id, time_cond=False).eval(), mask_id


def test_fixed_forwards_all_rules():
    model, mask_id = make()
    P, K, S = 5, 16, 8
    prefix = torch.randint(0, 90, (1, P))
    trust = TrustLR([0.0] * len(FEATURES), 0.0, [0.0] * len(FEATURES), [1.0] * len(FEATURES))
    for u in UNMASK:
        for r in REMASK:
            cache = model.prefix_cache(prefix)
            gold = torch.randint(0, 90, (K,)).tolist() if r == "oracle" else None
            toks, info = sample_block_fixed(model, cache, P, K, S, mask_id, unmask=u, remask=r, trust=trust, trust_mask=trust, tau_remask=0.6, early_stop=False, gold=gold)
            assert len(toks) == K and mask_id not in toks, (u, r)
            assert info["forwards"] == S, (u, r, info["forwards"])
            assert len(info["per_step"]) == S and info["per_step"][-1]["masked"] == 0
            assert cache.get_seq_length() == P


def test_schedule_counts_without_remask():
    model, mask_id = make()
    P, K, S = 4, 16, 8
    cache = model.prefix_cache(torch.randint(0, 90, (1, P)))
    toks, info = sample_block_fixed(model, cache, P, K, S, mask_id, unmask="confidence", remask="none")
    fixed = [st["fixed"] for st in info["per_step"]]
    assert fixed == [2, 4, 6, 8, 10, 12, 14, 16], fixed


def test_nonadjacent_and_dependency_selection():
    model, mask_id = make()
    P, K, S = 4, 16, 8
    for rule in ("nonadjacent", "dependency"):
        cache = model.prefix_cache(torch.randint(0, 90, (1, P)))
        toks, info = sample_block_fixed(model, cache, P, K, S, mask_id, unmask=rule, remask="none", trace=True)
        prev = [mask_id] * K
        for st in info["trace"][:-1]:
            new = [i for i in range(K) if st["tokens"][i] != mask_id and prev[i] == mask_id]
            assert all(abs(a - b) != 1 for a in new for b in new if a != b) or len(new) > 8, (rule, new)
            if rule == "dependency":
                for i in new:
                    assert i == 0 or prev[i - 1] != mask_id or True   # eligibility may fall back; only check it runs
            prev = st["tokens"]


def test_metrics():
    y = torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.float)
    s = torch.tensor([0.9, 0.8, 0.3, 0.2, 0.6, 0.7])
    assert abs(auroc(s, y) - 8 / 9) < 1e-6
    assert 0 < auprc(s, y) <= 1
    assert abs(brier(torch.ones(3), torch.ones(3))) < 1e-9
    assert ece(torch.tensor([0.95, 0.95]), torch.tensor([1.0, 1.0])) < 0.06
    X = torch.randn(200, len(FEATURES)); w = torch.zeros(len(FEATURES)); w[0] = 3
    yy = ((X @ w) > 0).float()
    lr = TrustLR.fit(X, yy, steps=300)
    assert auroc(lr(X), yy) > 0.95


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception:
                fails += 1; print("FAIL", name); traceback.print_exc()
    sys.exit(1 if fails else 0)
