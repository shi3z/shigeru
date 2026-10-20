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
from engine.cdlm.decoder import generate

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
