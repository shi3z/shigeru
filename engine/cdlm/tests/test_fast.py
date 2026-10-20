"""Plain-assert tests for engine.cdlm.fast_model / fast_decoder (CPU only, tiny random Qwen3 saved to a temp dir).

Run from the repo root:  CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -m engine.cdlm.tests.test_fast
"""
import copy
import os
import sys
import tempfile
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

from engine.cdlm import add_mask_token
from engine.cdlm.decoder import generate as hf_generate, window_attention_mask
from engine.cdlm.fast_decoder import generate as fast_generate
from engine.cdlm.fast_model import BUCKETS, FastQwen3, bucket_for

VOCAB = 97
MAX_LEN = 160
BASE_CFG = Qwen3Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                       tie_word_embeddings=False, initializer_range=0.2)


def hf_model(seed=0, tie=False):
    """HF Qwen3ForCausalLM (float32, sdpa) with the [MASK] row added by add_mask_token and then re-randomised."""
    torch.manual_seed(seed)
    cfg = copy.deepcopy(BASE_CFG)
    cfg.tie_word_embeddings = tie
    m = Qwen3ForCausalLM(cfg).float().eval()
    m.set_attn_implementation("sdpa")
    mask_id = add_mask_token(m)["mask_id"]
    with torch.no_grad():
        g = torch.Generator().manual_seed(seed + 100)
        m.get_input_embeddings().weight[mask_id] = torch.randn(cfg.hidden_size, generator=g)
        if not tie:
            m.get_output_embeddings().weight[mask_id] = torch.randn(cfg.hidden_size, generator=g)
    return m, mask_id


def pair(seed=0, tie=False, max_len=MAX_LEN):
    """(HF model, FastQwen3 loaded from the HF model saved to a temp dir, mask_id)."""
    m, mask_id = hf_model(seed, tie)
    d = tempfile.mkdtemp(prefix="cdlm_fast_")
    m.save_pretrained(d)
    f = FastQwen3.from_pretrained(d, device="cpu", dtype=torch.float32, max_len=max_len)
    assert f.cfg.mask_id == mask_id and f.cfg.vocab == VOCAB + 1 and f.cfg.tie == tie
    assert f.embed_tokens.weight.shape[0] == VOCAB + 1
    assert torch.equal(f.embed_tokens.weight[mask_id], m.get_input_embeddings().weight[mask_id])
    return m, f, mask_id


def random_prompts(g, k=6, lo=2, hi=24):
    return [torch.randint(0, VOCAB, (int(torch.randint(lo, hi, (1,), generator=g)),), generator=g) for _ in range(k)]


def maxdiff(a, b):
    return float((a - b).abs().max())


def test_logits_exact_vs_hf():
    """(1) logits of the fast model == transformers' Qwen3ForCausalLM (float32, max abs diff < 1e-4) for: a plain
    causal forward; chunked prefill + continuation; the observed-first window forward with explicit positions over
    the cache (vs HF with the decoder's dict mask on a DynamicCache); write=False; the bucketed static path."""
    g = torch.Generator().manual_seed(10)
    worst = 0.0
    for tie in (False, True):
        m, f, mask_id = pair(seed=int(tie), tie=tie)
        for prompt in random_prompts(g, 4, 3, 40):
            n = prompt.numel()
            ref = m(input_ids=prompt[None], use_cache=False).logits[0]
            # (a) plain causal forward from an empty cache, all rows
            f.kv.zero_()
            out = f(prompt, torch.arange(n), 0, write=True)
            assert out.shape == (n, VOCAB + 1)
            d = maxdiff(out, ref); worst = max(worst, d); assert d < 1e-4, d
            # (b) chunked prefill, then the rest in one window; rows gathered
            for chunk in (1, 3, 512):
                f.kv.zero_()
                k = max(1, n // 2)
                last, cl = f.prefill(prompt[:k], 0, chunk=chunk)
                assert cl == k and maxdiff(last, ref[k - 1]) < 1e-4
                if k < n:
                    rows = [0, n - k - 1]
                    out = f(prompt[k:], torch.arange(k, n), cl, write=True, rows=rows)
                    assert out.shape == (2, VOCAB + 1)
                    d = maxdiff(out, ref[[k, n - 1]]); worst = max(worst, d); assert d < 1e-4, (chunk, d)
            # (c) window forward in observed-first order with ORIGINAL positions vs HF on a DynamicCache
            W = int(torch.randint(1, 17, (1,), generator=g))
            n_filled = int(torch.randint(0, W + 1, (1,), generator=g))
            filled_slots = sorted(torch.randperm(W, generator=g)[:n_filled].tolist())
            mask_slots = [s for s in range(W) if s not in filled_slots]
            order = filled_slots + mask_slots
            win = torch.tensor([int(torch.randint(0, VOCAB, (1,), generator=g)) if s in filled_slots else mask_id
                                for s in order])
            pos = torch.tensor([n + s for s in order])
            cache = DynamicCache(config=m.config)
            m.model(input_ids=prompt[None], past_key_values=cache, use_cache=True)
            h = m.model(input_ids=win[None], position_ids=pos[None], attention_mask=window_attention_mask(m, n, W),
                        past_key_values=cache, use_cache=True).last_hidden_state[0]
            ref_w = m.lm_head(h)
            f.kv.zero_()
            _, cl = f.prefill(prompt, 0)
            out_nw = f(win, pos, cl, write=False)                       # no write: cache untouched beyond cl
            assert float(f.kv.K[:, :, :, cl:].abs().max()) == 0.0
            out = f(win, pos, cl, write=True)
            d = max(maxdiff(out, ref_w), maxdiff(out_nw, ref_w)); worst = max(worst, d); assert d < 1e-4, d
            # the written window K/V equal HF's cache rows for the window
            for li in range(m.config.num_hidden_layers):
                assert maxdiff(f.kv.K[li, :, :, cl:cl + W], cache.layers[li].keys[:, :, n:]) < 1e-4
                assert maxdiff(f.kv.V[li, :, :, cl:cl + W], cache.layers[li].values[:, :, n:]) < 1e-4
            # (d) static bucketed path (padded window, whole-cache context, index gather) == the same rows
            rows = list(range(W))[::-1][: max(1, W // 2)]
            out_s = f.run(win, pos, cl, rows, static=True)
            assert out_s.shape == (len(rows), VOCAB + 1) and bucket_for(W) in BUCKETS
            d = maxdiff(out_s, ref_w[rows]); worst = max(worst, d); assert d < 1e-4, d
            assert maxdiff(f.kv.K[:, :, :, cl:cl + W], torch.stack([c.keys[:, :, n:] for c in cache.layers])) < 1e-4
    print(f"  max |fast - HF| logits: {worst:.2e}")


def strip(stats):
    return {k: v for k, v in stats.items() if k not in ("cache", "cache_len", "prompt_len")}


def test_fast_decoder_matches_hf_decoder():
    """(2) fast_decoder.generate == decoder.generate token for token (and in stats) on the same weights, for random
    prompts x W in {1, 4, 16} x tau in {None, 0.5, 1e9} x predict in {mask, ar} x {no stop, random stop}; the static
    path gives the same tokens."""
    g = torch.Generator().manual_seed(20)
    n_cases = 0
    for tie in (False, True):
        m, f, mask_id = pair(seed=3 + int(tie), tie=tie)
        for prompt in random_prompts(g, 4):
            for stop_ids in ([], [int(torch.randint(0, VOCAB, (1,), generator=g))]):
                for W in (1, 4, 16):
                    for tau in (None, 0.5, 1e9):
                        for predict in ("mask", "ar"):
                            ref, rs = hf_generate(m, None, prompt, 30, window=W, tau=tau, lam=0.02, mask_id=mask_id,
                                                  stop_ids=stop_ids, predict=predict)
                            out, st = fast_generate(f, None, prompt, 30, window=W, tau=tau, lam=0.02, mask_id=mask_id,
                                                    stop_ids=stop_ids, predict=predict)
                            assert out == ref, (tie, W, tau, predict, stop_ids, out, ref)
                            assert strip(st) == strip(rs), (strip(st), strip(rs))
                            assert st["cache_len"] == prompt.numel() + len(out) == rs["cache"].get_seq_length()
                            if tau is not None and W == 16 and not stop_ids:
                                out_s, st_s = fast_generate(f, None, prompt, 30, window=W, tau=tau, mask_id=mask_id,
                                                            stop_ids=[], predict=predict, static=True)
                                assert out_s == ref and strip(st_s) == strip(rs), (W, tau, predict)
                            n_cases += 1
        # mask_id defaults to the checkpoint's cdlm_mask_token_id; both predictors differ on random weights
        prompt = random_prompts(g, 1)[0]
        a, _ = fast_generate(f, None, prompt, 20, window=4, tau=None, stop_ids=[], predict="ar")
        b, _ = fast_generate(f, None, prompt, 20, window=4, tau=None, stop_ids=[], predict="mask")
        assert a == hf_generate(m, None, prompt, 20, window=4, tau=None, stop_ids=[], predict="ar")[0] and a != b
    # tau=1e9 fills W per step; sampling path and max_new_tokens=0
    out, st = fast_generate(f, None, prompt, 37, window=16, tau=1e9, mask_id=mask_id, stop_ids=[])
    assert len(out) == 37 and st["steps"] == 4 and st["filled_per_step"] == {5: 1, 16: 2}, st
    out, st = fast_generate(f, None, prompt, 23, window=16, tau=0.5, mask_id=mask_id, stop_ids=[], greedy=False,
                            generator=torch.Generator().manual_seed(0))
    assert len(out) == 23 and st["finish_reason"] == "length"
    out, st = fast_generate(f, None, prompt, 0, window=16, tau=None, mask_id=mask_id, stop_ids=[])
    assert out == [] and st["steps"] == 0 and st["cache_len"] == prompt.numel()
    print(f"  {n_cases} decoder cases identical")


FILL_OPTION_SETS = [dict(no_adjacent=True), dict(max_fill=3), dict(max_fill=1), dict(margin_tau=0.3),
                    dict(fill_mode="leftmost_run"), dict(no_adjacent=True, max_fill=2, margin_tau=0.2),
                    dict(margin_tau=0.5, fill_mode="leftmost_run")]


def test_fast_decoder_matches_hf_decoder_fill_options():
    """(2b) fast_decoder.generate == decoder.generate (tokens, stats incl. filled_positions) for every fill-selection
    option set x W in {4, 16} x tau in {0.5, 2.5, 1e9} x predict in {mask, ar} on 4 prompts (+ a random stop id on
    half of them), the margin-only rule at tau=None, and the static path; the options do change the output vs the
    reference rule somewhere (non-vacuous)."""
    g = torch.Generator().manual_seed(21)
    m, f, mask_id = pair(seed=9)
    n_cases = differs = 0
    for i, prompt in enumerate(random_prompts(g, 4)):
        stop_ids = [] if i % 2 == 0 else [int(torch.randint(0, VOCAB, (1,), generator=g))]
        for W in (4, 16):
            for tau in (0.5, 2.5, 1e9, None):
                for predict in ("mask", "ar"):
                    for opts in FILL_OPTION_SETS:
                        if tau is None and opts.get("margin_tau") is None:
                            continue                      # tau=None without margin_tau is the leftmost mask (test 2)
                        ref, rs = hf_generate(m, None, prompt, 30, window=W, tau=tau, lam=0.02, mask_id=mask_id,
                                              stop_ids=stop_ids, predict=predict, **opts)
                        out, st = fast_generate(f, None, prompt, 30, window=W, tau=tau, lam=0.02, mask_id=mask_id,
                                                stop_ids=stop_ids, predict=predict, **opts)
                        assert out == ref, (W, tau, predict, opts, stop_ids, out, ref)
                        assert strip(st) == strip(rs), (W, tau, predict, opts, strip(st), strip(rs))
                        assert st["cache_len"] == prompt.numel() + len(out) == rs["cache"].get_seq_length()
                        if W == 16 and predict == "mask":
                            out_s, st_s = fast_generate(f, None, prompt, 30, window=W, tau=tau, lam=0.02, mask_id=mask_id,
                                                        stop_ids=stop_ids, predict=predict, static=True, **opts)
                            assert out_s == ref and strip(st_s) == strip(rs), (W, tau, predict, opts)
                        base, _ = fast_generate(f, None, prompt, 30, window=W, tau=tau, lam=0.02, mask_id=mask_id,
                                                stop_ids=stop_ids, predict=predict)
                        differs += base != out
                        n_cases += 1
    assert differs > 0
    print(f"  {n_cases} decoder cases with fill options identical ({differs} differ from the reference rule)")


def test_cache_len_bookkeeping():
    """(3) after generate, slots [0, cache_len) hold the K/V of a fresh causal forward of [prompt | output] (< 1e-5),
    the slots beyond cache_len are scratch (overwriting them changes nothing), and the cache can be continued."""
    g = torch.Generator().manual_seed(30)
    m, f, mask_id = pair(seed=7)
    worst = 0.0
    for prompt in random_prompts(g, 3):
        for tau, predict, W in ((None, "ar", 16), (None, "mask", 4), (0.5, "mask", 16), (1e9, "mask", 8), (1.0, "ar", 16)):
            out, st = fast_generate(f, None, prompt, 25, window=W, tau=tau, mask_id=mask_id, stop_ids=[3], predict=predict)
            cl = st["cache_len"]
            full = torch.tensor(prompt.tolist() + out)
            assert cl == full.numel() == st["prompt_len"] + st["generated"]
            ref = DynamicCache(config=m.config)
            ref_logits = m.model(input_ids=full[None], past_key_values=ref, use_cache=True).last_hidden_state[0]
            ref_logits = m.lm_head(ref_logits)
            for li in range(m.config.num_hidden_layers):
                d = max(maxdiff(f.kv.K[li, :, :, :cl], ref.layers[li].keys), maxdiff(f.kv.V[li, :, :, :cl], ref.layers[li].values))
                worst = max(worst, d); assert d < 1e-5, (tau, predict, W, li, d)
            # scratch beyond cache_len: scribble finite garbage there, then continue one token on the cache
            f.kv.K[:, :, :, cl:] = 7.0; f.kv.V[:, :, :, cl:] = -3.0
            nxt = torch.tensor([int(torch.randint(0, VOCAB, (1,), generator=g))])
            out1 = f(nxt, torch.tensor([cl]), cl, write=True)[0]
            ref1 = m(input_ids=torch.cat([full, nxt])[None], use_cache=False).logits[0, -1]
            d = maxdiff(out1, ref1); worst = max(worst, d); assert d < 1e-4, d
            # the cache is not disturbed by the garbage: a static-path step on top gives the same logits
            f.kv.K[:, :, :, cl + 1:] = 7.0
            out1s = f.run(nxt, torch.tensor([cl]), cl, [0], static=True)[0]
            assert maxdiff(out1s, ref1) < 1e-4
    # cache capacity is asserted, not silently exceeded
    try:
        fast_generate(f, None, random_prompts(g, 1)[0], MAX_LEN, window=16, mask_id=mask_id, stop_ids=[])
        raise RuntimeError("expected an AssertionError for prompt + budget + window > max_len")
    except AssertionError:
        pass
    print(f"  max |cache - full forward| / continuation: {worst:.2e}")


VERIFY_OPTION_SETS = [dict(no_adjacent=True), dict(max_fill=2), dict(margin_tau=0.3), dict(fill_mode="leftmost_run")]


def hf_greedy(m, prompt, n, stop_ids):
    # explicit all-ones attention mask: without it generate() treats every prompt token equal to pad_token_id (0, a
    # real id in this vocabulary) as padding and masks it out, which is not the greedy continuation of the prompt
    kw = dict(max_new_tokens=n, min_new_tokens=0, do_sample=False, pad_token_id=0, num_beams=1,
              attention_mask=torch.ones_like(prompt[None]))
    if stop_ids:
        kw["eos_token_id"] = list(stop_ids)
    return m.generate(prompt[None], **kw)[0][prompt.numel():].tolist()


def test_fast_decoder_matches_hf_decoder_verify():
    """(2c) verify=True: fast_decoder.generate == decoder.generate token for token and in every stat (incl. the
    verify stats and filled_positions) on 3 seeds (tied and untied) x 4 prompts x {no stop, random stop} x W in
    {1, 4, 16} x tau in {None, 0.5, 1e9} x predict in {mask, ar}, plus the fill rules; both equal HF greedy
    generate(); the static path agrees; tokens/step >= 1; slots [0, cache_len) hold [prompt | output] (incl. the
    completion forward of a final correction)."""
    g = torch.Generator().manual_seed(22)
    n_cases = n_static = 0
    tot_acc = tot_drafted = 0
    worst = 0.0
    for seed in (3, 4, 5):
        tie = seed % 2 == 0
        m, f, mask_id = pair(seed=seed, tie=tie)
        for i, prompt in enumerate(random_prompts(g, 4)):
            P = prompt.numel()
            for stop_ids in ([], [int(torch.randint(0, VOCAB, (1,), generator=g))]):
                ref_hf = hf_greedy(m, prompt, 30, stop_ids)
                cases = [(W, tau, predict, {}) for W in (1, 4, 16) for tau in (None, 0.5, 1e9) for predict in ("mask", "ar")]
                cases += [(W, tau, "mask", o) for o in VERIFY_OPTION_SETS for W in (4, 16) for tau in (0.5, 1e9)]
                for W, tau, predict, opts in cases:
                    ref, rs = hf_generate(m, None, prompt, 30, window=W, tau=tau, lam=0.02, mask_id=mask_id, stop_ids=stop_ids,
                                          predict=predict, verify=True, **opts)
                    out, st = fast_generate(f, None, prompt, 30, window=W, tau=tau, lam=0.02, mask_id=mask_id, stop_ids=stop_ids,
                                            predict=predict, verify=True, **opts)
                    assert out == ref == ref_hf, (seed, W, tau, predict, opts, stop_ids, out, ref, ref_hf)
                    assert strip(st) == strip(rs), (seed, W, tau, predict, opts, strip(st), strip(rs))
                    assert st["verify"] and st["tokens_per_step"] >= 1.0 and min(st["committed_per_step"]) >= 1
                    assert st["forwards"] == st["steps"] + 1 + int(st["completion_forward"])
                    assert st["cache_len"] == P + len(out) == rs["cache"].get_seq_length()
                    tot_acc += st["accepted"]; tot_drafted += st["drafted"]
                    if W == 16 and i % 2 == 0:
                        out_s, st_s = fast_generate(f, None, prompt, 30, window=W, tau=tau, lam=0.02, mask_id=mask_id,
                                                    stop_ids=stop_ids, predict=predict, verify=True, static=True, **opts)
                        assert out_s == ref and strip(st_s) == strip(rs), (seed, W, tau, predict, opts)
                        n_static += 1
                    if W == 16 and tau == 1e9 and predict == "mask" and not opts:
                        full = torch.tensor(prompt.tolist() + out)
                        cache = DynamicCache(config=m.config)
                        m.model(input_ids=full[None], past_key_values=cache, use_cache=True)
                        cl = st["cache_len"]
                        for li in range(m.config.num_hidden_layers):
                            d = max(maxdiff(f.kv.K[li, :, :, :cl], cache.layers[li].keys), maxdiff(f.kv.V[li, :, :, :cl], cache.layers[li].values))
                            worst = max(worst, d); assert d < 1e-5, (seed, li, d)
                    n_cases += 1
    assert 0 < tot_acc < tot_drafted
    out, st = fast_generate(f, None, prompt, 0, window=16, tau=None, mask_id=mask_id, stop_ids=[], verify=True)
    assert out == [] and st["steps"] == 0 and st["cache_len"] == prompt.numel()
    print(f"  {n_cases} verify cases identical in both decoders and == HF greedy ({n_static} static); drafts accepted"
          f" {tot_acc}/{tot_drafted}; max |cache - full forward| {worst:.2e}")


class SuppressMask(torch.nn.Module):
    """lm_head wrapper: the mask id is never emitted (as for a real model; this random one otherwise emits it, which
    would make an observed row holding it indistinguishable from a mask row for the oracle below)."""

    def __init__(self, head, mask_id):
        super().__init__(); self.head, self.mask_id = head, mask_id

    def forward(self, h):
        logits = self.head(h)
        logits[..., self.mask_id] -= 1e4
        return logits


class OracleFast(FastQwen3):
    """FastQwen3 whose mask rows are a deliberately good drafter: a window row holding the mask token gets a one-hot
    of the greedy token at its position (wrong at every multiple of wrong_every when set); the AR rows are real
    (with the mask id suppressed by SuppressMask on lm_head, installed by the test)."""
    oracle, mask_id, wrong_every = {}, None, None

    def run(self, tokens, pos_ids, cache_len, rows, static=False, use_graph=False):
        logits = super().run(tokens, pos_ids, cache_len, rows, static=static, use_graph=use_graph).clone()
        for i, r in enumerate(rows):
            p = int(pos_ids[r])
            if int(tokens[r]) == self.mask_id and p in self.oracle:
                t = self.oracle[p]
                if self.wrong_every and p % self.wrong_every == 0:
                    t = (t + 1) % VOCAB
                logits[i] = -50.0
                logits[i, t] = 50.0
        return logits


def test_fast_verify_good_drafter():
    """(2d) with a deliberately good mask head on the fast model, verify=True is still exactly HF greedy and reaches
    tokens/step > 1 (perfect drafter: every draft accepted, W tokens in a fully accepted step; a drafter wrong at every
    third position: in between), on the exact and the static path."""
    g = torch.Generator().manual_seed(23)
    m, mask_id = hf_model(seed=11)
    d = tempfile.mkdtemp(prefix="cdlm_fast_oracle_")
    m.save_pretrained(d)
    f = OracleFast.from_pretrained(d, device="cpu", dtype=torch.float32, max_len=MAX_LEN)
    f.mask_id = mask_id
    f.lm_head, m.lm_head = SuppressMask(f.lm_head, mask_id), SuppressMask(m.lm_head, mask_id)
    n_cases = 0
    tps16 = {}
    for prompt in random_prompts(g, 3):
        P = prompt.numel()
        ref = hf_greedy(m, prompt, 40, [])
        assert mask_id not in ref
        f.oracle = {P + i: t for i, t in enumerate(ref)}
        for wrong_every in (None, 3):
            f.wrong_every = wrong_every
            for W in (4, 16):
                for tau in (None, 1e9):
                    for static in (False, True):
                        out, st = fast_generate(f, None, prompt, 40, window=W, tau=tau, mask_id=mask_id, stop_ids=[],
                                                verify=True, static=static)
                        assert out == ref, (wrong_every, W, tau, static, out, ref)
                        assert st["cache_len"] == P + 40 and st["tokens_per_step"] > 1.0, st
                        if W == 16 and tau == 1e9 and not static:
                            tps16[wrong_every] = st["tokens_per_step"]
                        if wrong_every is None:
                            assert st["draft_acceptance_rate"] == 1.0 and st["tokens_per_step"] > 1.5
                            if tau == 1e9:
                                assert max(st["committed_per_step"]) == W and st["tokens_per_step"] >= W / 3, (W, st)
                        else:
                            assert 0.0 < st["draft_acceptance_rate"] < 1.0
                        n_cases += 1
    print(f"  {n_cases} good-drafter cases exact on the fast model; W=16, tau=1e9, 40 tokens: {tps16[None]:.2f} tokens/step"
          f" with the perfect drafter, {tps16[3]:.2f} with the every-third-wrong drafter")


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
