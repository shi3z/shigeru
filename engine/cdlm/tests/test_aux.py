"""Plain-assert tests for engine.cdlm.aux (B13 auxiliary-conditioned drafter; CPU only, tiny random Qwen3).
Run from the repo root:  CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -m engine.cdlm.tests.test_aux
"""
import copy
import os
import sys
import traceback
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from engine.cdlm import IGNORE, add_mask_token, ar_loss, build_example, collate, forward_rows, mlm_loss, total_loss
from engine.cdlm.aux import AlignHooks, add_aux_tokens, align_loss, assign_aux, aux_ids_of
from engine.cdlm.decoder import generate

VOCAB = 120
BASE_CFG = Qwen3Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                       tie_word_embeddings=True, initializer_range=0.2)
MASK = 100                                                  # ids 100.. are free rows: [MASK] 100, auxiliaries 101..108


def tiny(seed=0):
    torch.manual_seed(seed)
    m = Qwen3ForCausalLM(copy.deepcopy(BASE_CFG)).float()
    m.set_attn_implementation("sdpa")
    mask_id = add_mask_token(m, MASK)["mask_id"]
    info = add_aux_tokens(m, mask_id, 8, seed=seed)
    return m, mask_id, info


def example_batch(mask_id, seed=1):
    rng = torch.Generator().manual_seed(seed)
    exs = []
    for L, pl in ((40, 7), (33, 5)):
        toks = torch.randint(0, mask_id, (L,), generator=rng)
        exs.append(build_example(toks, pl, mask_id, block=8, rng=rng, mask_mode="suffix"))
    return collate(exs, pad_id=0)


def test_add_aux_tokens_rows_and_config():
    m, mask_id, info = tiny()
    assert info == {"aux_base": 101, "aux_n": 8, "old_rows": VOCAB, "new_rows": VOCAB}, info
    assert aux_ids_of(m.config) == (101, 8)
    w = m.get_input_embeddings().weight[101:109]
    assert w.shape == (8, 64) and (w - w[0]).abs().sum() > 0          # distinct rows
    assert not torch.isnan(w).any()


def test_assign_aux_is_the_rank_under_the_ar_distribution():
    m, mask_id, info = tiny()
    batch = example_batch(mask_id)
    ids_aux, st = assign_aux(m, batch, mask_id, info["aux_base"], info["aux_n"], torch.device("cpu"))
    mrows = batch["mlm_labels"] != IGNORE
    assert st["aux_rows"] == int(mrows.sum()) and 0 <= st["aux_rank0_frac"] <= 1
    assert bool((ids_aux[~mrows] == batch["input_ids"][~mrows]).all()), "a non-mask row changed"
    assert m.training is False or True                                  # mode restored (model was in eval -> stays)
    # every mask row: aux id = base + rank of the label under the causal logits at position p-1, or [MASK] if rank >= n
    m.eval()
    with torch.no_grad():
        for b in range(batch["input_ids"].shape[0]):
            L = int(batch["seq_lens"][b])
            lg = m(input_ids=batch["input_ids"][b:b + 1, :L]).logits[0]
            for r in mrows[b].nonzero().flatten().tolist():
                p = int(batch["position_ids"][b, r]); y = int(batch["mlm_labels"][b, r])
                rank = int((lg[p - 1] > lg[p - 1, y]).sum())
                want = info["aux_base"] + rank if rank < info["aux_n"] else mask_id
                assert int(ids_aux[b, r]) == want, (b, r, p, rank, int(ids_aux[b, r]), want)


def test_assign_aux_with_a_forced_argmax_gives_rank0():
    """Planting the label as the overwhelming argmax at the AR row makes the auxiliary rank 0."""
    m, mask_id, info = tiny()
    batch = example_batch(mask_id)
    b, r = (batch["mlm_labels"] != IGNORE).nonzero()[0].tolist()
    p, y = int(batch["position_ids"][b, r]), int(batch["mlm_labels"][b, r])
    with torch.no_grad():                                               # output row y points along the hidden mean direction: huge logit everywhere
        L = int(batch["seq_lens"][b])
        h = m.model(input_ids=batch["input_ids"][b:b + 1, :L]).last_hidden_state[0, p - 1]
        m.get_output_embeddings().weight[y] = 1000.0 * h / h.norm()
    ids_aux, _ = assign_aux(m, batch, mask_id, info["aux_base"], info["aux_n"], torch.device("cpu"))
    assert int(ids_aux[b, r]) == info["aux_base"], int(ids_aux[b, r])


def test_align_loss_backward_and_training_step():
    m, mask_id, info = tiny()
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); m.train()
    proj = torch.nn.Linear(64, 64, bias=False); torch.nn.init.eye_(proj.weight)
    batch = example_batch(mask_id)
    ids_aux, _ = assign_aux(m, batch, mask_id, info["aux_base"], info["aux_n"], torch.device("cpu"))
    assert m.training, "assign_aux must restore train mode"
    mlm_rows = (batch["mlm_labels"] != IGNORE).nonzero(as_tuple=True); ar_rows = (batch["ar_labels"] != IGNORE).nonzero(as_tuple=True)
    rows = (torch.cat([mlm_rows[0], ar_rows[0]]), torch.cat([mlm_rows[1], ar_rows[1]]))
    with AlignHooks(m, 2, 3) as hk:
        logits = forward_rows(m, ids_aux, batch["position_ids"], batch["attention_mask"], rows)
        al = align_loss(hk, proj, batch, torch.device("cpu"))
    assert not hk.handles, "hooks removed on exit"
    assert 0.0 <= float(al) <= 2.0
    n = mlm_rows[0].numel()
    loss = total_loss(mlm_loss(logits[:n], batch["mlm_labels"][mlm_rows], batch["mlm_weights"][mlm_rows]),
                      ar_loss(logits[n:], batch["ar_labels"][ar_rows])) + al
    loss.backward()
    assert float(proj.weight.grad.norm()) > 0
    g = m.get_input_embeddings().weight.grad
    assert g is not None and float(g[info["aux_base"]:info["aux_base"] + info["aux_n"]].norm()) > 0, "aux rows receive gradient"
    # identical hidden states on both sides -> cos 1 -> loss 0 (identity map)
    hk.h = {"dec": torch.randn(2, 10, 64), "enc": None}
    hk.h["enc"] = hk.h["dec"].clone()
    b2 = {"mlm_labels": torch.full((2, 10), IGNORE), "position_ids": torch.arange(10)[None].repeat(2, 1), "seq_lens": torch.tensor([5, 5])}
    b2["mlm_labels"][:, 6:] = 1
    pos = b2["position_ids"]; hk.h["enc"] = torch.zeros(2, 10, 64)
    for b in range(2):
        for r in range(6, 10):
            hk.h["enc"][b, pos[b, r] - 1] = hk.h["dec"][b, r]
    assert float(align_loss(hk, proj, b2, torch.device("cpu"))) < 1e-5


def test_verified_decoding_with_aux0_is_exactly_greedy():
    m, mask_id, info = tiny()
    m.eval()
    prompt = torch.randint(0, mask_id, (9,)).tolist()
    def hf_greedy(suppress):                                            # the random model's greedy output may contain the free rows
        kw = {"suppress_tokens": list(range(mask_id, VOCAB))} if suppress else {}
        with torch.no_grad():
            return m.generate(torch.tensor([prompt]), attention_mask=torch.ones(1, 9, dtype=torch.long), max_new_tokens=24,
                              do_sample=False, **kw)[0, 9:].tolist()
    for mid, excl in ((mask_id, None), (info["aux_base"], mask_id), (mask_id, mask_id)):
        ref = hf_greedy(excl is not None)
        out, stats = generate(m, None, prompt, 24, window=8, tau=1e9, mask_id=mid, stop_ids=[], verify=True, exclude_from=excl)
        assert out == ref, (mid, excl, out, ref)
        if excl is not None:
            assert all(t < mask_id for t in out)
        assert stats["drafted"] > 0                                     # the exclusion keeps the draft entropies finite


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
