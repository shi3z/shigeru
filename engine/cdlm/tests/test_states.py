"""Plain-assert tests for on-policy `states` in engine.cdlm.data.build_example and the trainer's state dataset (CPU).

Run from the repo root:  CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -m engine.cdlm.tests.test_states
"""
import os
import sys
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch

from engine.cdlm.data import IGNORE, build_example, collate
from engine.cdlm.mask import dense_mask

MASK_ID = 97
B = 8


def example(seed=0, L=40, P=10):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 90, (L,), generator=g)
    return tokens, P


def test_state_block_layout_and_labels():
    tokens, P = example()
    L = tokens.numel()
    # block 2 = positions 16..23 (all response): place wrong tokens at 16,17,20, mask the rest; block 3: one correct placed
    st2 = [91, 92, -1, -1, 93, -1, -1, -1]
    st3 = [-1] * B
    st3[1] = int(tokens[25])
    ex = build_example(tokens, P, MASK_ID, block=B, rng=torch.Generator().manual_seed(1), states={2: st2, 3: st3})
    ids, pos, lab, w = ex["input_ids"], ex["position_ids"], ex["mlm_labels"], ex["mlm_weights"]
    assert ex["n_corrupt"] == 3 and ex["n_state_blocks"] == 2, (ex["n_corrupt"], ex["n_state_blocks"])
    # physical rows of xt block 2: L + 16 .. L + 23, observed first (16, 17, 20) then masks (18, 19, 21, 22, 23)
    r = slice(L + 16, L + 24)
    assert pos[r].tolist() == [16, 17, 20, 18, 19, 21, 22, 23], pos[r].tolist()
    assert ids[r].tolist()[:3] == [91, 92, 93] and all(t == MASK_ID for t in ids[r].tolist()[3:]), ids[r].tolist()
    assert lab[r].tolist()[:3] == [IGNORE] * 3 and lab[r].tolist()[3:] == tokens[[18, 19, 21, 22, 23]].tolist()
    assert torch.allclose(w[r][3:], torch.full((5,), 1 / (5 / 8))) and bool((w[r][:3] == 0).all()), w[r]
    # the clean tokens of the corrupted positions appear nowhere in the xt rows of that block
    for p_ in (16, 17, 20):
        assert int(tokens[p_]) not in ids[r].tolist() or int(tokens[p_]) in (91, 92, 93)
    # block 3: the placed token equals the clean one -> not corrupt; weight 1/(7/8)
    r3 = slice(L + 24, L + 32)
    assert pos[r3].tolist()[0] == 25 and ids[r3].tolist()[0] == int(tokens[25])
    assert torch.allclose(w[r3][1:], torch.full((7,), 1 / (7 / 8)))
    # x0 half untouched
    assert ids[:L].tolist() == tokens.tolist() and ex["ar_labels"][:L - 1][P - 1:].tolist() == tokens[P:].tolist()


def test_other_blocks_follow_gamma_and_mask_is_unchanged():
    tokens, P = example(3)
    rng_a, rng_b = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    plain = build_example(tokens, P, MASK_ID, block=B, rng=rng_a)
    with_state = build_example(tokens, P, MASK_ID, block=B, rng=rng_b, states={4: [-1] * B})
    L = tokens.numel()
    # blocks 0..3 identical (same rng stream consumption: a state block draws nothing beyond gamma)
    assert with_state["input_ids"][: L + 32].tolist() == plain["input_ids"][: L + 32].tolist()
    assert with_state["mlm_labels"][: L + 32].tolist() == plain["mlm_labels"][: L + 32].tolist()
    # the attention mask depends only on (L, block), never on the states
    batch = collate([plain, with_state], pad_id=0)
    assert torch.equal(batch["attention_mask"][0], batch["attention_mask"][1])
    assert torch.equal(batch["attention_mask"][0, 0], dense_mask(L, B, 2 * L))


def test_prompt_positions_never_masked_or_corrupted():
    tokens, P = example(4, L=24, P=12)       # block 1 = positions 8..15 straddles the prompt boundary at 12
    st = [-1] * 4 + [50, -1, 51, -1]          # masks only on response positions 12..15 would be fine; this masks prompt ones
    try:
        build_example(tokens, P, MASK_ID, block=B, rng=torch.Generator().manual_seed(0), states={1: st})
        assert False, "prompt position masked by a state must fail"
    except AssertionError as e:
        assert "prompt" in str(e)
    st = [60, 61, 62, 63, 50, -1, 51, -1]     # placed tokens on prompt positions are ignored: prompt rows stay clean
    ex = build_example(tokens, P, MASK_ID, block=B, rng=torch.Generator().manual_seed(0), states={1: st})
    L = tokens.numel()
    rows = ex["input_ids"][L + 8: L + 16].tolist()
    assert rows[:4] == tokens[8:12].tolist(), rows
    assert ex["n_corrupt"] == sum(int(t != int(tokens[p])) for t, p in ((50, 12), (51, 14)))


def test_no_label_leakage_through_inputs():
    """Changing the clean token at a masked or corrupted position changes labels only, never any xt input row."""
    tokens, P = example(6)
    st = [91, -1, 92, -1, -1, -1, 93, -1]
    a = build_example(tokens, P, MASK_ID, block=B, rng=torch.Generator().manual_seed(2), states={2: st})
    t2 = tokens.clone()
    for p_ in (16, 17, 18, 19):               # 16, 18 corrupted; 17, 19 masked
        t2[p_] = (int(t2[p_]) + 7) % 90
    b = build_example(t2, P, MASK_ID, block=B, rng=torch.Generator().manual_seed(2), states={2: st})
    L = tokens.numel()
    assert a["input_ids"][L:].tolist() == b["input_ids"][L:].tolist()          # xt inputs identical
    assert a["mlm_labels"][L:].tolist() != b["mlm_labels"][L:].tolist()        # labels follow the clean tokens


def test_trainer_dataset_mixing():
    sys.path.insert(0, os.getcwd())
    from distill.train_cdlm import CDLMDataset
    tokens, P = example(8)
    items = [(tokens, P, {2: [91, 92, -1, -1, 93, -1, -1, -1]})] * 20
    ds_all = CDLMDataset(items, MASK_ID, B, 0.0, 1.0, seed=0, states_mix=1.0)
    ds_none = CDLMDataset(items, MASK_ID, B, 0.0, 1.0, seed=0, states_mix=0.0)
    ds_half = CDLMDataset(items, MASK_ID, B, 0.0, 1.0, seed=0, states_mix=0.5)
    assert all(ds_all[i]["n_state_blocks"] == 1 for i in range(20))
    assert all(ds_none[i]["n_state_blocks"] == 0 for i in range(20))
    n = sum(ds_half[i]["n_state_blocks"] for i in range(20))
    assert 3 <= n <= 17, n


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
