"""On-policy denoising states for training (SHIGERU Phase D §4, docs/pure_diffusion_research.md section 7).

For each example of a stage-2 JSONL (tokenized exactly as distill/train_cdlm.py does), a few response blocks are chosen
at random; each is denoised FROM SCRATCH by the current checkpoint with the self-correcting denoiser
(engine/cdlm/sc_decoder.denoise_block, teacher-forced clean prefix, write=False forwards) and the INTERMEDIATE states the
decoder actually visits are recorded: per step, each slot is a [MASK] (-1) or the token the model placed there — right
or wrong.  Those states (with the clean tokens as labels on the masks) are the training examples of the on-policy
objective: the model learns to predict the truth from contexts that contain its own mistakes, and to re-predict a
re-masked position it had filled wrongly.  Up to --states-per-block states per block are kept, evenly spaced over the
steps, preferring states that contain at least one wrong placed token.
Output JSONL rows: {id, tokens: [prompt + response ids], prompt_len, block, states: {block_index: [B ints]}, n_wrong:
{block_index: [per state]}}; train_cdlm.py reads them with `data_states:`.
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. .venv-train/bin/python distill/gen_onpolicy.py --model runs/phaseC_sft3_4b/final \
      --data data/distill/phaseA_stage2c_train.jsonl --limit 6000 --blocks-per-item 2 --states-per-block 3 \
      --out data/distill/pure_states_sft3.jsonl"""
import argparse, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from distill.train_cdlm import load_jsonl, tokenize_examples
from engine.cdlm.fast_model import FastQwen3
from engine.cdlm.sc_decoder import denoise_block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--blocks-per-item", type=int, default=2)
    ap.add_argument("--states-per-block", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    # denoiser configuration (the one the states should be representative of)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--remask", default="lowconf:0.9")
    ap.add_argument("--confirm-k", type=int, default=1)
    ap.add_argument("--no-anchor", action="store_true")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    mask_id = getattr(hf.config, "cdlm_mask_token_id", None) or tok.convert_tokens_to_ids("<|mask|>")
    model = FastQwen3.from_hf(hf, device="cuda", dtype=torch.bfloat16, max_len=4096)
    del hf
    torch.cuda.empty_cache()
    rows = load_jsonl(a.data)
    if a.limit:
        rng.shuffle(rows)
        rows = rows[: a.limit]
    items, dropped, trunc = tokenize_examples(rows, tok, a.max_len)
    print(f"{len(items)} items (dropped {dropped}, truncated {trunc})", file=sys.stderr, flush=True)
    t0 = time.time()
    n_states = n_wrong_states = n_fwd = 0
    with open(a.out, "w") as f:
        for n, ((tokens, P), row) in enumerate(zip(items, rows)):
            ids = tokens.tolist()
            L = len(ids)
            K = a.block
            # candidate blocks: fully inside the response
            cands = [b for b in range(-(-L // K)) if b * K >= P and (b + 1) * K <= L]
            if not cands:
                continue
            chosen = rng.sample(cands, min(a.blocks_per_item, len(cands)))
            model.kv.zero_()
            model.prefill(torch.tensor(ids, device="cuda"), 0)
            states, nwrong = {}, {}
            for b in chosen:
                s = b * K
                gold = ids[s:s + K]

                def fwd(i_, p_, r_, _s=s):
                    return model.forward(torch.tensor(i_, device="cuda"), torch.tensor(p_, device="cuda"), cache_len=_s,
                                         write=False, rows=r_)

                g = torch.Generator().manual_seed(a.seed * 7919 + n * 31 + b)
                _, info = denoise_block(fwd, s, K, mask_id, steps=None, tau=a.tau, anchor_leftmost=not a.no_anchor,
                                        remask=a.remask, confirm_k=a.confirm_k, trace=True, generator=g)
                n_fwd += info["forwards"]
                # input state of step t = output of step t-1 (tokens/status after step t-1); skip the all-mask start
                cand_states = []
                for tr in info["trace"][:-1]:
                    st = [tr["tokens"][i] if tr["status"][i] != 0 else -1 for i in range(K)]
                    if all(x == -1 for x in st) or all(x != -1 for x in st):
                        continue
                    wrong = sum(1 for i in range(K) if st[i] != -1 and st[i] != gold[i])
                    cand_states.append((st, wrong))
                if not cand_states:
                    continue
                with_wrong = [c for c in cand_states if c[1] > 0]
                pool = with_wrong if len(with_wrong) >= a.states_per_block else cand_states
                if len(pool) > a.states_per_block:
                    idx = [round(i * (len(pool) - 1) / (a.states_per_block - 1)) for i in range(a.states_per_block)] if a.states_per_block > 1 else [len(pool) // 2]
                    pool = [pool[i] for i in sorted(set(idx))]
                # one row per state (a block index can carry only one state per training example)
                for st, wrong in pool:
                    states.setdefault(b, []).append(st)
                    nwrong.setdefault(b, []).append(wrong)
                    n_states += 1
                    n_wrong_states += int(wrong > 0)
            # emit one JSONL row per state so each training example carries exactly one state block
            for b, sts in states.items():
                for st, w in zip(sts, nwrong[b]):
                    f.write(json.dumps({"id": row.get("id", n), "tokens": ids, "prompt_len": P, "block": K,
                                        "states": {str(b): st}, "n_wrong": {str(b): w}}) + "\n")
            if n % 50 == 0:
                print(f"{n}/{len(items)} items, {n_states} states ({n_wrong_states} with errors), {n_fwd} forwards, {time.time() - t0:.0f}s",
                      file=sys.stderr, flush=True)
    print(f"=== gen_onpolicy {n_states} states ({n_wrong_states} with >=1 wrong token) from {len(items)} items, {n_fwd} forwards, "
          f"{time.time() - t0:.0f}s -> {a.out}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
