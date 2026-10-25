"""Phase B8 §4 — training data for the pairwise compatibility head (docs/pairwise_joint_diffusion.md §3).

The head has to approximate, WITHOUT an extra forward, the dependency term the chain-rule joint uses:
    delta(a, b) = log p(b | a placed at i) - log p(b)        for the two positions (i, j) a fixed-budget step places.
Data are collected on the sampler's OWN free-generation trajectories (K = 16, S = 8, left-to-right, the deployment
distribution — no reference tokens needed, the target is the frozen model's own conditional): at every two-token step
  h_i, h_j        final-layer hidden states of the two masked positions from the step's forward (bf16)
  cand_i, cand_j  top-k token ids and log-probs at the two positions (k = 4)
  delta [k, k]    log p(b | a) - log p(b) from k extra forwards (candidate a placed at i, j still masked)
  x               the block state at the step (K token ids, mask id where masked) and, per prompt, the prompt ids and
                  the generated tokens (<out>_prompts.jsonl) so that every state can be rebuilt for training
  step, P, block  context (step index 1..7, prefix length, block index)
The sampler continues with the chain-rule joint choice (so later steps come from the better trajectory) — flag
--follow independent to keep the plain independent trajectory instead.  Shards of --shard rows (torch.save).
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv-train/bin/python distill/gen_pairs.py --model runs/zeta_sft_final_e5/final \
      --adapter runs/bdiff_e5_lr1e4/adapter --prompts bench/prompts_gsm8k_pairtrain.json --n 1200 --k 4 \
      --out data/distill/pairs_gsm8k"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts, score
from engine.bdiff.model import BDiffModel
from engine.bdiff.corrupt import gamma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--follow", default="joint", choices=["joint", "independent"])
    ap.add_argument("--shard", type=int, default=4000)
    ap.add_argument("--out", required=True, help="output prefix; writes <out>_<shard>.pt and <out>_meta.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    mask_id = getattr(lm.config, "cdlm_mask_token_id", None) or tok.convert_tokens_to_ids("<|mask|>")
    if a.adapter:
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, a.adapter).eval().merge_and_unload()
    model = BDiffModel(lm, mask_id, time_cond=False).to(dev).eval()
    K, S, k = a.K, a.steps, a.k
    stop_ids = {tok.eos_token_id} | ({tok.convert_tokens_to_ids("<|im_end|>")} if "<|im_end|>" in tok.get_vocab() else set())
    prompts = load_prompts(a.prompts)[a.skip: a.skip + a.n]
    rows = []; n_shard = 0; n_rows = 0; n_fwd = 0; t0 = time.time()
    pf = open(f"{a.out}_prompts.jsonl", "w")

    def flush():
        nonlocal rows, n_shard
        if rows:
            torch.save(dict(h=torch.stack([r["h"] for r in rows]), cand=torch.stack([r["cand"] for r in rows]), lp=torch.stack([r["lp"] for r in rows]),
                            delta=torch.stack([r["delta"] for r in rows]), meta=torch.tensor([r["meta"] for r in rows]), x=torch.stack([r["x"] for r in rows])), f"{a.out}_{n_shard:03d}.pt")
            n_shard += 1; rows = []

    for pi, p in enumerate(prompts):
        ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        ids = [int(t) for t in (ids[0] if torch.is_tensor(ids) else ids["input_ids"][0])]
        out = []; finish = None; blk = 0
        while finish is None:
            Kb = min(K, a.max_new_tokens - len(out))
            P = len(ids) + len(out)
            cache = model.prefix_cache(torch.tensor([ids + out], device=dev)); n_fwd += 1
            x = torch.full((Kb,), mask_id, dtype=torch.long, device=dev)
            for s in range(1, S + 1):
                is_mask = x == mask_id
                n_mask = int(is_mask.sum())
                if n_mask == 0:
                    break
                g = max(n_mask / Kb, 1e-3)
                lg, hid = model.denoise(cache, P, x[None], g, return_hidden=True); n_fwd += 1
                lg = lg.float(); lg[:, mask_id] = float("-inf")
                probs = torch.softmax(lg, -1)
                last = s == S
                target = 0 if last else int(round(Kb * gamma(1.0 - s / S)))
                n_un = max(0, n_mask - target)
                masked_idx = is_mask.nonzero().flatten()
                sel = masked_idx[:n_un] if not last else masked_idx
                pred = probs.argmax(-1)
                if len(sel) == 2 and not last:
                    i, j = sel.tolist()
                    ti = probs[i].topk(k); tj = probs[j].topk(k)
                    lp_i = ti.values.clamp_min(1e-12).log(); lp_j = tj.values.clamp_min(1e-12).log()
                    delta = torch.zeros(k, k, device=dev)
                    for ai in range(k):
                        x2 = x.clone(); x2[i] = ti.indices[ai]
                        l2 = model.denoise(cache, P, x2[None], g).float(); n_fwd += 1
                        l2[:, mask_id] = float("-inf")
                        pj = torch.softmax(l2[j], -1)
                        delta[ai] = pj[tj.indices].clamp_min(1e-12).log() - lp_j
                    rows.append(dict(h=torch.stack([hid[i], hid[j]]).to(torch.bfloat16).cpu(), cand=torch.stack([ti.indices, tj.indices]).cpu(),
                                     lp=torch.stack([lp_i, lp_j]).cpu(), delta=delta.cpu(), meta=[pi + a.skip, blk, s, P, i, j], x=x.clone().cpu()))
                    n_rows += 1
                    if a.follow == "joint":
                        jscore = lp_i[:, None] + lp_j[None, :] + delta
                        best = int(jscore.argmax()); ai, bj = divmod(best, k)
                        pred = pred.clone(); pred[i] = ti.indices[ai]; pred[j] = tj.indices[bj]
                    if len(rows) >= a.shard:
                        flush()
                x[sel] = pred[sel]
            toks = x.tolist()
            keep = Kb
            for q in range(Kb):
                if toks[q] in stop_ids:
                    keep, finish = q + 1, "stop"; break
            out += toks[:keep]; blk += 1
            if len(out) >= a.max_new_tokens:
                finish = finish or "length"
        pf.write(json.dumps(dict(idx=pi + a.skip, id=p["id"], ids=ids, out=out, correct=score(p, tok.decode(out, skip_special_tokens=True)))) + "\n"); pf.flush()
        if pi % 20 == 0:
            print(f"{pi} prompts, {n_rows} pair rows, {n_fwd} forwards, {time.time() - t0:.0f}s", flush=True)
    flush(); pf.close()
    json.dump(dict(model=a.model, adapter=a.adapter, prompts=a.prompts, n=len(prompts), k=k, K=K, steps=S, follow=a.follow, rows=n_rows, shards=n_shard,
                   forwards=n_fwd, seconds=round(time.time() - t0, 1), hidden=int(model.lm.config.hidden_size)), open(f"{a.out}_meta.json", "w"), indent=1)
    print(f"=== gen_pairs {n_rows} rows in {n_shard} shards, {n_fwd} forwards, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
