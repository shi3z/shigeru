"""Phase B11 §7 — training-state distribution vs inference-state distribution (docs/sampler_aligned_global_denoising.md §8).

Training side: the state shards of distill/gen_global_states.py (--train prefixes; the x_plain states, and x_corr for
the corrected variant) and, for the original B-series recipe, random masks over gold blocks (--gold-random, synthesised
from the dev reference blocks at the sampler's mask ratios).  Inference side: the states the --sampler visits on the
dev prompts in FREE generation (no reference), with the sequential teacher block T of each prefix as the "wrong
visible" reference (16 extra forwards per block, diagnostic only).
Features per state: mask ratio, islands, visible prefix length, right-visible flag, wrong-visible fraction (vs T on the
inference side; vs the stored teacher block on the training side: for plain states the teacher completion, for
corrected states 0 by construction), number of positions the sampler updates at this step and their mean pairwise
distance (inference side only).  Histograms (fixed bins) and the histogram intersection (overlap in [0, 1]) between each
training set and the inference distribution are reported per feature.
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv-train/bin/python bench/diag_state_dist.py --model runs/zeta_sft_final_e5/final \
      --adapter runs/bdiff_e5_lr1e4/adapter --train remask=data/distill/gstates_remask left=data/distill/gstates_left \
      --sampler remask --n 50 --out results/pure/b11_state_dist.json"""
import argparse, collections, glob, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts, gpu_info, save_json
from engine.bdiff.model import BDiffModel
from engine.bdiff.sampler2 import sample_block_fixed
from engine.bdiff.global_sampler import global_sample_block

BINS = dict(mask_ratio=[i / 16 for i in range(17)], islands=list(range(0, 9)), prefix=list(range(0, 17)), wrong_visible_frac=[i / 10 for i in range(11)],
            right_visible=[0, 0.5, 1], n_update=list(range(0, 17)), update_dist=list(range(0, 16)))


def feats(x, mask_id, ref):
    K = len(x); m = [t == mask_id for t in x]
    islands = sum(1 for q in range(K) if m[q] and (q == 0 or not m[q - 1]))
    prefix = 0
    while prefix < K and not m[prefix]:
        prefix += 1
    rv = int(any(m[q] and any(not m[r] for r in range(q + 1, K)) for q in range(K)))
    vis = [q for q in range(K) if not m[q]]
    wrong = sum(1 for q in vis if ref is not None and ref[q] >= 0 and x[q] != ref[q]) / max(1, len(vis)) if vis else 0.0
    return dict(mask_ratio=sum(m) / K, islands=islands, prefix=prefix, right_visible=rv, wrong_visible_frac=wrong)


def hist(values, edges):
    h = [0] * (len(edges) - 1)
    for v in values:
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1] or (i == len(edges) - 2 and v == edges[-1]):
                h[i] += 1; break
    n = max(1, sum(h))
    return [c / n for c in h]


def overlap(h1, h2):
    return round(sum(min(a, b) for a, b in zip(h1, h2)), 4)


@torch.no_grad()
def predict(model, cache, P, x, mask_id):
    xt = torch.tensor(x, device=model.device)
    lg = model.denoise(cache, P, xt[None], max(float((xt == mask_id).sum()) / len(x), 1e-3)).float(); lg[:, mask_id] = float("-inf")
    return lg.argmax(-1).tolist()


@torch.no_grad()
def seqfill(model, cache, P, x, mask_id):
    x = list(x)
    for q in range(len(x)):
        if x[q] == mask_id:
            x[q] = predict(model, cache, P, x, mask_id)[q]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--train", nargs="*", default=[], help="name=prefix pairs of gen_global_states outputs")
    ap.add_argument("--refs", default="results/pure/dev_refs_e5.json")
    ap.add_argument("--prompts", default="bench/prompts_gsm8k_dev.json")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--sampler", default="remask")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    K, S = a.K, a.steps
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    mask_id = getattr(lm.config, "cdlm_mask_token_id", None) or tok.convert_tokens_to_ids("<|mask|>")
    if a.adapter:
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, a.adapter).eval().merge_and_unload()
    model = BDiffModel(lm, mask_id, time_cond=False).to(dev).eval()
    # ---- training distributions
    dists = {}
    for spec in a.train:
        name, pref = spec.split("=")
        rows = collections.defaultdict(list)
        for f in sorted(glob.glob(f"{pref}_[0-9][0-9][0-9].pt")):
            d = torch.load(f)
            for n in range(d["meta"].shape[0]):
                Kb = int(d["meta"][n][4]); x = d["x_plain"][n][:Kb].tolist(); y = d["y_plain"][n][:Kb].tolist()
                # reference for "wrong visible" on the training side: visible tokens have no stored teacher value (they are kept);
                # use the corrected variant: x_corr re-masked exactly the visible tokens != T, so wrong fraction = (#masked in corr - #masked in plain)/#visible
                xc = d["x_corr"][n][:Kb].tolist()
                fe = feats(x, mask_id, None)
                vis = sum(1 for t in x if t != mask_id); extra = sum(1 for t in xc if t == mask_id) - sum(1 for t in x if t == mask_id)
                fe["wrong_visible_frac"] = extra / max(1, vis) if vis else 0.0
                for k, v in fe.items():
                    rows[k].append(v)
        dists[name] = {k: hist(v, BINS[k]) for k, v in rows.items()}; dists[name]["n"] = len(rows["mask_ratio"])
        dists[name]["means"] = {k: round(sum(v) / max(1, len(v)), 4) for k, v in rows.items()}
    # gold-random (the original recipe) from the dev references
    refs = json.load(open(a.refs))["refs"]
    prompts = load_prompts(a.prompts)[: a.n]
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        p["ids"] = [int(t) for t in (ids[0] if torch.is_tensor(ids) else ids["input_ids"][0])]
    rng = torch.Generator().manual_seed(0); rows = collections.defaultdict(list)
    for p in prompts:
        resp = refs[p["id"]]["response_ids"]
        for s0 in range(0, len(resp) - K + 1, K):
            for s in range(1, S + 1):
                n_mask = max(1, int(round(K * (1 - (s - 1) / S)))); perm = set(torch.randperm(K, generator=rng)[:n_mask].tolist())
                x = [mask_id if q in perm else resp[s0 + q] for q in range(K)]
                for k, v in feats(x, mask_id, resp[s0:s0 + K]).items():
                    rows[k].append(v)
    dists["gold_random"] = {k: hist(v, BINS[k]) for k, v in rows.items()}; dists["gold_random"]["n"] = len(rows["mask_ratio"])
    dists["gold_random"]["means"] = {k: round(sum(v) / max(1, len(v)), 4) for k, v in rows.items()}
    # ---- inference distribution: free generation with the sampler on the dev prompts
    stop_ids = {tok.eos_token_id} | ({tok.convert_tokens_to_ids("<|im_end|>")} if "<|im_end|>" in tok.get_vocab() else set())
    inf = collections.defaultdict(list); t0 = time.time()
    for p in prompts:
        ids = list(p["ids"]); out = []; finish = None
        while finish is None:
            Kb = min(K, a.max_new_tokens - len(out)); P = len(ids) + len(out)
            cache = model.prefix_cache(torch.tensor([ids + out], device=dev))
            if a.sampler in ("left", "confidence"):
                toks, info = sample_block_fixed(model, cache, P, Kb, S, mask_id, unmask=a.sampler, remask="none", early_stop=False, trace=True)
                states = []; prev = [mask_id] * Kb
                for st in info["trace"]:
                    states.append((list(prev), st["tokens"])); prev = st["tokens"]
            else:
                toks, info = global_sample_block(model, cache, P, Kb, S, mask_id, mode=a.sampler, trace=True)
                states = []; prev = [mask_id] * Kb
                for st in info["trace"]:
                    nxt = [mask_id if mk else g_ for g_, mk in zip(st["guess"], st["masked"])]
                    states.append((list(prev), nxt)); prev = nxt
            T = seqfill(model, cache, P, [mask_id] * Kb, mask_id)
            for x, nxt in states:
                fe = feats(x, mask_id, T)
                upd = [q for q in range(Kb) if x[q] != nxt[q]]
                fe["n_update"] = len(upd)
                fe["update_dist"] = (sum(abs(i - j) for i in upd for j in upd if i < j) / max(1, len(upd) * (len(upd) - 1) / 2)) if len(upd) > 1 else 0.0
                for k, v in fe.items():
                    inf[k].append(v)
            keep = Kb
            for q in range(Kb):
                if toks[q] in stop_ids:
                    keep, finish = q + 1, "stop"; break
            out += toks[:keep]
            if len(out) >= a.max_new_tokens:
                finish = finish or "length"
        print(f"{p['id']} states {len(inf['mask_ratio'])} {time.time() - t0:.0f}s", flush=True)
    inference = {k: hist(v, BINS[k]) for k, v in inf.items()}; inference["n"] = len(inf["mask_ratio"]); inference["means"] = {k: round(sum(v) / max(1, len(v)), 4) for k, v in inf.items()}
    ov = {name: {k: overlap(d[k], inference[k]) for k in ("mask_ratio", "islands", "prefix", "right_visible", "wrong_visible_frac")} for name, d in dists.items()}
    res = dict(sampler=a.sampler, inference=inference, training=dists, overlap_with_inference=ov)
    print("=== state_dist overlap", json.dumps(ov), flush=True)
    print("=== state_dist means", json.dumps(dict(inference=inference["means"], **{n: d["means"] for n, d in dists.items()})), flush=True)
    save_json(a.out, dict(model=a.model, adapter=a.adapter, gpu=gpu_info(), seconds=round(time.time() - t0, 1), **res))


if __name__ == "__main__":
    main()
