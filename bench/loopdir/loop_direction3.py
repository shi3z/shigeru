"""Is there a linear direction that marks repetition, and does it exist BEFORE the repetition starts?  (v3)

v2 reported a predictive AUC of 0.93, which is not evidence: its positives were the 24 positions before a loop
onset, which happens deep in a generation, while its negatives were sampled uniformly from generations averaging
473 tokens.  A direction encoding "how far into the generation am I" scores 0.93 on that split by itself.  v3
matches every negative to a positive's absolute position, reports a position-only baseline for comparison, and
saves the activations so later analyses need no regeneration.

The first attempt labelled with exact token-level period-8 repetition while the benchmark that found the loops used
a 12-word window occurring 4+ times in the decoded text.  That was simply the wrong criterion: measured on the same
150 bf16 generations, the word detector fires on 9.3 % of answers and the token one on 0.7 %, so the first run
collected 2 positives out of 309 generations and the probes had one class.  This run labels with the benchmark's
criterion, and takes the onset to be the first token position whose prefix satisfies it.

Collected per looping generation:
  descriptive  onset .. onset+64, against positions sampled from generations that never satisfy the criterion.
               Easy by construction (the context contains the repetition) and useless for prevention.
  predictive   onset-24 .. onset, against matched positions from non-looping generations.  Only a direction that
               separates these could be ablated to stop loops forming.

Per layer: the AUC of the difference-of-means direction (what an ablation actually uses) and of a ridge
discriminant (whether any hyperplane can do it), split by question so no generation appears on both sides.
"""
import argparse, json, random, time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

N_GRAM, TIMES = 12, 4


def fires(text):
    """The benchmark's detector: any N_GRAM-word window occurring TIMES or more."""
    w = text.split()
    if len(w) < N_GRAM * TIMES:
        return False
    seen = {}
    for i in range(len(w) - N_GRAM):
        k = " ".join(w[i:i + N_GRAM])
        seen[k] = seen.get(k, 0) + 1
        if seen[k] >= TIMES:
            return True
    return False


def onset_position(tok, ids, step=8):
    """First token index whose prefix satisfies the detector, scanning in `step`s then refining.  None if never."""
    lo = None
    for t in range(N_GRAM * TIMES, len(ids) + 1, step):
        if fires(tok.decode(ids[:t], skip_special_tokens=True)):
            lo = t
            break
    if lo is None:
        return None
    for t in range(max(N_GRAM * TIMES, lo - step), lo + 1):
        if fires(tok.decode(ids[:t], skip_special_tokens=True)):
            return t - 1
    return lo - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/shi3z/models/hf/ds9b-orig-hf")
    ap.add_argument("--gsm8k", default="/home/shi3z/shigeru/open-mercury/bench/prompts_gsm8k_full.json")
    ap.add_argument("--n", type=int, default=450)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--layers", default="4,8,12,16,20,24,28,31")
    ap.add_argument("--pre-window", type=int, default=24)
    ap.add_argument("--post-window", type=int, default=64)
    ap.add_argument("--out", default="/home/shi3z/loopdir/loop_direction2.json")
    ap.add_argument("--texts", default="/home/shi3z/loopdir/loop_texts.jsonl")
    ap.add_argument("--save-npz", default="/home/shi3z/loopdir/loop_acts.npz")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0},
                                                 trust_remote_code=True).eval()
    blocks = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
    print(f"loaded, {len(blocks)} layers, probing {layers}", flush=True)

    grabbed = {}

    def mk(i):
        def f(_m, _a, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            grabbed[i] = h[0, -1].detach().float().cpu()
        return f

    handles = [blocks[i].register_forward_hook(mk(i)) for i in layers]
    data = {k: {i: {"X": [], "y": [], "q": [], "pos": []} for i in layers} for k in ("descriptive", "predictive")}
    # every clean generation's activations stay available by position so negatives can be matched to a positive's
    # depth instead of drawn from the whole generation
    clean_pool = []   # list of (qid, n_tokens, {layer: {pos: tensor}}) kept sparse: one in 8 positions
    stats = {"generations": 0, "looped": 0, "clean": 0, "tokens": 0}
    texts = open(a.texts, "w")
    gsm = json.load(open(a.gsm8k))[: a.n]
    t0 = time.time()

    @torch.no_grad()
    def one(q):
        pr = tok.apply_chat_template([{"role": "user", "content": q["prompt"]}], tokenize=False,
                                     add_generation_prompt=True)
        ids = tok(pr, return_tensors="pt", add_special_tokens=False).to(model.device)
        past, cur, out, per = None, ids["input_ids"], [], []
        fired_at = None
        for _ in range(a.max_new):
            o = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = int(o.logits[0, -1].argmax())
            out.append(nxt)
            per.append({i: grabbed[i].clone() for i in layers})
            cur = torch.tensor([[nxt]], device=model.device)
            if nxt == tok.eos_token_id:
                break
            # once the criterion holds, keep post_window more positions and stop: the rest is more of the same
            if fired_at is None and len(out) % 8 == 0 and len(out) >= N_GRAM * TIMES:
                if fires(tok.decode(out, skip_special_tokens=True)):
                    fired_at = len(out)
            elif fired_at is not None and len(out) - fired_at >= a.post_window:
                break
        stats["generations"] += 1
        stats["tokens"] += len(out)
        text = tok.decode(out, skip_special_tokens=True)
        onset = onset_position(tok, out) if (fired_at is not None or fires(text)) else None
        texts.write(json.dumps({"id": q["id"], "tokens": len(out), "onset": onset,
                                "text": text}, ensure_ascii=False) + "\n")
        texts.flush()
        if onset is not None:
            stats["looped"] += 1
            for t in range(onset, min(len(out), onset + a.post_window)):
                for i in layers:
                    data["descriptive"][i]["X"].append(per[t][i]); data["descriptive"][i]["y"].append(1)
                    data["descriptive"][i]["q"].append(q["id"]); data["descriptive"][i]["pos"].append(t)
            for t in range(max(0, onset - a.pre_window), onset):
                for i in layers:
                    data["predictive"][i]["X"].append(per[t][i]); data["predictive"][i]["y"].append(1)
                    data["predictive"][i]["q"].append(q["id"]); data["predictive"][i]["pos"].append(t)
        else:
            stats["clean"] += 1
            keep = {i: {t: per[t][i] for t in range(0, len(out), 8)} for i in layers}
            clean_pool.append((q["id"], len(out), keep))

    for n, q in enumerate(gsm):
        one(q)
        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(gsm)} looped={stats['looped']} clean={stats['clean']} "
                  f"tokens={stats['tokens']} {time.time()-t0:.0f}s", flush=True)
    for h in handles:
        h.remove()
    texts.close()
    print("collection done", json.dumps(stats), flush=True)

    # Position-matched negatives: for each positive at position t, take a clean generation that reached t and use
    # its nearest kept position.  Without this the probe can win on depth alone.
    rng = random.Random(7)
    for kind in ("descriptive", "predictive"):
        want = list(data[kind][layers[0]]["pos"])
        for t in want:
            cands = [c for c in clean_pool if c[1] > t]
            if not cands:
                continue
            qid, n, keep = cands[rng.randrange(len(cands))]
            near = min(keep[layers[0]].keys(), key=lambda p: abs(p - t))
            for i in layers:
                data[kind][i]["X"].append(keep[i][near]); data[kind][i]["y"].append(0)
                data[kind][i]["q"].append(qid); data[kind][i]["pos"].append(near)
        pos_n = sum(data[kind][layers[0]]["y"])
        neg_n = len(data[kind][layers[0]]["y"]) - pos_n
        import statistics as st
        ps = [p for p, y in zip(data[kind][layers[0]]["pos"], data[kind][layers[0]]["y"]) if y == 1]
        ns = [p for p, y in zip(data[kind][layers[0]]["pos"], data[kind][layers[0]]["y"]) if y == 0]
        print(f"  {kind}: {pos_n} positives (median position {st.median(ps) if ps else 0:.0f}), "
              f"{neg_n} matched negatives (median {st.median(ns) if ns else 0:.0f})", flush=True)

    import numpy as np

    def auc(scores, labels):
        o = np.argsort(scores); r = np.empty(len(scores)); r[o] = np.arange(1, len(scores) + 1)
        n1, n0 = labels.sum(), (1 - labels).sum()
        return float((r[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")

    res = {"stats": stats, "layers": layers, "detector": f"{N_GRAM}-word window x{TIMES}", "probes": {}}
    for kind in ("descriptive", "predictive"):
        res["probes"][kind] = {}
        for i in layers:
            if not data[kind][i]["X"] or len(set(data[kind][i]["y"])) < 2:
                res["probes"][kind][i] = {"error": "not enough data"}
                continue
            X = np.stack([x.numpy() for x in data[kind][i]["X"]])
            y = np.array(data[kind][i]["y"]); qs = np.array(data[kind][i]["q"])
            uq = sorted(set(qs.tolist())); random.Random(1).shuffle(uq)
            test = set(uq[: max(1, len(uq) // 4)])
            te = np.array([q in test for q in qs]); tr = ~te
            if y[tr].std() == 0 or y[te].std() == 0:
                res["probes"][kind][i] = {"error": "a split has one class only"}
                continue
            d = X[tr & (y == 1)].mean(0) - X[tr & (y == 0)].mean(0)
            d /= np.linalg.norm(d) + 1e-9
            P = np.array(data[kind][i]["pos"], dtype=float)
            row = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
                   "pos_train": int(y[tr].sum()), "pos_test": int(y[te].sum()),
                   "median_pos_positive": float(np.median(P[y == 1])),
                   "median_pos_negative": float(np.median(P[y == 0])),
                   "position_only_auc": round(auc(P[te], y[te]), 4),
                   "diff_of_means_auc": round(auc(X[te] @ d, y[te]), 4)}
            try:
                mu = X[tr].mean(0); Xc = X[tr] - mu
                t = np.where(y[tr] == 1, 1.0, -1.0)
                lam = 1e-2 * np.trace(Xc.T @ Xc) / Xc.shape[1]
                w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ t)
                row["ridge_auc"] = round(auc((X[te] - mu) @ w, y[te]), 4)
            except Exception as exc:
                row["ridge_auc"] = None; row["ridge_error"] = str(exc)[:80]
            res["probes"][kind][i] = row
            print(f"  {kind:11s} layer {i:>2}: diff-of-means {row['diff_of_means_auc']:.3f}"
                  + (f"  ridge {row['ridge_auc']:.3f}" if row.get("ridge_auc") else "")
                  + f"  position-only {row['position_only_auc']:.3f}"
                  + f"  (test {row['n_test']}, pos {row['pos_test']})", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    if a.save_npz:
        import numpy as np
        np.savez_compressed(a.save_npz, **{
            f"{kind}_{i}_X": np.stack([x.numpy() for x in data[kind][i]["X"]]).astype(np.float16)
            for kind in ("descriptive", "predictive") for i in layers if data[kind][i]["X"]},
            **{f"{kind}_{i}_y": np.array(data[kind][i]["y"]) for kind in ("descriptive", "predictive")
               for i in layers if data[kind][i]["X"]},
            **{f"{kind}_{i}_pos": np.array(data[kind][i]["pos"]) for kind in ("descriptive", "predictive")
               for i in layers if data[kind][i]["X"]})
        print("saved activations to", a.save_npz, flush=True)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
