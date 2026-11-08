"""Confirm the winning ablation on questions it was not selected on.

The screen chose its winner on questions 450-649, where the difference between arms is a handful of answers
(n=200 puts the standard error near 1.5 points).  Re-measuring on the same questions would be measuring the
selection.  This runs the baseline and the chosen arm on 600 fresh questions in one process, and reports the paired
changes, where the signal is the flips rather than the rates.
"""
import argparse, json, re, time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

N_GRAM, TIMES = 12, 4
WRITERS = ("linear_attn.out_proj", "self_attn.o_proj", "mlp.down_proj")


def fires(text):
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


def final_number(text):
    for pat in (r"####\s*(-?\d[\d,]*(?:\.\d+)?)", r"(-?\d[\d,]*(?:\.\d+)?)"):
        m = re.findall(pat, text)
        if m:
            v = m[-1].replace(",", "").rstrip(".")
            try:
                float(v); return v
            except ValueError:
                continue
    return None


def get_mod(block, path):
    mod = block
    for part in path.split("."):
        mod = getattr(mod, part, None)
        if mod is None:
            return None
    return mod if getattr(mod, "weight", None) is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/shi3z/models/hf/ds9b-orig-hf")
    ap.add_argument("--acts", default="/home/shi3z/loopdir/loop_acts.npz")
    ap.add_argument("--gsm8k", default="/home/shi3z/shigeru/open-mercury/bench/prompts_gsm8k_full.json")
    ap.add_argument("--skip", type=int, default=650, help="past the 450 the direction used and the 200 of the screen")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--layers", default="16-31")
    ap.add_argument("--dir-layer", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--out", default="/home/shi3z/loopdir/confirm_ablation.json")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.layers.split("-"))

    npz = np.load(a.acts)
    X = npz[f"predictive_{a.dir_layer}_X"].astype(np.float32)
    y = npz[f"predictive_{a.dir_layer}_y"]
    d = X[y == 1].mean(0) - X[y == 0].mean(0)
    d /= np.linalg.norm(d) + 1e-9

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0},
                                                 trust_remote_code=True).eval()
    blocks = model.model.layers
    qs = json.load(open(a.gsm8k))[a.skip: a.skip + a.n]
    print(f"confirming alpha {a.alpha}, layers {lo}-{hi}, direction from layer {a.dir_layer}, "
          f"on questions {a.skip}..{a.skip + len(qs) - 1}", flush=True)

    @torch.no_grad()
    def evaluate(tag):
        rows, t0 = [], time.time()
        for n, q in enumerate(qs):
            pr = tok.apply_chat_template([{"role": "user", "content": q["prompt"]}], tokenize=False,
                                         add_generation_prompt=True)
            ids = tok(pr, return_tensors="pt", add_special_tokens=False).to(model.device)
            past, cur, out = None, ids["input_ids"], []
            for _ in range(a.max_new):
                o = model(input_ids=cur, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nxt = int(o.logits[0, -1].argmax())
                out.append(nxt)
                cur = torch.tensor([[nxt]], device=model.device)
                if nxt == tok.eos_token_id:
                    break
            text = tok.decode(out, skip_special_tokens=True)
            got = final_number(text)
            try:
                hit = got is not None and abs(float(got) - float(q["answer"])) < 1e-6
            except ValueError:
                hit = False
            rows.append({"id": q["id"], "hit": hit, "loop": fires(text), "tokens": len(out),
                         "capped": len(out) >= a.max_new})
            if (n + 1) % 50 == 0:
                print(f"    [{tag}] {n+1}/{len(qs)} acc {sum(r['hit'] for r in rows)/len(rows):.4f} "
                      f"loops {sum(r['loop'] for r in rows)} {time.time()-t0:.0f}s", flush=True)
                json.dump(rows, open(a.out + f".{tag}.partial", "w"))
        nn = len(rows)
        s = {"n": nn, "accuracy": round(sum(r["hit"] for r in rows) / nn, 4),
             "loops": sum(r["loop"] for r in rows), "loop_pct": round(100 * sum(r["loop"] for r in rows) / nn, 2),
             "errors": nn - sum(r["hit"] for r in rows),
             "errors_that_loop": sum(1 for r in rows if r["loop"] and not r["hit"]),
             "capped": sum(r["capped"] for r in rows),
             "mean_tokens": round(sum(r["tokens"] for r in rows) / nn, 1)}
        print(f"=== {tag} {json.dumps(s)}", flush=True)
        return s, rows

    before, rows_b = evaluate("baseline")
    with torch.no_grad():
        touched = 0
        for i in range(lo, hi + 1):
            for path in WRITERS:
                mod = get_mod(blocks[i], path)
                if mod is None:
                    continue
                W = mod.weight.data
                dv = torch.as_tensor(d, dtype=torch.float32, device=W.device)
                Wf = W.float()
                mod.weight.data = (Wf - a.alpha * torch.outer(dv, dv @ Wf)).to(W.dtype)
                touched += 1
    print(f"ablated {touched} matrices, alpha {a.alpha}, layers {lo}-{hi}", flush=True)
    after, rows_a = evaluate("ablated")

    f = {"loop_to_clean": 0, "clean_to_loop": 0, "right_to_wrong": 0, "wrong_to_right": 0}
    for b, c in zip(rows_b, rows_a):
        f["loop_to_clean"] += b["loop"] and not c["loop"]
        f["clean_to_loop"] += c["loop"] and not b["loop"]
        f["right_to_wrong"] += b["hit"] and not c["hit"]
        f["wrong_to_right"] += c["hit"] and not b["hit"]
    # McNemar on the accuracy flips: with b+c discordant pairs, is the split one-sided beyond chance?
    b_, c_ = f["right_to_wrong"], f["wrong_to_right"]
    import math
    n_disc = b_ + c_
    p = None
    if n_disc:
        tail = sum(math.comb(n_disc, k) for k in range(0, min(b_, c_) + 1)) * 2 / (2 ** n_disc)
        p = round(min(1.0, tail), 4)
    print(f"=== paired changes {json.dumps(f)}", flush=True)
    print(f"    accuracy {before['accuracy']:.4f} -> {after['accuracy']:.4f} "
          f"({after['accuracy'] - before['accuracy']:+.4f}), McNemar two-sided p = {p}", flush=True)
    print(f"    loops {before['loop_pct']:.2f} % -> {after['loop_pct']:.2f} % "
          f"({after['loop_pct'] - before['loop_pct']:+.2f})", flush=True)
    json.dump({"alpha": a.alpha, "layers": [lo, hi], "dir_layer": a.dir_layer, "questions": [q["id"] for q in qs],
               "before": before, "after": after, "flips": f, "mcnemar_p": p,
               "rows_before": rows_b, "rows_after": rows_a}, open(a.out, "w"), indent=1)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
