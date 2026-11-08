"""Remove the pre-loop direction from the weights and see whether the loops go with it.

The direction exists: with negatives matched to each positive's position, the difference of means at layer 20
separates the 24 positions before a repetition from the same positions in non-repeating generations with AUC 0.895,
while position alone gives 0.456.  It is a single direction, and it beats a trained hyperplane (0.842), which is
exactly the shape a Heretic-style ablation can use.

This projects that direction out of everything the chosen layers write to the residual stream
(linear_attn.out_proj, self_attn.o_proj, mlp.down_proj: W <- (I - d d^T) W) and evaluates GSM8K before and after in
one process, so the two arms differ only by the edit.  Questions come from after the 450 the direction was fitted
on.

Verdicts:
  loops fall and accuracy holds   the edit works, and loops are half this model's error budget
  loops unchanged                 the direction is detectable but not causal
  accuracy falls                  the edit takes something else with it
"""
import argparse, json, re, time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

N_GRAM, TIMES = 12, 4


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


def direction(npz, layer, kind="predictive"):
    X = npz[f"{kind}_{layer}_X"].astype(np.float32)
    y = npz[f"{kind}_{layer}_y"]
    d = X[y == 1].mean(0) - X[y == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-9)


def residual_writers(block):
    out = []
    for path in ("linear_attn.out_proj", "self_attn.o_proj", "mlp.down_proj"):
        mod = block
        for part in path.split("."):
            mod = getattr(mod, part, None)
            if mod is None:
                break
        if mod is not None and getattr(mod, "weight", None) is not None:
            out.append((path, mod))
    return out


@torch.no_grad()
def ablate(model, d_np, layers):
    """W <- (I - d d^T) W for every matrix those layers write the residual stream with."""
    blocks = model.model.layers
    touched = 0
    for i in layers:
        for _path, mod in residual_writers(blocks[i]):
            W = mod.weight.data
            d = torch.as_tensor(d_np, dtype=torch.float32, device=W.device)
            Wf = W.float()
            mod.weight.data = (Wf - torch.outer(d, d @ Wf)).to(W.dtype)
            touched += 1
    return touched


@torch.no_grad()
def evaluate(model, tok, questions, max_new, tag):
    rows, t0 = [], time.time()
    for n, q in enumerate(questions):
        pr = tok.apply_chat_template([{"role": "user", "content": q["prompt"]}], tokenize=False,
                                     add_generation_prompt=True)
        ids = tok(pr, return_tensors="pt", add_special_tokens=False).to(model.device)
        past, cur, out = None, ids["input_ids"], []
        for _ in range(max_new):
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
                     "capped": len(out) >= max_new, "got": got, "want": q["answer"], "tail": text[-160:]})
        if (n + 1) % 20 == 0:
            ok = sum(r["hit"] for r in rows); lp = sum(r["loop"] for r in rows)
            print(f"  [{tag}] {n+1}/{len(questions)} acc {ok/len(rows):.3f} loops {lp} "
                  f"mean_tok {sum(r['tokens'] for r in rows)/len(rows):.0f} {time.time()-t0:.0f}s", flush=True)
    n = len(rows)
    s = {"n": n, "accuracy": round(sum(r["hit"] for r in rows) / n, 4),
         "loops": sum(r["loop"] for r in rows), "loop_pct": round(100 * sum(r["loop"] for r in rows) / n, 2),
         "errors": n - sum(r["hit"] for r in rows),
         "errors_that_loop": sum(1 for r in rows if r["loop"] and not r["hit"]),
         "capped": sum(r["capped"] for r in rows),
         "mean_tokens": round(sum(r["tokens"] for r in rows) / n, 1)}
    print(f"=== {tag} {json.dumps(s)}", flush=True)
    return s, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/shi3z/models/hf/ds9b-orig-hf")
    ap.add_argument("--acts", default="/home/shi3z/loopdir/loop_acts.npz")
    ap.add_argument("--gsm8k", default="/home/shi3z/shigeru/open-mercury/bench/prompts_gsm8k_full.json")
    ap.add_argument("--skip", type=int, default=450, help="questions the direction was fitted on")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dir-layer", type=int, default=20)
    ap.add_argument("--ablate-layers", default="9-24")
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--out", default="/home/shi3z/loopdir/ablate_loop.json")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.ablate_layers.split("-"))
    layers = list(range(lo, hi + 1))

    npz = np.load(a.acts)
    d = direction(npz, a.dir_layer)
    print(f"direction from the predictive set of layer {a.dir_layer}: dim {d.shape[0]}, norm {np.linalg.norm(d):.4f}",
          flush=True)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0},
                                                 trust_remote_code=True).eval()
    qs = json.load(open(a.gsm8k))[a.skip: a.skip + a.n]
    print(f"evaluating on questions {a.skip}..{a.skip + len(qs) - 1} (unseen by the direction)", flush=True)

    before, rows_b = evaluate(model, tok, qs, a.max_new, "unablated")
    touched = ablate(model, d, layers)
    print(f"ablated {touched} matrices across layers {lo}-{hi}", flush=True)
    after, rows_a = evaluate(model, tok, qs, a.max_new, f"ablated L{lo}-{hi}")

    flip = {"loop_to_clean": 0, "clean_to_loop": 0, "right_to_wrong": 0, "wrong_to_right": 0}
    for b, c in zip(rows_b, rows_a):
        flip["loop_to_clean"] += b["loop"] and not c["loop"]
        flip["clean_to_loop"] += c["loop"] and not b["loop"]
        flip["right_to_wrong"] += b["hit"] and not c["hit"]
        flip["wrong_to_right"] += c["hit"] and not b["hit"]
    print("=== paired changes", json.dumps(flip), flush=True)
    json.dump({"direction_layer": a.dir_layer, "ablated_layers": layers, "before": before, "after": after,
               "flips": flip, "rows_before": rows_b, "rows_after": rows_a},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
