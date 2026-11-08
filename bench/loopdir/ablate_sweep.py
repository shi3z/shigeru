"""Find a strength, layer range and matrix set where removing the pre-loop direction keeps the accuracy.

Full removal from layers 9-24 works on the loops and costs accuracy: on 200 unseen questions the loops went
5.5 % -> 1.0 % (all 11 looping answers stopped looping, 2 new ones appeared) while accuracy went 95.0 % -> 93.5 %
(6 right answers became wrong, 3 wrong became right), and the generations got shorter, 430 -> 398 tokens.  Something
the direction carries is also needed to keep going.

Three knobs, screened in stages so the cost stays bounded:
  strength  W <- (I - alpha d d^T) W for alpha in 0.25, 0.5, 0.75 (1.0 is the measurement above)
  layers    the AUC peaks at 16-24, so try narrower and wider ranges at the best alpha
  matrices  mlp.down_proj alone against the attention output projection alone

One model load, one shared baseline, the touched matrices kept pristine on the host and restored between arms, and
results written after every arm so a crash costs one arm.
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
    ap.add_argument("--skip", type=int, default=450)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--out", default="/home/shi3z/loopdir/ablate_sweep.json")
    a = ap.parse_args()

    npz = np.load(a.acts)

    def direction(layer):
        X = npz[f"predictive_{layer}_X"].astype(np.float32)
        y = npz[f"predictive_{layer}_y"]
        d = X[y == 1].mean(0) - X[y == 0].mean(0)
        return d / (np.linalg.norm(d) + 1e-9)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0},
                                                 trust_remote_code=True).eval()
    blocks = model.model.layers
    qs = json.load(open(a.gsm8k))[a.skip: a.skip + a.n]
    print(f"{len(blocks)} layers, {len(qs)} questions from {a.skip}", flush=True)

    pristine = {}
    for i in range(len(blocks)):
        for path in WRITERS:
            mod = get_mod(blocks[i], path)
            if mod is not None:
                pristine[(i, path)] = mod.weight.data.detach().to("cpu", copy=True)
    print(f"kept {len(pristine)} matrices pristine on the host", flush=True)

    @torch.no_grad()
    def restore():
        for (i, path), W in pristine.items():
            get_mod(blocks[i], path).weight.data.copy_(W.to(get_mod(blocks[i], path).weight.device))

    @torch.no_grad()
    def apply_ablation(d_np, layers, paths, alpha):
        n = 0
        for i in layers:
            for path in paths:
                mod = get_mod(blocks[i], path)
                if mod is None:
                    continue
                W = mod.weight.data
                d = torch.as_tensor(d_np, dtype=torch.float32, device=W.device)
                Wf = W.float()
                mod.weight.data = (Wf - alpha * torch.outer(d, d @ Wf)).to(W.dtype)
                n += 1
        return n

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
            rows.append({"id": q["id"], "hit": hit, "loop": fires(text), "tokens": len(out)})
            if (n + 1) % 50 == 0:
                print(f"    [{tag}] {n+1}/{len(qs)} acc {sum(r['hit'] for r in rows)/len(rows):.3f} "
                      f"loops {sum(r['loop'] for r in rows)} {time.time()-t0:.0f}s", flush=True)
        nn = len(rows)
        return {"n": nn, "accuracy": round(sum(r["hit"] for r in rows) / nn, 4),
                "loops": sum(r["loop"] for r in rows),
                "loop_pct": round(100 * sum(r["loop"] for r in rows) / nn, 2),
                "errors": nn - sum(r["hit"] for r in rows),
                "errors_that_loop": sum(1 for r in rows if r["loop"] and not r["hit"]),
                "mean_tokens": round(sum(r["tokens"] for r in rows) / nn, 1)}, rows

    def flips(before, after):
        f = {"loop_to_clean": 0, "clean_to_loop": 0, "right_to_wrong": 0, "wrong_to_right": 0}
        for b, c in zip(before, after):
            f["loop_to_clean"] += b["loop"] and not c["loop"]
            f["clean_to_loop"] += c["loop"] and not b["loop"]
            f["right_to_wrong"] += b["hit"] and not c["hit"]
            f["wrong_to_right"] += c["hit"] and not b["hit"]
        return f

    res = {"questions": [q["id"] for q in qs], "arms": []}

    def save():
        json.dump(res, open(a.out, "w"), indent=1)

    print("\n### baseline", flush=True)
    base, base_rows = evaluate("baseline")
    res["baseline"] = base
    print("=== baseline", json.dumps(base), flush=True)
    save()

    def arm(label, layer_for_direction, layers, paths, alpha):
        restore()
        d = direction(layer_for_direction)
        n = apply_ablation(d, layers, paths, alpha)
        print(f"\n### {label}: {n} matrices, alpha {alpha}, direction from layer {layer_for_direction}", flush=True)
        s, rows = evaluate(label)
        f = flips(base_rows, rows)
        rec = {"label": label, "alpha": alpha, "direction_layer": layer_for_direction,
               "layers": [layers[0], layers[-1]], "paths": list(paths), "matrices": n,
               "after": s, "flips": f,
               "delta_accuracy": round(s["accuracy"] - base["accuracy"], 4),
               "delta_loop_pct": round(s["loop_pct"] - base["loop_pct"], 2)}
        res["arms"].append(rec)
        print(f"=== {label} {json.dumps(s)}\n    flips {json.dumps(f)} "
              f"d_acc {rec['delta_accuracy']:+.4f} d_loop {rec['delta_loop_pct']:+.2f}", flush=True)
        save()
        return rec

    # stage 1: strength
    stage1 = [arm(f"alpha{al}", 20, list(range(9, 25)), WRITERS, al) for al in (0.25, 0.5, 0.75)]
    best_alpha = max(stage1, key=lambda r: r["after"]["accuracy"] - 0.01 * r["after"]["loop_pct"])["alpha"]
    print(f"\n>>> best alpha by accuracy minus 0.01 x loop_pct: {best_alpha}", flush=True)

    # stage 2: layer range at that strength
    for lo, hi in ((16, 24), (16, 31), (9, 31)):
        arm(f"L{lo}-{hi}_a{best_alpha}", 20, list(range(lo, hi + 1)), WRITERS, best_alpha)
    best_layers = max([r for r in res["arms"] if r["alpha"] == best_alpha],
                      key=lambda r: r["after"]["accuracy"] - 0.01 * r["after"]["loop_pct"])["layers"]
    print(f"\n>>> best layer range: {best_layers}", flush=True)

    # stage 3: which matrices
    lo, hi = best_layers
    arm(f"mlp_only_a{best_alpha}", 20, list(range(lo, hi + 1)), ("mlp.down_proj",), best_alpha)
    arm(f"attn_only_a{best_alpha}", 20, list(range(lo, hi + 1)),
        ("linear_attn.out_proj", "self_attn.o_proj"), best_alpha)
    # and the direction taken from a different layer, since 16 scored nearly as well as 20
    arm(f"dir16_a{best_alpha}", 16, list(range(lo, hi + 1)), WRITERS, best_alpha)

    print("\n### summary (baseline acc "
          f"{base['accuracy']:.4f}, loops {base['loop_pct']:.2f} %)", flush=True)
    for r in res["arms"]:
        print(f"  {r['label']:22s} acc {r['after']['accuracy']:.4f} ({r['delta_accuracy']:+.4f})  "
              f"loops {r['after']['loop_pct']:5.2f} % ({r['delta_loop_pct']:+.2f})  "
              f"loop_fixed {r['flips']['loop_to_clean']:>2} new_loop {r['flips']['clean_to_loop']:>2}  "
              f"broke {r['flips']['right_to_wrong']:>2} fixed {r['flips']['wrong_to_right']:>2}  "
              f"tok {r['after']['mean_tokens']:.0f}", flush=True)
    save()
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
