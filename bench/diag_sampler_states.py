"""Phase B11-A/B — sampler state distributions and teacher-correction feasibility (docs/sampler_aligned_global_denoising.md §2–3).

Teacher-forced on the reference blocks of dev items (REF = the base model's own greedy answer); each sampler is run for
S steps from all masks and EVERY intermediate state x_s (K tokens: [MASK] or placed) is recorded with
  mask ratio, number of mask islands (runs), longest visible prefix from position 0, whether some masked position has a
  visible token to its right (impossible under left-to-right monotonic unmasking), number / fraction of visible tokens
  that differ from REF ("wrong visible"), Hamming distance of the state's argmax block to REF and to the run's FINAL.
The TRAINING distribution is sampled for comparison: random masks on the REF block at the same mask ratios (visible
tokens always correct — the B-series mask-only recipe).
Teacher correction from each visited state (Gate 2), with the frozen B-series model as teacher:
  fill        one forward, argmax at the masked positions, visible kept                      (1 forward)
  seqfill     masked positions filled one per forward, left-to-right (chain rule over masks)   (n_mask forwards)
  oracle      visible tokens != REF are re-masked (uses REF — diagnostic only), then seqfill  (n_mask' forwards)
reported as block token accuracy vs REF after the update, wrong->correct and correct->wrong counts over the block, and
block exact match — per sampler and per step.  Forwards are counted.
Samplers: left, confidence (engine/bdiff/sampler2.py), remask (non-monotonic confidence re-masking), sync (full-block
Jacobi) (engine/bdiff/global_sampler.py).
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv-train/bin/python bench/diag_sampler_states.py --model runs/zeta_sft_final_e5/final \
      --adapter runs/bdiff_e5_lr1e4/adapter --refs results/pure/dev_refs_e5.json --n 50 --samplers left,remask --out results/pure/b11_states_left_remask.json"""
import argparse, collections, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts, gpu_info, save_json
from engine.bdiff.model import BDiffModel
from engine.bdiff.sampler2 import sample_block_fixed
from engine.bdiff.global_sampler import global_sample_block


def geometry(x, mask_id, ref):
    K = len(x)
    m = [t == mask_id for t in x]
    islands = sum(1 for q in range(K) if m[q] and (q == 0 or not m[q - 1]))
    prefix = 0
    while prefix < K and not m[prefix]:
        prefix += 1
    right_visible = any(m[q] and any(not m[r] for r in range(q + 1, K)) for q in range(K))
    wrong_vis = sum(1 for q in range(K) if not m[q] and x[q] != ref[q])
    n_vis = K - sum(m)
    return dict(mask_ratio=sum(m) / K, islands=islands, prefix=prefix, right_visible=int(right_visible), wrong_visible=wrong_vis, n_visible=n_vis)


@torch.no_grad()
def predict(model, cache, P, x, mask_id):
    xt = torch.tensor(x, device=model.device)
    g = max(float((xt == mask_id).sum()) / len(x), 1e-3)
    lg = model.denoise(cache, P, xt[None], g).float(); lg[:, mask_id] = float("-inf")
    return lg.argmax(-1).tolist()


@torch.no_grad()
def seqfill(model, cache, P, x, mask_id):
    x = list(x); n = 0
    for q in range(len(x)):
        if x[q] == mask_id:
            pred = predict(model, cache, P, x, mask_id); n += 1
            x[q] = pred[q]
    return x, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--adapter2", default=None)
    ap.add_argument("--prompts", default="bench/prompts_gsm8k_dev.json")
    ap.add_argument("--refs", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--samplers", default="left,confidence,remask,sync")
    ap.add_argument("--max-blocks", type=int, default=6)
    ap.add_argument("--no-correction", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    mask_id = getattr(lm.config, "cdlm_mask_token_id", None) or tok.convert_tokens_to_ids("<|mask|>")
    from peft import PeftModel
    for ad in (a.adapter, a.adapter2):
        if ad:
            lm = PeftModel.from_pretrained(lm, ad).eval().merge_and_unload()
    model = BDiffModel(lm, mask_id, time_cond=False).to(dev).eval()
    K, S = a.K, a.steps
    prompts = load_prompts(a.prompts)[: a.n]
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        p["ids"] = [int(t) for t in (ids[0] if torch.is_tensor(ids) else ids["input_ids"][0])]
    refs = json.load(open(a.refs))["refs"]
    rng = torch.Generator().manual_seed(0)
    t0 = time.time(); n_fwd = 0
    res = {}
    for sampler in a.samplers.split(",") + ["train_random"]:
        G = collections.defaultdict(lambda: collections.defaultdict(float)); N = collections.Counter()
        C = collections.defaultdict(lambda: collections.Counter())
        final_acc = [0, 0]
        for p in prompts:
            resp = refs[p["id"]]["response_ids"]; full = p["ids"] + resp; P0 = len(p["ids"])
            for s0 in list(range(P0, P0 + len(resp) - K + 1, K))[: a.max_blocks]:
                ref = full[s0:s0 + K]
                cache = model.prefix_cache(torch.tensor([full[:s0]], device=dev)); n_fwd += 1
                states = []                                       # (step, x) states BEFORE each forward
                if sampler == "train_random":
                    for s in range(1, S + 1):
                        ratio = 1.0 - (s - 1) / S                 # the same mask ratios the samplers visit
                        n_mask = max(1, int(round(K * ratio)))
                        perm = torch.randperm(K, generator=rng)[:n_mask].tolist()
                        x = [mask_id if q in perm else ref[q] for q in range(K)]
                        states.append((s, x))
                    final = None
                elif sampler in ("left", "confidence"):
                    toks, info = sample_block_fixed(model, cache, s0, K, S, mask_id, unmask=sampler, remask="none", early_stop=False, trace=True)
                    n_fwd += info["forwards"]
                    prev = [mask_id] * K
                    for st in info["trace"]:
                        states.append((st["step"], list(prev))); prev = st["tokens"]
                    final = toks
                else:
                    toks, info = global_sample_block(model, cache, s0, K, S, mask_id, mode=sampler, trace=True)
                    n_fwd += info["forwards"]
                    prev = [mask_id] * K
                    for st in info["trace"]:
                        states.append((st["step"], list(prev)))
                        prev = [mask_id if mk else g_ for g_, mk in zip(st["guess"], st["masked"])]
                    final = toks
                if final is not None:
                    final_acc[0] += sum(final[q] == ref[q] for q in range(K)); final_acc[1] += K
                for s, x in states:
                    geo = geometry(x, mask_id, ref)
                    for k, v in geo.items():
                        G[s][k] += v
                    G[s]["wrong_visible_frac"] += geo["wrong_visible"] / max(1, geo["n_visible"]) if geo["n_visible"] else 0.0
                    G[s]["has_wrong_visible"] += int(geo["wrong_visible"] > 0)
                    N[s] += 1
                    if a.no_correction:
                        continue
                    # --- teacher correction from this state (REF is only read for the oracle variant and for scoring)
                    before_ok = [x[q] == ref[q] for q in range(K)]
                    pred = predict(model, cache, s0, x, mask_id); n_fwd += 1
                    fill = [pred[q] if x[q] == mask_id else x[q] for q in range(K)]
                    sq, nf = seqfill(model, cache, s0, x, mask_id); n_fwd += nf
                    xo = [mask_id if (x[q] != mask_id and x[q] != ref[q]) else x[q] for q in range(K)]
                    oc, no = seqfill(model, cache, s0, xo, mask_id); n_fwd += no
                    for name, y in (("fill", fill), ("seqfill", sq), ("oracle_remask_seqfill", oc)):
                        c = C[(s, name)]
                        after_ok = [y[q] == ref[q] for q in range(K)]
                        c["n"] += 1; c["tok_ok"] += sum(after_ok); c["exact"] += int(all(after_ok))
                        c["w2c"] += sum(1 for q in range(K) if not before_ok[q] and after_ok[q]); c["c2w"] += sum(1 for q in range(K) if before_ok[q] and not after_ok[q])
                        c["masked_ok"] += sum(1 for q in range(K) if x[q] == mask_id and after_ok[q]); c["masked"] += sum(1 for q in range(K) if x[q] == mask_id)
            print(f"[{sampler}] {p['id']} forwards {n_fwd} {time.time() - t0:.0f}s", flush=True)
        per_step = {}
        for s in sorted(N):
            n = N[s]; g = G[s]
            per_step[str(s)] = dict(n=n, mask_ratio=round(g["mask_ratio"] / n, 3), islands=round(g["islands"] / n, 2), prefix=round(g["prefix"] / n, 2),
                                    right_visible_rate=round(g["right_visible"] / n, 3), wrong_visible_per_state=round(g["wrong_visible"] / n, 3),
                                    wrong_visible_frac=round(g["wrong_visible_frac"] / n, 4), states_with_wrong_visible=round(g["has_wrong_visible"] / n, 3))
            for name in ("fill", "seqfill", "oracle_remask_seqfill"):
                c = C.get((s, name))
                if c and c["n"]:
                    per_step[str(s)][name] = dict(token_acc=round(c["tok_ok"] / (c["n"] * K), 4), block_exact=round(c["exact"] / c["n"], 4), w2c=round(c["w2c"] / c["n"], 3), c2w=round(c["c2w"] / c["n"], 3),
                                                  masked_acc=round(c["masked_ok"] / max(1, c["masked"]), 4))
        res[sampler] = dict(per_step=per_step, final_token_acc=round(final_acc[0] / final_acc[1], 4) if final_acc[1] else None)
        print("=== states", sampler, json.dumps({s: {k: v for k, v in r.items() if not isinstance(v, dict)} for s, r in per_step.items()}), flush=True)
    save_json(a.out, dict(model=a.model, adapter=a.adapter, adapter2=a.adapter2, K=K, steps=S, n=len(prompts), gpu=gpu_info(), forwards=n_fwd, seconds=round(time.time() - t0, 1), results=res))


if __name__ == "__main__":
    main()
