"""Diagnostic 1 of docs/pure_diffusion_research.md: teacher-forced mask recovery of a converted checkpoint.

For each dev prompt the model's own greedy AR answer (HF generate, cached in --refs) is the reference response.  The
response is cut into blocks of K after the prompt; for every block, mask rate gamma and mask layout (random subset /
contiguous suffix) ONE forward of the block in observed-first order on top of the clean prefix (prompt + earlier
blocks) predicts the masked tokens — exactly the context a denoising step of engine/cdlm/block_decoder.py provides.
No AR logits are used.  Reported per (layout, gamma): token accuracy of the argmax, block exact-match rate, accuracy
by distance to the nearest observed token on the LEFT (1 = the token right after an observed token), accuracy by
distance to the nearest observed token on either side, and a confidence calibration table (max prob bins).
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. .venv-train/bin/python bench/diag_mask_recovery.py --model runs/phaseC_sft3_4b/final \
      --prompts bench/prompts_gsm8k_dev.json --n 100 --refs results/pure/dev_refs_sft3.json --out results/pure/diag_recovery_sft3.json"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts, gpu_info, save_json, score
from engine.cdlm.fast_model import FastQwen3

GAMMAS = (0.1, 0.25, 0.5, 0.75, 1.0)
LAYOUTS = ("random", "suffix")


def make_refs(model, tok, prompts, max_new, stop_ids, path):
    refs = {}
    if os.path.exists(path):
        refs = json.load(open(path))["refs"]
    todo = [p for p in prompts if p["id"] not in refs]
    for p in todo:
        ids = torch.tensor([p["ids"]]).cuda()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False, eos_token_id=stop_ids)
        new = out[0, ids.shape[1]:].tolist()
        text = tok.decode(new, skip_special_tokens=True)
        refs[p["id"]] = {"response_ids": new, "text": text, "correct": score(p, text)}
        print(f"ref {p['id']} {len(new)} tok correct={refs[p['id']]['correct']}", flush=True)
    if todo:
        save_json(path, {"model": model.name_or_path, "max_new_tokens": max_new, "refs": refs})
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="bench/prompts_gsm8k_dev.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refs", required=True, help="cache of the model's own greedy AR responses on these prompts")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    mask_id = getattr(hf.config, "cdlm_mask_token_id", None) or tok.convert_tokens_to_ids("<|mask|>")
    stop_ids = [tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|im_end|>")] if "<|im_end|>" in tok.get_vocab() else [])
    prompts = load_prompts(a.prompts)[: a.n]
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True,
                                      enable_thinking=False, return_tensors="pt")
        p["ids"] = ids[0].tolist() if torch.is_tensor(ids) else list(ids["input_ids"][0])
    refs = make_refs(hf, tok, prompts, a.max_new_tokens, stop_ids, a.refs)
    model = FastQwen3.from_hf(hf, device="cuda", dtype=torch.bfloat16, max_len=4096)
    del hf
    torch.cuda.empty_cache()
    g = torch.Generator().manual_seed(a.seed)

    # accumulators: (layout, gamma) -> counts
    acc = {}
    def bump(key, sub, k, correct):
        d = acc.setdefault(key, {}).setdefault(sub, {}).setdefault(k, [0, 0])
        d[0] += int(correct); d[1] += 1
    conf_bins = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
    t0 = time.time()
    n_fwd = 0
    for p in prompts:
        resp = refs[p["id"]]["response_ids"]
        full = torch.tensor(p["ids"] + resp, dtype=torch.long, device="cuda")
        P = len(p["ids"])
        model.kv.zero_()
        model.prefill(full, 0)                                   # clean K/V of prompt + whole response
        for s in range(P, P + len(resp), a.block):
            e = min(P + len(resp), s + a.block)
            K = e - s
            true = full[s:e]
            for layout in LAYOUTS:
                for gamma in GAMMAS:
                    n_mask = max(1, int(round(K * gamma)))
                    if layout == "suffix":
                        masked = torch.zeros(K, dtype=torch.bool); masked[K - n_mask:] = True
                    else:
                        masked = torch.zeros(K, dtype=torch.bool); masked[torch.randperm(K, generator=g)[:n_mask]] = True
                    obs = (~masked).nonzero().flatten(); msk = masked.nonzero().flatten()
                    order = torch.cat([obs, msk]).cuda()
                    ids = true[order].clone(); ids[len(obs):] = mask_id
                    pos = (torch.arange(K, device="cuda") + s)[order]
                    logits = model.forward(ids, pos, cache_len=s, write=False, rows=list(range(len(obs), K)))
                    n_fwd += 1
                    probs = torch.softmax(logits, -1)
                    conf, pred = probs.max(-1)
                    ok = (pred == true[msk.cuda()]).tolist()
                    conf = conf.tolist()
                    key = f"{layout}:{gamma}"
                    for i, m in enumerate(msk.tolist()):
                        bump(key, "token", "all", ok[i])
                        # distance to the nearest observed token on the left (within the block; the prefix counts as
                        # observed at position -1)
                        left = [o for o in obs.tolist() if o < m]
                        dl = m - (left[-1] if left else -1)
                        right = [o for o in obs.tolist() if o > m]
                        dr = (right[0] - m) if right else None
                        bump(key, "dist_left", str(min(dl, 8)), ok[i])
                        bump(key, "dist_any", str(min(dl if dr is None else min(dl, dr), 8)), ok[i])
                        bump(key, "has_right_obs", "yes" if right else "no", ok[i])
                        for b in range(len(conf_bins) - 1):
                            if conf_bins[b] <= conf[i] < conf_bins[b + 1]:
                                bump(key, "conf_bin", f"{conf_bins[b]:.2f}", ok[i]); break
                    bump(key, "block", "exact", all(ok))
        print(f"{p['id']} resp {len(resp)} tok, forwards so far {n_fwd}, {time.time() - t0:.0f}s", flush=True)
    # summarise
    summary = {}
    for key, subs in acc.items():
        summary[key] = {sub: {k: {"acc": round(v[0] / v[1], 4), "n": v[1]} for k, v in sorted(d.items())} for sub, d in subs.items()}
    ref_correct = [refs[p["id"]]["correct"] for p in prompts]
    headline = {k: {"token_acc": v["token"]["all"]["acc"], "block_exact": v["block"]["exact"]["acc"]} for k, v in summary.items()}
    print("=== recovery", json.dumps(headline), flush=True)
    save_json(a.out, {"model": a.model, "prompts": a.prompts, "n": len(prompts), "block": a.block, "seed": a.seed,
                      "gpu": gpu_info(), "ref_ar_accuracy": sum(bool(c) for c in ref_correct) / len(ref_correct),
                      "forwards": n_fwd, "seconds": round(time.time() - t0, 1), "summary": summary, "headline": headline})


if __name__ == "__main__":
    main()
