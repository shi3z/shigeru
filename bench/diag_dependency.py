"""Phase E3-A — dependency between block positions (docs/dependency_aware_refinement.md §3).

Teacher-forced blocks of K from the model's own greedy answers (--refs).  For each block: the first all-mask forward gives
the model's own placement t (its errors included); the leave-one-out probe gives the base distribution P_i for every
position.  Then for each position j, t_j is replaced by its probe alternative (the probe argmax if it differs from t_j,
else the second-best token) and the probe is run again -> P_i^(j).  Dependency of i on j:
  top1_change   argmax P_i != argmax P_i^(j)
  JSD           Jensen-Shannon divergence (base 2, in [0, 1]) between P_i and P_i^(j), computed in float32 over the
                full vocabulary with 1e-12 smoothing
  KL            KL(P_i || P_i^(j)) with the same smoothing (reported, unbounded)
  conf_delta    |max P_i - max P_i^(j)|
aggregated by distance |i - j| (and by direction: j left / right of i).  Cost: 1 + K probe forwards of K*K rows per
block (recorded).  Also, for a Jacobi refineN run on the same blocks, every adopted change is tagged with the maximum
JSD-dependency to the other positions changed in the same round, and its correctness is scored, to test whether
simultaneous changes between strongly dependent positions are the ones that go wrong.
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv-train/bin/python bench/diag_dependency.py --model runs/phaseC_sft3_4b/final \
      --refs results/pure/dev_refs_sft3.json --n 50 --K 8 --out results/pure/diag_dependency_K8_sft3.json"""
import argparse, collections, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts, gpu_info, save_json
from engine.cdlm.fast_model import FastQwen3
from engine.cdlm.k4_decoder import k4_denoise_block, probe_key_mask, probe_rows

EPS = 1e-12


def jsd_kl(p, q):
    """p, q [N, V] float32 probability rows -> (jsd [N] in bits, kl [N])."""
    p = p.clamp_min(EPS); q = q.clamp_min(EPS)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(-1)
    kl_qm = (q * (q / m).log()).sum(-1)
    jsd = 0.5 * (kl_pm + kl_qm) / torch.log(torch.tensor(2.0))
    kl = (p * (p / q).log()).sum(-1)
    return jsd, kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="bench/prompts_gsm8k_dev.json")
    ap.add_argument("--refs", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--max-blocks", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    K = a.K
    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    mask_id = getattr(hf.config, "cdlm_mask_token_id", None) or tok.convert_tokens_to_ids("<|mask|>")
    prompts = load_prompts(a.prompts)[: a.n]
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        p["ids"] = ids[0].tolist() if torch.is_tensor(ids) else list(ids["input_ids"][0])
    refs = json.load(open(a.refs))["refs"]
    model = FastQwen3.from_hf(hf, device="cuda", dtype=torch.bfloat16, max_len=4096)
    del hf
    torch.cuda.empty_cache()
    dev = torch.device("cuda")
    by_dist = collections.defaultdict(lambda: dict(n=0, top1=0, jsd=0.0, kl=0.0, dconf=0.0, jsd_sq=0.0))
    by_dir = {"left": dict(n=0, jsd=0.0, top1=0), "right": dict(n=0, jsd=0.0, top1=0)}
    jsd_hist = collections.Counter()
    # Jacobi changes tagged with the max dependency to co-changed positions
    change_bins = collections.defaultdict(lambda: dict(n=0, to_right=0, to_wrong=0, ww=0))
    solo = dict(n=0, to_right=0, to_wrong=0, ww=0)
    n_fwd = 0
    t0 = time.time()
    for p in prompts:
        resp = refs[p["id"]]["response_ids"]
        full = torch.tensor(p["ids"] + resp, dtype=torch.long, device=dev)
        P = len(p["ids"])
        model.kv.zero_()
        model.prefill(full, 0)
        starts = list(range(P, P + len(resp) - K + 1, K))[: a.max_blocks]
        for s in starts:
            gold = full[s:s + K].tolist()

            def fwd_std(ids, pos, rows, _s=s):
                return model.forward(torch.tensor(ids, device=dev), torch.tensor(pos, device=dev), _s, write=False, rows=rows)

            def fwd_probe(ids, pos, Kp, rows, _s=s):
                return model.forward(torch.tensor(ids, device=dev), torch.tensor(pos, device=dev), _s, write=False, rows=rows,
                                     key_mask=probe_key_mask(_s, Kp, device=dev))

            def probe_probs(tokens):
                pi, pp, pr = probe_rows(tokens, s, mask_id)
                lg = fwd_probe(pi, pp, K, pr).float()
                lg[:, mask_id] = float("-inf")
                return torch.softmax(lg, -1)

            # model's own placement = first all-mask forward
            lg = fwd_std([mask_id] * K, [s + i for i in range(K)], list(range(K))).float(); lg[:, mask_id] = float("-inf")
            t = lg.argmax(-1).tolist()
            n_fwd += 1
            base = probe_probs(t); n_fwd += 1
            base_top = base.argmax(-1); base_conf = base.max(-1).values
            top2 = base.topk(2, dim=-1).indices
            dep = torch.zeros(K, K)                                  # dep[i, j] = JSD of i when j changes
            for j in range(K):
                alt = int(top2[j, 0]) if int(top2[j, 0]) != t[j] else int(top2[j, 1])
                t2 = list(t); t2[j] = alt
                q = probe_probs(t2); n_fwd += 1
                jsd, kl = jsd_kl(base, q)
                top1 = (q.argmax(-1) != base_top)
                dconf = (q.max(-1).values - base_conf).abs()
                for i in range(K):
                    if i == j:
                        continue
                    d = abs(i - j)
                    b = by_dist[d]
                    b["n"] += 1; b["top1"] += int(top1[i]); b["jsd"] += float(jsd[i]); b["kl"] += float(kl[i]); b["dconf"] += float(dconf[i]); b["jsd_sq"] += float(jsd[i]) ** 2
                    dd = by_dir["left" if j < i else "right"]
                    dd["n"] += 1; dd["jsd"] += float(jsd[i]); dd["top1"] += int(top1[i])
                    jsd_hist[f"{min(int(float(jsd[i]) * 10), 9) / 10:.1f}"] += 1
                    dep[i, j] = float(jsd[i])
            # Jacobi refineN with change tagging
            toks, info = k4_denoise_block(fwd_std, fwd_probe, s, K, mask_id, strategy="refineN", adopt="always", max_rounds=3, trace=True)
            n_fwd += info["forwards"]
            by_round = collections.defaultdict(list)
            for (rnd, i, old, new) in info["changes"]:
                by_round[rnd].append((i, old, new))
            for rnd, chs in by_round.items():
                for (i, old, new) in chs:
                    others = [j for (j, _, _) in chs if j != i]
                    cat = "to_right" if new == gold[i] and old != gold[i] else "to_wrong" if old == gold[i] and new != gold[i] else "ww"
                    if not others:
                        solo["n"] += 1; solo[cat] += 1
                    else:
                        mx = max(float(dep[i, j]) for j in others)
                        key = "jsd<0.1" if mx < 0.1 else "jsd<0.3" if mx < 0.3 else "jsd>=0.3"
                        cb = change_bins[key]; cb["n"] += 1; cb[cat] += 1
        print(f"{p['id']} {len(resp)} tok forwards {n_fwd} {time.time() - t0:.0f}s", flush=True)
    out_dist = {}
    for d, b in sorted(by_dist.items()):
        n = b["n"]
        mean = b["jsd"] / n
        out_dist[str(d)] = dict(n=n, top1_change_rate=round(b["top1"] / n, 4), jsd_mean=round(mean, 4),
                                jsd_std=round((b["jsd_sq"] / n - mean ** 2) ** 0.5, 4), kl_mean=round(b["kl"] / n, 4),
                                conf_delta_mean=round(b["dconf"] / n, 4))
    out_dir = {k: dict(n=v["n"], jsd_mean=round(v["jsd"] / max(1, v["n"]), 4), top1_change_rate=round(v["top1"] / max(1, v["n"]), 4)) for k, v in by_dir.items()}
    out_changes = {k: dict(n=v["n"], to_right=round(v["to_right"] / v["n"], 4), to_wrong=round(v["to_wrong"] / v["n"], 4), wrong_to_wrong=round(v["ww"] / v["n"], 4))
                   for k, v in {**change_bins, "solo": solo}.items() if v["n"]}
    summary = dict(by_distance=out_dist, by_direction=out_dir, jsd_histogram=dict(sorted(jsd_hist.items())), jacobi_changes_by_max_codependency=out_changes,
                   forwards=n_fwd, probe_forwards_per_block_for_dependency=K + 1)
    print("=== dependency", json.dumps({"by_distance": {d: (v["jsd_mean"], v["top1_change_rate"]) for d, v in out_dist.items()}, "changes": out_changes}), flush=True)
    save_json(a.out, {"model": a.model, "refs": a.refs, "n": len(prompts), "K": K, "gpu": gpu_info(), "seconds": round(time.time() - t0, 1), "summary": summary})


if __name__ == "__main__":
    main()
