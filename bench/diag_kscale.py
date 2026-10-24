"""K-scaling diagnostics (docs/k_scaling_failure_analysis.md, experiments B, C and 10), teacher-forced on the model's own
greedy answers (--refs from bench/diag_mask_recovery.py), blocks of K at stride K, clean prefix, write=False forwards.

B  first-forward quality per K: accuracy by position, block exact match, distribution of the number of correct tokens,
   mean error position, longest run of consecutive errors, confidence calibration (bins of max prob), duplicate rate.
10 oracle conditional accuracy: the leave-one-out probe with the K-1 TRUE neighbours, by position (the ceiling).
C  oracle seed injection: n gold tokens fixed (patterns left / even / random / wrongfirst), the rest denoised with
   refineN (adopt always and one) -> accuracy of the un-fixed positions, block exact, rounds, outcome counts
   (converged / cycle / budget).  Fixed tokens are never changed and never scored.
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. .venv-train/bin/python bench/diag_kscale.py --model runs/phaseC_sft3_4b/final \
      --refs results/pure/dev_refs_sft3.json --n 50 --K 3 4 5 6 7 8 --inject-K 6 8 --out results/pure/diag_kscale_sft3.json"""
import argparse, collections, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts, gpu_info, save_json
from engine.cdlm.fast_model import FastQwen3
from engine.cdlm.k4_decoder import k4_denoise_block, probe_key_mask, probe_rows

BINS = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
PATTERNS = ("left", "even", "random", "wrongfirst")


def pick_fixed(pattern, n, K, first_ok, rng):
    if n == 0:
        return []
    if pattern == "left":
        return list(range(n))
    if pattern == "even":
        return sorted({round(i * (K - 1) / max(1, n - 1)) if n > 1 else 0 for i in range(n)})[:n] if n > 1 else [0]
    if pattern == "random":
        return sorted(rng.sample(range(K), n))
    if pattern == "wrongfirst":
        wrong = [i for i in range(K) if not first_ok[i]]
        right = [i for i in range(K) if first_ok[i]]
        return sorted((wrong + right)[:n])
    raise ValueError(pattern)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="bench/prompts_gsm8k_dev.json")
    ap.add_argument("--refs", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--K", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8])
    ap.add_argument("--inject-K", type=int, nargs="*", default=[6, 8])
    ap.add_argument("--inject-n", type=int, nargs="*", default=[0, 1, 2, 3, 4, 6, 8])
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    mask_id = getattr(hf.config, "cdlm_mask_token_id", None) or tok.convert_tokens_to_ids("<|mask|>")
    prompts = load_prompts(a.prompts)[: a.n]
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True,
                                      enable_thinking=False, return_tensors="pt")
        p["ids"] = ids[0].tolist() if torch.is_tensor(ids) else list(ids["input_ids"][0])
    refs = json.load(open(a.refs))["refs"]
    model = FastQwen3.from_hf(hf, device="cuda", dtype=torch.bfloat16, max_len=4096)
    del hf
    torch.cuda.empty_cache()
    B = {K: dict(blocks=0, pos_ok=[0] * K, exact=0, n_correct=collections.Counter(), err_pos_sum=0, err_n=0, max_run=collections.Counter(),
                 bins=[[0, 0] for _ in BINS[:-1]], dup=0.0, probe_ok=[0] * K) for K in a.K}
    C = {}
    t0 = time.time()
    for p in prompts:
        resp = refs[p["id"]]["response_ids"]
        full = torch.tensor(p["ids"] + resp, dtype=torch.long, device="cuda")
        P = len(p["ids"])
        model.kv.zero_()
        model.prefill(full, 0)
        for K in a.K:
            for s in range(P, P + len(resp) - K + 1, K):
                gold = full[s:s + K].tolist()

                def fwd_std(ids, pos, rows, _s=s):
                    return model.forward(torch.tensor(ids, device="cuda"), torch.tensor(pos, device="cuda"), _s, write=False, rows=rows)

                def fwd_probe(ids, pos, Kp, rows, _s=s):
                    return model.forward(torch.tensor(ids, device="cuda"), torch.tensor(pos, device="cuda"), _s, write=False, rows=rows,
                                         key_mask=probe_key_mask(_s, Kp, device="cuda"))

                # B: first forward
                lg = fwd_std([mask_id] * K, [s + i for i in range(K)], list(range(K)))
                lgf = lg.float(); lgf[:, mask_id] = float("-inf")
                probs = torch.softmax(lgf, -1)
                conf, pred = probs.max(-1)
                pred, conf = pred.tolist(), conf.tolist()
                ok = [pred[i] == gold[i] for i in range(K)]
                b = B[K]
                b["blocks"] += 1
                for i in range(K):
                    b["pos_ok"][i] += int(ok[i])
                    for j in range(len(BINS) - 1):
                        if BINS[j] <= conf[i] < BINS[j + 1]:
                            b["bins"][j][0] += int(ok[i]); b["bins"][j][1] += 1; break
                b["exact"] += int(all(ok))
                b["n_correct"][sum(ok)] += 1
                errs = [i for i in range(K) if not ok[i]]
                if errs:
                    b["err_pos_sum"] += sum(errs) / len(errs); b["err_n"] += 1
                run = best = 0
                for o in ok:
                    run = 0 if o else run + 1; best = max(best, run)
                b["max_run"][best] += 1
                b["dup"] += 1.0 - len(set(pred)) / K
                # 10: oracle conditional (probe with true neighbours)
                pi, pp, pr = probe_rows(gold, s, mask_id)
                lg2 = fwd_probe(pi, pp, K, pr).float(); lg2[:, mask_id] = float("-inf")
                pr2 = lg2.argmax(-1).tolist()
                for i in range(K):
                    b["probe_ok"][i] += int(pr2[i] == gold[i])
                # C: oracle seed injection
                if K in a.inject_K:
                    for pattern in PATTERNS:
                        for n in [x for x in a.inject_n if x < K]:
                            fx = pick_fixed(pattern, n, K, ok, rng)
                            fixed = {i: gold[i] for i in fx}
                            for adopt in ("always", "one"):
                                key = (K, pattern, n, adopt)
                                c = C.setdefault(key, dict(blocks=0, free=0, free_ok=0, exact=0, rounds=0, outcomes=collections.Counter(),
                                                           first_free_ok=0, corrections=0))
                                toks, info = k4_denoise_block(fwd_std, fwd_probe, s, K, mask_id, strategy="refineN", adopt=adopt,
                                                              max_rounds=a.max_rounds, fixed=fixed)
                                free = [i for i in range(K) if i not in fixed]
                                c["blocks"] += 1; c["free"] += len(free)
                                c["free_ok"] += sum(int(toks[i] == gold[i]) for i in free)
                                c["first_free_ok"] += sum(int(info["first"][i] == gold[i]) for i in free)
                                c["exact"] += int(toks == gold); c["rounds"] += info["rounds"]
                                c["outcomes"][info["outcome"]] += 1; c["corrections"] += info["corrections"]
        print(f"{p['id']} {len(resp)} tok {time.time() - t0:.0f}s", flush=True)
    outB = {}
    for K, b in B.items():
        nb = b["blocks"]
        outB[str(K)] = dict(blocks=nb, first_acc_by_position=[round(x / nb, 4) for x in b["pos_ok"]],
                            first_token_acc=round(sum(b["pos_ok"]) / (nb * K), 4), first_block_exact=round(b["exact"] / nb, 4),
                            n_correct_dist={str(k): v for k, v in sorted(b["n_correct"].items())},
                            mean_error_position=round(b["err_pos_sum"] / b["err_n"], 3) if b["err_n"] else None,
                            max_error_run_dist={str(k): v for k, v in sorted(b["max_run"].items())},
                            calibration={f"{BINS[j]:.2f}": (round(c[0] / c[1], 4) if c[1] else None, c[1]) for j, c in enumerate(b["bins"])},
                            duplicate_rate=round(b["dup"] / nb, 4),
                            oracle_conditional_acc_by_position=[round(x / nb, 4) for x in b["probe_ok"]],
                            oracle_conditional_acc=round(sum(b["probe_ok"]) / (nb * K), 4))
    outC = {}
    for (K, pattern, n, adopt), c in sorted(C.items()):
        outC.setdefault(str(K), {}).setdefault(pattern, {}).setdefault(str(n), {})[adopt] = dict(
            blocks=c["blocks"], free_acc=round(c["free_ok"] / c["free"], 4) if c["free"] else None,
            first_free_acc=round(c["first_free_ok"] / c["free"], 4) if c["free"] else None,
            block_exact=round(c["exact"] / c["blocks"], 4), rounds_per_block=round(c["rounds"] / c["blocks"], 3),
            outcomes=dict(c["outcomes"]), corrections_per_block=round(c["corrections"] / c["blocks"], 3))
    print("=== kscale_B", json.dumps({K: {k: v[k] for k in ("first_acc_by_position", "first_block_exact", "duplicate_rate", "oracle_conditional_acc_by_position")} for K, v in outB.items()}), flush=True)
    print("=== kscale_C", json.dumps({K: {pat: {n: {ad: (vv["free_acc"], vv["block_exact"], vv["outcomes"]) for ad, vv in d.items()} for n, d in pats.items()} for pat, pats in per.items()} for K, per in outC.items()}), flush=True)
    save_json(a.out, {"model": a.model, "refs": a.refs, "n": len(prompts), "K": a.K, "inject_K": a.inject_K, "inject_n": a.inject_n,
                      "max_rounds": a.max_rounds, "seed": a.seed, "gpu": gpu_info(), "seconds": round(time.time() - t0, 1), "B": outB, "C": outC})


if __name__ == "__main__":
    main()
