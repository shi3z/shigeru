"""Phase B8 §2 — pairwise dependency of the two tokens placed per step at S = 8 (docs/pairwise_joint_diffusion.md §1–2).

Teacher-forced: gold prefix + K reference tokens; the fixed-budget sampler is run left-to-right (or confidence order)
for S steps WITHOUT re-ranking, and at every step that places exactly two tokens (i, j) the step's marginals are
re-ranked post hoc under several rules, each scored against the reference pair:
  independent   argmax each (what the sampler does)
  joint_fwd     log p(a) + log p(b | a)          k extra forwards (chain rule, research ceiling)
  joint_both    symmetric version               2k extra forwards
  oracle        reference pair if it is among the k x k candidates ("reranking ceiling"; else independent)
Dependency diagnostics per step: top-1 change rate of position j when a1 vs a2 is placed at i, JSD between p(b|a1)
and p(b|a2), the fraction of steps whose k x k candidate set contains the reference pair, and the fraction of steps
where independent is wrong but some candidate pair is right (what re-ranking could fix at most).  Pair exact match,
token accuracy and W->C / C->W (independent -> re-ranked) are reported per rule, for k = 2 and k = 4.
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv-train/bin/python bench/diag_pairwise.py --model runs/zeta_sft_final_e5/final \
      --adapter runs/bdiff_e5_lr1e4/adapter --no-time --refs results/pure/dev_refs_e5.json --n 50 --out results/pure/diag_pairwise_lr1e4.json"""
import argparse, collections, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts, gpu_info, save_json
from engine.bdiff.model import BDiffModel
from engine.bdiff.sampler2 import sample_block_fixed
from engine.bdiff.pairwise import rerank_pair


def jsd(p, q):
    p = p.clamp_min(1e-12); q = q.clamp_min(1e-12); m = 0.5 * (p + q)
    return float(0.5 * ((p * (p / m).log()).sum() + (q * (q / m).log()).sum()) / torch.log(torch.tensor(2.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--prompts", default="bench/prompts_gsm8k_dev.json")
    ap.add_argument("--refs", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--unmask", default="left")
    ap.add_argument("--ks", default="2,4")
    ap.add_argument("--max-blocks", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    mask_id = getattr(lm.config, "cdlm_mask_token_id", None) or tok.convert_tokens_to_ids("<|mask|>")
    if a.adapter:
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, a.adapter).eval().merge_and_unload()
    model = BDiffModel(lm, mask_id, time_cond=not a.no_time).to(dev).eval()
    K, S = a.K, a.steps
    ks = [int(x) for x in a.ks.split(",")]
    prompts = load_prompts(a.prompts)[a.skip: a.skip + a.n]
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        p["ids"] = ids[0].tolist() if torch.is_tensor(ids) else list(ids["input_ids"][0])
    refs = json.load(open(a.refs))["refs"]
    t0 = time.time()
    C = collections.Counter()
    jsds = []; jsd_by_dist = collections.defaultdict(list); top1_by_dist = collections.defaultdict(list)
    per_k = {k: collections.Counter() for k in ks}
    n_fwd = 0
    for p in prompts:
        resp = refs[p["id"]]["response_ids"]
        full = p["ids"] + resp
        P0 = len(p["ids"])
        for s0 in list(range(P0, P0 + len(resp) - K + 1, K))[: a.max_blocks]:
            gold = full[s0:s0 + K]
            cache = model.prefix_cache(torch.tensor([full[:s0]], device=dev))
            log = []
            toks, info = sample_block_fixed(model, cache, s0, K, S, mask_id, unmask=a.unmask, remask="none", early_stop=False, feature_log=log, gold=gold, trace=True)
            n_fwd += info["forwards"]
            # replay: at each step the sampler placed `sel`; recover the two positions from consecutive trace states
            prev = [mask_id] * K
            for row, tr in zip(log, info["trace"]):
                placed = [q for q in range(K) if prev[q] == mask_id and tr["tokens"][q] != mask_id]
                prev = tr["tokens"]
                if len(placed) != 2:
                    continue
                i, j = placed
                x = torch.tensor(row["tokens"], device=dev)
                # recompute the step's probs (one forward; identical to the sampler's — deterministic) to get candidates
                g = max(int((x == mask_id).sum()) / K, 1e-3)
                lg = model.denoise(cache, s0, x[None], g).float(); n_fwd += 1
                lg[:, mask_id] = float("-inf")
                probs = torch.softmax(lg, -1)
                gi, gj = gold[i], gold[j]
                ind_i, ind_j = int(probs[i].argmax()), int(probs[j].argmax())
                C["steps"] += 1
                C["indep_pair_ok"] += int(ind_i == gi and ind_j == gj)
                C["indep_tok_ok"] += int(ind_i == gi) + int(ind_j == gj)
                C["ref_i_in_top2"] += int(gi in probs[i].topk(2).indices.tolist()); C["ref_j_in_top2"] += int(gj in probs[j].topk(2).indices.tolist())
                for k in ks:
                    ci = probs[i].topk(k).indices.tolist(); cj = probs[j].topk(k).indices.tolist()
                    in_set = gi in ci and gj in cj
                    pk = per_k[k]
                    pk["steps"] += 1; pk["ref_pair_in_candidates"] += int(in_set)
                    pk["fixable"] += int(in_set and not (ind_i == gi and ind_j == gj))
                    ti, tj, extra, inf = rerank_pair(model, cache, s0, x, probs, i, j, mask_id, k=k, rule="joint_both")
                    n_fwd += extra
                    for rule, mat in (("joint_fwd", inf["fwd"]), ("joint_both", inf["score"]), ("indep_check", inf["indep"])):
                        b = int(mat.argmax()); ai, bj = divmod(b, k)
                        ri, rj = ci[ai], cj[bj]
                        ok = ri == gi and rj == gj
                        pk[f"{rule}_pair_ok"] += int(ok)
                        pk[f"{rule}_tok_ok"] += int(ri == gi) + int(rj == gj)
                        ind_ok = ind_i == gi and ind_j == gj
                        if ok and not ind_ok: pk[f"{rule}_W2C"] += 1
                        if ind_ok and not ok: pk[f"{rule}_C2W"] += 1
                        pk[f"{rule}_changed"] += int((ri, rj) != (ind_i, ind_j))
                    # oracle: reference pair if in candidates else independent
                    pk["oracle_pair_ok"] += int(in_set or (ind_i == gi and ind_j == gj))
                    if k == 2:
                        # dependency of j on the choice at i
                        pj_a1 = inf["cond_j"][0].exp(); pj_a2 = inf["cond_j"][1].exp()   # only over the 2 candidates -> use full dists instead
                        # full-distribution dependency: re-run the two conditionals here (already done inside rerank; cheap enough to redo)
                        x1 = x.clone(); x1[i] = ci[0]; x2 = x.clone(); x2[i] = ci[1]
                        l1 = model.denoise(cache, s0, x1[None], g).float(); l2 = model.denoise(cache, s0, x2[None], g).float(); n_fwd += 2
                        l1[:, mask_id] = float("-inf"); l2[:, mask_id] = float("-inf")
                        q1 = torch.softmax(l1[j], -1); q2 = torch.softmax(l2[j], -1)
                        d = abs(i - j); jv = jsd(q1, q2); jsds.append(jv); jsd_by_dist[d].append(jv)
                        top1_by_dist[d].append(int(q1.argmax() != q2.argmax()))
                        C["j_top1_changes_with_a"] += int(q1.argmax() != q2.argmax())
                        # does conditioning on the REFERENCE token at i fix j?
                        xr = x.clone(); xr[i] = gi
                        lr = model.denoise(cache, s0, xr[None], g).float(); n_fwd += 1; lr[:, mask_id] = float("-inf")
                        C["j_ok_given_ref_i"] += int(int(lr[j].argmax()) == gj)
                        C["j_ok_indep"] += int(ind_j == gj)
        print(f"{p['id']} steps {C['steps']} fwd {n_fwd} {time.time() - t0:.0f}s", flush=True)
    n = max(1, C["steps"])
    out = dict(steps=n, independent=dict(pair_exact=round(C["indep_pair_ok"] / n, 4), token_acc=round(C["indep_tok_ok"] / (2 * n), 4),
                                         ref_in_top2_i=round(C["ref_i_in_top2"] / n, 4), ref_in_top2_j=round(C["ref_j_in_top2"] / n, 4)),
               dependency=dict(j_top1_change_rate_a1_vs_a2=round(C["j_top1_changes_with_a"] / n, 4), jsd_mean=round(sum(jsds) / max(1, len(jsds)), 4),
                               jsd_by_distance={str(d): round(sum(v) / len(v), 4) for d, v in sorted(jsd_by_dist.items())},
                               top1_change_by_distance={str(d): round(sum(v) / len(v), 4) for d, v in sorted(top1_by_dist.items())},
                               j_acc_independent=round(C["j_ok_indep"] / n, 4), j_acc_given_reference_i=round(C["j_ok_given_ref_i"] / n, 4)),
               rerank={})
    for k in ks:
        pk = per_k[k]; m = max(1, pk["steps"])
        out["rerank"][str(k)] = dict(ref_pair_in_candidates=round(pk["ref_pair_in_candidates"] / m, 4), fixable_steps=round(pk["fixable"] / m, 4),
                                    oracle_pair_exact=round(pk["oracle_pair_ok"] / m, 4),
                                    **{rule: dict(pair_exact=round(pk[f"{rule}_pair_ok"] / m, 4), token_acc=round(pk[f"{rule}_tok_ok"] / (2 * m), 4),
                                                  W2C=pk[f"{rule}_W2C"], C2W=pk[f"{rule}_C2W"], changed=round(pk[f"{rule}_changed"] / m, 4),
                                                  extra_forwards_per_step=(k if rule == "joint_fwd" else 2 * k if rule == "joint_both" else 0))
                                       for rule in ("indep_check", "joint_fwd", "joint_both")})
    print("=== pairwise", json.dumps(out), flush=True)
    save_json(a.out, dict(model=a.model, adapter=a.adapter, K=K, steps=S, unmask=a.unmask, n=len(prompts), gpu=gpu_info(), forwards=n_fwd, seconds=round(time.time() - t0, 1), summary=out))


if __name__ == "__main__":
    main()
