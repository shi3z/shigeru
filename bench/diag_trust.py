"""Phase B7 §2 — trust diagnostic for B-series decoding (docs/trust_aware_diffusion.md §2).

Teacher-forced trajectories: for every reference block (gold prefix + K reference tokens), the fixed-budget sampler
(engine/bdiff/sampler2.py) is run from all masks for S steps WITHOUT re-masking, and at every step every position's
features (engine/bdiff/trust.py) are logged together with its correctness:
  visible positions   label = placed token == reference token     ("should this token be re-masked?")
  masked positions    label = current argmax == reference token   ("is it safe to unmask this now?")
Each feature is scored as a predictor of WRONGNESS (AUROC, AUPRC; positive = wrong) and, where it is a probability,
as a calibrated P(correct) (Brier, ECE).  --oracle adds the leave-one-out feature (re-mask the position alone and run
one extra forward; research diagnostic only, K extra forwards per step) on the first --oracle-blocks blocks.
A logistic-regression trust score is fitted on the visible-position rows of the first half of the prompts and
evaluated on the second half; the weights are saved (results/pure/trust_lr_<tag>.json) for the "trust" re-masking
rule of the sampler.  The reference tokens are the sft3 model's greedy answers (results/pure/dev_refs_sft3.json), as
in every other B-series teacher-forced measurement; "correct" therefore means "equal to the reference".
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv-train/bin/python bench/diag_trust.py --model runs/zeta_sft_final_e5/final \
      --adapter runs/bdiff_e5_lr1e4/adapter --no-time --refs results/pure/dev_refs_sft3.json --n 50 --steps 8 \
      --unmask confidence,left --oracle --oracle-blocks 40 --out results/pure/diag_trust_lr1e4.json"""
import argparse, collections, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts, gpu_info, save_json
from engine.bdiff.model import BDiffModel
from engine.bdiff.sampler2 import sample_block_fixed
from engine.bdiff.trust import FEATURES, TrustLR, auroc, auprc, brier, ece

# feature -> wrongness score direction (+1: larger = more likely wrong)
WRONG_SIGN = dict(conf=-1, margin=-1, entropy=+1, p_tok=-1, agree=-1, stable=-1, age=+1, conf_at=-1, dconf=+1, changes=+1, nb_changed=+1)
PROB_FEATS = ("conf", "p_tok", "conf_at", "margin")


def metrics(feat, label_correct, oracle=None):
    """feat [N, F], label_correct [N] bool -> per-feature wrongness AUROC/AUPRC (+ Brier/ECE as P(correct))."""
    wrong = (~label_correct).float()
    out = {}
    for j, name in enumerate(FEATURES):
        col = feat[:, j]
        sc = col * WRONG_SIGN[name]
        m = dict(auroc=round(auroc(sc, wrong), 4), auprc=round(auprc(sc, wrong), 4))
        if name in PROB_FEATS:
            p = col.clamp(0, 1)
            m.update(brier=round(brier(p, label_correct.float()), 4), ece=round(ece(p, label_correct.float()), 4))
        out[name] = m
    if oracle is not None:
        o_agree, o_conf = oracle
        out["oracle_loo_agree"] = dict(auroc=round(auroc(1 - o_agree, wrong), 4), auprc=round(auprc(1 - o_agree, wrong), 4))
        out["oracle_loo_conf"] = dict(auroc=round(auroc(-o_conf, wrong), 4), auprc=round(auprc(-o_conf, wrong), 4),
                                      brier=round(brier(o_conf, label_correct.float()), 4), ece=round(ece(o_conf, label_correct.float()), 4))
    out["base_rate_wrong"] = round(float(wrong.mean()), 4)
    out["n"] = int(len(wrong))
    return out


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
    ap.add_argument("--unmask", default="confidence,left")
    ap.add_argument("--max-blocks", type=int, default=6)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--oracle-blocks", type=int, default=40)
    ap.add_argument("--tag", default=None)
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
    prompts = load_prompts(a.prompts)[a.skip: a.skip + a.n]
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        p["ids"] = ids[0].tolist() if torch.is_tensor(ids) else list(ids["input_ids"][0])
    refs = json.load(open(a.refs))["refs"]
    half = set(p["id"] for p in prompts[: len(prompts) // 2])
    t0 = time.time()
    result = {}
    for rule in a.unmask.split(","):
        rows_vis, rows_mask = [], []          # (prompt_id, step, feat, correct)
        oracle_vis = []                        # (agree, conf) aligned with the oracle subset of rows_vis
        oracle_idx = []
        n_blocks = 0; exact = 0; final_wrong = 0; final_tokens = 0
        detectable = collections.Counter()    # of the final wrong tokens: was there ever a disagreement / low p_tok / change?
        per_step_wrong = collections.defaultdict(lambda: [0, 0])
        forwards = 0
        for p in prompts:
            resp = refs[p["id"]]["response_ids"]
            full = p["ids"] + resp
            P0 = len(p["ids"])
            starts = list(range(P0, P0 + len(resp) - K + 1, K))[: a.max_blocks]
            for s0 in starts:
                gold = full[s0:s0 + K]
                cache = model.prefix_cache(torch.tensor([full[:s0]], device=dev))
                log = []
                toks, info = sample_block_fixed(model, cache, s0, K, S, mask_id, unmask=rule, remask="none", early_stop=False, feature_log=log, gold=gold)
                forwards += info["forwards"]
                n_blocks += 1
                exact += int(toks == gold)
                wrong_final = [i for i in range(K) if toks[i] != gold[i]]
                final_wrong += len(wrong_final); final_tokens += K
                ever_dis = set(); ever_low = set(); ever_change = set()
                do_oracle = a.oracle and n_blocks <= a.oracle_blocks
                for row in log:
                    s = row["step"]; feat = row["feat"]; im = row["is_mask"]
                    vis = (~im).nonzero().flatten().tolist()
                    msk = im.nonzero().flatten().tolist()
                    for i in vis:
                        rows_vis.append((p["id"], s, feat[i], bool(row["tok_correct"][i])))
                        if feat[i][FEATURES.index("agree")] < 0.5: ever_dis.add(i)
                        if feat[i][FEATURES.index("p_tok")] < 0.5: ever_low.add(i)
                        if feat[i][FEATURES.index("changes")] > 0: ever_change.add(i)
                        per_step_wrong[s][0] += int(not row["tok_correct"][i]); per_step_wrong[s][1] += 1
                    for i in msk:
                        rows_mask.append((p["id"], s, feat[i], bool(row["pred_correct"][i])))
                    if do_oracle and vis:
                        x = torch.tensor(row["tokens"], device=dev)
                        for i in vis:
                            x2 = x.clone(); x2[i] = mask_id
                            lg = model.denoise(cache, s0, x2[None], max(int((x2 == mask_id).sum()) / K, 1e-3)).float(); lg[:, mask_id] = float("-inf")
                            pr = torch.softmax(lg[i], -1)
                            oracle_vis.append((float(pr.argmax() == x[i]), float(pr[x[i]])))
                            oracle_idx.append(len(rows_vis) - len(vis) + vis.index(i))
                for i in wrong_final:
                    detectable["total"] += 1
                    detectable["ever_disagree"] += int(i in ever_dis)
                    detectable["ever_p_tok<0.5"] += int(i in ever_low)
                    detectable["ever_changed"] += int(i in ever_change)
            print(f"[{rule}] {p['id']} blocks {n_blocks} vis rows {len(rows_vis)} {time.time() - t0:.0f}s", flush=True)
        Fv = torch.stack([r[2] for r in rows_vis]); yv = torch.tensor([r[3] for r in rows_vis])
        Fm = torch.stack([r[2] for r in rows_mask]); ym = torch.tensor([r[3] for r in rows_mask])
        res = dict(blocks=n_blocks, block_exact=round(exact / n_blocks, 4), final_token_error=round(final_wrong / final_tokens, 4), forwards=forwards,
                   wrong_rate_by_step={str(s): round(v[0] / max(1, v[1]), 4) for s, v in sorted(per_step_wrong.items())},
                   final_wrong_detectable={k: (v if k == "total" else round(v / max(1, detectable["total"]), 4)) for k, v in detectable.items()},
                   visible=metrics(Fv, yv), masked=metrics(Fm, ym))
        if oracle_vis:
            oi = torch.tensor(oracle_idx)
            oa = torch.tensor([o[0] for o in oracle_vis]); oc = torch.tensor([o[1] for o in oracle_vis])
            res["visible_oracle_subset"] = metrics(Fv[oi], yv[oi], oracle=(oa, oc))
        # by step (visible, p_tok and agree only)
        by_step = {}
        for s in sorted(set(r[1] for r in rows_vis)):
            idx = torch.tensor([k for k, r in enumerate(rows_vis) if r[1] == s])
            if len(idx) > 20 and yv[idx].float().mean() < 1:
                by_step[str(s)] = dict(n=len(idx), wrong=round(1 - float(yv[idx].float().mean()), 4),
                                       auroc_p_tok=round(auroc(-Fv[idx, FEATURES.index("p_tok")], (~yv[idx]).float()), 4),
                                       auroc_agree=round(auroc(-Fv[idx, FEATURES.index("agree")], (~yv[idx]).float()), 4))
        res["visible_by_step"] = by_step
        # logistic trust score: fit on first half of prompts, evaluate on second half
        tr_idx = torch.tensor([k for k, r in enumerate(rows_vis) if r[0] in half]); te_idx = torch.tensor([k for k, r in enumerate(rows_vis) if r[0] not in half])
        lr = TrustLR.fit(Fv[tr_idx], yv[tr_idx].float())
        pt = lr(Fv[te_idx])
        res["trust_lr_heldout_half"] = dict(n_train=len(tr_idx), n_test=len(te_idx), auroc=round(auroc(1 - pt, (~yv[te_idx]).float()), 4),
                                            auprc=round(auprc(1 - pt, (~yv[te_idx]).float()), 4), brier=round(brier(pt, yv[te_idx].float()), 4),
                                            ece=round(ece(pt, yv[te_idx].float()), 4), weights=dict(zip(FEATURES, [round(w, 3) for w in lr.w.tolist()])))
        tag = a.tag or os.path.basename(os.path.dirname(a.adapter.rstrip("/"))) if a.adapter else "base"
        lr.save(f"results/pure/trust_lr_{tag}_{rule}_half.json", extra=dict(fit="prompts first half", rule=rule))
        TrustLR.fit(Fv, yv.float()).save(f"results/pure/trust_lr_{tag}_{rule}.json", extra=dict(fit="all prompts", rule=rule))
        # masked-position LR too (unmask safety)
        lrm = TrustLR.fit(Fm[torch.tensor([k for k, r in enumerate(rows_mask) if r[0] in half])], ym[torch.tensor([k for k, r in enumerate(rows_mask) if r[0] in half])].float())
        tem = torch.tensor([k for k, r in enumerate(rows_mask) if r[0] not in half])
        pm = lrm(Fm[tem])
        res["masked_lr_heldout_half"] = dict(auroc=round(auroc(1 - pm, (~ym[tem]).float()), 4), brier=round(brier(pm, ym[tem].float()), 4),
                                             weights=dict(zip(FEATURES, [round(w, 3) for w in lrm.w.tolist()])))
        lrm.save(f"results/pure/trust_lr_{tag}_{rule}_masked_half.json", extra=dict(fit="prompts first half, masked positions", rule=rule))
        TrustLR.fit(Fm, ym.float()).save(f"results/pure/trust_lr_{tag}_{rule}_masked.json", extra=dict(fit="all prompts, masked positions", rule=rule))
        result[rule] = res
        print("=== trust", rule, json.dumps({k: v for k, v in res.items() if k in ("block_exact", "final_token_error", "final_wrong_detectable", "trust_lr_heldout_half")}), flush=True)
        print("=== trust-visible", rule, json.dumps({k: v.get("auroc") for k, v in res["visible"].items() if isinstance(v, dict)}), flush=True)
        print("=== trust-masked", rule, json.dumps({k: v.get("auroc") for k, v in res["masked"].items() if isinstance(v, dict)}), flush=True)
    save_json(a.out, dict(model=a.model, adapter=a.adapter, K=K, steps=S, n=len(prompts), gpu=gpu_info(), seconds=round(time.time() - t0, 1), results=result))


if __name__ == "__main__":
    main()
