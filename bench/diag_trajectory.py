"""Phase B9-A/B/D — generation-trajectory analysis of the B-series K = 16 sampler (docs/global_block_denoising.md §2–3, §5).

Free generation (no reference read at inference) with the fixed-budget sampler (S steps, left-to-right or
confidence order).  At EVERY step and for ALL 16 positions the following is recorded from the step's forward:
top-k token ids + log-probs, entropy, max-prob, the mask pattern, the placed tokens, and (optionally) the final-layer
hidden states.  Offline analysis against two targets, kept strictly apart:
  FINAL  the block the same run ends with (what the trajectory converges to — available only after the run)
  REF    the base model's own greedy AR answer, teacher-forced prefix (--refs; only for the teacher-forced mode)
Mode --mode free: prompts are generated freely (targets: FINAL only, plus the GSM8K answer).
Mode --mode tf: each reference block is decoded from its gold prefix (targets: FINAL and REF).
Per-step quantities (averaged over blocks): for still-masked positions, top-1 / top-k recall of the FINAL (REF) token;
the step at which each position's argmax first equals its final token and never changes again (stabilisation);
JSD between consecutive steps' distributions at positions still masked in both ("how much placing others moves the
rest"); Hamming distance of the current argmax block (masked positions take their current argmax) to FINAL / REF;
entropy; hidden-state displacement; the independent-argmax block of the FIRST forward vs the FINAL block (block exact,
position accuracy, 4-gram repetition, adjacent duplicates); and the log-probability of the FINAL block under the
marginals of step s (how much of the final block is already the mode at s).  Everything is written to one json;
hidden states go to a .pt when --save-hidden.
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv-train/bin/python bench/diag_trajectory.py --model runs/zeta_sft_final_e5/final \
      --adapter runs/bdiff_e5_lr1e4/adapter --n 50 --steps 16 --unmask left --mode tf --refs results/pure/dev_refs_e5.json \
      --out results/pure/traj_s16_left_tf.json"""
import argparse, collections, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts, gpu_info, save_json, score
from engine.bdiff.model import BDiffModel
from engine.bdiff.corrupt import gamma


def jsd_rows(p, q):
    p = p.clamp_min(1e-12); q = q.clamp_min(1e-12); m = 0.5 * (p + q)
    return 0.5 * ((p * (p / m).log()).sum(-1) + (q * (q / m).log()).sum(-1)) / torch.log(torch.tensor(2.0, device=p.device))


def rep4(toks):
    if len(toks) < 8:
        return 0.0
    grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


@torch.no_grad()
def run_block(model, cache, P, K, S, mask_id, unmask, topk, want_hidden):
    """Returns final tokens and the per-step records."""
    dev = model.device
    x = torch.full((K,), mask_id, dtype=torch.long, device=dev)
    recs = []
    for s in range(1, S + 1):
        is_mask = x == mask_id
        n_mask = int(is_mask.sum())
        if n_mask == 0:
            break
        g = max(n_mask / K, 1e-3)
        if want_hidden:
            lg, hid = model.denoise(cache, P, x[None], g, return_hidden=True)
        else:
            lg = model.denoise(cache, P, x[None], g); hid = None
        lg = lg.float(); lg[:, mask_id] = float("-inf")
        probs = torch.softmax(lg, -1)
        tk = probs.topk(topk)
        ent = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(-1)
        last = s == S
        target = 0 if last else int(round(K * gamma(1.0 - s / S)))
        n_un = max(0, n_mask - target)
        masked_idx = is_mask.nonzero().flatten()
        conf, pred = probs.max(-1)
        if last:
            sel = masked_idx
        elif unmask == "left":
            sel = masked_idx[:n_un]
        else:
            sel = masked_idx[conf[masked_idx].argsort(descending=True)[:n_un]]
        recs.append(dict(step=s, mask=is_mask.cpu(), tokens_before=x.clone().cpu(), pred=pred.cpu(), conf=conf.cpu(), ent=ent.cpu(),
                         topk_ids=tk.indices.cpu(), topk_lp=tk.values.clamp_min(1e-12).log().cpu(), probs=probs.cpu(), hidden=None if hid is None else hid.to(torch.bfloat16).cpu(),
                         placed=sel.cpu()))
        x[sel] = pred[sel]
    return x.tolist(), recs


def analyse(blocks, K, S, topk):
    """blocks: list of dict(final, ref or None, recs).  Returns the aggregated per-step tables."""
    A = collections.defaultdict(lambda: collections.defaultdict(float))
    N = collections.defaultdict(lambda: collections.defaultdict(int))
    stab_final = []; stab_ref = []
    first_vs_final = dict(blocks=0, exact=0, pos_ok=0, rep4_first=0.0, rep4_final=0.0, dup_adj_first=0, dup_adj_final=0)
    for b in blocks:
        final = torch.tensor(b["final"]); ref = torch.tensor(b["ref"]) if b["ref"] is not None else None
        recs = b["recs"]
        # first-forward independent argmax block
        f0 = recs[0]["pred"]
        first_vs_final["blocks"] += 1; first_vs_final["exact"] += int(torch.equal(f0, final)); first_vs_final["pos_ok"] += int((f0 == final).sum())
        first_vs_final["rep4_first"] += rep4(f0.tolist()); first_vs_final["rep4_final"] += rep4(final.tolist())
        first_vs_final["dup_adj_first"] += int((f0[1:] == f0[:-1]).sum()); first_vs_final["dup_adj_final"] += int((final[1:] == final[:-1]).sum())
        # stabilisation step per position: first step from which argmax == final at every later step (while masked) and the placed token == final
        for q in range(K):
            seq = [int(r["pred"][q]) for r in recs]
            st = None
            for s in range(len(seq)):
                if all(t == int(final[q]) for t in seq[s:]):
                    st = s + 1; break
            stab_final.append(st if st is not None else S + 1)
            if ref is not None:
                st = None
                for s in range(len(seq)):
                    if all(t == int(ref[q]) for t in seq[s:]):
                        st = s + 1; break
                stab_ref.append(st if st is not None else S + 1)
        prev = None
        for r in recs:
            s = r["step"]; m = r["mask"]
            nm = int(m.sum())
            cur = torch.where(m, r["pred"], r["tokens_before"])          # current best guess of the whole block
            A[s]["hamming_final_all"] += int((cur != final).sum()); N[s]["hamming_final_all"] += 1
            A[s]["hamming_final_placed"] += int(((cur != final) & ~m).sum()); A[s]["n_placed"] += int((~m).sum())
            if ref is not None:
                A[s]["hamming_ref_all"] += int((cur != ref).sum())
            if nm:
                mk = m.nonzero().flatten()
                A[s]["masked_top1_final"] += int((r["pred"][mk] == final[mk]).sum())
                A[s]["masked_topk_final"] += int((r["topk_ids"][mk] == final[mk, None]).any(-1).sum())
                A[s]["masked_ent"] += float(r["ent"][mk].sum()); A[s]["masked_conf"] += float(r["conf"][mk].sum())
                A[s]["lp_final_masked"] += float(r["probs"][mk].gather(-1, final[mk, None]).clamp_min(1e-12).log().sum())
                if ref is not None:
                    A[s]["masked_top1_ref"] += int((r["pred"][mk] == ref[mk]).sum())
                    A[s]["masked_topk_ref"] += int((r["topk_ids"][mk] == ref[mk, None]).any(-1).sum())
                N[s]["masked"] += nm
            if prev is not None:
                both = m & prev["mask"]
                if both.any():
                    bi = both.nonzero().flatten()
                    A[s]["jsd_prev_masked"] += float(jsd_rows(prev["probs"][bi], r["probs"][bi]).sum())
                    A[s]["top1_change_masked"] += int((prev["pred"][bi] != r["pred"][bi]).sum())
                    A[s]["topk_set_jaccard"] += float(sum(len(set(prev["topk_ids"][q].tolist()) & set(r["topk_ids"][q].tolist())) / len(set(prev["topk_ids"][q].tolist()) | set(r["topk_ids"][q].tolist())) for q in bi.tolist()))
                    N[s]["both_masked"] += len(bi)
                    if r["hidden"] is not None:
                        A[s]["hidden_disp_masked"] += float((r["hidden"][bi].float() - prev["hidden"][bi].float()).norm(dim=-1).sum())
                        A[s]["hidden_norm_masked"] += float(r["hidden"][bi].float().norm(dim=-1).sum())
            prev = r
    nb = max(1, first_vs_final["blocks"])
    per_step = {}
    for s in sorted(A):
        a = A[s]; n = N[s]; nmk = max(1, n["masked"]); nbm = max(1, n["both_masked"]); nblk = max(1, n["hamming_final_all"])
        row = dict(blocks=n["hamming_final_all"], masked_positions=round(n["masked"] / nblk, 2),
                   hamming_to_final_all16=round(a["hamming_final_all"] / nblk, 3), hamming_to_final_placed_only=round(a["hamming_final_placed"] / max(1, a["n_placed"]), 4),
                   masked_top1_recall_final=round(a["masked_top1_final"] / nmk, 4), masked_topk_recall_final=round(a["masked_topk_final"] / nmk, 4),
                   masked_entropy=round(a["masked_ent"] / nmk, 3), masked_conf=round(a["masked_conf"] / nmk, 4),
                   logp_final_per_masked_token=round(a["lp_final_masked"] / nmk, 3))
        if "hamming_ref_all" in a:
            row.update(hamming_to_ref_all16=round(a["hamming_ref_all"] / nblk, 3), masked_top1_recall_ref=round(a["masked_top1_ref"] / nmk, 4), masked_topk_recall_ref=round(a["masked_topk_ref"] / nmk, 4))
        if n["both_masked"]:
            row.update(jsd_vs_prev_step_masked=round(a["jsd_prev_masked"] / nbm, 4), top1_change_vs_prev_masked=round(a["top1_change_masked"] / nbm, 4),
                       topk_jaccard_vs_prev=round(a["topk_set_jaccard"] / nbm, 4))
            if a.get("hidden_norm_masked"):
                row.update(hidden_rel_displacement_masked=round(a["hidden_disp_masked"] / a["hidden_norm_masked"], 4))
        per_step[str(s)] = row
    hist_f = collections.Counter(stab_final); hist_r = collections.Counter(stab_ref)
    out = dict(per_step=per_step, topk=topk,
               stabilisation_step_final=dict(mean=round(sum(stab_final) / len(stab_final), 3), hist={str(k): v for k, v in sorted(hist_f.items())},
                                             frac_by_step=[round(sum(v for k, v in hist_f.items() if k <= s) / len(stab_final), 4) for s in range(1, S + 1)]),
               first_forward_vs_final=dict(block_exact=round(first_vs_final["exact"] / nb, 4), position_acc=round(first_vs_final["pos_ok"] / (nb * K), 4),
                                           rep4_first=round(first_vs_final["rep4_first"] / nb, 4), rep4_final=round(first_vs_final["rep4_final"] / nb, 4),
                                           adjacent_duplicates_per_block_first=round(first_vs_final["dup_adj_first"] / nb, 3), adjacent_duplicates_per_block_final=round(first_vs_final["dup_adj_final"] / nb, 3)))
    if stab_ref:
        out["stabilisation_step_ref"] = dict(mean=round(sum(stab_ref) / len(stab_ref), 3), frac_by_step=[round(sum(v for k, v in hist_r.items() if k <= s) / len(stab_ref), 4) for s in range(1, S + 1)],
                                             never=round(hist_r.get(S + 1, 0) / len(stab_ref), 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--no-time", action="store_true", default=True)
    ap.add_argument("--prompts", default="bench/prompts_gsm8k_dev.json")
    ap.add_argument("--refs", default=None)
    ap.add_argument("--mode", default="tf", choices=["tf", "free"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--unmask", default="left")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--max-blocks", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--save-hidden", default=None, help=".pt with hidden states of every step (bf16) for the probe analysis")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    mask_id = getattr(lm.config, "cdlm_mask_token_id", None) or tok.convert_tokens_to_ids("<|mask|>")
    if a.adapter:
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, a.adapter).eval().merge_and_unload()
    model = BDiffModel(lm, mask_id, time_cond=False).to(dev).eval()
    K, S = a.K, a.steps
    stop_ids = {tok.eos_token_id} | ({tok.convert_tokens_to_ids("<|im_end|>")} if "<|im_end|>" in tok.get_vocab() else set())
    prompts = load_prompts(a.prompts)[a.skip: a.skip + a.n]
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        p["ids"] = ids[0].tolist() if torch.is_tensor(ids) else list(ids["input_ids"][0])
    refs = json.load(open(a.refs))["refs"] if a.refs else {}
    t0 = time.time(); blocks = []; hidden_dump = []; gen_rows = []
    want_hidden = True
    for p in prompts:
        if a.mode == "tf":
            resp = refs[p["id"]]["response_ids"]; full = p["ids"] + resp; P0 = len(p["ids"])
            for s0 in list(range(P0, P0 + len(resp) - K + 1, K))[: a.max_blocks]:
                cache = model.prefix_cache(torch.tensor([full[:s0]], device=dev))
                final, recs = run_block(model, cache, s0, K, S, mask_id, a.unmask, a.topk, want_hidden)
                blocks.append(dict(final=final, ref=full[s0:s0 + K], recs=recs, id=p["id"], start=s0))
        else:
            ids = list(p["ids"]); out = []; finish = None
            while finish is None:
                Kb = min(K, a.max_new_tokens - len(out))
                cache = model.prefix_cache(torch.tensor([ids + out], device=dev))
                final, recs = run_block(model, cache, len(ids) + len(out), Kb, S, mask_id, a.unmask, a.topk, want_hidden)
                if Kb == K:
                    blocks.append(dict(final=final, ref=None, recs=recs, id=p["id"], start=len(ids) + len(out)))
                keep = Kb
                for q in range(Kb):
                    if final[q] in stop_ids:
                        keep, finish = q + 1, "stop"; break
                out += final[:keep]
                if len(out) >= a.max_new_tokens:
                    finish = finish or "length"
            text = tok.decode(out, skip_special_tokens=True)
            gen_rows.append(dict(id=p["id"], correct=score(p, text), n_tokens=len(out)))
        print(f"{p['id']} blocks {len(blocks)} {time.time() - t0:.0f}s", flush=True)
    res = analyse(blocks, K, S, a.topk)
    if gen_rows:
        res["generation"] = dict(accuracy=sum(bool(r["correct"]) for r in gen_rows) / len(gen_rows), n=len(gen_rows))
    # intermediate texts of a few blocks for the report
    samples = []
    for b in blocks[:6]:
        samples.append(dict(id=b["id"], final=tok.decode(b["final"]), ref=None if b["ref"] is None else tok.decode(b["ref"]),
                            steps=[dict(step=r["step"], guess=tok.decode(torch.where(r["mask"], r["pred"], r["tokens_before"]).tolist())) for r in b["recs"]]))
    res["samples"] = samples
    if a.save_hidden:
        torch.save(dict(blocks=[dict(id=b["id"], start=b["start"], final=b["final"], ref=b["ref"],
                                     steps=[dict(step=r["step"], mask=r["mask"], pred=r["pred"], conf=r["conf"], hidden=r["hidden"], topk_ids=r["topk_ids"], topk_lp=r["topk_lp"]) for r in b["recs"]]) for b in blocks]), a.save_hidden)
    print("=== trajectory", json.dumps({k: v for k, v in res.items() if k not in ("samples",)})[:3000], flush=True)
    save_json(a.out, dict(model=a.model, adapter=a.adapter, mode=a.mode, unmask=a.unmask, K=K, steps=S, n=len(prompts), blocks=len(blocks), gpu=gpu_info(),
                          seconds=round(time.time() - t0, 1), **res))


if __name__ == "__main__":
    main()
