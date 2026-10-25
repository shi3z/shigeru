"""Phase B7 §3/§4/§6 — fixed-budget sampler evaluation (docs/trust_aware_diffusion.md).

For every (unmask rule, remask rule) pair:
  TF  teacher-forced trajectories on the reference blocks (gold prefix, K reference tokens, S forwards from all masks):
      final token error, block exact match, and for the RE-MASKED positions: wrong->correct, correct->wrong, wrong->wrong
      (token before re-masking vs final token, both against the reference), correction success / damage rates, and the
      per-step counts of fixed / masked / re-masked positions.
  GEN free generation on the prompts (GSM8K accuracy, tok/s, tokens per forward incl. the prefix pass, premature EOS,
      length-budget hits, re-masks per block, per-step fixed/masked/re-masked counts).
Exactly S network calls per block in both modes; forwards are counted and reported, never hidden.
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv-train/bin/python bench/eval_trust.py --model runs/zeta_sft_final_e5/final \
      --adapter runs/bdiff_e5_lr1e4/adapter --no-time --refs results/pure/dev_refs_sft3.json --n 50 --steps 8 \
      --unmask confidence,left --remask none,disagree,lowprob,stability,dependency,trust \
      --trust results/pure/trust_lr_bdiff_e5_lr1e4_confidence_half.json --out results/pure/b7_dev50_lr1e4.json"""
import argparse, collections, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts, gpu_info, save_json, score
from engine.bdiff.model import BDiffModel
from engine.bdiff.sampler2 import sample_block_fixed
from engine.bdiff.trust import TrustLR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--prompts", default="bench/prompts_gsm8k_dev.json")
    ap.add_argument("--refs", default=None, help="reference answers for the teacher-forced part (omit with --skip-tf)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--unmask", default="confidence")
    ap.add_argument("--remask", default="none")
    ap.add_argument("--remask-budget", type=int, default=2)
    ap.add_argument("--remask-start", type=int, default=2)
    ap.add_argument("--tau-unmask", type=float, default=0.9)
    ap.add_argument("--tau-remask", type=float, default=0.5)
    ap.add_argument("--trust", default=None, help="TrustLR weights json for remask=trust")
    ap.add_argument("--trust-mask", default=None, help="TrustLR weights json (masked positions) for unmask=trust")
    ap.add_argument("--schedule", default="linear")
    ap.add_argument("--skip-tf", action="store_true")
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--max-blocks", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=384)
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
    trust = TrustLR.load(a.trust) if a.trust else None
    trust_mask = TrustLR.load(a.trust_mask) if a.trust_mask else None
    stop_ids = {tok.eos_token_id} | ({tok.convert_tokens_to_ids("<|im_end|>")} if "<|im_end|>" in tok.get_vocab() else set())
    prompts = load_prompts(a.prompts)[a.skip: a.skip + a.n]
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        p["ids"] = ids[0].tolist() if torch.is_tensor(ids) else list(ids["input_ids"][0])
    refs = json.load(open(a.refs))["refs"] if a.refs else {}
    t0 = time.time()
    kw = dict(schedule=a.schedule, remask_budget=a.remask_budget, remask_start=a.remask_start, tau_unmask=a.tau_unmask, tau_remask=a.tau_remask, trust=trust, trust_mask=trust_mask)
    out_tf, out_gen = {}, {}
    torch.cuda.reset_peak_memory_stats()
    for u in a.unmask.split(","):
        for r in a.remask.split(","):
            name = f"S{S}_u-{u}_r-{r}" + (f"_b{a.remask_budget}" if r != "none" else "") + (f"_t{a.tau_remask}" if r in ("lowprob", "trust") else "") + (f"_tu{a.tau_unmask}" if u == "adaptive" else "")
            # ---------------- teacher-forced
            if not a.skip_tf and refs:
                c = dict(blocks=0, exact=0, tok_err=0, toks=0, remasked=0, wc=0, cw=0, ww=0, cc=0, forwards=0)
                steps_acc = collections.defaultdict(lambda: [0, 0, 0])
                for p in prompts:
                    if p["id"] not in refs:
                        continue
                    resp = refs[p["id"]]["response_ids"]
                    full = p["ids"] + resp
                    P0 = len(p["ids"])
                    for s0 in list(range(P0, P0 + len(resp) - K + 1, K))[: a.max_blocks]:
                        gold = full[s0:s0 + K]
                        cache = model.prefix_cache(torch.tensor([full[:s0]], device=dev))
                        toks, info = sample_block_fixed(model, cache, s0, K, S, mask_id, unmask=u, remask=r, trace=True, early_stop=False, gold=gold if r == "oracle" else None, **kw)
                        c["blocks"] += 1; c["forwards"] += info["forwards"]
                        c["exact"] += int(toks == gold); c["toks"] += K; c["tok_err"] += sum(toks[i] != gold[i] for i in range(K))
                        # token before each re-mask (from the trace: tokens at the previous step) vs final
                        prev_tokens = [mask_id] * K
                        for st in info["trace"]:
                            for i in st["remasked"]:
                                before = prev_tokens[i]
                                b_ok = before == gold[i]; f_ok = toks[i] == gold[i]
                                c["remasked"] += 1
                                if b_ok and f_ok: c["cc"] += 1
                                elif b_ok: c["cw"] += 1
                                elif f_ok: c["wc"] += 1
                                else: c["ww"] += 1
                            prev_tokens = st["tokens"]
                        for st in info["per_step"]:
                            s = steps_acc[st["step"]]; s[0] += st["fixed"]; s[1] += st["masked"]; s[2] += st["remasked"]
                nb = max(1, c["blocks"])
                out_tf[name] = dict(blocks=c["blocks"], forwards_per_block=round(c["forwards"] / nb, 3), block_exact=round(c["exact"] / nb, 4),
                                    final_token_error=round(c["tok_err"] / max(1, c["toks"]), 4), remasked_per_block=round(c["remasked"] / nb, 3),
                                    wrong_to_correct=c["wc"], correct_to_wrong=c["cw"], wrong_to_wrong=c["ww"], correct_to_correct=c["cc"],
                                    correction_success=round(c["wc"] / max(1, c["wc"] + c["ww"]), 4), damage_rate=round(c["cw"] / max(1, c["cw"] + c["cc"]), 4),
                                    net_gain_per_block=round((c["wc"] - c["cw"]) / nb, 3),
                                    per_step=[dict(step=s, fixed=round(v[0] / nb, 2), masked=round(v[1] / nb, 2), remasked=round(v[2] / nb, 2)) for s, v in sorted(steps_acc.items())])
                print("=== b7_tf", name, json.dumps({k: v for k, v in out_tf[name].items() if k != "per_step"}), flush=True)
            # ---------------- free generation
            if not a.skip_gen and r != "oracle":
                rows = []
                steps_acc = collections.defaultdict(lambda: [0, 0, 0]); nblocks = 0; nrem = 0
                for p in prompts:
                    ids = list(p["ids"]); out = []; forwards = 0; finish = None
                    torch.cuda.synchronize(); tt = time.time()
                    while finish is None:
                        Kb = min(K, a.max_new_tokens - len(out))
                        cache = model.prefix_cache(torch.tensor([ids + out], device=dev)); forwards += 1
                        toks, info = sample_block_fixed(model, cache, len(ids) + len(out), Kb, S, mask_id, unmask=u, remask=r, **kw)
                        forwards += info["forwards"]; nblocks += 1; nrem += info["remasked"]
                        for st in info["per_step"]:
                            s = steps_acc[st["step"]]; s[0] += st["fixed"]; s[1] += st["masked"]; s[2] += st["remasked"]
                        keep = Kb
                        for j in range(Kb):
                            if toks[j] in stop_ids:
                                keep, finish = j + 1, "stop"; break
                        out += toks[:keep]
                        if len(out) >= a.max_new_tokens:
                            finish = finish or "length"
                    torch.cuda.synchronize(); dt = time.time() - tt
                    text = tok.decode(out, skip_special_tokens=True)
                    rows.append(dict(id=p["id"], n_tokens=len(out), seconds=round(dt, 2), tok_s=round(len(out) / dt, 1), forwards=forwards, correct=score(p, text),
                                     finish=finish, has_answer="####" in text, text=text))
                    print(f"[{name}] {p['id']} {len(out)} tok {rows[-1]['tok_s']} tok/s fwd {forwards} correct={rows[-1]['correct']}", flush=True)
                tot = sum(x["n_tokens"] for x in rows)
                out_gen[name] = dict(accuracy=sum(bool(x["correct"]) for x in rows) / len(rows), n=len(rows), tok_s_mean=round(sum(x["tok_s"] for x in rows) / len(rows), 1),
                                     tokens_per_forward=round(tot / max(1, sum(x["forwards"] for x in rows)), 3), mean_tokens=round(tot / len(rows), 1),
                                     latency_s_mean=round(sum(x["seconds"] for x in rows) / len(rows), 2),
                                     premature_eos=sum(1 for x in rows if x["finish"] == "stop" and not x["has_answer"]), finish_length=sum(x["finish"] == "length" for x in rows),
                                     remasked_per_block=round(nrem / max(1, nblocks), 3),
                                     per_step=[dict(step=s, fixed=round(v[0] / max(1, nblocks), 2), masked=round(v[1] / max(1, nblocks), 2), remasked=round(v[2] / max(1, nblocks), 2)) for s, v in sorted(steps_acc.items())],
                                     rows=rows)
                print("=== b7_gen", name, json.dumps({k: v for k, v in out_gen[name].items() if k not in ("rows", "per_step")}), flush=True)
    save_json(a.out, dict(model=a.model, adapter=a.adapter, K=K, steps=S, n=len(prompts), skip=a.skip, prompts=a.prompts, gpu=gpu_info(),
                          peak_mem_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), seconds=round(time.time() - t0, 1), args=vars(a), TF=out_tf, GEN=out_gen))


if __name__ == "__main__":
    main()
