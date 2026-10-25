"""Phase B9-C — evaluation of the global refinement samplers (docs/global_block_denoising.md §4–5).

For every (mode, S):
  TF   teacher-forced from the reference prefix (dev refs): per-step argmax-block accuracy against the reference,
       wrong->correct / correct->wrong between consecutive steps, positions changed per step (self-copying = 0 changes
       at a fully visible block), final token accuracy, block exact match, 4-gram repetition of the final block,
       mean entropy per step (distribution collapse / lock-in).
  GEN  free generation on the prompts: GSM8K accuracy, tokens/forward (incl. the prefix pass), tok/s, latency,
       premature EOS, length-budget hits, repetition.
Modes: monotonic (engine/bdiff/sampler2.py, left-to-right; the C1 baseline), sync, remask, soft, anneal
(engine/bdiff/global_sampler.py).  Exactly S forwards per block in every mode.
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. .venv-train/bin/python bench/eval_global.py --model runs/zeta_sft_final_e5/final \
      --adapter runs/bdiff_e5_lr1e4/adapter --refs results/pure/dev_refs_e5.json --n 50 --modes monotonic,sync,remask,soft,anneal \
      --steps 4,8,16 --out results/pure/b9_global_dev50.json"""
import argparse, collections, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts, gpu_info, save_json, score
from engine.bdiff.model import BDiffModel
from engine.bdiff.sampler2 import sample_block_fixed
from engine.bdiff.global_sampler import global_sample_block


def rep4(toks):
    if len(toks) < 8:
        return 0.0
    grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--prompts", default="bench/prompts_gsm8k_dev.json")
    ap.add_argument("--refs", default=None)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--steps", default="4,8,16")
    ap.add_argument("--modes", default="monotonic,sync,remask,soft,anneal")
    ap.add_argument("--schedule", default="linear")
    ap.add_argument("--t-end", type=float, default=0.2)
    ap.add_argument("--alpha-end", type=float, default=1.0)
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
    model = BDiffModel(lm, mask_id, time_cond=False).to(dev).eval()
    K = a.K
    stop_ids = {tok.eos_token_id} | ({tok.convert_tokens_to_ids("<|im_end|>")} if "<|im_end|>" in tok.get_vocab() else set())
    prompts = load_prompts(a.prompts)[a.skip: a.skip + a.n]
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        p["ids"] = ids[0].tolist() if torch.is_tensor(ids) else list(ids["input_ids"][0])
    refs = json.load(open(a.refs))["refs"] if a.refs else {}

    def sample(cache, P, Kb, S, mode, gold=None):
        if mode == "monotonic":
            toks, info = sample_block_fixed(model, cache, P, Kb, S, mask_id, unmask="left", remask="none", early_stop=False, trace=gold is not None, gold=gold)
            if gold is not None:
                gt = torch.tensor(gold); prev = None; ps = []
                for st in info["trace"]:
                    cur = torch.tensor(st["tokens"]); guess = torch.where(cur == mask_id, torch.tensor(st["pred"]), cur)
                    row = dict(step=st["step"], acc=round(float((guess == gt).float().mean()), 4), masked=int((cur == mask_id).sum()), changed=int((prev != guess).sum()) if prev is not None else Kb)
                    if prev is not None:
                        row["w2c"] = int(((prev != gt) & (guess == gt)).sum()); row["c2w"] = int(((prev == gt) & (guess != gt)).sum())
                    ps.append(row); prev = guess
                info["per_step"] = ps
            else:
                info["per_step"] = [dict(step=st["step"], changed=st["unmasked"], masked=st["masked"]) for st in info["per_step"]]
            return toks, info
        return global_sample_block(model, cache, P, Kb, S, mask_id, mode=mode, schedule=a.schedule, temperature_end=a.t_end, alpha_end=a.alpha_end, gold=gold)

    t0 = time.time(); TF = {}; GEN = {}
    torch.cuda.reset_peak_memory_stats()
    for mode in a.modes.split(","):
        for S in [int(x) for x in a.steps.split(",")]:
            name = f"{mode}_S{S}"
            if not a.skip_tf and refs:
                c = collections.Counter(); steps_acc = collections.defaultdict(lambda: collections.Counter()); rep = 0.0
                for p in prompts:
                    if p["id"] not in refs:
                        continue
                    resp = refs[p["id"]]["response_ids"]; full = p["ids"] + resp; P0 = len(p["ids"])
                    for s0 in list(range(P0, P0 + len(resp) - K + 1, K))[: a.max_blocks]:
                        gold = full[s0:s0 + K]
                        cache = model.prefix_cache(torch.tensor([full[:s0]], device=dev))
                        toks, info = sample(cache, s0, K, S, mode, gold=gold)
                        c["blocks"] += 1; c["forwards"] += info["forwards"]; c["exact"] += int(toks == gold); c["tok_ok"] += sum(toks[q] == gold[q] for q in range(K))
                        rep += rep4(toks)
                        for st in info["per_step"]:
                            sa = steps_acc[st["step"]]; sa["n"] += 1; sa["acc"] += st.get("acc", 0.0); sa["changed"] += st["changed"]; sa["masked"] += st["masked"]
                            sa["w2c"] += st.get("w2c", 0); sa["c2w"] += st.get("c2w", 0); sa["ent"] += st.get("entropy", 0.0)
                nb = max(1, c["blocks"])
                TF[name] = dict(blocks=c["blocks"], forwards_per_block=round(c["forwards"] / nb, 2), token_acc=round(c["tok_ok"] / (nb * K), 4), block_exact=round(c["exact"] / nb, 4), rep4=round(rep / nb, 4),
                                per_step=[dict(step=s, acc=round(v["acc"] / v["n"], 4), changed=round(v["changed"] / v["n"], 2), masked=round(v["masked"] / v["n"], 2), w2c=round(v["w2c"] / v["n"], 3), c2w=round(v["c2w"] / v["n"], 3), entropy=round(v["ent"] / v["n"], 3)) for s, v in sorted(steps_acc.items())])
                print("=== b9_tf", name, json.dumps(TF[name]), flush=True)
            if not a.skip_gen:
                rows = []; nblk = 0; ch = collections.Counter()
                for p in prompts:
                    ids = list(p["ids"]); out = []; forwards = 0; finish = None
                    torch.cuda.synchronize(); tt = time.time()
                    while finish is None:
                        Kb = min(K, a.max_new_tokens - len(out))
                        cache = model.prefix_cache(torch.tensor([ids + out], device=dev)); forwards += 1
                        toks, info = sample(cache, len(ids) + len(out), Kb, S, mode)
                        forwards += info["forwards"]; nblk += 1
                        for st in info["per_step"]:
                            ch[st["step"]] += st["changed"]
                        keep = Kb
                        for q in range(Kb):
                            if toks[q] in stop_ids:
                                keep, finish = q + 1, "stop"; break
                        out += toks[:keep]
                        if len(out) >= a.max_new_tokens:
                            finish = finish or "length"
                    torch.cuda.synchronize(); dt = time.time() - tt
                    text = tok.decode(out, skip_special_tokens=True)
                    rows.append(dict(id=p["id"], n_tokens=len(out), seconds=round(dt, 2), tok_s=round(len(out) / dt, 1), forwards=forwards, correct=score(p, text), finish=finish,
                                     has_answer="####" in text, rep4=round(rep4(out), 3), text=text))
                    print(f"[{name}] {p['id']} {len(out)} tok {rows[-1]['tok_s']} tok/s fwd {forwards} correct={rows[-1]['correct']}", flush=True)
                tot = sum(r["n_tokens"] for r in rows)
                GEN[name] = dict(accuracy=sum(bool(r["correct"]) for r in rows) / len(rows), n=len(rows), tok_s_mean=round(sum(r["tok_s"] for r in rows) / len(rows), 1),
                                 tokens_per_forward=round(tot / max(1, sum(r["forwards"] for r in rows)), 3), mean_tokens=round(tot / len(rows), 1), latency_s_mean=round(sum(r["seconds"] for r in rows) / len(rows), 2),
                                 premature_eos=sum(1 for r in rows if r["finish"] == "stop" and not r["has_answer"]), finish_length=sum(r["finish"] == "length" for r in rows),
                                 rep4_mean=round(sum(r["rep4"] for r in rows) / len(rows), 4), changed_per_step={str(s): round(v / max(1, nblk), 2) for s, v in sorted(ch.items())}, rows=rows)
                print("=== b9_gen", name, json.dumps({k: v for k, v in GEN[name].items() if k != "rows"}), flush=True)
    save_json(a.out, dict(model=a.model, adapter=a.adapter, K=K, n=len(prompts), skip=a.skip, prompts=a.prompts, gpu=gpu_info(), peak_mem_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                          seconds=round(time.time() - t0, 1), args=vars(a), TF=TF, GEN=GEN))


if __name__ == "__main__":
    main()
