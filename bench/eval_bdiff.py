"""B-series evaluation (brief §5, §7, §8): Stage B1 masked-denoising accuracy, Stage B3 correction effectiveness, Stage B4
reverse-diffusion free generation on GSM8K dev.  Loads a LoRA adapter (+ time embedding) or a merged dir.

B1 (teacher-forced, --refs blocks of K): mask rates 10/25/50/75/90/100 % -> masked-token accuracy, block exact match,
   accuracy by position (for 100 %).
B3 (teacher-forced; ONE denoising forward, argmax at every position):
   A  random corruption: 25 % of the block replaced by random tokens (no masks)
   B  partial reconstruction: 50 % masked, rest clean
   C  on-policy corruption: the A-series decoder's recorded states (--states, K=32 states cut to the first 16) or, if
      not given, the B-series sampler's own intermediate state at step S/2 from all masks
   -> error count before/after, W->C, C->W, W->W', C->C, Hamming distance to the target, block exact.
B4 (free generation, --prompts dev items, --n): all-mask K-block reverse diffusion with S steps (--steps list), orders,
   optional remask; GSM8K accuracy, tokens/forward (K / network calls incl. the prefix pass per block), tok/s, premature EOS.
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-train/bin/python bench/eval_bdiff.py --model runs/zeta_sft_final_e5/final \
      --adapter runs/bdiff_b1/adapter --time-embed runs/bdiff_b1/time_embed.pt --refs results/pure/dev_refs_sft3.json --n 50 \
      --steps 16,8,4 --out results/pure/bdiff_b1_eval.json"""
import argparse, collections, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts, gpu_info, save_json, score
from engine.bdiff.corrupt import corrupt_block
from engine.bdiff.model import BDiffModel
from engine.bdiff.sampler import sample_block

RATES = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--time-embed", default=None)
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--prompts", default="bench/prompts_gsm8k_dev.json")
    ap.add_argument("--refs", required=True)
    ap.add_argument("--states", default=None)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--steps", default="16,8,4")
    ap.add_argument("--orders", default="confidence")
    ap.add_argument("--remask", action="store_true")
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--max-blocks", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    mask_id = getattr(lm.config, "cdlm_mask_token_id", None) or tok.convert_tokens_to_ids("<|mask|>")
    if a.adapter:
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, a.adapter).eval()
        lm = lm.merge_and_unload()
    model = BDiffModel(lm, mask_id, time_cond=not a.no_time).to(dev).eval()
    if model.time is not None and a.time_embed:
        model.time.load_state_dict(torch.load(a.time_embed, map_location="cpu"))
        model.time.to(dev)
    K = a.K
    stop_ids = {tok.eos_token_id} | ({tok.convert_tokens_to_ids("<|im_end|>")} if "<|im_end|>" in tok.get_vocab() else set())
    prompts = load_prompts(a.prompts)[: a.n]
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        p["ids"] = ids[0].tolist() if torch.is_tensor(ids) else list(ids["input_ids"][0])
    refs = json.load(open(a.refs))["refs"]
    rng = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    # ---------------- B1 + B3 (teacher-forced)
    b1 = {r: dict(masked=0, masked_ok=0, blocks=0, exact=0, pos_ok=[0] * K) for r in RATES}
    b3 = {c: dict(blocks=0, err_before=0, err_after=0, wc=0, cw=0, ww=0, cc=0, exact=0) for c in ("A_random", "B_partial", "C_onpolicy")}
    states = collections.defaultdict(list)
    if a.states:
        for line in open(a.states):
            r = json.loads(line)
            b = int(list(r["states"].keys())[0]); B = int(r["block"]); s0 = b * B
            states[r["id"]].append((s0, r["states"][str(b)][:K], r["tokens"]))
    for p in prompts:
        resp = refs[p["id"]]["response_ids"]
        full = p["ids"] + resp
        P0 = len(p["ids"])
        starts = list(range(P0, P0 + len(resp) - K + 1, K))[: a.max_blocks]
        for s in starts:
            gold = full[s:s + K]
            cache = model.prefix_cache(torch.tensor([full[:s]], device=dev))
            for r in RATES:
                xt, im, _, g = corrupt_block(gold, r, mask_id, rng, schedule="linear")
                lg = model.denoise(cache, s, torch.tensor([xt], device=dev), g).float(); lg[:, mask_id] = float("-inf")
                pred = lg.argmax(-1).tolist()
                ok = [pred[i] == gold[i] for i in range(K) if im[i]]
                c = b1[r]; c["masked"] += len(ok); c["masked_ok"] += sum(ok); c["blocks"] += 1; c["exact"] += int(all(ok))
                if r == 1.0:
                    for i in range(K):
                        c["pos_ok"][i] += int(pred[i] == gold[i])

            def one_step(xt, g, name):
                lg = model.denoise(cache, s, torch.tensor([xt], device=dev), g).float(); lg[:, mask_id] = float("-inf")
                pred = lg.argmax(-1).tolist()
                c = b3[name]; c["blocks"] += 1
                for i in range(K):
                    before_ok = xt[i] == gold[i]; after_ok = pred[i] == gold[i]
                    c["err_before"] += int(not before_ok); c["err_after"] += int(not after_ok)
                    if before_ok and after_ok: c["cc"] += 1
                    elif before_ok: c["cw"] += 1
                    elif after_ok: c["wc"] += 1
                    elif pred[i] != xt[i]: c["ww"] += 1
                c["exact"] += int(pred == gold)

            xa = list(gold)
            for i in torch.randperm(K, generator=rng)[: K // 4].tolist():
                xa[i] = int(torch.randint(151643, (1,), generator=rng))
            one_step(xa, 0.05, "A_random")
            xb, _, _, gb = corrupt_block(gold, 0.5, mask_id, rng)
            one_step(xb, gb, "B_partial")
            st = [x for x in states.get(p["id"], []) if x[0] == s]
            if st:
                xc = [mask_id if v == -1 else v for v in st[0][1]]
                one_step(xc, max(sum(v == -1 for v in st[0][1]) / K, 1e-3), "C_onpolicy")
            else:
                # the sampler's own half-way state from all masks (S = 8 -> stop after 4 steps)
                toks, info = sample_block(model, cache, s, K, 8, mask_id, order="confidence", trace=True)
                mid = info["trace"][3]["tokens"]
                one_step(mid, max(sum(t == mask_id for t in mid) / K, 1e-3), "C_onpolicy")
        print(f"B1/B3 {p['id']} {time.time() - t0:.0f}s", flush=True)
    outB1 = {str(r): dict(masked_acc=round(c["masked_ok"] / max(1, c["masked"]), 4), block_exact=round(c["exact"] / max(1, c["blocks"]), 4),
                          **({"acc_by_position": [round(x / c["blocks"], 3) for x in c["pos_ok"]]} if r == 1.0 else {})) for r, c in b1.items()}
    outB3 = {n: dict(blocks=c["blocks"], errors_before_per_block=round(c["err_before"] / max(1, c["blocks"]), 3), errors_after_per_block=round(c["err_after"] / max(1, c["blocks"]), 3),
                     wrong_to_correct=c["wc"], correct_to_wrong=c["cw"], wrong_to_wrong=c["ww"], correct_to_correct=c["cc"], block_exact_after=round(c["exact"] / max(1, c["blocks"]), 4),
                     correction_success_rate=round(c["wc"] / max(1, c["wc"] + c["ww"] + (c["err_before"] - c["wc"] - c["ww"])), 4),
                     error_introduction_rate=round(c["cw"] / max(1, c["cw"] + c["cc"]), 4)) for n, c in b3.items() if c["blocks"]}
    # ---------------- B4 free generation
    gen = {}
    if not a.skip_gen:
        steps_list = [int(x) for x in a.steps.split(",")]
        for order in a.orders.split(","):
            for S in steps_list:
                name = f"S{S}_{order}" + ("_remask" if a.remask else "")
                rows = []
                for p in prompts:
                    ids = list(p["ids"])
                    out = []
                    forwards = 0
                    torch.cuda.synchronize(); tt = time.time()
                    finish = None
                    while finish is None:
                        Kb = min(K, a.max_new_tokens - len(out))
                        cache = model.prefix_cache(torch.tensor([ids + out], device=dev)); forwards += 1
                        toks, info = sample_block(model, cache, len(ids) + len(out), Kb, S, mask_id, order=order, remask=a.remask,
                                                  generator=torch.Generator().manual_seed(a.seed))
                        forwards += info["forwards"]
                        keep = Kb
                        for j in range(Kb):
                            if toks[j] in stop_ids:
                                keep, finish = j + 1, "stop"; break
                        out += toks[:keep]
                        if len(out) >= a.max_new_tokens:
                            finish = finish or "length"
                    torch.cuda.synchronize(); dt = time.time() - tt
                    text = tok.decode(out, skip_special_tokens=True)
                    rows.append(dict(id=p["id"], n_tokens=len(out), seconds=round(dt, 2), tok_s=round(len(out) / dt, 1), forwards=forwards,
                                     correct=score(p, text), finish=finish, has_answer="####" in text, text=text))
                    print(f"[{name}] {p['id']} {len(out)} tok {rows[-1]['tok_s']} tok/s fwd {forwards} correct={rows[-1]['correct']}", flush=True)
                tot = sum(r["n_tokens"] for r in rows)
                gen[name] = dict(accuracy=sum(bool(r["correct"]) for r in rows) / len(rows), n=len(rows), tok_s_mean=round(sum(r["tok_s"] for r in rows) / len(rows), 1),
                                 tokens_per_forward=round(tot / max(1, sum(r["forwards"] for r in rows)), 3), mean_tokens=round(tot / len(rows), 1),
                                 premature_eos=sum(1 for r in rows if r["finish"] == "stop" and not r["has_answer"]), finish_length=sum(r["finish"] == "length" for r in rows),
                                 rows=rows)
                print("=== bdiff_gen", name, json.dumps({k: v for k, v in gen[name].items() if k != "rows"}), flush=True)
    print("=== bdiff_B1", json.dumps(outB1), flush=True)
    print("=== bdiff_B3", json.dumps(outB3), flush=True)
    save_json(a.out, {"model": a.model, "adapter": a.adapter, "K": K, "n": len(prompts), "gpu": gpu_info(), "seconds": round(time.time() - t0, 1),
                      "B1": outB1, "B3": outB3, "B4": gen})


if __name__ == "__main__":
    main()
