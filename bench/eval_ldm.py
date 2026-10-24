"""SHIGERU-LDM evaluation (teacher-forced, the protocol of bench/diag_pffp.py so the numbers are directly comparable
with masked diffusion): for every block of K reference tokens after a clean prefix,
  oracle          : decode(codec(gold))                                   -> Phase H1 reconstruction of the projection
  noisy oracle    : z_t = (1-t) eps + t codec(gold), denoise from t -> 1, decode   (t = 0.3, 0.6)    -> Phase H2
  generated       : from pure noise with S Euler steps (S in --steps), decode                       -> Phase H4
  no-cond control : generated with the condition zeroed (what the prior alone gives)
Metrics per setting: position accuracy (and by distance), multiset F1 (matching), duplicate rate, block exact match,
forwards of the denoiser per block.  Data: GSM8K dev (own AR references, --refs) + genre files as diag_pffp.
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-train/bin/python bench/eval_ldm.py --model runs/phaseC_sft3_4b/final \
      --ckpt runs/ldm_pilot_k16/denoiser.pt --refs results/pure/dev_refs_sft3.json --n 50 --out results/pure/ldm_pilot_k16_eval.json"""
import argparse, collections, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts, gpu_info, save_json
from bench.diag_pffp import GENRE_OF, max_matching, multiset_inter
from engine.ldm.codec import EmbedCodec
from engine.ldm.denoiser import LatentDenoiser, sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompts", default="bench/prompts_gsm8k_dev.json")
    ap.add_argument("--refs", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--genre-files", nargs="*", default=["data/distill/phaseA_medium_test.jsonl", "data/distill/phaseA_medium_val.jsonl",
                                                          "data/distill/phaseA_codecomp_test.jsonl"])
    ap.add_argument("--steps", default="16,8,4,2")
    ap.add_argument("--method", default="euler")
    ap.add_argument("--max-blocks", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    codec = EmbedCodec(lm.get_input_embeddings().weight, d_lat=ck["codec"]["d_lat"], seed=ck["codec"]["seed"], exclude=ck["codec"]["exclude"]).to(dev)
    den = LatentDenoiser(**ck["hparams"]).to(dev).eval()
    den.load_state_dict(ck["denoiser"])
    K = den.K
    cond_max = ck["args"].get("cond_max", 768)
    items = []
    prompts = load_prompts(a.prompts)[: a.n]
    refs = json.load(open(a.refs))["refs"]
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        ids = ids[0].tolist() if torch.is_tensor(ids) else list(ids["input_ids"][0])
        items.append(("gsm8k", ids, refs[p["id"]]["response_ids"]))
    for f in a.genre_files:
        for line in open(f):
            ex = json.loads(line)
            g = GENRE_OF.get(ex.get("task_category"))
            if not g:
                continue
            text = tok.apply_chat_template([{"role": "user", "content": ((ex["context"] + "\n\n") if ex.get("context") else "") + ex["instruction"]}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False)
            p_ids = tok(text, add_special_tokens=False)["input_ids"]
            r_ids = tok(ex["response"] + tok.eos_token, add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(r_ids) > a.max_len or len(r_ids) < K:
                continue
            items.append((g, p_ids, r_ids))
    steps_list = [int(s) for s in a.steps.split(",")]
    settings = ["oracle", "noisy_t0.6", "noisy_t0.3"] + [f"gen_S{s}" for s in steps_list] + [f"nocond_S{steps_list[0]}"]
    agg = {}

    def A(genre, name):
        return agg.setdefault(genre, {}).setdefault(name, dict(blocks=0, pos_ok=[0] * K, inter=0, exact=0, dup=0.0, fwd=0))

    def score(genre, name, pred, gold, fwd):
        c = A(genre, name)
        c["blocks"] += 1; c["fwd"] += fwd
        for i in range(K):
            c["pos_ok"][i] += int(pred[i] == gold[i])
        c["inter"] += multiset_inter(pred, gold)
        c["exact"] += int(pred == gold)
        c["dup"] += 1.0 - len(set(pred)) / K

    g_cpu = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    for n, (genre, p_ids, r_ids) in enumerate(items):
        full = torch.tensor([p_ids + r_ids], dtype=torch.long, device=dev)
        P = len(p_ids)
        with torch.no_grad():
            hid = lm.model(input_ids=full).last_hidden_state[0]                      # [L, D]
        starts = list(range(P, P + len(r_ids) - K + 1, K))[: a.max_blocks]
        for s in starts:
            gold = full[0, s:s + K].tolist()
            cond = hid[max(0, s - cond_max): s][None]
            pad = torch.zeros((1, cond.shape[1]), dtype=torch.bool, device=dev)
            z0 = codec.encode(full[0, s:s + K])[None]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                score(genre, "oracle", codec.decode(z0)[0].tolist(), gold, 0)
                for t in (0.6, 0.3):
                    eps = torch.randn(z0.shape, generator=g_cpu).to(dev)
                    z_t = (1 - t) * eps + t * z0
                    S = max(2, int(round(steps_list[0] * (1 - t))))
                    z = sample(den, cond, pad, K, codec.d_lat, steps=S, method=a.method, z_t=z_t, from_t=t)
                    score(genre, f"noisy_t{t}", codec.decode(z)[0].tolist(), gold, S)
                for S in steps_list:
                    z = sample(den, cond, pad, K, codec.d_lat, steps=S, method=a.method, generator=g_cpu, device=dev)
                    score(genre, f"gen_S{S}", codec.decode(z)[0].tolist(), gold, S)
                z = sample(den, torch.zeros_like(cond), pad, K, codec.d_lat, steps=steps_list[0], method=a.method, generator=g_cpu, device=dev)
                score(genre, f"nocond_S{steps_list[0]}", codec.decode(z)[0].tolist(), gold, steps_list[0])
        if n % 25 == 0:
            print(f"{n}/{len(items)} {genre} {time.time() - t0:.0f}s", flush=True)
    summary = {}
    for genre, per in agg.items():
        summary[genre] = {}
        for name in settings:
            c = per.get(name)
            if not c:
                continue
            nt = c["blocks"] * K
            summary[genre][name] = dict(blocks=c["blocks"], position_acc=round(sum(c["pos_ok"]) / nt, 4),
                                        by_distance=[round(x / c["blocks"], 3) for x in c["pos_ok"]],
                                        multiset_f1=round(c["inter"] / nt, 4), block_exact=round(c["exact"] / c["blocks"], 4),
                                        duplicate_rate=round(c["dup"] / c["blocks"], 4), denoiser_forwards_per_block=round(c["fwd"] / c["blocks"], 2))
    print("=== ldm", json.dumps({g: {k: {kk: v[kk] for kk in ("position_acc", "multiset_f1", "block_exact", "duplicate_rate")} for k, v in per.items()} for g, per in summary.items()}), flush=True)
    save_json(a.out, {"model": a.model, "ckpt": a.ckpt, "ckpt_step": ck.get("step"), "hparams": ck["hparams"], "train_args": ck["args"], "K": K,
                      "n_items": collections.Counter(g for g, _, _ in items), "gpu": gpu_info(), "seconds": round(time.time() - t0, 1), "summary": summary})


if __name__ == "__main__":
    main()
