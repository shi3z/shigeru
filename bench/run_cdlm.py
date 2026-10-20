"""Benchmark a converted causal-diffusion checkpoint (engine/cdlm) on bench prompts: speed + GSM8K-style accuracy.
Runs in .venv-train. Usage:
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. .venv-train/bin/python bench/run_cdlm.py --model runs/phaseC_pilot_4b/final \
      --prompts bench/prompts_gsm8k50.json --n 10 --tau none,0.4,1.0 --window 16 --max-new-tokens 384 --out results/cdlm_pilot.json
tau=none: one token per step (leftmost mask, predict=mask) — the exactness gate vs the AR student (compare with --ar).
--ar: also decode the same model with plain HF greedy generate (AR rows) for the τ=None comparison.
--engine hf|fast|fast_static|fast_graph: HF DynamicCache decoder (engine/cdlm/decoder.py) or the own engine
(engine/cdlm/fast_model.py + fast_decoder.py; static = bucketed static shapes, graph = CUDA graphs)."""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts, gpu_info, save_json, score
from engine.cdlm.decoder import generate as generate_hf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="bench/prompts_gsm8k50.json")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--tau", default="none,0.4")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--ar", action="store_true")
    ap.add_argument("--engine", default="hf", choices=["hf", "fast", "fast_static", "fast_graph"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    mask_id = getattr(model.config, "cdlm_mask_token_id", None) or tok.convert_tokens_to_ids("<|mask|>")
    stop_ids = [tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|im_end|>")] if "<|im_end|>" in tok.get_vocab() else [])
    prompts = load_prompts(a.prompts)[: a.n]
    if a.engine == "hf":
        generate = generate_hf; dec_model = model
    else:
        from engine.cdlm.fast_model import FastQwen3
        from engine.cdlm.fast_decoder import generate as generate_fast
        dec_model = FastQwen3.from_hf(model, device="cuda", dtype=torch.bfloat16, max_len=4096)
        kw = {"static": a.engine != "fast", "use_graph": a.engine == "fast_graph"}
        generate = lambda m, *args, **k: generate_fast(m, *args, **k, **kw)
        # warm-up (graph capture happens on the first use of each window bucket)
        generate(dec_model, tok, tok("warm up", return_tensors="pt")["input_ids"][0].tolist(), 48, window=a.window, tau=None, mask_id=mask_id, stop_ids=stop_ids)
    configs = [("ar", None)] if a.ar else []
    configs += [(f"tau={t}", None if t == "none" else float(t)) for t in a.tau.split(",")]
    runs = []
    for name, tau in configs:
        for p in prompts:
            ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
            ids = ids.cuda() if torch.is_tensor(ids) else torch.tensor(ids["input_ids"]).cuda()
            torch.cuda.synchronize(); t0 = time.time()
            if name == "ar":
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=a.max_new_tokens, do_sample=False, eos_token_id=stop_ids)
                new = out[0, ids.shape[1]:].tolist(); stats = {"steps": len(new), "forwards": len(new)}
            else:
                new, stats = generate(dec_model, tok, ids[0].tolist(), a.max_new_tokens, window=a.window, tau=tau, lam=a.lam,
                                      mask_id=mask_id, stop_ids=stop_ids, greedy=True)
            torch.cuda.synchronize(); dt = time.time() - t0
            text = tok.decode(new, skip_special_tokens=True)
            r = {"config": name, "id": p["id"], "n_tokens": len(new), "seconds": round(dt, 3), "tok_s": round(len(new) / dt, 1),
                 "steps": stats.get("steps"), "forwards": stats.get("forwards"), "correct": score(p, text) if "answer" in p else None,
                 "text": text}
            runs.append(r)
            print(f"[{name}] {p['id']} {r['n_tokens']} tok in {dt:.1f}s = {r['tok_s']} tok/s, steps {r['steps']}, correct={r['correct']}", flush=True)
    summary = {}
    for name, _ in configs:
        rs = [r for r in runs if r["config"] == name]
        summary[name] = {"tok_s_mean": round(sum(r["tok_s"] for r in rs) / len(rs), 1), "tokens_per_step": round(sum(r["n_tokens"] for r in rs) / max(1, sum(r["steps"] or 0 for r in rs)), 2),
                         "accuracy": (sum(bool(r["correct"]) for r in rs) / len(rs)) if rs and rs[0]["correct"] is not None else None, "n": len(rs)}
    # exactness gate: τ=None vs AR greedy on identical prompts
    if a.ar and any(n == "tau=none" for n, _ in configs):
        ar = {r["id"]: r["text"] for r in runs if r["config"] == "ar"}
        summary["tau_none_equals_ar"] = sum(r["text"] == ar[r["id"]] for r in runs if r["config"] == "tau=none")
    print("=== cdlm", json.dumps(summary), flush=True)
    save_json(a.out, {"model": a.model, "engine": a.engine, "args": vars(a), "gpu": gpu_info(), "runs": runs, "summary": summary})


if __name__ == "__main__":
    main()
