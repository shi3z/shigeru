#!/usr/bin/env python3
"""Phase-1: LLaDA2.1-mini (16B MoE, 1.4B active) through its Hugging Face remote `generate` (block diffusion with confidence
threshold and token editing). Same prompts and metrics as the other runners; forward passes counted by wrapping the model.
usage: run_llada2.py --threshold 0.5 --editing-threshold 0 --gen-length 256 --out results/llada2_mini_t0.5.json"""
import argparse, os, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_prompts, gpu_info, other_tenants, save_json, summarize, score, accuracy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/ssd/models/hf/LLaDA2.1-mini")
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")); ap.add_argument("--ids", nargs="*")
    ap.add_argument("--gen-length", type=int, default=256); ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.5); ap.add_argument("--editing-threshold", type=float, default=0.0)
    ap.add_argument("--repeat", type=int, default=1); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
    load_s = time.time() - t_load
    eos = tok.eos_token_id
    state = {"n": 0, "t_first": None}; _fwd = model.forward
    def fwd(*args, **kw):
        out = _fwd(*args, **kw)
        if state["t_first"] is None:
            torch.cuda.synchronize(); state["t_first"] = time.perf_counter()
        state["n"] += 1; return out
    model.forward = fwd
    kw = dict(eos_early_stop=True, gen_length=a.gen_length, block_length=a.block_length, threshold=a.threshold, editing_threshold=a.editing_threshold, temperature=0.0)
    with torch.inference_mode():
        model.generate(inputs=tok.apply_chat_template([{"role": "user", "content": "Hello"}], add_generation_prompt=True, tokenize=True, return_tensors="pt").to("cuda"), **{**kw, "gen_length": 32})
    runs = []
    for p in load_prompts(a.prompts, a.ids):
        ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, tokenize=True, return_tensors="pt").to("cuda")
        for rep in range(a.repeat):
            state.update(n=0, t_first=None); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(inputs=ids, **kw)
            torch.cuda.synchronize(); e2e = time.perf_counter() - t0
            seq = out[0].tolist()
            if len(seq) > ids.shape[1] and seq[:ids.shape[1]] == ids[0].tolist():
                seq = seq[ids.shape[1]:]
            eff = seq.index(eos) if eos in seq else len(seq)
            answer = tok.decode(seq[:eff], skip_special_tokens=True)
            r = {"id": p["id"], "lang": p["lang"], "rep": rep, "prompt_tokens": ids.shape[1], "gen_length": a.gen_length, "effective_tokens": eff,
                 "ttft_s": state["t_first"] - t0, "e2e_s": e2e, "n_forward": state["n"], "tok_s_effective": eff / e2e, "tok_s_raw": len(seq) / e2e,
                 "forward_per_s": state["n"] / e2e, "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30, "answer": answer, "correct": score(p, answer)}
            runs.append(r)
            print(f"[{p['id']}] eff {eff}/{len(seq)}, ttft {r['ttft_s']*1000:.0f} ms, e2e {e2e:.2f} s, {r['tok_s_effective']:.1f} tok/s, {state['n']} fwd | {answer[:70]!r}", flush=True)
    res = {"engine": "llada2-hf-generate", "model": a.model, "config": vars(a), "gpu": gpu_info(), "load_s": load_s, "gpu_state": other_tenants(),
           "runs": runs, "accuracy": accuracy(runs), "summary": summarize(runs, ["ttft_s", "e2e_s", "tok_s_effective", "tok_s_raw", "n_forward", "effective_tokens", "peak_mem_gib"])}
    save_json(a.out, res); s = res["summary"]
    print(f"=== LLaDA2.1-mini thr={a.threshold} edit={a.editing_threshold} block={a.block_length}: mean {s['tok_s_effective']['mean']:.1f} tok/s effective, {s['n_forward']['mean']:.0f} fwd, ttft {s['ttft_s']['mean']*1000:.0f} ms, peak {s['peak_mem_gib']['max']:.1f} GiB, acc {res['accuracy']}")

if __name__ == "__main__":
    main()
