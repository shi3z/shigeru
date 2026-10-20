#!/usr/bin/env python3
"""Benchmark the Open Mercury engine (engine/llada.py + engine/sampler.py) with the shared prompts and metrics.
usage: run_ours.py --mode threshold --threshold 0.9 --gen-length 256 --block-length 32 --out results/ours_t0.9_b32.json
       run_ours.py --mode fixed --steps 256 --no-cache ...   (parity with the official sampler)"""
import argparse, os, sys, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer
from engine.llada import LLaDA, EOS_ID
from engine.sampler import generate
from common import load_prompts, gpu_info, other_tenants, save_json, summarize, score, accuracy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/ssd/models/hf/LLaDA-8B-Instruct")
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")); ap.add_argument("--ids", nargs="*")
    ap.add_argument("--mode", choices=["threshold", "fixed"], default="threshold"); ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--steps", type=int, default=None); ap.add_argument("--gen-length", type=int, default=256); ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--no-cache", action="store_true"); ap.add_argument("--repeat", type=int, default=1); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    t_load = time.time(); model = LLaDA.from_pretrained(a.model); load_s = time.time() - t_load
    kw = dict(gen_length=a.gen_length, block_length=a.block_length, steps=a.steps, mode=a.mode, threshold=a.threshold, use_cache=not a.no_cache)
    warm = tok.apply_chat_template([{"role": "user", "content": "Hello"}], add_generation_prompt=True, tokenize=False)
    generate(model, tok(warm, return_tensors="pt").input_ids.cuda(), **{**kw, "gen_length": 32, "block_length": 32, "steps": 32 if a.steps else None})
    runs = []
    for p in load_prompts(a.prompts, a.ids):
        ids = tok(tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, tokenize=False), return_tensors="pt").input_ids.cuda()
        for rep in range(a.repeat):
            st = {}; torch.cuda.reset_peak_memory_stats()
            x = generate(model, ids, stats=st, **kw)
            seq = x[0, ids.shape[1]:].tolist(); eff = seq.index(EOS_ID) if EOS_ID in seq else len(seq)
            answer = tok.decode(seq[:eff], skip_special_tokens=True)
            r = {"id": p["id"], "lang": p["lang"], "rep": rep, "prompt_tokens": ids.shape[1], "gen_length": a.gen_length, "effective_tokens": eff, **st,
                 "tok_s_effective": eff / st["e2e_s"], "tok_s_raw": a.gen_length / st["e2e_s"], "forward_per_s": st["n_forward"] / st["e2e_s"],
                 "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30, "answer": answer, "correct": score(p, answer)}
            runs.append(r)
            print(f"[{p['id']}] eff {eff}/{a.gen_length}, ttft {st['ttft_s']*1000:.0f} ms, e2e {st['e2e_s']:.2f} s, {r['tok_s_effective']:.1f} tok/s (raw {r['tok_s_raw']:.1f}), {st['n_forward']} fwd | {answer[:70]!r}", flush=True)
    res = {"engine": f"open-mercury/{a.mode}{'' if not a.no_cache else '-nocache'}", "model": a.model, "config": vars(a), "gpu": gpu_info(), "load_s": load_s,
           "gpu_state": other_tenants(), "runs": runs, "accuracy": accuracy(runs),
           "summary": summarize(runs, ["ttft_s", "e2e_s", "tok_s_effective", "tok_s_raw", "n_forward", "effective_tokens", "peak_mem_gib"])}
    save_json(a.out, res); s = res["summary"]
    print(f"=== ours {a.mode} thr={a.threshold} steps={a.steps} block={a.block_length} cache={not a.no_cache}: mean {s['tok_s_effective']['mean']:.1f} tok/s effective, {s['tok_s_raw']['mean']:.1f} raw, {s['n_forward']['mean']:.0f} fwd, ttft {s['ttft_s']['mean']*1000:.0f} ms, peak {s['peak_mem_gib']['max']:.1f} GiB, acc {res['accuracy']}")

if __name__ == "__main__":
    main()
