#!/usr/bin/env python3
"""Phase-1: Fast-dLLM v1 (NVlabs) samplers for LLaDA-8B-Instruct — prefix KV cache, dual cache, confidence-threshold
parallel decoding — on the same prompts and metrics as run_llada.py. Forward passes are counted by wrapping the model.
usage: run_fastdllm_v1.py --mode dual_cache --threshold 0.9 --gen-length 256 --block-length 32 --out results/fastdllm_v1_dual_t0.9.json"""
import argparse, os, sys, time
import torch
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "Fast-dLLM", "v1", "llada")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM          # Fast-dLLM's KV-cache-enabled LLaDA
import generate as fd                                   # Fast-dLLM v1 samplers
from common import load_prompts, gpu_info, other_tenants, save_json, summarize, score, accuracy

EOS_ID = 126081

class Counter:
    """wraps model.forward to count forward passes and timestamp the first one"""
    def __init__(self, model):
        self.model, self.n, self.t_first, self.t0 = model, 0, None, None
        self._fwd = model.forward
        def fwd(*a, **k):
            out = self._fwd(*a, **k)
            if self.t_first is None:
                torch.cuda.synchronize(); self.t_first = time.perf_counter()
            self.n += 1
            return out
        model.forward = fwd
    def reset(self): self.n, self.t_first = 0, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/ssd/models/hf/LLaDA-8B-Instruct")
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")); ap.add_argument("--ids", nargs="*")
    ap.add_argument("--mode", choices=["vanilla", "prefix_cache", "dual_cache"], default="dual_cache")
    ap.add_argument("--gen-length", type=int, default=256); ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--block-length", type=int, default=32); ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--factor", type=float, default=None); ap.add_argument("--repeat", type=int, default=1); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    t_load = time.time()
    model = LLaDAModelLM.from_pretrained(a.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
    load_s = time.time() - t_load
    gen = {"vanilla": fd.generate, "prefix_cache": fd.generate_with_prefix_cache, "dual_cache": fd.generate_with_dual_cache}[a.mode]
    kw = dict(steps=a.steps, gen_length=a.gen_length, block_length=a.block_length, temperature=0.0, remasking="low_confidence",
              threshold=a.threshold, factor=a.factor)
    cnt = Counter(model)
    warm = tok.apply_chat_template([{"role": "user", "content": "Hello"}], add_generation_prompt=True, tokenize=False)
    with torch.inference_mode():
        gen(model, tok(warm, return_tensors="pt").input_ids.to("cuda"), **{**kw, "steps": 32, "gen_length": 32, "block_length": 32})
    runs = []
    for p in load_prompts(a.prompts, a.ids):
        text = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt").input_ids.to("cuda")
        for rep in range(a.repeat):
            cnt.reset(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.inference_mode():
                out = gen(model, ids, **kw)
            torch.cuda.synchronize(); e2e = time.perf_counter() - t0
            x = out[0] if isinstance(out, tuple) else out
            seq = x[0, ids.shape[1]:].tolist()
            eff = seq.index(EOS_ID) if EOS_ID in seq else len(seq)
            answer = tok.decode(seq[:eff], skip_special_tokens=True)
            r = {"id": p["id"], "lang": p["lang"], "rep": rep, "prompt_tokens": ids.shape[1], "gen_length": a.gen_length, "effective_tokens": eff,
                 "ttft_s": cnt.t_first - t0, "e2e_s": e2e, "n_forward": cnt.n, "tok_s_effective": eff / e2e, "tok_s_raw": a.gen_length / e2e,
                 "forward_per_s": cnt.n / e2e, "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30, "answer": answer, "correct": score(p, answer)}
            runs.append(r)
            print(f"[{p['id']}] eff {eff}/{a.gen_length}, ttft {r['ttft_s']*1000:.0f} ms, e2e {e2e:.2f} s, {r['tok_s_effective']:.1f} tok/s (raw {r['tok_s_raw']:.1f}), {cnt.n} fwd | {answer[:70]!r}", flush=True)
    res = {"engine": f"fast-dllm-v1/{a.mode}", "model": a.model, "config": vars(a), "gpu": gpu_info(), "load_s": load_s, "gpu_state": other_tenants(),
           "runs": runs, "accuracy": accuracy(runs), "summary": summarize(runs, ["ttft_s", "e2e_s", "tok_s_effective", "tok_s_raw", "n_forward", "effective_tokens", "peak_mem_gib"])}
    save_json(a.out, res); s = res["summary"]
    print(f"=== Fast-dLLM v1 {a.mode} thr={a.threshold} factor={a.factor} gen={a.gen_length} block={a.block_length}: mean {s['tok_s_effective']['mean']:.1f} tok/s effective, {s['tok_s_raw']['mean']:.1f} raw, {s['n_forward']['mean']:.0f} fwd, ttft {s['ttft_s']['mean']*1000:.0f} ms, peak {s['peak_mem_gib']['max']:.1f} GiB")

if __name__ == "__main__":
    main()
