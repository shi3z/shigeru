#!/usr/bin/env python3
"""Phase-1 autoregressive baseline (Qwen3) with Hugging Face transformers, greedy, bf16, no thinking.
Measures TTFT (prefill + first token), decode tokens/s, end-to-end latency, output tokens, peak GPU memory. Saves JSON.
usage: run_ar.py --model /mnt/ssd/models/hf/Qwen3-8B --max-new-tokens 256 --out results/qwen3_8b_hf_256.json"""
import argparse, os, sys, time, threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(__file__))
from common import load_prompts, gpu_info, other_tenants, save_json, summarize

class FirstTokenTimer:
    """records the wall time of the first generated token through a logits processor hook"""
    def __init__(self): self.t_first = None
    def __call__(self, input_ids, scores):
        if self.t_first is None:
            torch.cuda.synchronize(); self.t_first = time.perf_counter()
        return scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/ssd/models/hf/Qwen3-8B")
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__), "prompts.json")); ap.add_argument("--ids", nargs="*")
    ap.add_argument("--max-new-tokens", type=int, default=256); ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--thinking", action="store_true"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to("cuda").eval()
    load_s = time.time() - t_load
    prompts = load_prompts(a.prompts, a.ids)
    warm = tok.apply_chat_template([{"role": "user", "content": "Hello"}], add_generation_prompt=True, tokenize=False, enable_thinking=a.thinking)
    with torch.no_grad():
        model.generate(**tok(warm, return_tensors="pt").to("cuda"), max_new_tokens=16, do_sample=False)
    runs = []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, tokenize=False, enable_thinking=a.thinking)
        enc = tok(text, return_tensors="pt").to("cuda")
        for rep in range(a.repeat):
            torch.cuda.reset_peak_memory_stats(); ft = FirstTokenTimer()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False, logits_processor=[ft])
            torch.cuda.synchronize(); e2e = time.perf_counter() - t0
            gen = out[0, enc.input_ids.shape[1]:].tolist()
            n = len(gen); ttft = (ft.t_first - t0) if ft.t_first else None
            decode_s = e2e - ttft if ttft else None
            r = {"id": p["id"], "lang": p["lang"], "rep": rep, "prompt_tokens": enc.input_ids.shape[1], "effective_tokens": n,
                 "ttft_s": ttft, "e2e_s": e2e, "tok_s_effective": n / e2e, "decode_tok_s": (n - 1) / decode_s if decode_s and n > 1 else None,
                 "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30, "answer": tok.decode(gen, skip_special_tokens=True)}
            runs.append(r)
            print(f"[{p['id']}] {n} tok, ttft {ttft*1000:.0f} ms, e2e {e2e:.2f} s, {r['tok_s_effective']:.1f} tok/s e2e, decode {r['decode_tok_s']:.1f} tok/s | {r['answer'][:70]!r}", flush=True)
    res = {"engine": "hf-transformers-generate", "model": a.model, "config": vars(a), "gpu": gpu_info(), "load_s": load_s, "gpu_state": other_tenants(),
           "runs": runs, "summary": summarize(runs, ["ttft_s", "e2e_s", "tok_s_effective", "decode_tok_s", "effective_tokens", "peak_mem_gib"])}
    save_json(a.out, res)
    s = res["summary"]
    print(f"=== AR {os.path.basename(a.model)}: mean {s['tok_s_effective']['mean']:.1f} tok/s e2e, decode {s['decode_tok_s']['mean']:.1f} tok/s, ttft {s['ttft_s']['mean']*1000:.0f} ms, peak mem {s['peak_mem_gib']['max']:.1f} GiB")

if __name__ == "__main__":
    main()
