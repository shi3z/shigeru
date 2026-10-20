#!/usr/bin/env python3
"""Phase-1b: Fast-dLLM v2 7B (Qwen2.5-7B-Instruct adapted to block diffusion, block 32, sub-block 8, hierarchical cache)
through its Hugging Face remote `generate`. Same prompts and metrics as the other runners; forward passes counted by wrapping
the model. usage: run_fastdllm_v2.py --max-new-tokens 256 --threshold 0.9 --small-block-size 8 --out results/fastdllm_v2_t0.9.json"""
import argparse, os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_prompts, gpu_info, other_tenants, save_json, summarize

STOP_ID = 151645  # <|im_end|>

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/ssd/models/hf/Fast_dLLM_v2_7B")
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")); ap.add_argument("--ids", nargs="*")
    ap.add_argument("--max-new-tokens", type=int, default=256); ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--small-block-size", type=int, default=8); ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--block-cache", action="store_true"); ap.add_argument("--repeat", type=int, default=1); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    load_s = time.time() - t_load
    state = {"n": 0, "t_first": None}
    _fwd = model.forward
    def fwd(*args, **kw):
        out = _fwd(*args, **kw)
        if state["t_first"] is None:
            torch.cuda.synchronize(); state["t_first"] = time.perf_counter()
        state["n"] += 1
        return out
    model.forward = fwd
    def chat(p):
        return tok.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    kw = dict(tokenizer=tok, threshold=a.threshold, small_block_size=a.small_block_size, block_size=a.block_size, temperature=0, use_block_cache=a.block_cache)
    with torch.inference_mode():
        model.generate(tok([chat("Hello")], return_tensors="pt").input_ids.to("cuda"), max_new_tokens=32, **kw)
    runs = []
    for p in load_prompts(a.prompts, a.ids):
        ids = tok([chat(p["prompt"])], return_tensors="pt").input_ids.to("cuda")
        for rep in range(a.repeat):
            state.update(n=0, t_first=None); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(ids, max_new_tokens=a.max_new_tokens, **kw)
            torch.cuda.synchronize(); e2e = time.perf_counter() - t0
            seq = out[0, ids.shape[1]:].tolist()
            eff = seq.index(STOP_ID) if STOP_ID in seq else len(seq)
            answer = tok.decode(seq[:eff], skip_special_tokens=True)
            r = {"id": p["id"], "lang": p["lang"], "rep": rep, "prompt_tokens": ids.shape[1], "gen_length": a.max_new_tokens, "effective_tokens": eff,
                 "ttft_s": state["t_first"] - t0, "e2e_s": e2e, "n_forward": state["n"], "tok_s_effective": eff / e2e, "tok_s_raw": len(seq) / e2e,
                 "forward_per_s": state["n"] / e2e, "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30, "answer": answer}
            runs.append(r)
            print(f"[{p['id']}] eff {eff}/{len(seq)}, ttft {r['ttft_s']*1000:.0f} ms, e2e {e2e:.2f} s, {r['tok_s_effective']:.1f} tok/s, {state['n']} fwd | {answer[:70]!r}", flush=True)
    res = {"engine": "fast-dllm-v2-hf-generate", "model": a.model, "config": vars(a), "gpu": gpu_info(), "load_s": load_s, "gpu_state": other_tenants(),
           "runs": runs, "summary": summarize(runs, ["ttft_s", "e2e_s", "tok_s_effective", "tok_s_raw", "n_forward", "effective_tokens", "peak_mem_gib"])}
    save_json(a.out, res); s = res["summary"]
    print(f"=== Fast-dLLM v2 thr={a.threshold} sub={a.small_block_size} cache={a.block_cache}: mean {s['tok_s_effective']['mean']:.1f} tok/s effective, {s['n_forward']['mean']:.0f} fwd, ttft {s['ttft_s']['mean']*1000:.0f} ms, peak {s['peak_mem_gib']['max']:.1f} GiB")

if __name__ == "__main__":
    main()
