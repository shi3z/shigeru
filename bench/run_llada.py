#!/usr/bin/env python3
"""Phase-1 baseline: LLaDA (masked diffusion LM) with the official sampling loop, instrumented for timing.
Measures per prompt: time to first unmasked token (TTFT analog), end-to-end latency, forward passes, time per forward,
effective output tokens (up to the first EOS), tokens/s (effective and raw), peak GPU memory. Saves JSON.
usage: run_llada.py --model /mnt/ssd/models/hf/LLaDA-8B-Instruct --gen-length 256 --steps 256 --block-length 32 --out results/llada_g256_s256_b32.json"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
sys.path.insert(0, os.path.dirname(__file__))
from common import load_prompts, gpu_info, other_tenants, save_json, summarize

MASK_ID = 126336
EOS_ID = 126081

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num[i, :remainder[i]] += 1
    return num

@torch.no_grad()
def generate_timed(model, prompt, steps, gen_length, block_length, temperature=0.0, remasking="low_confidence", sync_each_step=True):
    """The official LLaDA sampler (semi-autoregressive blocks, low-confidence remasking) with timing hooks.
    Returns (x, timing dict). Timing uses cuda synchronize per step when sync_each_step is set."""
    x = torch.full((1, prompt.shape[1] + gen_length), MASK_ID, dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    step_times, n_forward, t_first = [], 0, None
    t0 = time.perf_counter()
    for nb in range(num_blocks):
        bs, be = prompt.shape[1] + nb * block_length, prompt.shape[1] + (nb + 1) * block_length
        block_mask = (x[:, bs:be] == MASK_ID)
        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)
        for i in range(steps_per_block):
            ts = time.perf_counter()
            mask_index = (x == MASK_ID)
            logits = model(x).logits
            n_forward += 1
            if temperature > 0:
                logits64 = logits.to(torch.float64)
                noise = torch.rand_like(logits64)
                logits_noisy = logits64.exp() / ((-torch.log(noise)) ** temperature)
                x0 = torch.argmax(logits_noisy, dim=-1)
            else:
                x0 = torch.argmax(logits, dim=-1)
            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
            else:
                x0_p = torch.rand(x0.shape, device=x0.device)
            x0_p[:, be:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            transfer = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                _, sel = torch.topk(confidence[j], k=int(num_transfer[j, i]))
                transfer[j, sel] = True
            x[transfer] = x0[transfer]
            if sync_each_step:
                torch.cuda.synchronize()
            te = time.perf_counter()
            if t_first is None:
                t_first = te - t0
            step_times.append(te - ts)
    torch.cuda.synchronize()
    return x, {"ttft_s": t_first, "e2e_s": time.perf_counter() - t0, "n_forward": n_forward,
               "step_ms_mean": 1000 * sum(step_times) / len(step_times), "step_ms_first": 1000 * step_times[0]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/ssd/models/hf/LLaDA-8B-Instruct")
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__), "prompts.json"))
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--gen-length", type=int, default=256); ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--block-length", type=int, default=32); ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--remasking", default="low_confidence"); ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    t_load = time.time()
    model = AutoModel.from_pretrained(a.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
    load_s = time.time() - t_load
    prompts = load_prompts(a.prompts, a.ids)
    # warmup (compiles kernels, fills caches): one short generation
    warm = tok.apply_chat_template([{"role": "user", "content": "Hello"}], add_generation_prompt=True, tokenize=False)
    generate_timed(model, tok(warm, return_tensors="pt").input_ids.to("cuda"), steps=8, gen_length=32, block_length=32)
    runs = []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt").input_ids.to("cuda")
        for rep in range(a.repeat):
            torch.cuda.reset_peak_memory_stats()
            x, tm = generate_timed(model, ids, a.steps, a.gen_length, a.block_length, a.temperature, a.remasking)
            out = x[0, ids.shape[1]:].tolist()
            eff = out.index(EOS_ID) if EOS_ID in out else len(out)
            answer = tok.decode(out[:eff], skip_special_tokens=True)
            r = {"id": p["id"], "lang": p["lang"], "rep": rep, "prompt_tokens": ids.shape[1], "gen_length": a.gen_length,
                 "effective_tokens": eff, **tm,
                 "tok_s_effective": eff / tm["e2e_s"], "tok_s_raw": a.gen_length / tm["e2e_s"],
                 "forward_per_s": tm["n_forward"] / tm["e2e_s"], "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
                 "answer": answer}
            runs.append(r)
            print(f"[{p['id']}] eff {eff}/{a.gen_length} tok, ttft {tm['ttft_s']*1000:.0f} ms, e2e {tm['e2e_s']:.2f} s, {r['tok_s_effective']:.1f} tok/s (raw {r['tok_s_raw']:.1f}), {tm['n_forward']} fwd @ {tm['step_ms_mean']:.1f} ms | {answer[:70]!r}", flush=True)
    res = {"engine": "llada-official-sampler", "model": a.model, "config": vars(a), "gpu": gpu_info(), "load_s": load_s,
           "gpu_state": other_tenants(), "runs": runs,
           "summary": summarize(runs, ["ttft_s", "e2e_s", "tok_s_effective", "tok_s_raw", "step_ms_mean", "effective_tokens", "peak_mem_gib"])}
    save_json(a.out, res)
    s = res["summary"]
    print(f"=== LLaDA gen={a.gen_length} steps={a.steps} block={a.block_length}: mean {s['tok_s_effective']['mean']:.1f} tok/s effective, {s['tok_s_raw']['mean']:.1f} raw, ttft {s['ttft_s']['mean']*1000:.0f} ms, step {s['step_ms_mean']['mean']:.1f} ms, peak mem {s['peak_mem_gib']['max']:.1f} GiB")

if __name__ == "__main__":
    main()
