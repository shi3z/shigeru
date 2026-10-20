#!/usr/bin/env python3
"""Open Mercury OpenAI-compatible API server (Phase 4 skeleton) over the block-diffusion engine.

Endpoints: GET /v1/models, POST /v1/chat/completions (stream=true/false, response_format json_object via a system instruction).
Concurrency: requests are queued and served by one engine worker (continuous batching comes later); streaming emits text as
each diffusion block completes (tokens inside a block are finalized in confidence order, so block granularity is the honest unit).
usage: python -m engine.server --model /mnt/ssd/models/hf/LLaDA-8B-Instruct --port 8030 [--graph] [--threshold 0.9]"""
import argparse, asyncio, json, os, sys, time, uuid, threading, queue
import torch
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from transformers import AutoTokenizer
from engine.llada import LLaDA, MASK_ID, EOS_ID
from engine.graphs import GraphRunner, bucket
import torch.nn.functional as F

app = FastAPI()
STATE = {}

def decode_blocks(model, runner, prompt, gen_length, block_length, threshold, use_graph, on_block):
    """threshold block decoding (same algorithm as engine.sampler.generate) that calls on_block(block_token_ids) per finished block
    and stops after an all-EOS block."""
    dev = prompt.device; P = prompt.shape[1]; L = P + gen_length; Lb = bucket(L)
    cache = runner.get_cache(Lb)
    key_mask = torch.zeros(Lb, dtype=torch.bool, device=dev); key_mask[:L] = True
    x = torch.full((1, Lb), EOS_ID, dtype=torch.long, device=dev); x[:, P:L] = MASK_ID; x[:, :P] = prompt
    pos_all = torch.arange(Lb, device=dev); n_fwd = 0
    for nb in range(gen_length // block_length):
        bs, be = P + nb * block_length, P + (nb + 1) * block_length
        logits = runner.run(x, pos_all, key_mask, use_graph=use_graph)[:, bs:be].clone(); n_fwd += 1
        while True:
            masked = (x[:, bs:be] == MASK_ID)
            if not masked.any():
                break
            x0 = logits.argmax(-1); p = torch.gather(F.softmax(logits, -1), -1, x0.unsqueeze(-1)).squeeze(-1)
            conf = torch.where(masked, p, torch.full_like(p, -1.0)); sel = (conf >= threshold) & masked
            if not sel.any():
                sel = torch.zeros_like(masked); sel[0, conf[0].argmax()] = True
            blk = x[:, bs:be]; blk[sel] = x0[sel]; x[:, bs:be] = blk
            if not (x[:, bs:be] == MASK_ID).any():
                break
            logits = runner.run(x[:, bs:be], pos_all[bs:be], key_mask, use_graph=use_graph).clone(); n_fwd += 1
        ids = x[0, bs:be].tolist()
        on_block(ids)
        if all(t == EOS_ID for t in ids) or EOS_ID in ids:
            break
    return n_fwd

def worker():
    tok, model, runner, a = STATE["tok"], STATE["model"], STATE["runner"], STATE["args"]
    while True:
        job = STATE["queue"].get()
        try:
            text = tok.apply_chat_template(job["messages"], add_generation_prompt=True, tokenize=False)
            ids = tok(text, return_tensors="pt").input_ids.cuda()
            gen_length = min(job["max_tokens"], a.max_gen); gen_length = max(a.block_length, (gen_length // a.block_length) * a.block_length)
            t0 = time.perf_counter(); n_tokens = [0]; done_eos = [False]
            def on_block(block_ids):
                if done_eos[0]:
                    return
                eff = block_ids.index(EOS_ID) if EOS_ID in block_ids else len(block_ids)
                piece = tok.decode(block_ids[:eff], skip_special_tokens=True); n_tokens[0] += eff
                if EOS_ID in block_ids:
                    done_eos[0] = True
                job["out"].put(("text", piece))
            n_fwd = decode_blocks(model, runner, ids, gen_length, a.block_length, job["threshold"], a.graph, on_block)
            job["out"].put(("done", {"prompt_tokens": ids.shape[1], "completion_tokens": n_tokens[0], "forwards": n_fwd, "seconds": time.perf_counter() - t0,
                                     "finish_reason": "stop" if done_eos[0] else "length"}))
        except Exception as e:
            job["out"].put(("error", str(e)))

@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": STATE["model_id"], "object": "model", "owned_by": "open-mercury"}]}

@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    messages = body.get("messages", [])
    if (body.get("response_format") or {}).get("type") == "json_object":
        messages = [{"role": "system", "content": "Respond with a single valid JSON object and nothing else."}] + messages
    job = {"messages": messages, "max_tokens": int(body.get("max_tokens") or 256), "threshold": float(body.get("threshold") or STATE["args"].threshold), "out": queue.Queue()}
    STATE["queue"].put(job)
    rid = "chatcmpl-" + uuid.uuid4().hex[:24]; created = int(time.time()); model_id = STATE["model_id"]
    if body.get("stream"):
        async def gen():
            yield "data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_id, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}) + "\n\n"
            while True:
                kind, val = await asyncio.get_event_loop().run_in_executor(None, job["out"].get)
                if kind == "text":
                    yield "data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_id, "choices": [{"index": 0, "delta": {"content": val}, "finish_reason": None}]}) + "\n\n"
                elif kind == "done":
                    yield "data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_id, "choices": [{"index": 0, "delta": {}, "finish_reason": val["finish_reason"]}],
                                                 "usage": {"prompt_tokens": val["prompt_tokens"], "completion_tokens": val["completion_tokens"], "total_tokens": val["prompt_tokens"] + val["completion_tokens"]}, "timings": val}) + "\n\n"
                    yield "data: [DONE]\n\n"; return
                else:
                    yield "data: " + json.dumps({"error": val}) + "\n\n"; return
        return StreamingResponse(gen(), media_type="text/event-stream")
    parts = []
    while True:
        kind, val = await asyncio.get_event_loop().run_in_executor(None, job["out"].get)
        if kind == "text":
            parts.append(val)
        elif kind == "done":
            return {"id": rid, "object": "chat.completion", "created": created, "model": model_id,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)}, "finish_reason": val["finish_reason"]}],
                    "usage": {"prompt_tokens": val["prompt_tokens"], "completion_tokens": val["completion_tokens"], "total_tokens": val["prompt_tokens"] + val["completion_tokens"]}, "timings": val}
        else:
            return JSONResponse({"error": val}, status_code=500)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/ssd/models/hf/LLaDA-8B-Instruct"); ap.add_argument("--port", type=int, default=8030); ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--threshold", type=float, default=0.9); ap.add_argument("--block-length", type=int, default=32); ap.add_argument("--max-gen", type=int, default=1024)
    ap.add_argument("--graph", action="store_true")
    a = ap.parse_args()
    STATE["args"] = a; STATE["model_id"] = os.path.basename(a.model.rstrip("/")) + "-open-mercury"
    STATE["tok"] = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    STATE["model"] = LLaDA.from_pretrained(a.model); STATE["runner"] = GraphRunner(STATE["model"]); STATE["queue"] = queue.Queue()
    threading.Thread(target=worker, daemon=True).start()
    print(f"open-mercury server: {STATE['model_id']} on http://{a.host}:{a.port}/v1 (threshold {a.threshold}, block {a.block_length}, graph {a.graph})", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")

if __name__ == "__main__":
    main()
