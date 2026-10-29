"""GSM8K-style prompts through an OpenAI-compatible chat API (llama-server, SGLang, dsv41), greedy, one request at a
time, timing from the client: used for external comparison rows (Clover INT4 on llama.cpp, 2026-10-29).
  .venv/bin/python bench/run_api.py --base-url http://127.0.0.1:8080/v1 --model clover --prompts bench/prompts_gsm8k_dev2.json \
      --n 150 --max-tokens 384 --out results/clover_int4_dev2.json"""
import argparse, json, os, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import load_prompts, score, save_json


def chat(base_url, model, prompt, max_tokens, temperature, extra, timeout):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": temperature, "stream": False, **extra}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    dt = time.time() - t0
    ch = d["choices"][0]
    text = (ch.get("message") or {}).get("content") or ""
    usage = d.get("usage") or {}
    return text, dt, usage, ch.get("finish_reason"), d.get("timings")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="default")
    ap.add_argument("--prompts", default="bench/prompts_gsm8k_dev2.json")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--extra", default="{}", help="extra JSON merged into the request body")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--tag", default="api")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    extra = json.loads(a.extra)
    prompts = load_prompts(a.prompts)[: a.n]
    runs = []
    for p in prompts:
        try:
            text, dt, usage, finish, timings = chat(a.base_url, a.model, p["prompt"], a.max_tokens, a.temperature, extra, a.timeout)
        except Exception as e:
            print(f"{p['id']} ERROR {e}", flush=True)
            runs.append({"id": p["id"], "error": str(e)})
            continue
        n_out = usage.get("completion_tokens")
        r = {"id": p["id"], "n_tokens": n_out, "seconds": round(dt, 3), "tok_s": round(n_out / dt, 1) if n_out else None,
             "finish_reason": finish, "correct": score(p, text) if "answer" in p else None, "text": text, "usage": usage}
        if timings:  # llama-server reports predicted_per_second etc.
            r["server_timings"] = dict(timings)
        runs.append(r)
        print(f"[{a.tag}] {p['id']} {n_out} tok in {dt:.1f}s = {r['tok_s']} tok/s, correct={r['correct']}", flush=True)
    ok = [r for r in runs if "error" not in r]
    summary = {"n": len(runs), "errors": len(runs) - len(ok),
               "accuracy": (sum(bool(r["correct"]) for r in ok) / len(ok)) if ok and ok[0]["correct"] is not None else None,
               "tok_s_mean": round(sum(r["tok_s"] for r in ok if r["tok_s"]) / max(1, sum(1 for r in ok if r["tok_s"])), 1),
               "tokens_mean": round(sum(r["n_tokens"] or 0 for r in ok) / max(1, len(ok)), 1),
               "seconds_total": round(sum(r["seconds"] for r in ok), 1)}
    srv = [r["server_timings"].get("predicted_per_second") for r in ok if r.get("server_timings")]
    if srv and all(s is not None for s in srv):
        summary["server_predicted_tok_s_mean"] = round(sum(srv) / len(srv), 1)
    print("=== api", json.dumps(summary), flush=True)
    save_json(a.out, {"base_url": a.base_url, "model": a.model, "args": vars(a), "runs": runs, "summary": summary})


if __name__ == "__main__":
    main()
