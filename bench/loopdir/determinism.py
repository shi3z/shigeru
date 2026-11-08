"""Is the llama.cpp path in the GSM8K runs actually greedy?

Everything about the loops looked trajectory-dependent: the same question loops in one model and not the other, the
sets swap when the quantization changes, and the same model in bf16 through transformers with a plain argmax barely
loops at all.  Sampling would explain all three at once, and "temperature: 0" in the request body is an assumption
until it is tested.

Twice through the same server, same prompts, same body: identical output means greedy, divergence means the server
sampled.  Also reports what the server says its sampler is.
"""
import json, subprocess, time, urllib.request

MODEL = "/home/shi3z/models/gguf/ds9b/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf"
BIN = "/home/shi3z/bench-gguf/llama.cpp/build/bin/llama-server"
PORT = 18244
N = 12


def ask(prompt, extra=None):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 400, "temperature": 0, "stream": False}
    if extra:
        body.update(extra)
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    m = d["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + (m.get("content") or "")


p = subprocess.Popen([BIN, "-m", MODEL, "--port", str(PORT), "-ngl", "99", "-c", "8192", "-t", "16",
                      "--host", "127.0.0.1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read()
            break
        except Exception:
            time.sleep(1)
    props = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=10).read())
    sp = props.get("default_generation_settings", {}).get("params", props.get("default_generation_settings", {}))
    print("server sampler defaults:", json.dumps({k: sp.get(k) for k in
          ("temperature", "top_k", "top_p", "min_p", "typ_p", "repeat_penalty", "repeat_last_n", "seed",
           "dynatemp_range", "mirostat", "samplers")}, ensure_ascii=False)[:400], flush=True)

    gsm = json.load(open("/home/shi3z/shigeru/open-mercury/bench/prompts_gsm8k_full.json"))[:N]
    same_default = same_forced = 0
    for q in gsm:
        a1 = ask(q["prompt"])
        a2 = ask(q["prompt"])
        ok = a1 == a2
        same_default += ok
        # the same question with the sampler nailed down explicitly
        b1 = ask(q["prompt"], {"top_k": 1, "seed": 1234, "repeat_penalty": 1.0})
        b2 = ask(q["prompt"], {"top_k": 1, "seed": 1234, "repeat_penalty": 1.0})
        ok2 = b1 == b2
        same_forced += ok2
        if not ok:
            i = next((j for j in range(min(len(a1), len(a2))) if a1[j] != a2[j]), min(len(a1), len(a2)))
            print(f"  {q['id']}: temperature-0 runs DIVERGE at char {i}; "
                  f"...{a1[max(0,i-40):i+40]!r} vs ...{a2[max(0,i-40):i+40]!r}", flush=True)
        else:
            print(f"  {q['id']}: temperature-0 identical ({len(a1)} chars), top_k=1 identical={ok2}", flush=True)
    print(f"\n=== temperature 0 alone: {same_default}/{N} runs reproducible")
    print(f"=== top_k 1 + seed + no repeat penalty: {same_forced}/{N} reproducible")
finally:
    p.kill(); p.wait()
