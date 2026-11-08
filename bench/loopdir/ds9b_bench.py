"""What a Heretic-style weight ablation costs in capability, measured on the model and its un-ablated original.

DeepSeek-V4-Pro-Qwen3.5-9B (a reasoning distillation into Qwen3.5-9B, ~250K math/STEM samples) against the same
model after Heretic v2.0.0 Arbitrary-Rank Ablation on layers 9-24, both as mradermacher's i1-Q4_K_M GGUF, served by
llama.cpp's OpenAI-compatible server on one RTX PRO 6000.

Measured per model: GSM8K accuracy (the model's advertised strength, greedy, "#### <number>" on the last line),
tokens generated per answer (an ablation that breaks the reasoning shows up as longer or looping output), the rate of
answers that repeat themselves, and a set of Japanese prompts printed side by side for reading.

This measures capability, not refusal behaviour.
"""
import argparse, json, re, subprocess, time, urllib.request

JA = [
    ("指示追従", "日本の梅雨について、小学生にもわかる言葉で、箇条書きを3つだけ、各行30字以内で説明してください。"),
    ("推論", "ある店ではりんご3個とみかん5個で960円、りんご5個とみかん3個で1120円です。りんご1個とみかん1個の値段を求め、考え方も日本語で説明してください。"),
    ("要約", "次の文章を40字以内の日本語1文で要約してください。『近年、都市部では高齢化と人口減少が同時に進み、空き家が増加している。空き家は防犯や景観の問題を引き起こすため、自治体は改修費の補助や移住者への貸し出しなどの対策を進めているが、所有者が不明な物件が多く、手続きが難航している。』"),
    ("文体変換", "次の文を、取引先に送るメール向けの丁寧な敬語に書き換えてください。『資料まだもらってないので早く送ってほしい。明日の会議で使うから。』"),
    ("STEM", "内部抵抗0.5Ωの乾電池（起電力1.5V）に2Ωの抵抗をつなぎました。流れる電流と抵抗での消費電力を求め、計算過程も日本語で示してください。"),
]


def serve(binary, model, port, ngl=99, ctx=8192, extra=()):
    p = subprocess.Popen([binary, "-m", model, "--port", str(port), "-ngl", str(ngl), "-c", str(ctx),
                          "-t", "16", "--host", "127.0.0.1", *extra],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
            return p
        except Exception:
            if p.poll() is not None:
                raise RuntimeError("server exited")
            time.sleep(1)
    p.kill()
    raise RuntimeError("server did not come up")


def ask(port, prompt, max_tokens=2048, temperature=0.0):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "temperature": temperature, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    dt = time.time() - t0
    msg = d["choices"][0]["message"]
    text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    u = d.get("usage") or {}
    return text, {"seconds": round(dt, 2), "completion_tokens": u.get("completion_tokens"),
                  "prompt_tokens": u.get("prompt_tokens"),
                  "tok_s": round((u.get("completion_tokens") or 0) / dt, 1) if dt else None}


def looping(text, n=12, times=4):
    """A crude repetition detector: any n-word window that occurs `times` or more."""
    w = text.split()
    if len(w) < n * times:
        return False
    seen = {}
    for i in range(len(w) - n):
        k = " ".join(w[i:i + n])
        seen[k] = seen.get(k, 0) + 1
        if seen[k] >= times:
            return True
    return False


def final_number(text):
    """The number after the last '####', else the last number in the text.  Commas are thousands separators, so a
    match has to contain a digit: a bare ',' used to survive the regex and strip down to an empty string."""
    for pat in (r"####\s*(-?\d[\d,]*(?:\.\d+)?)", r"(-?\d[\d,]*(?:\.\d+)?)"):
        m = re.findall(pat, text)
        if m:
            v = m[-1].replace(",", "").rstrip(".")
            try:
                float(v)
                return v
            except ValueError:
                continue
    return None


def score(got, want):
    if got is None:
        return False
    try:
        return abs(float(got) - float(want)) < 1e-6
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="/home/shi3z/bench-gguf/llama.cpp/build/bin/llama-server")
    ap.add_argument("--models", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--gsm8k", default="/home/shi3z/shigeru/open-mercury/bench/prompts_gsm8k_dev2.json")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--port", type=int, default=18231)
    ap.add_argument("--out", default="/home/shi3z/bench-ds9b/results.json")
    a = ap.parse_args()
    gsm = json.load(open(a.gsm8k))[: a.n]
    res = {}
    for spec in a.models:
        name, path = spec.split("=", 1)
        print(f"\n######## {name}", flush=True)
        p = serve(a.binary, path, a.port)
        try:
            ok = loops = 0
            toks, secs, rows = [], [], []
            for i, q in enumerate(gsm):
                text, st = ask(a.port, q["prompt"])
                got = final_number(text)
                hit = score(got, q["answer"])
                lp = looping(text)
                ok += hit; loops += lp
                toks.append(st["completion_tokens"] or 0); secs.append(st["seconds"])
                rows.append({"id": q["id"], "hit": hit, "got": got, "want": q["answer"], "loop": lp, **st})
                if (i + 1) % 10 == 0:
                    print(f"  {name} {i+1}/{len(gsm)} acc {ok/(i+1):.3f} loops {loops} "
                          f"mean_tok {sum(toks)/len(toks):.0f}", flush=True)
            ja = []
            for tag, q in JA:
                text, st = ask(a.port, q, max_tokens=1536)
                ja.append({"task": tag, "answer": text, "loop": looping(text), **st})
                print(f"\n--- [{name}][{tag}] {json.dumps(st)} loop={looping(text)}\n{text[:700]}", flush=True)
            res[name] = {"gsm8k": {"n": len(gsm), "accuracy": round(ok / len(gsm), 4), "looping_answers": loops,
                                   "mean_completion_tokens": round(sum(toks) / len(toks), 1),
                                   "mean_seconds": round(sum(secs) / len(secs), 2),
                                   "tok_s": round(sum(toks) / sum(secs), 1)},
                         "gsm8k_rows": rows, "japanese": ja}
            print(f"=== {name} {json.dumps(res[name]['gsm8k'])}", flush=True)
        finally:
            p.kill(); p.wait(); time.sleep(3)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("\n=== summary")
    for name, r in res.items():
        print(f"  {name:10s} {json.dumps(r['gsm8k'])}", flush=True)


if __name__ == "__main__":
    main()
