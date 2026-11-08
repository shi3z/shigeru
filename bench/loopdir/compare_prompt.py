"""Do the two paths send the model the same prompt?

The loops survive a quantization change (Q4 8.5 %, Q6 7.7 %) and the server is provably greedy (12/12 reproducible),
yet the same model in bf16 through transformers barely loops.  If precision and sampling are both ruled out, the
remaining difference is what each path puts in front of the model: llama.cpp applies the chat template embedded in
the GGUF, the transformers run applied tokenizer_config's, and a reasoning model's template often decides whether a
thinking block is opened at all.
"""
import json, subprocess, time, urllib.request

MODEL = "/home/shi3z/models/gguf/ds9b-orig/DeepSeek-V4-Pro-Qwen3.5-9B.i1-Q4_K_M.gguf"
HF = "/home/shi3z/models/hf/ds9b-orig-hf"
BIN = "/home/shi3z/bench-gguf/llama.cpp/build/bin/llama-server"
PORT = 18245
Q = "Anna has 50 pencils and Harry has twice the number of Anna's Pencils but he lost 19 of them. How many pencils are left with Harry?"

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(HF, trust_remote_code=True)
hf_prompt = tok.apply_chat_template([{"role": "user", "content": Q}], tokenize=False, add_generation_prompt=True)
print("=== transformers template output")
print(repr(hf_prompt))
print("tokens:", len(tok(hf_prompt, add_special_tokens=False)["input_ids"]))
for kw in ("think", "reason"):
    print(f"  mentions {kw!r}:", kw in hf_prompt.lower())

p = subprocess.Popen([BIN, "-m", MODEL, "--port", str(PORT), "-ngl", "99", "-c", "4096", "-t", "16",
                      "--host", "127.0.0.1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read()
            break
        except Exception:
            time.sleep(1)
    props = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=10).read())
    gguf_tmpl = props.get("chat_template") or ""
    print("\n=== GGUF chat template (first 600 chars)")
    print(gguf_tmpl[:600])
    print("  template mentions 'think':", "think" in gguf_tmpl.lower())
    # what llama.cpp actually formats for this message, via its own apply-template endpoint
    body = json.dumps({"messages": [{"role": "user", "content": Q}]}).encode()
    try:
        r = urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{PORT}/apply-template", body, {"Content-Type": "application/json"}), timeout=30)
        formatted = json.loads(r.read()).get("prompt", "")
        print("\n=== llama.cpp formatted prompt")
        print(repr(formatted))
        print("same as transformers:", formatted == hf_prompt)
        if formatted != hf_prompt:
            i = next((j for j in range(min(len(formatted), len(hf_prompt))) if formatted[j] != hf_prompt[j]),
                     min(len(formatted), len(hf_prompt)))
            print(f"first difference at char {i}:")
            print("  llama.cpp   ...", repr(formatted[max(0, i - 60):i + 80]))
            print("  transformers...", repr(hf_prompt[max(0, i - 60):i + 80]))
    except Exception as exc:
        print("apply-template not available:", str(exc)[:120])
    # and what it actually generates, to compare token counts with the bf16 run
    body = json.dumps({"messages": [{"role": "user", "content": Q}], "max_tokens": 600,
                       "temperature": 0, "stream": False}).encode()
    d = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=600).read())
    m = d["choices"][0]["message"]
    print("\n=== llama.cpp generation")
    print("usage:", json.dumps(d.get("usage")))
    print("has reasoning_content:", bool(m.get("reasoning_content")))
    print("head:", repr(((m.get("reasoning_content") or "") + (m.get("content") or ""))[:300]))
finally:
    p.kill(); p.wait()
