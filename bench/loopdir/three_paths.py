"""Is the GGUF the same model as the HF weights?

Q4_K_M and Q6_K behave almost identically (8.5 % and 7.7 % looping answers, 92.80 % and 91.66 % accuracy) while the
same model in bf16 through transformers barely loops.  Quantization noise cannot produce that pattern: Q6 should sit
between Q4 and bf16.  Something the two GGUFs share and the HF weights do not would, and the checkpoint carries MTP
layers (`mtp_num_hidden_layers`, `unsloth_fixed_mtp`) that a conversion can get wrong.

Same questions, same greedy decoding, three paths.  Where the first token differs says which kind of difference it
is: agreement for a while then divergence is numerics, immediate or structural divergence is not.
"""
import json, subprocess, time, urllib.request

HF = "/home/shi3z/models/hf/ds9b-orig-hf"
GGUF_Q4 = "/home/shi3z/models/gguf/ds9b-orig/DeepSeek-V4-Pro-Qwen3.5-9B.i1-Q4_K_M.gguf"
GGUF_Q6H = "/home/shi3z/models/gguf/ds9b/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q6_K.gguf"
BIN = "/home/shi3z/bench-gguf/llama.cpp/build/bin/llama-server"
PORT = 18246
N, MAXNEW = 8, 200


def serve(path):
    p = subprocess.Popen([BIN, "-m", path, "--port", str(PORT), "-ngl", "99", "-c", "4096", "-t", "16",
                          "--host", "127.0.0.1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read(); return p
        except Exception:
            time.sleep(1)
    p.kill(); raise RuntimeError("server did not start")


def gguf_tokens(prompt_text):
    """Token ids llama.cpp generates for a raw prompt, so the comparison is token level and template-free."""
    body = json.dumps({"prompt": prompt_text, "n_predict": MAXNEW, "temperature": 0, "top_k": 1,
                       "repeat_penalty": 1.0, "stream": False, "cache_prompt": False}).encode()
    d = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion", body, {"Content-Type": "application/json"}), timeout=600).read())
    return d.get("content", ""), d.get("tokens_predicted")


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(HF, trust_remote_code=True)
    gsm = json.load(open("/home/shi3z/shigeru/open-mercury/bench/prompts_gsm8k_full.json"))[:N]
    prompts = [tok.apply_chat_template([{"role": "user", "content": q["prompt"]}], tokenize=False,
                                       add_generation_prompt=True) for q in gsm]

    print("=== bf16 (transformers, argmax)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16, device_map={"": 0},
                                                 trust_remote_code=True).eval()
    bf16 = []
    with torch.no_grad():
        for pr in prompts:
            ids = tok(pr, return_tensors="pt", add_special_tokens=False).to(model.device)
            past, cur, out = None, ids["input_ids"], []
            for _ in range(MAXNEW):
                o = model(input_ids=cur, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nxt = int(o.logits[0, -1].argmax())
                out.append(nxt)
                cur = torch.tensor([[nxt]], device=model.device)
                if nxt == tok.eos_token_id:
                    break
            bf16.append(out)
            print(f"  {len(out):>4} tokens: {tok.decode(out)[:90]!r}", flush=True)
    del model
    torch.cuda.empty_cache()

    for label, path in (("Q4_K_M original", GGUF_Q4), ("Q6_K heretic", GGUF_Q6H)):
        print(f"\n=== {label} (llama.cpp, top_k 1)", flush=True)
        p = serve(path)
        try:
            for i, pr in enumerate(prompts):
                text, n = gguf_tokens(pr)
                g = tok(text, add_special_tokens=False)["input_ids"]
                b = bf16[i]
                first = next((j for j in range(min(len(g), len(b))) if g[j] != b[j]), None)
                same = "identical" if first is None and len(g) == len(b) else f"diverges at token {first}"
                print(f"  {n:>4} tokens vs bf16 {len(b):>4}: {same}", flush=True)
                if first is not None and first < 12:
                    print(f"      bf16 {tok.decode(b[:30])!r}", flush=True)
                    print(f"      gguf {tok.decode(g[:30])!r}", flush=True)
        finally:
            p.kill(); p.wait(); time.sleep(3)


if __name__ == "__main__":
    main()
