"""The bf16 loop rate under the SAME detector the GGUF runs used.

Outstanding question: the GGUF paths loop on about 8 % of GSM8K answers, and the bf16 model looked like it loops on
0.6 % -- but that figure came from a stricter detector (exact token-level period 8) than the benchmark's
(a 12-word window occurring 4+ times in the decoded text).  Until the same detector is applied to full-length bf16
generations, "bf16 does not loop" is not a measurement.

Generates with the same greedy decoding and the same 2048-token cap as the benchmark, keeps the text, and scores it
with both detectors.
"""
import argparse, json, time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def looping_words(text, n=12, times=4):
    """The benchmark's detector (ds9b_bench.py): any n-word window occurring `times` or more."""
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


def looping_tokens(ids, w=8):
    """The stricter one used for the activation run: exact token-level period w."""
    for t in range(2 * w, len(ids)):
        if ids[t - w:t] == ids[t - 2 * w:t - w]:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/shi3z/models/hf/ds9b-orig-hf")
    ap.add_argument("--gsm8k", default="/home/shi3z/shigeru/open-mercury/bench/prompts_gsm8k_full.json")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--out", default="/home/shi3z/loopdir/bf16_looprate.json")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0},
                                                 trust_remote_code=True).eval()
    gsm = json.load(open(a.gsm8k))[: a.n]
    rows, t0 = [], time.time()
    with torch.no_grad():
        for i, q in enumerate(gsm):
            pr = tok.apply_chat_template([{"role": "user", "content": q["prompt"]}], tokenize=False,
                                         add_generation_prompt=True)
            ids = tok(pr, return_tensors="pt", add_special_tokens=False).to(model.device)
            past, cur, out = None, ids["input_ids"], []
            for _ in range(a.max_new):
                o = model(input_ids=cur, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nxt = int(o.logits[0, -1].argmax())
                out.append(nxt)
                cur = torch.tensor([[nxt]], device=model.device)
                if nxt == tok.eos_token_id:
                    break
            text = tok.decode(out, skip_special_tokens=True)
            import re
            m = re.findall(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text) or re.findall(r"(-?\d[\d,]*(?:\.\d+)?)", text)
            got = m[-1].replace(",", "") if m else None
            try:
                hit = got is not None and abs(float(got) - float(q["answer"])) < 1e-6
            except ValueError:
                hit = False
            rows.append({"id": q["id"], "hit": hit, "tokens": len(out), "capped": len(out) >= a.max_new,
                         "loop_words": looping_words(text), "loop_tokens": looping_tokens(out),
                         "tail": text[-200:]})
            if (i + 1) % 10 == 0:
                lw = sum(r["loop_words"] for r in rows); lt = sum(r["loop_tokens"] for r in rows)
                ok = sum(r["hit"] for r in rows)
                print(f"  {i+1}/{len(gsm)} acc {ok/len(rows):.3f} loop_words {lw} loop_tokens {lt} "
                      f"mean_tok {sum(r['tokens'] for r in rows)/len(rows):.0f} {time.time()-t0:.0f}s", flush=True)
                json.dump(rows, open(a.out, "w"), ensure_ascii=False, indent=1)
    n = len(rows)
    lw = sum(r["loop_words"] for r in rows); lt = sum(r["loop_tokens"] for r in rows)
    wrong = [r for r in rows if not r["hit"]]
    print(f"\n=== bf16 n={n} accuracy {sum(r['hit'] for r in rows)/n:.4f} errors {len(wrong)}")
    print(f"    word-detector loops  {lw} ({100*lw/n:.1f} %)   <- the benchmark's detector, GGUF gave 7.7-8.5 %")
    print(f"    token-detector loops {lt} ({100*lt/n:.1f} %)   <- the stricter one")
    print(f"    errors that loop (word detector): {sum(r['loop_words'] for r in wrong)}/{len(wrong)}")
    print(f"    answers at the {a.max_new}-token cap: {sum(r['capped'] for r in rows)}")
    print(f"    mean tokens {sum(r['tokens'] for r in rows)/n:.1f}")
    json.dump(rows, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
