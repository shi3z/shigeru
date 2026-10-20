#!/usr/bin/env python3
"""Collect generated samples (EN/JA) from results/*.json into docs/samples.md, grouped by engine."""
import glob, json, os
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SHOW = ["en_explain", "en_code", "ja_explain", "ja_reason"]
out = ["# Generated samples (greedy, 256 tokens unless noted)", "", "Verbatim outputs taken from `results/*.json` (same prompts for every engine; see `bench/prompts.json`). Speed figures are the per-prompt effective tokens/s of that run.", ""]
for f in sorted(glob.glob(os.path.join(root, "results", "*.json"))):
    if os.path.basename(f).startswith(("gsm8k50_", "humaneval_", "mbpp_")):
        continue
    r = json.load(open(f))
    runs = {x["id"]: x for x in r["runs"] if x.get("rep", 0) == 0}
    if not any(i in runs for i in SHOW):
        continue
    c = r["config"]; cfg = " ".join(f"{k}={c[k]}" for k in ("mode", "steps", "block_length", "threshold", "small_block_size", "graph", "stop_on_eos") if k in c and c[k] not in (None, False))
    out.append(f"## {r['engine']} — {os.path.basename(r['model'])} ({cfg})"); out.append(f"`{os.path.basename(f)}`"); out.append("")
    for i in SHOW:
        if i not in runs: continue
        x = runs[i]
        out.append(f"**{i}** — {x['effective_tokens']} tokens, {x['tok_s_effective']:.1f} tok/s" + (f", correct={x['correct']}" if x.get("correct") is not None else ""))
        out.append(""); out.append("```text"); out.append(x["answer"].strip()[:1500]); out.append("```"); out.append("")
open(os.path.join(root, "docs", "samples.md"), "w").write("\n".join(out))
print("docs/samples.md:", len(out), "lines")
