#!/usr/bin/env python3
"""Collect results/*.json into a Markdown table (mean over prompts)."""
import glob, json, os, sys
rows = []
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "*.json"))):
    r = json.load(open(f)); s = r["summary"]; c = r["config"]
    cfg = {k: c[k] for k in ("mode", "steps", "block_length", "threshold", "factor", "small_block_size", "block_cache", "max_new_tokens", "gen_length") if k in c and c[k] not in (None, False)}
    rows.append((os.path.basename(f), r["engine"], os.path.basename(r["model"]), cfg, s))
print("| result file | engine | model | config | tok/s effective | tok/s raw | decode tok/s | TTFT ms | E2E s | fwd/req | eff tokens | peak GiB |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for f, e, m, cfg, s in rows:
    g = lambda k, fmt="%.1f": (fmt % s[k]["mean"]) if k in s else "-"
    print(f"| {f} | {e} | {m} | {' '.join(f'{k}={v}' for k,v in cfg.items())} | {g('tok_s_effective')} | {g('tok_s_raw')} | {g('decode_tok_s')} | {('%.0f' % (1000*s['ttft_s']['mean'])) if 'ttft_s' in s else '-'} | {g('e2e_s','%.2f')} | {g('n_forward','%.0f')} | {g('effective_tokens','%.0f')} | {g('peak_mem_gib')} |")
