"""Make a config.json written by transformers 5.x loadable with the SAME semantics by transformers 4.57 (bench venv).
transformers 5 stores RoPE as {"rope_parameters": {"rope_theta": ..., "rope_type": ...}}; 4.57 only reads "rope_theta" /
"rope_scaling" and silently falls back to the class default (10 000 for Qwen3, instead of 1 000 000), which degrades
generation (measured 2026-10-20: doubled brackets in code, HumanEval 24 % instead of ~80 %). Usage: python -m distill.fix_config <dir>..."""
import json, os, sys


def fix(d):
    p = os.path.join(d, "config.json")
    c = json.load(open(p))
    rp = c.get("rope_parameters")
    changed = False
    if isinstance(rp, dict):
        if "rope_theta" in rp and c.get("rope_theta") != rp["rope_theta"]:
            c["rope_theta"] = rp["rope_theta"]; changed = True
        if rp.get("rope_type", "default") == "default":
            if c.get("rope_scaling") is not None:
                c["rope_scaling"] = None; changed = True
        else:
            c["rope_scaling"] = {k: v for k, v in rp.items() if k != "rope_theta"}; changed = True
    if "dtype" in c and "torch_dtype" not in c:
        c["torch_dtype"] = c["dtype"]; changed = True
    if changed:
        json.dump(c, open(p, "w"), indent=2)
    return changed


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(d, "fixed" if fix(d) else "unchanged")
