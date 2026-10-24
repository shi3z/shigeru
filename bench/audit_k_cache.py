"""Attention / KV-cache audit for the probe forward (docs/k_scaling_failure_analysis.md §10): for K = 4, 6, 8 and a few
prefixes, compare the probe logits computed (a) through the cache (prefill the prefix, then the K*K-row probe forward
with probe_key_mask, write=False) and (b) without a cache (one forward over [prefix rows | probe rows] with the
equivalent dense mask: prefix causal, probe rows see the prefix and their own group causally).  Also (c) each group
alone (K rows: K-1 placed + 1 mask, observed-first standard forward) — the per-group independence check, and the
static/graph path run_masked vs the plain path.  Reports max |Δlogit|, argmax agreement and top-1 prob difference in
bf16, per K, so a K-dependent implementation error would show as a disagreement that grows with K.
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. .venv-train/bin/python bench/audit_k_cache.py --model runs/phaseC_sft3_4b/final \
      --refs results/pure/dev_refs_sft3.json --n 8 --out results/pure/audit_k_cache_sft3.json"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts, gpu_info, save_json
from engine.cdlm.fast_model import FastQwen3
from engine.cdlm.k4_decoder import probe_key_mask, probe_rows, probe_win_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="bench/prompts_gsm8k_dev.json")
    ap.add_argument("--refs", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--K", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    mask_id = getattr(hf.config, "cdlm_mask_token_id", None) or tok.convert_tokens_to_ids("<|mask|>")
    prompts = load_prompts(a.prompts)[: a.n]
    refs = json.load(open(a.refs))["refs"]
    model = FastQwen3.from_hf(hf, device="cuda", dtype=torch.bfloat16, max_len=4096)
    del hf
    torch.cuda.empty_cache()
    dev = torch.device("cuda")
    res = {str(K): dict(n=0, cache_vs_nocache_maxdiff=0.0, cache_vs_nocache_argmax_agree=0, group_vs_probe_maxdiff=0.0,
                        group_vs_probe_argmax_agree=0, static_vs_plain_maxdiff=0.0, static_vs_plain_argmax_agree=0, rows=0) for K in a.K}
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        ids = ids[0].tolist() if torch.is_tensor(ids) else list(ids["input_ids"][0])
        resp = refs[p["id"]]["response_ids"]
        for K in a.K:
            s = len(ids) + 8                                   # block start a few tokens into the response
            if s + K > len(ids) + len(resp):
                continue
            prefix = ids + resp[: s - len(ids)]
            gold = resp[s - len(ids): s - len(ids) + K]
            pi, pp, pr = probe_rows(gold, s, mask_id)
            # (a) cache path
            model.kv.zero_()
            model.prefill(torch.tensor(prefix, device=dev), 0)
            la = model.forward(torch.tensor(pi, device=dev), torch.tensor(pp, device=dev), s, write=False, rows=pr,
                               key_mask=probe_key_mask(s, K, device=dev))
            # (b) no cache: prefix + probe rows in one forward with the dense equivalent mask
            T = K * K
            allids = torch.tensor(prefix + pi, device=dev)
            allpos = torch.tensor(list(range(s)) + pp, device=dev)
            km = torch.zeros(s + T, s + T, dtype=torch.bool, device=dev)
            km[:s, :s] = torch.tril(torch.ones(s, s, dtype=torch.bool, device=dev))
            km[s:, :] = probe_key_mask(s, K, device=dev)
            model.kv.zero_()
            lb = model.forward(allids, allpos, 0, write=False, rows=[s + r for r in pr], key_mask=km)
            # (c) each group alone, observed-first standard forward on the cache
            model.kv.zero_()
            model.prefill(torch.tensor(prefix, device=dev), 0)
            lc = []
            for i in range(K):
                obs = [j for j in range(K) if j != i]
                gi = torch.tensor([gold[j] for j in obs] + [mask_id], device=dev)
                gp = torch.tensor([s + j for j in obs] + [s + i], device=dev)
                lc.append(model.forward(gi, gp, s, write=False, rows=[K - 1])[0])
            lc = torch.stack(lc)
            # (d) static path run_masked (writes probe K/V at s..; harmless here)
            ld = model.run_masked(torch.tensor(pi), torch.tensor(pp), s, probe_win_mask(K, device=dev), pr, static=True) if T <= 32 else None
            r = res[str(K)]
            r["n"] += 1; r["rows"] += K
            r["cache_vs_nocache_maxdiff"] = max(r["cache_vs_nocache_maxdiff"], float((la - lb).abs().max()))
            r["cache_vs_nocache_argmax_agree"] += int((la.argmax(-1) == lb.argmax(-1)).sum())
            r["group_vs_probe_maxdiff"] = max(r["group_vs_probe_maxdiff"], float((la - lc).abs().max()))
            r["group_vs_probe_argmax_agree"] += int((la.argmax(-1) == lc.argmax(-1)).sum())
            if ld is not None:
                r["static_vs_plain_maxdiff"] = max(r["static_vs_plain_maxdiff"], float((la - ld).abs().max()))
                r["static_vs_plain_argmax_agree"] += int((la.argmax(-1) == ld.argmax(-1)).sum())
                r["static_rows"] = r.get("static_rows", 0) + K
            else:
                r["static_vs_plain_maxdiff"] = r["static_vs_plain_argmax_agree"] = "n/a (K*K rows exceed the largest static bucket)"
    for K, r in res.items():
        if r["rows"]:
            for k in ("cache_vs_nocache_argmax_agree", "group_vs_probe_argmax_agree"):
                r[k] = round(r[k] / r["rows"], 4)
            if isinstance(r["static_vs_plain_argmax_agree"], int):
                r["static_vs_plain_argmax_agree"] = round(r["static_vs_plain_argmax_agree"] / r["static_rows"], 4)
    print("=== audit", json.dumps(res), flush=True)
    save_json(a.out, {"model": a.model, "dtype": "bfloat16", "gpu": gpu_info(), "results": res})


if __name__ == "__main__":
    main()
