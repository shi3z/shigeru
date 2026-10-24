"""Benchmark the pure blockwise masked-diffusion decoder (engine/cdlm/block_decoder.py): accuracy + measured speed.

No AR verification anywhere.  One JSON per invocation with every condition (model, prompts, block sizes, steps,
order, schedule, remask, tau, engine path, seed, GPU, dtype, torch version, max_new_tokens, warm-up) and per-prompt
rows (tokens, seconds, tok/s, denoising steps, forwards, remasked, correct, text).  Runs in .venv-train.
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. .venv-train/bin/python bench/run_block.py --model runs/phaseC_sft3_4b/final \
      --prompts bench/prompts_gsm8k_dev.json --n 100 --block 8,16,32 --steps 1,2,4,8 --order confidence \
      --engine plain --out results/pure/block_sft3_dev.json
--remask lowconf:0.2:0.75 | tau:0.9:0.5 ; --steps adaptive with --tau for the threshold schedule.
--ar-ref <json of bench/run_ar.py or diag refs> adds the AR accuracy of the same prompts to the summary (reference only,
never used in decoding)."""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts, gpu_info, save_json, score
from engine.cdlm.fast_model import FastQwen3, BUCKETS, bucket_for
from engine.cdlm.block_decoder import block_generate
from engine.cdlm.sc_decoder import sc_generate
from engine.cdlm.k4_decoder import k4_generate


def parse_remask(s):
    if not s or s == "none":
        return None
    kind, val, until = s.split(":")
    return (kind, float(val), float(until))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="bench/prompts_gsm8k_dev.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--block", default="32")
    ap.add_argument("--steps", default="8", help="comma list of ints, or 'adaptive' (then --tau)")
    ap.add_argument("--schedule", default="linear")
    ap.add_argument("--order", default="confidence")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--min-fill", type=int, default=1)
    ap.add_argument("--anchor-leftmost", action="store_true", help="always fill the leftmost mask of the block as well")
    ap.add_argument("--remask", default="none", help="block decoder: lowconf:f:until | tau:c:until ; sc decoder: none|all|lowconf:t|entropy:h|random:p|scheduled:f0|drop:d")
    ap.add_argument("--decoder", default="block", choices=["block", "sc", "k4"], help="sc = self-correcting (engine/cdlm/sc_decoder.py); k4 = small-block leave-one-out refinement (engine/cdlm/k4_decoder.py)")
    ap.add_argument("--k4-strategy", default="refine", help="k4: s1|conf2|refine|refineN|conf2refine|seq")
    ap.add_argument("--adopt", default="always", choices=["always", "conf", "nonadjacent", "one"])
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--confirm-k", type=int, default=1, help="sc: identical re-predictions needed to confirm a placed token")
    ap.add_argument("--max-steps", type=int, default=None, help="sc: step budget per block (default K+1 adaptive, S+2 fixed)")
    ap.add_argument("--max-remask-frac", type=float, default=0.5, help="sc: cap on re-evaluated slots per step as a fraction of K")
    ap.add_argument("--min-reeval-steps", type=int, default=0)
    ap.add_argument("--stop", default="confirmed", choices=["confirmed", "conf", "stable"])
    ap.add_argument("--stop-tau", type=float, default=0.99)
    ap.add_argument("--engine", default="plain", choices=["plain", "static", "graph"])
    ap.add_argument("--no-merge-commit", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", action="store_true", help="sample instead of argmax (temperature 1)")
    ap.add_argument("--ar-ref", default=None)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    mask_id = getattr(hf.config, "cdlm_mask_token_id", None) or tok.convert_tokens_to_ids("<|mask|>")
    stop_ids = [tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|im_end|>")] if "<|im_end|>" in tok.get_vocab() else [])
    prompts = load_prompts(a.prompts)[: a.n]
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True,
                                      enable_thinking=False, return_tensors="pt")
        p["ids"] = ids[0].tolist() if torch.is_tensor(ids) else list(ids["input_ids"][0])
    model = FastQwen3.from_hf(hf, device="cuda", dtype=torch.bfloat16, max_len=4096)
    del hf
    torch.cuda.empty_cache()
    static, use_graph = a.engine != "plain", a.engine == "graph"
    if a.decoder == "k4":
        assert a.engine == "plain", "k4 decoder: probe forwards need the plain path (custom key mask)"
    merge = not a.no_merge_commit
    blocks = [int(x) for x in a.block.split(",")]
    steps_list = [None] if a.steps == "adaptive" else [int(x) for x in a.steps.split(",")]
    remask = parse_remask(a.remask) if a.decoder == "block" else a.remask
    if static:
        need = max(blocks) * (2 if merge else 1)
        assert need <= BUCKETS[-1], f"static path: block {max(blocks)} (x2 with merge_commit) exceeds the largest bucket {BUCKETS[-1]}"
    # warm-up (kernel selection; graphs captured lazily on first use per bucket pair, excluded by a second warm-up)
    wu = tok("warm up", return_tensors="pt")["input_ids"][0].tolist()
    def decode(ids, K, S, gen, trace):
        if a.decoder == "k4":
            return k4_generate(model, tok, ids, a.max_new_tokens, block=K, strategy=a.k4_strategy, adopt=a.adopt,
                               max_rounds=a.max_rounds, anchor_leftmost=a.anchor_leftmost, mask_id=mask_id, stop_ids=stop_ids,
                               trace=trace)
        if a.decoder == "block":
            return block_generate(model, tok, ids, a.max_new_tokens, block=K, steps=S, schedule=a.schedule, order=a.order,
                                  tau=a.tau, min_fill=a.min_fill, anchor_leftmost=a.anchor_leftmost, remask=remask,
                                  mask_id=mask_id, stop_ids=stop_ids, greedy=not a.sample, generator=gen, static=static,
                                  use_graph=use_graph, merge_commit=merge, trace=trace)
        return sc_generate(model, tok, ids, a.max_new_tokens, block=K, mask_id=mask_id, stop_ids=stop_ids, static=static,
                           use_graph=use_graph, merge_commit=merge, trace=trace, steps=S, schedule=a.schedule, order=a.order,
                           tau=a.tau if a.tau is not None else 0.99, min_fill=a.min_fill, anchor_leftmost=a.anchor_leftmost,
                           remask=remask, confirm_k=a.confirm_k, max_steps=a.max_steps, max_remask_frac=a.max_remask_frac,
                           min_reeval_steps=a.min_reeval_steps, stop=a.stop, stop_tau=a.stop_tau, greedy=not a.sample,
                           generator=gen)

    for _ in range(2):
        decode(wu, blocks[0], steps_list[0], torch.Generator().manual_seed(0), False)
    ar_ref = None
    if a.ar_ref:
        d = json.load(open(a.ar_ref))
        rows = d.get("runs") or [{"id": k, **v} for k, v in d.get("refs", {}).items()]
        ar_ref = {r["id"]: r.get("correct") for r in rows}
    def cfg_name(K, S):
        if a.decoder == "k4":
            return f"K={K},k4:{a.k4_strategy}" + (f":{a.adopt}" if a.k4_strategy.startswith("refine") else "") + (f":r{a.max_rounds}" if a.k4_strategy == "refineN" else "")
        return f"K={K},S={'adaptive' if S is None else S}" + (",anchor" if a.anchor_leftmost else "") + (f",sc:{a.remask}:k{a.confirm_k}" if a.decoder == "sc" else "")

    def rep4(ids):
        if len(ids) < 8:
            return 0.0
        grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
        return round(1.0 - len(set(grams)) / len(grams), 4)

    runs = []
    configs = [(K, S) for K in blocks for S in steps_list]
    if a.decoder == "k4":
        configs = [(K, None) for K in blocks]
    for K, S in configs:
        name = cfg_name(K, S)
        gen = torch.Generator().manual_seed(a.seed)
        for p in prompts:
            torch.cuda.synchronize(); t0 = time.time()
            new, st = decode(p["ids"], K, S, gen, a.trace)
            torch.cuda.synchronize(); dt = time.time() - t0
            text = tok.decode(new, skip_special_tokens=True)
            r = {"config": name, "id": p["id"], "n_tokens": len(new), "seconds": round(dt, 3), "tok_s": round(len(new) / dt, 1),
                 "steps": st["steps"], "forwards": st["forwards"], "blocks": st["blocks"], "remasked": st["remasked"],
                 "corrections": st.get("corrections"), "confirmed": st.get("confirmed_by_agree"), "accepted_tentative": st.get("accepted_tentative"),
                 "finish": st["finish_reason"], "rep4": rep4(new), "has_answer": ("####" in text), "correct": score(p, text) if "answer" in p else None, "text": text}
            if a.trace:
                r["trace"] = st["trace"]
            runs.append(r)
            print(f"[{name}] {p['id']} {len(new)} tok in {dt:.1f}s = {r['tok_s']} tok/s, steps {st['steps']} fwd {st['forwards']} "
                  f"remask {st['remasked']} correct={r['correct']}", flush=True)
    summary = {}
    for K, S in configs:
        name = cfg_name(K, S)
        rs = [r for r in runs if r["config"] == name]
        tot_tok = sum(r["n_tokens"] for r in rs)
        summary[name] = {"accuracy": sum(bool(r["correct"]) for r in rs) / len(rs), "n": len(rs),
                         "tok_s_mean": round(sum(r["tok_s"] for r in rs) / len(rs), 1),
                         "tok_s_total": round(tot_tok / sum(r["seconds"] for r in rs), 1),
                         "tokens_per_step": round(tot_tok / max(1, sum(r["steps"] for r in rs)), 3),
                         "tokens_per_forward": round(tot_tok / max(1, sum(r["forwards"] for r in rs)), 3),
                         "remask_rate": round(sum(r["remasked"] for r in rs) / max(1, tot_tok), 4),
                         "mean_tokens": round(tot_tok / len(rs), 1),
                         "finish_length": sum(r["finish"] == "length" for r in rs),
                         "corrections_per_100tok": round(100 * sum(r["corrections"] or 0 for r in rs) / max(1, tot_tok), 2),
                         "rep4_mean": round(sum(r["rep4"] for r in rs) / len(rs), 4),
                         "premature_eos": sum(1 for r in rs if r["finish"] == "stop" and not r["has_answer"]),
                         "accepted_tentative_per_100tok": round(100 * sum(r["accepted_tentative"] or 0 for r in rs) / max(1, tot_tok), 2)}
        if ar_ref:
            summary[name]["ar_ref_accuracy"] = sum(bool(ar_ref.get(r["id"])) for r in rs) / len(rs)
    print("=== block", json.dumps(summary), flush=True)
    for p in prompts:
        p.pop("ids", None)
    save_json(a.out, {"model": a.model, "decoder": {"block": "engine/cdlm/block_decoder.py", "sc": "engine/cdlm/sc_decoder.py", "k4": "engine/cdlm/k4_decoder.py"}[a.decoder], "args": vars(a), "gpu": gpu_info(),
                      "torch": torch.__version__, "dtype": "bfloat16", "batch_size": 1, "max_new_tokens": a.max_new_tokens,
                      "warmup": "2 generations of 48 tokens", "runs": runs, "summary": summary})


if __name__ == "__main__":
    main()
