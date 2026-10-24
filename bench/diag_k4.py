"""Teacher-forced diagnostic of the small-block decoder (docs/four_token_diffusion.md §6): per-distance accuracy of the
FIRST forward (all K masked) and of the FINAL tokens, block exact match, correction success / regression rates.

For each dev prompt the model's greedy AR answer (--refs, from bench/diag_mask_recovery.py) is the reference; every
block of K response tokens is denoised from scratch on the clean prefix (write=False forwards) with each strategy of
engine/cdlm/k4_decoder.k4_denoise_block.  "distance" d = 1..K is the position inside the block (1 = right after the
last committed token).  Correction success rate = positions wrong after the first forward and right at the end /
positions wrong after the first forward; regression rate = right after the first forward and wrong at the end / right
after the first forward.  Also the mask-recovery accuracy of the probe (leave-one-out with the K-1 TRUE neighbours) per
distance, which is the ceiling of what a second step can recover when the other tokens are right.
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. .venv-train/bin/python bench/diag_k4.py --model runs/phaseC_sft3_4b/final \
      --refs results/pure/dev_refs_sft3.json --n 50 --block 4 --strategies s1 conf2 refine refineN seq --out results/pure/diag_k4_sft3.json"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts, gpu_info, save_json
from engine.cdlm.fast_model import FastQwen3
from engine.cdlm.k4_decoder import k4_denoise_block, probe_key_mask, probe_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="bench/prompts_gsm8k_dev.json")
    ap.add_argument("--refs", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--strategies", nargs="+", default=["s1", "conf2", "refine", "refineN", "conf2refine", "seq"])
    ap.add_argument("--adopt", default="always")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    mask_id = getattr(hf.config, "cdlm_mask_token_id", None) or tok.convert_tokens_to_ids("<|mask|>")
    prompts = load_prompts(a.prompts)[: a.n]
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True,
                                      enable_thinking=False, return_tensors="pt")
        p["ids"] = ids[0].tolist() if torch.is_tensor(ids) else list(ids["input_ids"][0])
    refs = json.load(open(a.refs))["refs"]
    model = FastQwen3.from_hf(hf, device="cuda", dtype=torch.bfloat16, max_len=4096)
    del hf
    torch.cuda.empty_cache()
    K = a.block
    agg = {s: dict(blocks=0, first_ok=[0] * K, final_ok=[0] * K, n=[0] * K, exact=0, first_exact=0, wrong1=0, wrong1_fixed=0,
                   right1=0, right1_broken=0, forwards=0, corrections=0) for s in a.strategies}
    probe_ok, probe_n = [0] * K, [0] * K
    t0 = time.time()
    for p in prompts:
        resp = refs[p["id"]]["response_ids"]
        full = torch.tensor(p["ids"] + resp, dtype=torch.long, device="cuda")
        P = len(p["ids"])
        model.kv.zero_()
        model.prefill(full, 0)
        for s in range(P, P + len(resp) - K + 1, K):
            gold = full[s:s + K].tolist()

            def fwd_std(ids, pos, rows, _s=s):
                return model.forward(torch.tensor(ids, device="cuda"), torch.tensor(pos, device="cuda"), _s, write=False, rows=rows)

            def fwd_probe(ids, pos, Kp, rows, _s=s):
                return model.forward(torch.tensor(ids, device="cuda"), torch.tensor(pos, device="cuda"), _s, write=False, rows=rows,
                                     key_mask=probe_key_mask(_s, Kp, device="cuda"))

            # probe ceiling: leave-one-out with the true neighbours
            pi, pp, pr = probe_rows(gold, s, mask_id)
            lg = fwd_probe(pi, pp, K, pr)
            pred = lg.argmax(-1).tolist()
            for i in range(K):
                probe_ok[i] += int(pred[i] == gold[i]); probe_n[i] += 1
            for strat in a.strategies:
                toks, info = k4_denoise_block(fwd_std, fwd_probe, s, K, mask_id, strategy=strat, adopt=a.adopt, trace=False)
                A = agg[strat]
                A["blocks"] += 1
                A["forwards"] += info["forwards"]
                A["corrections"] += info["corrections"]
                f_ok = [info["first"][i] == gold[i] for i in range(K)]
                e_ok = [toks[i] == gold[i] for i in range(K)]
                for i in range(K):
                    A["first_ok"][i] += int(f_ok[i]); A["final_ok"][i] += int(e_ok[i]); A["n"][i] += 1
                A["exact"] += int(all(e_ok)); A["first_exact"] += int(all(f_ok))
                for i in range(K):
                    if not f_ok[i]:
                        A["wrong1"] += 1; A["wrong1_fixed"] += int(e_ok[i])
                    else:
                        A["right1"] += 1; A["right1_broken"] += int(not e_ok[i])
        print(f"{p['id']} {len(resp)} tok {time.time() - t0:.0f}s", flush=True)
    summary = {}
    for strat, A in agg.items():
        summary[strat] = dict(blocks=A["blocks"], forwards_per_block=round(A["forwards"] / A["blocks"], 3),
                              tokens_per_forward=round(K * A["blocks"] / A["forwards"], 3),
                              first_acc_by_distance=[round(A["first_ok"][i] / A["n"][i], 4) for i in range(K)],
                              final_acc_by_distance=[round(A["final_ok"][i] / A["n"][i], 4) for i in range(K)],
                              final_token_acc=round(sum(A["final_ok"]) / sum(A["n"]), 4),
                              first_block_exact=round(A["first_exact"] / A["blocks"], 4), block_exact=round(A["exact"] / A["blocks"], 4),
                              correction_success_rate=round(A["wrong1_fixed"] / A["wrong1"], 4) if A["wrong1"] else None,
                              correction_regression_rate=round(A["right1_broken"] / A["right1"], 4) if A["right1"] else None,
                              corrections_per_block=round(A["corrections"] / A["blocks"], 3))
    probe = dict(acc_by_distance=[round(probe_ok[i] / probe_n[i], 4) for i in range(K)], n=probe_n[0])
    print("=== k4diag", json.dumps({"probe_ceiling": probe, **summary}), flush=True)
    save_json(a.out, {"model": a.model, "refs": a.refs, "n": len(prompts), "block": K, "adopt": a.adopt, "gpu": gpu_info(),
                      "seconds": round(time.time() - t0, 1), "probe_ceiling": probe, "summary": summary})


if __name__ == "__main__":
    main()
