"""Diagnostic for the self-correcting denoiser (docs/pure_diffusion_research.md section 6): teacher-forced blocks with
per-step ground truth, so corrections can be scored.

For each dev prompt the model's own greedy AR answer (the --refs cache of bench/diag_mask_recovery.py) is the reference;
each block of K response tokens is denoised FROM SCRATCH (all K masked) on the clean prefix with engine/cdlm/sc_decoder
.denoise_block (write=False forwards: the cache keeps the clean sequence), under one or more sampler configurations.
Per step the placed tokens are compared with the reference:
  correction success rate = positions that were wrong at some step and are right at the end / positions ever wrong
  correction regression rate = positions that were right at some step and are wrong at the end / positions ever right
  plus per-step token accuracy of the placed tokens, final block token accuracy and exact match, steps, re-evaluations,
  corrections (token changed on re-evaluation), of which to-correct / to-wrong.
Reference accuracy is the AR path's own text, so "right" means "the token the AR path would have written" — the
teacher-forced analogue of the AR-equivalence the brief does not require but that makes per-position scoring possible.
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv-train/bin/python bench/diag_selfcorrect.py --model runs/phaseC_sft3_4b/final \
      --refs results/pure/dev_refs_sft3.json --n 50 --configs none:k0 all:k1 lowconf0.9:k1 --out results/pure/diag_sc_sft3.json
Config syntax: <remask spec with ':' replaced by nothing or '='>:k<confirm_k>[:tau<t>][:S<steps>] e.g. "lowconf=0.9:k1:tau0.95"."""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts, gpu_info, save_json
from engine.cdlm.fast_model import FastQwen3
from engine.cdlm.sc_decoder import denoise_block


def parse_config(s):
    parts = s.split(":")
    remask = parts[0].replace("=", ":")
    kw = dict(remask=remask, confirm_k=1, steps=None, tau=0.99, anchor_leftmost=True)
    for p in parts[1:]:
        if p.startswith("k"):
            kw["confirm_k"] = int(p[1:])
        elif p.startswith("tau"):
            kw["tau"] = float(p[3:])
        elif p.startswith("S"):
            kw["steps"] = int(p[1:])
        elif p.startswith("frac"):
            kw["max_remask_frac"] = float(p[4:])
        elif p.startswith("max"):
            kw["max_steps"] = int(p[3:])
        elif p == "noanchor":
            kw["anchor_leftmost"] = False
        else:
            raise ValueError(p)
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="bench/prompts_gsm8k_dev.json")
    ap.add_argument("--refs", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--configs", nargs="+", default=["none:k0", "all:k1", "lowconf=0.9:k1", "drop=0.1:k1"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
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
    configs = {c: parse_config(c) for c in a.configs}
    agg = {c: dict(blocks=0, tokens=0, final_correct=0, exact=0, steps=0, reeval=0, corrections=0, corr_to_right=0,
                   corr_to_wrong=0, corr_wrong_to_wrong=0, ever_wrong=0, ever_wrong_fixed=0, ever_right=0,
                   ever_right_broken=0, step_acc=[], accepted_tentative=0) for c in configs}
    t0 = time.time()
    for p in prompts:
        resp = refs[p["id"]]["response_ids"]
        full = torch.tensor(p["ids"] + resp, dtype=torch.long, device="cuda")
        P = len(p["ids"])
        model.kv.zero_()
        model.prefill(full, 0)
        for s in range(P, P + len(resp), a.block):
            e = min(P + len(resp), s + a.block)
            K = e - s
            gold = full[s:e].tolist()

            def fwd(ids, pos, rows, _s=s):
                return model.forward(torch.tensor(ids, device="cuda"), torch.tensor(pos, device="cuda"), cache_len=_s,
                                     write=False, rows=rows)

            for c, kw in configs.items():
                g = torch.Generator().manual_seed(a.seed)
                toks, info = denoise_block(fwd, s, K, mask_id, trace=True, generator=g, **kw)
                A = agg[c]
                A["blocks"] += 1
                A["tokens"] += K
                A["steps"] += info["steps"]
                A["reeval"] += info["reeval"]
                A["corrections"] += info["corrections"]
                A["accepted_tentative"] += info["accepted_tentative"]
                A["final_correct"] += sum(int(t == gd) for t, gd in zip(toks, gold))
                A["exact"] += int(toks == gold)
                ever_wrong = [False] * K
                ever_right = [False] * K
                for ti, tr in enumerate(info["trace"]):
                    placed = [i for i in range(K) if tr["status"][i] != 0]
                    ok = [tr["tokens"][i] == gold[i] for i in placed]
                    if len(A["step_acc"]) <= ti:
                        A["step_acc"].append([0, 0])
                    A["step_acc"][ti][0] += sum(ok)
                    A["step_acc"][ti][1] += len(ok)
                    for i, o in zip(placed, ok):
                        if o:
                            ever_right[i] = True
                        else:
                            ever_wrong[i] = True
                    for (i, old, new) in tr["corrected"]:
                        if new == gold[i] and old != gold[i]:
                            A["corr_to_right"] += 1
                        elif old == gold[i] and new != gold[i]:
                            A["corr_to_wrong"] += 1
                        else:
                            A["corr_wrong_to_wrong"] += 1
                for i in range(K):
                    if ever_wrong[i]:
                        A["ever_wrong"] += 1
                        A["ever_wrong_fixed"] += int(toks[i] == gold[i])
                    if ever_right[i]:
                        A["ever_right"] += 1
                        A["ever_right_broken"] += int(toks[i] != gold[i])
        print(f"{p['id']} resp {len(resp)} tok, {time.time() - t0:.0f}s", flush=True)
    summary = {}
    for c, A in agg.items():
        summary[c] = dict(blocks=A["blocks"], tokens=A["tokens"], token_acc=round(A["final_correct"] / A["tokens"], 4),
                          block_exact=round(A["exact"] / A["blocks"], 4), steps_per_block=round(A["steps"] / A["blocks"], 2),
                          tokens_per_step=round(A["tokens"] / max(1, A["steps"]), 3),
                          reeval_per_token=round(A["reeval"] / A["tokens"], 3),
                          corrections_per_100tok=round(100 * A["corrections"] / A["tokens"], 2),
                          corr_to_right=A["corr_to_right"], corr_to_wrong=A["corr_to_wrong"], corr_wrong_to_wrong=A["corr_wrong_to_wrong"],
                          correction_success_rate=round(A["ever_wrong_fixed"] / A["ever_wrong"], 4) if A["ever_wrong"] else None,
                          correction_regression_rate=round(A["ever_right_broken"] / A["ever_right"], 4) if A["ever_right"] else None,
                          accepted_tentative_frac=round(A["accepted_tentative"] / A["tokens"], 4),
                          step_acc=[round(x[0] / x[1], 3) if x[1] else None for x in A["step_acc"]])
    print("=== selfcorrect", json.dumps(summary), flush=True)
    save_json(a.out, {"model": a.model, "refs": a.refs, "n": len(prompts), "block": a.block, "seed": a.seed, "gpu": gpu_info(),
                      "configs": {c: {k: v for k, v in kw.items()} for c, kw in configs.items()}, "seconds": round(time.time() - t0, 1),
                      "summary": summary})


if __name__ == "__main__":
    main()
