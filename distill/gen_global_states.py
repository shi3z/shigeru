"""Phase B11-C — sampler-aligned training states with teacher targets (docs/sampler_aligned_global_denoising.md §4).

The STUDENT sampler (--sampler remask: non-monotonic confidence re-masking; left: left-to-right monotonic) is run
freely on training prompts (K = 16, S steps) with the frozen B-series model, following ITS OWN choices (on-policy).
Every state x_s (the block before the s-th forward: [MASK] / placed tokens) is recorded with two teacher targets:
  plain      target at the masked positions = the teacher's sequential completion conditioned on the CURRENT visible
             tokens (masked positions filled one per forward, left to right — the chain-rule joint over the masks;
             n_mask extra forwards).  Visible tokens are kept and are never targets (no visible-correction CE).
  corrected  the teacher block T for this prefix (sequential left-to-right from all masks, 16 forwards, computed once per
             block); the state is rewritten so that visible tokens that differ from T are RE-MASKED, and every masked
             position gets T as target.  This is the "teacher-corrected state": wrong visible tokens are corrected only by
             making them masks and targets, as the brief requires.
Both are stored per state (x_plain, y_plain, x_corr, y_corr, with -100 at non-target positions), plus the prefix
(prompt ids + generated tokens so far), step, sampler and the final block of the student run.  Forwards are counted.
Also records, per state, how many visible tokens differ from T (the wrong-visible statistic on free-generation states).
  SHIGERU_GPU_GUARD=off CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-train/bin/python distill/gen_global_states.py \
      --model runs/zeta_sft_final_e5/final --adapter runs/bdiff_e5_lr1e4/adapter --prompts bench/prompts_gsm8k_pairtrain.json \
      --skip 800 --n 300 --sampler remask --out data/distill/gstates_remask"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts, score
from engine.bdiff.model import BDiffModel
from engine.bdiff.sampler2 import sample_block_fixed
from engine.bdiff.global_sampler import global_sample_block


@torch.no_grad()
def predict(model, cache, P, x, mask_id):
    xt = torch.tensor(x, device=model.device)
    g = max(float((xt == mask_id).sum()) / len(x), 1e-3)
    lg = model.denoise(cache, P, xt[None], g).float(); lg[:, mask_id] = float("-inf")
    return lg.argmax(-1).tolist()


@torch.no_grad()
def seqfill(model, cache, P, x, mask_id):
    x = list(x); n = 0
    for q in range(len(x)):
        if x[q] == mask_id:
            x[q] = predict(model, cache, P, x, mask_id)[q]; n += 1
    return x, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--sampler", default="remask", choices=["remask", "left", "confidence", "sync"])
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--shard", type=int, default=4000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    mask_id = getattr(lm.config, "cdlm_mask_token_id", None) or tok.convert_tokens_to_ids("<|mask|>")
    if a.adapter:
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, a.adapter).eval().merge_and_unload()
    model = BDiffModel(lm, mask_id, time_cond=False).to(dev).eval()
    K, S = a.K, a.steps
    stop_ids = {tok.eos_token_id} | ({tok.convert_tokens_to_ids("<|im_end|>")} if "<|im_end|>" in tok.get_vocab() else set())
    prompts = load_prompts(a.prompts)[a.skip: a.skip + a.n]
    rows = []; n_shard = 0; n_rows = 0; n_fwd = 0; t0 = time.time(); wrong_vis_total = 0; vis_total = 0
    pf = open(f"{a.out}_prompts.jsonl", "w")

    def flush():
        nonlocal rows, n_shard
        if rows:
            torch.save(dict(x_plain=torch.tensor([r["x_plain"] for r in rows]), y_plain=torch.tensor([r["y_plain"] for r in rows]),
                            x_corr=torch.tensor([r["x_corr"] for r in rows]), y_corr=torch.tensor([r["y_corr"] for r in rows]),
                            meta=torch.tensor([r["meta"] for r in rows])), f"{a.out}_{n_shard:03d}.pt")
            n_shard += 1; rows = []

    for pi, p in enumerate(prompts):
        ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        ids = [int(t) for t in (ids[0] if torch.is_tensor(ids) else ids["input_ids"][0])]
        out = []; finish = None; blk = 0
        while finish is None:
            Kb = min(K, a.max_new_tokens - len(out))
            P = len(ids) + len(out)
            cache = model.prefix_cache(torch.tensor([ids + out], device=dev)); n_fwd += 1
            # student run (on-policy), traced
            if a.sampler in ("left", "confidence"):
                toks, info = sample_block_fixed(model, cache, P, Kb, S, mask_id, unmask=a.sampler, remask="none", early_stop=False, trace=True)
                states = []; prev = [mask_id] * Kb
                for st in info["trace"]:
                    states.append((st["step"], list(prev))); prev = st["tokens"]
            else:
                toks, info = global_sample_block(model, cache, P, Kb, S, mask_id, mode=a.sampler, trace=True)
                states = []; prev = [mask_id] * Kb
                for st in info["trace"]:
                    states.append((st["step"], list(prev)))
                    prev = [mask_id if mk else g_ for g_, mk in zip(st["guess"], st["masked"])]
            n_fwd += info["forwards"]
            # teacher block for this prefix: sequential left-to-right from all masks
            T, nT = seqfill(model, cache, P, [mask_id] * Kb, mask_id); n_fwd += nT
            for s, x in states:
                y_plain, nf = seqfill(model, cache, P, x, mask_id); n_fwd += nf
                yp = [y_plain[q] if x[q] == mask_id else -100 for q in range(Kb)]
                x_corr = [mask_id if (x[q] != mask_id and x[q] != T[q]) else x[q] for q in range(Kb)]
                yc = [T[q] if x_corr[q] == mask_id else -100 for q in range(Kb)]
                wrong = sum(1 for q in range(Kb) if x[q] != mask_id and x[q] != T[q]); vis = sum(1 for q in range(Kb) if x[q] != mask_id)
                wrong_vis_total += wrong; vis_total += vis
                pad = K - Kb
                rows.append(dict(x_plain=x + [mask_id] * pad, y_plain=yp + [-100] * pad, x_corr=x_corr + [mask_id] * pad, y_corr=yc + [-100] * pad,
                                 meta=[pi + a.skip, blk, s, P, Kb, wrong, vis]))
                n_rows += 1
                if len(rows) >= a.shard:
                    flush()
            keep = Kb
            for q in range(Kb):
                if toks[q] in stop_ids:
                    keep, finish = q + 1, "stop"; break
            out += toks[:keep]; blk += 1
            if len(out) >= a.max_new_tokens:
                finish = finish or "length"
        pf.write(json.dumps(dict(idx=pi + a.skip, id=p["id"], ids=ids, out=out, correct=score(p, tok.decode(out, skip_special_tokens=True)))) + "\n"); pf.flush()
        if pi % 10 == 0:
            print(f"{pi} prompts, {n_rows} states, {n_fwd} forwards, wrong-visible {wrong_vis_total}/{vis_total}, {time.time() - t0:.0f}s", flush=True)
    flush(); pf.close()
    json.dump(dict(model=a.model, adapter=a.adapter, prompts=a.prompts, skip=a.skip, n=len(prompts), sampler=a.sampler, K=K, steps=S, rows=n_rows, shards=n_shard,
                   forwards=n_fwd, wrong_visible=wrong_vis_total, visible=vis_total, seconds=round(time.time() - t0, 1)), open(f"{a.out}_meta.json", "w"), indent=1)
    print(f"=== gen_global_states {a.sampler}: {n_rows} states in {n_shard} shards, {n_fwd} forwards, wrong-visible {wrong_vis_total}/{vis_total}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
