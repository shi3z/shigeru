"""Phase B11 §2/§5 — separating reference mismatch from real generation failure, and the dynamics of the global sampler
(docs/sampler_aligned_global_denoising.md §3.1, §7).

For a sampler (remask = non-monotonic confidence re-masking; left; confidence) run FREELY on dev prompts (no reference read
at inference), every intermediate state is scored against three different targets, kept apart:
  A  reference mismatch     visible tokens != REF (the base model's greedy answer) — only where the generated prefix is
                            still the reference's (teacher-forced mode, --tf), otherwise undefined
  B  self-trajectory        visible tokens != the same run's FINAL block
  C  teacher-trajectory     visible tokens != T, the sequential teacher block for the same prefix (16 extra forwards,
                            research only)
  D  structural error       adjacent duplicate tokens in the current block, 4-gram repetition of the final answer
  E  task failure           GSM8K answer wrong (free mode)
and the "paraphrase" check: among the final blocks of task-CORRECT answers, how many tokens differ from T anyway.
Dynamics of the global sampler per step: positions changed, positions re-masked, tokens that changed value more than
once over the run (oscillation), W->C and C->W against the run's own FINAL and against T, Hamming distance to T and to
FINAL, mean confidence of visible tokens; and order sensitivity: from the SAME step-4 state, finish with the
non-monotonic sampler vs left-to-right monotonic filling (both 4 more forwards) and compare the two completions'
agreement with T and with each other.
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv-train/bin/python bench/diag_mismatch.py --model runs/zeta_sft_final_e5/final \
      --adapter runs/bdiff_e5_lr1e4/adapter --n 50 --sampler remask --out results/pure/b11_mismatch_remask.json"""
import argparse, collections, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts, gpu_info, save_json, score
from engine.bdiff.model import BDiffModel
from engine.bdiff.sampler2 import sample_block_fixed
from engine.bdiff.global_sampler import global_sample_block


@torch.no_grad()
def predict(model, cache, P, x, mask_id):
    xt = torch.tensor(x, device=model.device)
    g = max(float((xt == mask_id).sum()) / len(x), 1e-3)
    lg = model.denoise(cache, P, xt[None], g).float(); lg[:, mask_id] = float("-inf")
    return lg.argmax(-1).tolist(), torch.softmax(lg, -1).max(-1).values.tolist()


@torch.no_grad()
def seqfill(model, cache, P, x, mask_id):
    x = list(x); n = 0
    for q in range(len(x)):
        if x[q] == mask_id:
            x[q] = predict(model, cache, P, x, mask_id)[0][q]; n += 1
    return x, n


def rep4(toks):
    if len(toks) < 8:
        return 0.0
    g = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return 1.0 - len(set(g)) / len(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--adapter2", default=None)
    ap.add_argument("--prompts", default="bench/prompts_gsm8k_dev.json")
    ap.add_argument("--refs", default="results/pure/dev_refs_e5.json")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--sampler", default="remask", choices=["remask", "left", "confidence"])
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    mask_id = getattr(lm.config, "cdlm_mask_token_id", None) or tok.convert_tokens_to_ids("<|mask|>")
    from peft import PeftModel
    for ad in (a.adapter, a.adapter2):
        if ad:
            lm = PeftModel.from_pretrained(lm, ad).eval().merge_and_unload()
    model = BDiffModel(lm, mask_id, time_cond=False).to(dev).eval()
    K, S = a.K, a.steps
    stop_ids = {tok.eos_token_id} | ({tok.convert_tokens_to_ids("<|im_end|>")} if "<|im_end|>" in tok.get_vocab() else set())
    prompts = load_prompts(a.prompts)[: a.n]
    refs = json.load(open(a.refs))["refs"] if a.refs and os.path.exists(a.refs) else {}
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        p["ids"] = [int(t) for t in (ids[0] if torch.is_tensor(ids) else ids["input_ids"][0])]
    t0 = time.time(); n_fwd = 0
    ST = collections.defaultdict(lambda: collections.defaultdict(float)); NS = collections.Counter()
    osc = collections.Counter(); fin = collections.Counter(); order = collections.Counter(); para = collections.Counter()
    rows = []
    for p in prompts:
        ids = list(p["ids"]); out = []; finish = None; blk = 0
        ref_resp = refs.get(p["id"], {}).get("response_ids")
        while finish is None:
            Kb = min(K, a.max_new_tokens - len(out)); P = len(ids) + len(out)
            cache = model.prefix_cache(torch.tensor([ids + out], device=dev)); n_fwd += 1
            if a.sampler in ("left", "confidence"):
                toks, info = sample_block_fixed(model, cache, P, Kb, S, mask_id, unmask=a.sampler, remask="none", early_stop=False, trace=True)
                states = []; prev = [mask_id] * Kb
                for st in info["trace"]:
                    states.append((st["step"], list(prev), st["pred"])); prev = st["tokens"]
            else:
                toks, info = global_sample_block(model, cache, P, Kb, S, mask_id, mode=a.sampler, trace=True)
                states = []; prev = [mask_id] * Kb
                for st in info["trace"]:
                    states.append((st["step"], list(prev), st["guess"]))
                    prev = [mask_id if mk else g_ for g_, mk in zip(st["guess"], st["masked"])]
            n_fwd += info["forwards"]
            T, nT = seqfill(model, cache, P, [mask_id] * Kb, mask_id); n_fwd += nT
            # REF alignment: the block of the reference at this prefix position, valid only while out == ref prefix
            ref_ok = ref_resp is not None and out == ref_resp[: len(out)] and len(ref_resp) >= len(out) + Kb
            R = ref_resp[len(out): len(out) + Kb] if ref_ok else None
            value_changes = [0] * Kb; last_val = [None] * Kb; prev_guess = None
            for s, x, guess in states:
                vis = [q for q in range(Kb) if x[q] != mask_id]
                NS[s] += 1
                ST[s]["n_vis"] += len(vis)
                ST[s]["mm_self"] += sum(1 for q in vis if x[q] != toks[q])
                ST[s]["mm_teacher"] += sum(1 for q in vis if x[q] != T[q])
                if R is not None:
                    ST[s]["n_vis_ref"] += len(vis); ST[s]["mm_ref"] += sum(1 for q in vis if x[q] != R[q])
                ST[s]["dup_adjacent"] += sum(1 for q in range(1, Kb) if x[q] != mask_id and x[q] == x[q - 1])
                ST[s]["ham_T_guess"] += sum(1 for q in range(Kb) if guess[q] != T[q]); ST[s]["ham_final_guess"] += sum(1 for q in range(Kb) if guess[q] != toks[q])
                if prev_guess is not None:
                    ST[s]["changed"] += sum(1 for q in range(Kb) if guess[q] != prev_guess[q])
                    ST[s]["w2c_T"] += sum(1 for q in range(Kb) if prev_guess[q] != T[q] and guess[q] == T[q]); ST[s]["c2w_T"] += sum(1 for q in range(Kb) if prev_guess[q] == T[q] and guess[q] != T[q])
                    ST[s]["w2c_final"] += sum(1 for q in range(Kb) if prev_guess[q] != toks[q] and guess[q] == toks[q]); ST[s]["c2w_final"] += sum(1 for q in range(Kb) if prev_guess[q] == toks[q] and guess[q] != toks[q])
                for q in range(Kb):
                    if last_val[q] is not None and guess[q] != last_val[q]:
                        value_changes[q] += 1
                    last_val[q] = guess[q]
                prev_guess = guess
                # order sensitivity from the mid-run state (step S/2 + 1): finish with remask (S/2 more) vs left monotonic fill (S/2 more)
                if s == S // 2 + 1 and a.sampler == "remask":
                    n_mask = sum(1 for q in range(Kb) if x[q] == mask_id)
                    if n_mask:
                        # left-to-right monotonic completion of the same state in S/2 forwards (n_mask/(S/2) per forward)
                        xl = list(x); per = max(1, -(-n_mask // (S // 2))); nf = 0
                        for _ in range(S // 2):
                            m_idx = [q for q in range(Kb) if xl[q] == mask_id]
                            if not m_idx:
                                break
                            pred, _ = predict(model, cache, P, xl, mask_id); nf += 1
                            for q in m_idx[:per]:
                                xl[q] = pred[q]
                        n_fwd += nf
                        if any(t == mask_id for t in xl):
                            pred, _ = predict(model, cache, P, xl, mask_id); n_fwd += 1
                            xl = [pred[q] if xl[q] == mask_id else xl[q] for q in range(Kb)]
                        order["n"] += 1
                        order["left_eq_T"] += sum(1 for q in range(Kb) if xl[q] == T[q]); order["remask_eq_T"] += sum(1 for q in range(Kb) if toks[q] == T[q])
                        order["left_eq_remask"] += sum(1 for q in range(Kb) if xl[q] == toks[q]); order["K"] += Kb
            osc["positions"] += Kb; osc["changed_ge2"] += sum(1 for v in value_changes if v >= 2); osc["changed_ge1"] += sum(1 for v in value_changes if v >= 1)
            fin["blocks"] += 1; fin["final_ne_T"] += sum(1 for q in range(Kb) if toks[q] != T[q]); fin["final_dup"] += sum(1 for q in range(1, Kb) if toks[q] == toks[q - 1])
            if R is not None:
                fin["blocks_ref"] += 1; fin["final_ne_R"] += sum(1 for q in range(Kb) if toks[q] != R[q])
            keep = Kb
            for q in range(Kb):
                if toks[q] in stop_ids:
                    keep, finish = q + 1, "stop"; break
            out += toks[:keep]; blk += 1
            if len(out) >= a.max_new_tokens:
                finish = finish or "length"
            # paraphrase check: collected per block, task correctness known after the answer
            rows.append(dict(id=p["id"], blk=blk - 1, ne_T=sum(1 for q in range(Kb) if toks[q] != T[q]), Kb=Kb))
        text = tok.decode(out, skip_special_tokens=True); ok = score(p, text)
        for r in rows:
            if r["id"] == p["id"]:
                para["correct" if ok else "wrong"] += r["ne_T"]; para["correct_K" if ok else "wrong_K"] += r["Kb"]
        fin["correct"] += int(bool(ok)); fin["rep4"] += rep4(out)
        print(f"{p['id']} correct={ok} blocks {blk} forwards {n_fwd} {time.time() - t0:.0f}s", flush=True)
    per_step = {}
    for s in sorted(NS):
        n = NS[s]; v = ST[s]; nv = max(1, v["n_vis"])
        per_step[str(s)] = dict(n=n, visible_per_state=round(v["n_vis"] / n, 2),
                                B_self_mismatch_frac=round(v["mm_self"] / nv, 4), C_teacher_mismatch_frac=round(v["mm_teacher"] / nv, 4),
                                A_ref_mismatch_frac=round(v["mm_ref"] / max(1, v["n_vis_ref"]), 4) if v["n_vis_ref"] else None, A_states_with_ref=int(v["n_vis_ref"] > 0),
                                D_adjacent_duplicates=round(v["dup_adjacent"] / n, 3), hamming_T=round(v["ham_T_guess"] / n, 3), hamming_final=round(v["ham_final_guess"] / n, 3),
                                changed=round(v["changed"] / n, 2), w2c_T=round(v["w2c_T"] / n, 3), c2w_T=round(v["c2w_T"] / n, 3), w2c_final=round(v["w2c_final"] / n, 3), c2w_final=round(v["c2w_final"] / n, 3))
    nb = max(1, fin["blocks"])
    summary = dict(sampler=a.sampler, n_prompts=len(prompts), gsm8k_accuracy=round(fin["correct"] / len(prompts), 4), rep4_mean=round(fin["rep4"] / len(prompts), 4),
                   final_block_vs_teacher_mismatch=round(fin["final_ne_T"] / (nb * K), 4), final_block_vs_ref_mismatch=round(fin["final_ne_R"] / max(1, fin["blocks_ref"] * K), 4) if fin["blocks_ref"] else None,
                   blocks_with_ref=fin["blocks_ref"], final_adjacent_duplicates_per_block=round(fin["final_dup"] / nb, 3),
                   oscillation=dict(positions_changed_value_ge1=round(osc["changed_ge1"] / max(1, osc["positions"]), 4), positions_changed_value_ge2=round(osc["changed_ge2"] / max(1, osc["positions"]), 4)),
                   paraphrase=dict(teacher_mismatch_in_task_correct_answers=round(para["correct"] / max(1, para["correct_K"]), 4), teacher_mismatch_in_task_wrong_answers=round(para["wrong"] / max(1, para["wrong_K"]), 4)),
                   order_sensitivity=dict(n_states=order["n"], left_completion_eq_T=round(order["left_eq_T"] / max(1, order["K"]), 4), remask_completion_eq_T=round(order["remask_eq_T"] / max(1, order["K"]), 4),
                                          left_eq_remask=round(order["left_eq_remask"] / max(1, order["K"]), 4)) if order["n"] else None,
                   per_step=per_step, forwards=n_fwd)
    print("=== mismatch", json.dumps({k: v for k, v in summary.items() if k != "per_step"}), flush=True)
    print("=== mismatch-steps", json.dumps(per_step), flush=True)
    save_json(a.out, dict(model=a.model, adapter=a.adapter, adapter2=a.adapter2, gpu=gpu_info(), seconds=round(time.time() - t0, 1), **summary))


if __name__ == "__main__":
    main()
