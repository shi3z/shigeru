"""Phase B10-B Stage 2 — conditional distribution distillation from the B8 joint teacher into a parallel student
(docs/global_consistency_distillation.md §3–4).

Teacher = the frozen B-series model (e5 base + lr1e-4 adapter merged) that may spend extra forwards; student = the same
weights plus a NEW trainable LoRA, one forward per step at inference.  Data = states of the student's own S = 8
left-to-right trajectories where two positions (i, j) are placed (distill/gen_pairs.py: block state x, top-k candidates,
delta = log p(b | a) - log p(b), and the generated tokens of every prompt).  For each state:
  a*, b*        the teacher's pair = argmax_{a,b} log p(a) + log p(b | a)        (chain-rule joint, 2 forwards)
  p_T(. | a*)   the teacher's full conditional at j with a* placed at i          (recomputed online, 1 forward)
  student       ONE forward on the state (i and j both masked)
  loss  = CE(student_i, a*)                                  the first token: what the joint picks
        + lambda_kl * KL(p_T(. | a*) || student_j)           the second token: the resolved conditional, not the marginal
        + lambda_mask * CE(student_q, final_q) over the other masked positions q (targets = the block the teacher-
          followed trajectory ended with) — keeps the ordinary denoising ability on the rest of the block
The student therefore learns to output, at j, in the same forward, the distribution the teacher only reaches after
committing i — the pair becomes coherent under independent argmax.  Nothing sequential is added at inference.
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv-train/bin/python distill/train_consist.py --model runs/zeta_sft_final_e5/final \
      --adapter runs/bdiff_e5_lr1e4/adapter --data data/distill/pairs_gsm8k --steps 1500 --out runs/consist_s2"""
import argparse, glob, json, math, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from engine.bdiff.masks import train_mask

MASK_FALLBACK = "<|mask|>"


def load_rows(prefix, max_rows=None, holdout_prompts=0.05, seed=0):
    prompts = {}
    for line in open(f"{prefix}_prompts.jsonl"):
        r = json.loads(line); prompts[r["idx"]] = r
    rows = []
    for f in sorted(glob.glob(f"{prefix}_[0-9][0-9][0-9].pt")):
        d = torch.load(f)
        for n in range(d["meta"].shape[0]):
            pi, blk, s, P, i, j = d["meta"][n].tolist()
            if pi not in prompts:
                continue
            rows.append(dict(pi=pi, blk=blk, step=s, P=P, i=i, j=j, x=d["x"][n], cand=d["cand"][n], lp=d["lp"][n], delta=d["delta"][n]))
    rng = random.Random(seed)
    pids = sorted(prompts)
    rng.shuffle(pids)
    n_hold = max(1, int(len(pids) * holdout_prompts))
    hold = set(pids[:n_hold])
    train = [r for r in rows if r["pi"] not in hold]; held = [r for r in rows if r["pi"] in hold]
    if max_rows:
        train = train[:max_rows]
    return train, held, prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True, help="teacher adapter (merged into both teacher and student base)")
    ap.add_argument("--data", required=True, nargs="+", help="one or more gen_pairs output prefixes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lambda-kl", type=float, default=1.0)
    ap.add_argument("--lambda-mask", type=float, default=1.0)
    ap.add_argument("--kl-topk", type=int, default=0, help="0 = full-vocabulary KL; else KL over the teacher's top-k plus the rest mass")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.model)
    from peft import PeftModel, LoraConfig, get_peft_model
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    base = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    mask_id = getattr(base.config, "cdlm_mask_token_id", None) or tok.convert_tokens_to_ids(MASK_FALLBACK)
    teacher = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    teacher = PeftModel.from_pretrained(teacher, a.adapter).merge_and_unload().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    student = get_peft_model(base, lcfg)
    student.print_trainable_parameters()
    train, held, prompts = [], [], {}
    for pref in a.data:
        tr, hd, pr = load_rows(pref, a.max_rows, seed=a.seed)
        train += tr; held += hd; prompts.update({(pref, k): v for k, v in pr.items()})
        for r in tr + hd:
            r["pref"] = pref
    print(f"train rows {len(train)}, held-out rows {len(held)} (prompts held out: {len(set((r['pref'], r['pi']) for r in held))})", flush=True)
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))

    def block_forward(model, prefix_ids, block_ids):
        """[prefix | block] with the block-bidirectional train mask -> logits of the block rows [K, V]."""
        ids = torch.tensor([prefix_ids + block_ids], device=dev)
        P, K = len(prefix_ids), len(block_ids)
        mask = train_mask(P, K, device=dev)[None, None]
        pos = torch.arange(P + K, device=dev)[None]
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos)
        return out.logits[0, P:]

    def teacher_targets(prefix, x, i, j, cand, lp, delta):
        """a*, b* by the chain-rule joint over the stored candidates; full conditional at j given a*."""
        k = cand.shape[1]
        jscore = lp[0][:, None] + lp[1][None, :] + delta
        best = int(jscore.argmax()); ai, bj = divmod(best, k)
        a_star = int(cand[0][ai]); b_star = int(cand[1][bj])
        x2 = list(x); x2[i] = a_star
        with torch.no_grad():
            lg = block_forward(teacher, prefix, x2).float()
            lg[:, mask_id] = -1e4                       # finite: the KL below multiplies p_t by log-probs
            p_t = torch.softmax(lg[j], -1)
        return a_star, b_star, p_t

    def state_of(r):
        pr = prompts[(r["pref"], r["pi"])]
        K = r["x"].shape[0]
        prefix = pr["ids"] + pr["out"][: r["blk"] * K]
        final = pr["out"][r["blk"] * K: r["blk"] * K + K]
        if len(final) < K:
            final = final + [mask_id] * (K - len(final))
        return prefix, r["x"].tolist(), final

    def losses(r, train_mode=True):
        prefix, x, final = state_of(r)
        i, j = r["i"], r["j"]
        a_star, b_star, p_t = teacher_targets(prefix, x, i, j, r["cand"], r["lp"], r["delta"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = block_forward(student, prefix, x)
        lg = lg.float(); lg[:, mask_id] = -1e4
        logp = torch.log_softmax(lg, -1)
        l_i = -logp[i, a_star]
        if a.kl_topk:
            tk = p_t.topk(a.kl_topk).indices
            pt = torch.cat([p_t[tk], (1 - p_t[tk].sum()).clamp_min(1e-8)[None]])
            ps = torch.cat([logp[j, tk].exp(), (1 - logp[j, tk].exp().sum()).clamp_min(1e-8)[None]])
            l_kl = (pt * (pt.clamp_min(1e-12).log() - ps.clamp_min(1e-12).log())).sum()
        else:
            l_kl = (p_t * (p_t.clamp_min(1e-12).log() - logp[j])).sum()
        others = [q for q in range(len(x)) if x[q] == mask_id and q not in (i, j) and final[q] != mask_id]
        l_mask = -torch.stack([logp[q, final[q]] for q in others]).mean() if others else lg.new_zeros(())
        with torch.no_grad():
            pair_ok = float(int(lg[i].argmax()) == a_star and int(lg[j].argmax()) == b_star)
            j_ok = float(int(lg[j].argmax()) == b_star)
            j_ok_teacher_marginal = float(int(r["cand"][1][int(r["lp"][1].argmax())]) == b_star)   # what the un-distilled marginal would pick
        return l_i, l_kl, l_mask, pair_ok, j_ok, j_ok_teacher_marginal

    def evaluate(rows, n=200):
        student.eval(); acc = dict(l_i=0.0, l_kl=0.0, l_mask=0.0, pair=0.0, j=0.0, j_marg=0.0); m = 0
        with torch.no_grad():
            for r in rows[:n]:
                l_i, l_kl, l_mask, pair_ok, j_ok, j_m = losses(r, False)
                acc["l_i"] += float(l_i); acc["l_kl"] += float(l_kl); acc["l_mask"] += float(l_mask); acc["pair"] += pair_ok; acc["j"] += j_ok; acc["j_marg"] += j_m; m += 1
        student.train()
        return {k: round(v / max(1, m), 4) for k, v in acc.items()}

    log = open(os.path.join(a.out, "log.jsonl"), "a")
    ev0 = evaluate(held); print("step 0 held-out", json.dumps(ev0), flush=True); log.write(json.dumps(dict(step=0, heldout=ev0)) + "\n")
    student.train(); t0 = time.time(); step = 0; run = dict(l_i=0.0, l_kl=0.0, l_mask=0.0, pair=0.0, n=0)
    order = list(range(len(train)))
    while step < a.steps:
        random.shuffle(order)
        for idx in order:
            l_i, l_kl, l_mask, pair_ok, _, _ = losses(train[idx])
            loss = (l_i + a.lambda_kl * l_kl + a.lambda_mask * l_mask) / a.accum
            loss.backward()
            run["l_i"] += float(l_i); run["l_kl"] += float(l_kl); run["l_mask"] += float(l_mask); run["pair"] += pair_ok; run["n"] += 1
            if run["n"] % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 25 == 0:
                    n = run["n"]
                    rec = dict(step=step, l_i=round(run["l_i"] / n, 4), l_kl=round(run["l_kl"] / n, 4), l_mask=round(run["l_mask"] / n, 4), pair_acc=round(run["pair"] / n, 4),
                               lr=sched.get_last_lr()[0], elapsed=round(time.time() - t0, 1), peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
                    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
                    run = dict(l_i=0.0, l_kl=0.0, l_mask=0.0, pair=0.0, n=0)
                if step % a.eval_every == 0:
                    ev = evaluate(held); print(f"step {step} held-out", json.dumps(ev), flush=True); log.write(json.dumps(dict(step=step, heldout=ev)) + "\n"); log.flush()
                if step % a.save_every == 0 or step == a.steps:
                    student.save_pretrained(os.path.join(a.out, "adapter"))
                if step >= a.steps:
                    break
    student.save_pretrained(os.path.join(a.out, "adapter"))
    print(f"=== train_consist done {step} steps {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
