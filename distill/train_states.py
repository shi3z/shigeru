"""Phase B11-D — sampler-aligned masked CE (docs/sampler_aligned_global_denoising.md §5).

Trains a new LoRA on the merged B-series model with the plain masked cross-entropy of B10.5 (arm B), but on the
STATES of the sampler that will be used at inference (distill/gen_global_states.py): for a state x (masks + visible
tokens, possibly wrong) the loss is CE at the masked positions against the teacher target; visible positions are never
targets.  --target plain uses the teacher completion conditioned on the current visible tokens; --target corr uses the
teacher-corrected state (wrong visible tokens re-masked, teacher block as target); --target mix alternates.
Several data prefixes can be given (e.g. remask + left states) and are sampled uniformly over rows.
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv-train/bin/python distill/train_states.py --model runs/zeta_sft_final_e5/final \
      --adapter runs/bdiff_e5_lr1e4/adapter --data data/distill/gstates_remask --target plain --steps 1000 --out runs/sa_remask_plain"""
import argparse, glob, json, math, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from engine.bdiff.masks import train_mask


def load(prefixes, holdout=0.05, seed=0, max_rows=None):
    rows = []; prompts = {}
    for pref in prefixes:
        pr = {}
        for line in open(f"{pref}_prompts.jsonl"):
            r = json.loads(line); pr[r["idx"]] = r
        for f in sorted(glob.glob(f"{pref}_[0-9][0-9][0-9].pt")):
            d = torch.load(f)
            for n in range(d["meta"].shape[0]):
                pi, blk, s, P, Kb, wrong, vis = d["meta"][n].tolist()
                if pi not in pr:
                    continue
                rows.append(dict(pref=pref, pi=pi, blk=blk, step=s, Kb=Kb, x_plain=d["x_plain"][n].tolist(), y_plain=d["y_plain"][n].tolist(),
                                 x_corr=d["x_corr"][n].tolist(), y_corr=d["y_corr"][n].tolist(), wrong=wrong))
        prompts.update({(pref, k): v for k, v in pr.items()})
    rng = random.Random(seed)
    keys = sorted(prompts); rng.shuffle(keys)
    hold = set(keys[: max(1, int(len(keys) * holdout))])
    train = [r for r in rows if (r["pref"], r["pi"]) not in hold]; held = [r for r in rows if (r["pref"], r["pi"]) in hold]
    if max_rows:
        rng.shuffle(train); train = train[:max_rows]
    return train, held, prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--target", default="plain", choices=["plain", "corr", "mix"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.model)
    from peft import PeftModel, LoraConfig, get_peft_model
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    base = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    mask_id = getattr(base.config, "cdlm_mask_token_id", None) or tok.convert_tokens_to_ids("<|mask|>")
    student = get_peft_model(base, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none",
                                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    student.print_trainable_parameters()
    train, held, prompts = load(a.data, seed=a.seed, max_rows=a.max_rows)
    print(f"train states {len(train)}, held-out {len(held)}; wrong-visible per state (train) {sum(r['wrong'] for r in train) / max(1, len(train)):.2f}", flush=True)
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))

    def loss_of(r, which):
        pr = prompts[(r["pref"], r["pi"])]; K = 16
        prefix = pr["ids"] + pr["out"][: r["blk"] * K]
        x = r[f"x_{which}"][: r["Kb"]]; y = r[f"y_{which}"][: r["Kb"]]
        tgt = [q for q in range(len(x)) if y[q] >= 0]
        if not tgt:
            return None, 0
        ids = torch.tensor([prefix + x], device=dev); P = len(prefix); Kb = len(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = student(input_ids=ids, attention_mask=train_mask(P, Kb, device=dev)[None, None], position_ids=torch.arange(P + Kb, device=dev)[None])
        lg = out.logits[0, P:].float(); lg[:, mask_id] = -1e4
        logp = torch.log_softmax(lg, -1)
        l = -torch.stack([logp[q, y[q]] for q in tgt]).mean()
        acc = sum(int(lg[q].argmax()) == y[q] for q in tgt) / len(tgt)
        return l, acc

    def evaluate(rows, n=300):
        student.eval(); tot = dict(plain=[0.0, 0.0, 0], corr=[0.0, 0.0, 0])
        with torch.no_grad():
            for r in rows[:n]:
                for which in ("plain", "corr"):
                    l, acc = loss_of(r, which)
                    if l is not None:
                        tot[which][0] += float(l); tot[which][1] += acc; tot[which][2] += 1
        student.train()
        return {w: dict(loss=round(v[0] / max(1, v[2]), 4), masked_acc=round(v[1] / max(1, v[2]), 4), n=v[2]) for w, v in tot.items()}

    log = open(os.path.join(a.out, "log.jsonl"), "a")
    ev = evaluate(held); print("step 0 held-out", json.dumps(ev), flush=True); log.write(json.dumps(dict(step=0, heldout=ev)) + "\n")
    student.train(); step = 0; t0 = time.time(); run = [0.0, 0.0, 0]; k = 0
    while step < a.steps:
        order = list(range(len(train))); random.shuffle(order)
        for idx in order:
            which = a.target if a.target != "mix" else ("plain" if k % 2 == 0 else "corr"); k += 1
            l, acc = loss_of(train[idx], which)
            if l is None:
                continue
            (l / a.accum).backward(); run[0] += float(l); run[1] += acc; run[2] += 1
            if run[2] % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 25 == 0:
                    rec = dict(step=step, loss=round(run[0] / run[2], 4), masked_acc=round(run[1] / run[2], 4), lr=sched.get_last_lr()[0], elapsed=round(time.time() - t0, 1),
                               peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
                    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush(); run = [0.0, 0.0, 0]
                if step % a.eval_every == 0:
                    ev = evaluate(held); print(f"step {step} held-out", json.dumps(ev), flush=True); log.write(json.dumps(dict(step=step, heldout=ev)) + "\n"); log.flush()
                if step >= a.steps:
                    break
    student.save_pretrained(os.path.join(a.out, "adapter"))
    print(f"=== train_states done {step} steps {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
