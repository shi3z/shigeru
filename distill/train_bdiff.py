"""B-series trainer (Stage B1 masked denoising, Stage B2 self-correction; brief §5-6).  LoRA on the 4B (base weights
untouched) + the time embedding; block-bidirectional attention over [prefix | K-block]; absorbing-mask corruption with a
schedule; optional visible-token corruption (random / on-policy states from distill/gen_onpolicy.py).

Loss per block (all positions get an x0 target = the clean token):
  L = mean_masked(1/gamma * CE) + lambda_vis * mean_corrupt_visible(CE) + lambda_id * mean_correct_visible(CE)
Data: stage-2 JSONL tokenized as train_cdlm; prefix = prompt + response[:c], block = response[c:c+K] (c uniform in the
response, blocks fully inside the response).  Batch = sequences of equal prefix length are not required: each sample
is forwarded alone (batch 1) and gradients are accumulated (`--accum`), which keeps the 4D mask simple.
Gate 1: `--overfit N` restricts the data to N fixed blocks and evaluates masked accuracy on them per mask-rate bucket.
Checkpoint: {out}/adapter (peft) + {out}/time_embed.pt + args.json; `--merge` writes {out}/final (merged HF weights).
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-train/bin/python distill/train_bdiff.py --model runs/zeta_sft_final_e5/final \
      --data data/distill/phaseA_stage2c_train.jsonl --out runs/bdiff_b1 --steps 3000"""
import argparse, json, math, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from distill.train_cdlm import load_jsonl, tokenize_examples
from engine.bdiff.corrupt import corrupt_block, SCHEDULES
from engine.bdiff.model import BDiffModel

BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--schedule", default="linear", choices=SCHEDULES)
    ap.add_argument("--p-replace", type=float, default=0.0, help="probability that a block also gets corrupted visible tokens")
    ap.add_argument("--r-replace", type=float, default=0.25)
    ap.add_argument("--states", default=None, help="on-policy states JSONL (gen_onpolicy); used for a fraction --p-state of the blocks")
    ap.add_argument("--p-state", type=float, default=0.0)
    ap.add_argument("--states-gold", action="store_true", help="use only the MASK PATTERN of the recorded states (visible tokens = clean tokens): on-policy masking without corruption")
    ap.add_argument("--lambda-vis", type=float, default=1.0)
    ap.add_argument("--lambda-id", type=float, default=0.1)
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--init-adapter", default=None, help="continue from an existing LoRA adapter dir (+ time_embed.pt next to it)")
    ap.add_argument("--t-min", type=float, default=0.0, help="sample t in [t_min, 1] (step-distillation emphasis on high noise)")
    ap.add_argument("--blocks", default=None, help="JSONL of {prefix_ids, x0} rows (e.g. teacher samples from gen_bstates --mode samples) used instead of --data cuts")
    ap.add_argument("--p-blocks", type=float, default=1.0, help="fraction of steps drawn from --blocks when both --blocks and --data are given")
    ap.add_argument("--data-tokens", default=None, help="pre-tokenized uint32 stream (distill/pretokenize.py) for raw-text blocks: prefix = a window of up to --max-len tokens, block = the next K")
    ap.add_argument("--p-tokens", type=float, default=0.5, help="fraction of samples drawn from --data-tokens when --data is also given")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); random.seed(a.seed)
    rng = torch.Generator().manual_seed(a.seed)
    prng = random.Random(a.seed)
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    mask_id = getattr(lm.config, "cdlm_mask_token_id", None) or tok.convert_tokens_to_ids("<|mask|>")
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    if a.init_adapter:
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, a.init_adapter, is_trainable=True)
    else:
        lm = get_peft_model(lm, lcfg)
    model = BDiffModel(lm, mask_id, time_cond=not a.no_time).to(dev)
    if model.time is not None:
        model.time.float()
        te = os.path.join(os.path.dirname(a.init_adapter.rstrip("/")), "time_embed.pt") if a.init_adapter else None
        if te and os.path.exists(te):
            model.time.load_state_dict(torch.load(te, map_location="cpu")); model.time.to(dev)
    params = [p for p in lm.parameters() if p.requires_grad] + (list(model.time.parameters()) if model.time is not None else [])
    n_train = sum(p.numel() for p in params)
    rows = load_jsonl(a.data)
    if a.limit:
        prng.shuffle(rows); rows = rows[: a.limit]
    items, dropped, trunc = tokenize_examples(rows, tok, a.max_len)
    items = [(t, P) for t, P in items if t.numel() - P >= a.K]
    states = []
    if a.states and a.p_state > 0:
        for r in load_jsonl(a.states):
            if "start" in r:                       # B-sampler states (gen_bstates): explicit start + state
                s0, st = int(r["start"]), r["state"]
            else:                                   # A-series states (gen_onpolicy): block index x block size
                b = int(list(r["states"].keys())[0]); st = r["states"][str(b)]; B = int(r["block"]); s0 = b * B
            if s0 + a.K <= len(r["tokens"]) and s0 >= r["prompt_len"] and len(st) >= a.K:
                states.append((r["tokens"], s0, st[: a.K]))
    blocks = [(r["prefix_ids"], r["x0"]) for r in load_jsonl(a.blocks)] if a.blocks else []
    tok_stream = None
    if a.data_tokens:
        import numpy as np
        tok_stream = np.memmap(a.data_tokens, dtype=np.uint32, mode="r")
        print(f"[train_bdiff] token stream {len(tok_stream) / 1e6:.0f} M tokens", flush=True)
    print(f"[train_bdiff] {len(items)} items, {len(states)} on-policy states, {len(blocks)} teacher blocks, trainable {n_train / 1e6:.1f} M, K={a.K}, time={not a.no_time}", flush=True)
    # fixed blocks for the overfit gate
    fixed = None
    if a.overfit:
        fixed = []
        for _ in range(a.overfit):
            t, P = items[prng.randrange(len(items))]
            c = prng.randint(P, t.numel() - a.K)
            fixed.append((t[:c].tolist(), t[c:c + a.K].tolist()))
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time()
    acc_bins = [[0, 0] for _ in BINS]
    model.train()
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        tot = dict(loss=0.0, mask=0.0, vis=0.0, idn=0.0)
        for _ in range(a.accum):
            use_state = states and float(torch.rand(1, generator=rng)) < a.p_state
            if fixed is not None:
                prefix, x0 = fixed[prng.randrange(len(fixed))]
                state = None
            elif blocks and (not items or float(torch.rand(1, generator=rng)) < a.p_blocks):
                prefix, x0 = blocks[prng.randrange(len(blocks))]
                prefix, x0, state = list(prefix), list(x0)[: a.K], None
            elif tok_stream is not None and (not items or float(torch.rand(1, generator=rng)) < a.p_tokens):
                plen = prng.randint(64, a.max_len - a.K)
                s0 = prng.randrange(0, len(tok_stream) - plen - a.K)
                seg = tok_stream[s0: s0 + plen + a.K].astype("int64").tolist()
                prefix, x0, state = seg[:plen], seg[plen:], None
            elif use_state:
                toks, s0, st = states[prng.randrange(len(states))]
                prefix, x0, state = toks[:s0], toks[s0:s0 + a.K], st
                if a.states_gold:
                    state = [-1 if v == -1 else x0[i] for i, v in enumerate(st)]
            else:
                t, P = items[prng.randrange(len(items))]
                c = prng.randint(P, t.numel() - a.K)
                prefix, x0, state = t[:c].tolist(), t[c:c + a.K].tolist(), None
            tt = a.t_min + (1.0 - a.t_min) * float(torch.rand(1, generator=rng))
            tt = max(tt, 1e-3)
            xt, is_mask, is_cor, g = corrupt_block(x0, tt, mask_id, rng, schedule=a.schedule, p_replace=a.p_replace,
                                                  r_replace=a.r_replace, vocab=151643, state=state)
            pre = torch.tensor([prefix], device=dev); blk = torch.tensor([xt], device=dev); tgt = torch.tensor(x0, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model.forward_block(pre, blk, torch.tensor([g], device=dev))[0].float()
            ce = F.cross_entropy(logits, tgt, reduction="none")
            m = torch.tensor(is_mask, device=dev); cr = torch.tensor(is_cor, device=dev)
            vis_ok = ~m & ~cr
            l_mask = (ce[m] / g).mean() if m.any() else ce.sum() * 0
            l_vis = ce[cr].mean() if cr.any() else ce.sum() * 0
            l_id = ce[vis_ok].mean() if vis_ok.any() else ce.sum() * 0
            loss = (l_mask + a.lambda_vis * l_vis + a.lambda_id * l_id) / a.accum
            loss.backward()
            tot["loss"] += loss.item(); tot["mask"] += l_mask.item() / a.accum; tot["vis"] += l_vis.item() / a.accum; tot["idn"] += l_id.item() / a.accum
            with torch.no_grad():
                if m.any():
                    ok = (logits.argmax(-1)[m] == tgt[m]).float().mean().item()
                    for j, (lo, hi) in enumerate(BINS):
                        if lo <= g < hi:
                            acc_bins[j][0] += ok; acc_bins[j][1] += 1; break
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % 10 == 0 or step == 1:
            rec = dict(step=step, loss=round(tot["loss"], 4), l_mask=round(tot["mask"], 4), l_vis=round(tot["vis"], 4), l_id=round(tot["idn"], 4),
                       lr=sched.get_last_lr()[0], grad_norm=round(float(gn), 3), elapsed=round(time.time() - t0, 1),
                       masked_acc_by_gamma={f"{lo:.2f}-{hi:.2f}": round(c[0] / c[1], 3) for (lo, hi), c in zip(BINS, acc_bins) if c[1]},
                       peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
            acc_bins = [[0, 0] for _ in BINS]
        if step % a.save_every == 0 or step == a.steps:
            lm.save_pretrained(os.path.join(a.out, "adapter"))
            if model.time is not None:
                torch.save(model.time.state_dict(), os.path.join(a.out, "time_embed.pt"))
    if a.merge:
        merged = lm.merge_and_unload()
        merged.save_pretrained(os.path.join(a.out, "final"), safe_serialization=True)
        tok.save_pretrained(os.path.join(a.out, "final"))
        try:
            from distill.fix_config import fix as _fix_cfg
            _fix_cfg(os.path.join(a.out, "final"))
        except Exception as e:
            print("fix_config skipped:", e)
    print(f"=== train_bdiff done {a.steps} steps in {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
