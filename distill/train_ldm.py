"""SHIGERU-LDM pilot trainer (Phase H2/H3/H4 in one): flow matching over K future-token latents conditioned on the frozen
4B's prefix hidden states.  docs/latent_diffusion_research.md.

Each step: a batch of tokenized stage-2 examples (prompt + response, as distill/train_cdlm.py tokenizes them) goes
through the FROZEN 4B once (bf16, no grad) -> last hidden state at every position; for every sequence `--cuts` cut points
c are drawn uniformly in [prompt_len, L - K]; the condition is hidden[:c] (the last --cond-max positions), the target is
z0 = codec(tokens[c:c+K]).  t ~ U(0,1), eps ~ N(0,I), z_t = (1-t) eps + t z0, loss = MSE(v_pred, z0 - eps)
(+ --token-loss * CE(codec.logits(z0_hat), tokens) with z0_hat = z_t + (1-t) v_pred).  Only the denoiser trains.
Checkpoints: {out}/denoiser.pt (state dict + hparams + codec seed/d_lat + args) every --save-every steps; log.jsonl.
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-train/bin/python distill/train_ldm.py --model runs/phaseC_sft3_4b/final \
      --data data/distill/phaseA_stage2c_train.jsonl --out runs/ldm_pilot_k16 --steps 6000"""
import argparse, json, math, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from distill.train_cdlm import load_jsonl, tokenize_examples
from engine.ldm.codec import EmbedCodec
from engine.ldm.denoiser import LatentDenoiser, flow_interpolate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--d-lat", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--predict", default="v", choices=["v", "z0"])
    ap.add_argument("--no-cond", action="store_true", help="ablation E: no prefix conditioning")
    ap.add_argument("--batch", type=int, default=8, help="sequences per step")
    ap.add_argument("--cuts", type=int, default=4, help="cut points per sequence")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--cond-max", type=int, default=768)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--token-loss", type=float, default=0.0)
    ap.add_argument("--token-tau", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    mask_id = getattr(lm.config, "cdlm_mask_token_id", None)
    codec = EmbedCodec(lm.get_input_embeddings().weight, d_lat=a.d_lat, seed=a.seed, exclude=mask_id).to(dev)
    den = LatentDenoiser(K=a.K, d_lat=a.d_lat, d_model=a.d_model, n_layers=a.layers, n_heads=a.heads,
                         d_cond=lm.config.hidden_size, predict=a.predict, use_cond=not a.no_cond).to(dev)
    n_params = sum(p.numel() for p in den.parameters())
    rows = load_jsonl(a.data)
    if a.limit:
        random.shuffle(rows); rows = rows[: a.limit]
    items, dropped, trunc = tokenize_examples(rows, tok, a.max_len)
    items = [(t, P) for t, P in items if t.numel() - P >= a.K + 1]
    print(f"[train_ldm] {len(items)} items (dropped {dropped}, truncated {trunc}); denoiser {n_params / 1e6:.1f} M params; "
          f"codec d={a.d_lat}; K={a.K}", flush=True)
    opt = torch.optim.AdamW(den.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    rng = random.Random(a.seed)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    t0 = time.time()
    ema = None
    for step in range(1, a.steps + 1):
        batch = rng.sample(items, a.batch)
        L = max(t.numel() for t, _ in batch)
        ids = torch.full((a.batch, L), tok.pad_token_id or tok.eos_token_id, dtype=torch.long)
        att = torch.zeros((a.batch, L), dtype=torch.long)
        for i, (t, _) in enumerate(batch):
            ids[i, : t.numel()] = t; att[i, : t.numel()] = 1
        with torch.no_grad():
            hid = lm.model(input_ids=ids.to(dev), attention_mask=att.to(dev)).last_hidden_state      # [B, L, D] bf16
        # cut points
        conds, pads, targets = [], [], []
        for i, (t, P) in enumerate(batch):
            n = t.numel()
            for _ in range(a.cuts):
                c = rng.randint(P, n - a.K)
                lo = max(0, c - a.cond_max)
                conds.append(hid[i, lo:c]); targets.append(t[c:c + a.K])
        Lc = max(x.shape[0] for x in conds)
        N = len(conds)
        cond = torch.zeros((N, Lc, hid.shape[-1]), dtype=hid.dtype, device=dev)
        pad = torch.ones((N, Lc), dtype=torch.bool, device=dev)
        for j, x in enumerate(conds):
            cond[j, : x.shape[0]] = x; pad[j, : x.shape[0]] = False
        tgt = torch.stack(targets).to(dev)                                                             # [N, K]
        z0 = codec.encode(tgt)                                                                         # [N, K, d]
        eps = torch.randn_like(z0)
        tt = torch.rand(N, device=dev)
        z_t, v_tgt = flow_interpolate(z0, eps, tt)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = den(z_t, tt, None if a.no_cond else cond, None if a.no_cond else pad)
        out = out.float()
        if a.predict == "v":
            loss_diff = F.mse_loss(out, v_tgt)
            z0_hat = z_t + (1 - tt)[:, None, None] * out
        else:
            loss_diff = F.mse_loss(out, z0)
            z0_hat = out
        loss = loss_diff
        loss_tok = torch.zeros(())
        if a.token_loss > 0:
            loss_tok = F.cross_entropy(codec.logits(z0_hat, a.token_tau).reshape(-1, codec.V), tgt.reshape(-1))
            loss = loss + a.token_loss * loss_tok
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(den.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            acc0 = (codec.decode(z0_hat) == tgt).float().mean().item()
        ema = loss_diff.item() if ema is None else 0.98 * ema + 0.02 * loss_diff.item()
        if step % 20 == 0 or step == 1:
            rec = dict(step=step, loss_diff=round(loss_diff.item(), 5), loss_ema=round(ema, 5), loss_tok=round(loss_tok.item(), 4),
                       acc_z0hat=round(acc0, 4), lr=sched.get_last_lr()[0], grad_norm=round(float(gn), 3),
                       elapsed=round(time.time() - t0, 1), peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
        if step % a.save_every == 0 or step == a.steps:
            torch.save(dict(denoiser=den.state_dict(), hparams=den.hparams, codec=dict(d_lat=a.d_lat, seed=a.seed, exclude=mask_id),
                            args=vars(a), step=step), os.path.join(a.out, "denoiser.pt"))
    print(f"=== train_ldm done {a.steps} steps in {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
