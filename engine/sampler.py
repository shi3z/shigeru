"""Block-diffusion sampling for the Open Mercury engine (LLaDA family).

generate(): semi-autoregressive blocks of `block_length` tokens; inside a block, denoising steps unmask tokens by confidence.
  mode "fixed":     exactly steps_per_block steps, k tokens per step (parity with the official sampler, low-confidence remasking)
  remask_threshold: selective remasking — decoded tokens of the current block whose confidence falls below it (and whose
                    argmax changed) are re-masked and re-decoded (bounded by remask_max per step and max_steps_per_block)
  mode "threshold": confidence-aware parallel decoding — unmask every masked position whose argmax probability >= threshold
                    (at least one per step), so the number of steps adapts to the text (dynamic step scheduling)
Caching: at each block start a full forward refreshes the dual cache (prompt + decoded blocks + masked future blocks);
subsequent steps of the block run a block-only forward against the cache. use_cache=False reproduces the no-cache baseline.
CUDA graphs (runner=GraphRunner): the cache is padded to a 64-token bucket with a key mask; block and full forwards are
captured once per shape and replayed.
"""
import math, time, torch, torch.nn.functional as F
from .llada import MASK_ID, EOS_ID
from .graphs import GraphRunner, bucket

@torch.no_grad()
def generate(model, prompt, gen_length=256, block_length=32, steps=None, mode="threshold", threshold=0.9,
             use_cache=True, runner=None, use_graph=False, stop_on_eos=False, remask_threshold=None, remask_max=2, max_steps_per_block=64, stats=None):
    B, P = prompt.shape
    assert B == 1, "single sequence for now"
    L = P + gen_length
    dev = prompt.device
    n_blocks = gen_length // block_length
    steps_per_block = (steps or gen_length) // n_blocks
    if runner is not None:                      # padded, masked cache + optional CUDA graphs
        Lb = bucket(L); cache = runner.get_cache(Lb)
        key_mask = torch.zeros(Lb, dtype=torch.bool, device=dev); key_mask[:L] = True
        x = torch.full((B, Lb), EOS_ID, dtype=torch.long, device=dev)
    else:
        Lb = L; cache = model.new_cache(B, L, dev) if use_cache else None; key_mask = None
        x = torch.full((B, L), MASK_ID, dtype=torch.long, device=dev)
    x[:, P:L] = MASK_ID; x[:, :P] = prompt
    pos_all = torch.arange(Lb, device=dev)
    def full_forward():
        if runner is not None:
            return runner.run(x, pos_all, key_mask, use_graph=use_graph)
        return model(x, cache=cache, start=0) if use_cache else model(x)
    def block_forward(bs, be):
        if runner is not None:
            return runner.run(x[:, bs:be], pos_all[bs:be], key_mask, use_graph=use_graph)
        return model(x[:, bs:be], cache=cache, start=bs)
    n_forward = 0; t_first = None; t0 = time.perf_counter()
    for nb in range(n_blocks):
        bs, be = P + nb * block_length, P + (nb + 1) * block_length
        logits = full_forward()[:, bs:be].clone(); n_forward += 1
        step = 0
        while True:
            masked = (x[:, bs:be] == MASK_ID)
            if not masked.any():
                break
            x0 = logits.argmax(-1)
            p = torch.gather(F.softmax(logits, -1), -1, x0.unsqueeze(-1)).squeeze(-1)
            conf = torch.where(masked, p, torch.full_like(p, -1.0))
            if mode == "threshold":
                sel = (conf >= threshold) & masked
                if not sel.any():
                    sel = torch.zeros_like(masked); sel[0, conf[0].argmax()] = True
            else:
                n_masked = int(masked.sum()); remaining = max(1, steps_per_block - step)
                k = n_masked if remaining == 1 else max(1, math.ceil(n_masked / remaining))
                sel = torch.zeros_like(masked); sel[0, conf[0].topk(k).indices] = True
            blk = x[:, bs:be]; blk[sel] = x0[sel]
            if remask_threshold is not None and step > 0:
                # selective remasking: a decoded (non-prompt) token of this block whose argmax probability dropped below the
                # remask threshold and whose argmax changed is put back to [MASK] (at most remask_max per step)
                decoded = (blk != MASK_ID) & ~sel
                bad = decoded & (p < remask_threshold) & (x0 != blk)
                if bad.any():
                    idx = torch.nonzero(bad[0]).flatten()
                    if idx.numel() > remask_max:
                        idx = idx[p[0, idx].argsort()[:remask_max]]
                    blk[0, idx] = MASK_ID
            x[:, bs:be] = blk
            step += 1
            if remask_threshold is not None and step > max_steps_per_block:
                # safety: finish the block greedily if remasking keeps the block open
                blk = x[:, bs:be]; m = (blk == MASK_ID); blk[m] = x0[m]; x[:, bs:be] = blk
            if t_first is None:
                torch.cuda.synchronize(); t_first = time.perf_counter() - t0
            if not (x[:, bs:be] == MASK_ID).any():
                break
            logits = block_forward(bs, be) if use_cache or runner is not None else model(x)[:, bs:be]
            logits = logits.clone() if runner is not None else logits
            n_forward += 1
        if stop_on_eos and (x[:, bs:be] == EOS_ID).all():
            break
    torch.cuda.synchronize()
    if stats is not None:
        stats.update(n_forward=n_forward, ttft_s=t_first, e2e_s=time.perf_counter() - t0, cache_len=Lb)
    return x[:, :L]
