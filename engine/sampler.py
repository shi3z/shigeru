"""Block-diffusion sampling for the Open Mercury engine (LLaDA family).

generate(): semi-autoregressive blocks of `block_length` tokens; inside a block, denoising steps unmask tokens by confidence.
  mode "fixed":     exactly steps_per_block steps, k tokens per step (parity with the official sampler, low-confidence remasking)
  mode "threshold": confidence-aware parallel decoding — unmask every masked position whose argmax probability >= threshold
                    (at least one per step), so the number of steps adapts to the text (dynamic step scheduling)
Caching: at each block start a full forward refreshes the dual cache (prompt + decoded blocks + masked future blocks);
subsequent steps of the block run a block-only forward against the cache. cache=False reproduces the no-cache baseline.
"""
import math, time, torch, torch.nn.functional as F
from .llada import MASK_ID, EOS_ID

@torch.no_grad()
def generate(model, prompt, gen_length=256, block_length=32, steps=None, mode="threshold", threshold=0.9,
             use_cache=True, remask_eos_after_first=False, stats=None):
    B, P = prompt.shape
    assert B == 1, "single sequence for now"
    L = P + gen_length
    x = torch.full((B, L), MASK_ID, dtype=torch.long, device=prompt.device)
    x[:, :P] = prompt
    n_blocks = gen_length // block_length
    steps_per_block = (steps or gen_length) // n_blocks
    cache = model.new_cache(B, L, prompt.device) if use_cache else None
    n_forward = 0; t_first = None; t0 = time.perf_counter()
    for nb in range(n_blocks):
        bs, be = P + nb * block_length, P + (nb + 1) * block_length
        # block start: full forward (refreshes the whole cache)
        logits = model(x, cache=cache, start=0) if use_cache else model(x)
        n_forward += 1
        step = 0
        while True:
            blk_logits = logits[:, bs:be]                     # [B, block, V]
            masked = (x[:, bs:be] == MASK_ID)                 # [B, block]
            if not masked.any():
                break
            x0 = blk_logits.argmax(-1)
            p = torch.gather(F.softmax(blk_logits, -1), -1, x0.unsqueeze(-1)).squeeze(-1)
            conf = torch.where(masked, p, torch.full_like(p, -1.0))
            if mode == "threshold":
                sel = (conf >= threshold) & masked
                if not sel.any():
                    sel = torch.zeros_like(masked); sel[0, conf[0].argmax()] = True
            else:  # fixed number of tokens per step, as the official sampler
                n_masked = int(masked.sum()); remaining_steps = max(1, steps_per_block - step)
                k = max(1, math.ceil(n_masked / remaining_steps)) if remaining_steps > 1 else n_masked
                sel = torch.zeros_like(masked); sel[0, conf[0].topk(k).indices] = True
            x[:, bs:be][sel] = x0[sel]
            step += 1
            if t_first is None:
                torch.cuda.synchronize(); t_first = time.perf_counter() - t0
            if not (x[:, bs:be] == MASK_ID).any():
                break
            if use_cache:
                logits_blk = model(x[:, bs:be], cache=cache, start=bs)    # block-only forward against the cache
                logits = torch.cat([logits[:, :bs], logits_blk, logits[:, be:]], 1)
            else:
                logits = model(x)
            n_forward += 1
    torch.cuda.synchronize()
    if stats is not None:
        stats.update(n_forward=n_forward, ttft_s=t_first, e2e_s=time.perf_counter() - t0)
    return x

