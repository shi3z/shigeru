"""Latent denoiser for SHIGERU-LDM: a small pre-LN transformer over K latent slots, conditioned on the frozen 4B's prefix
hidden states by cross-attention, with a sinusoidal time embedding.  Predicts the flow-matching velocity v = z0 - eps
(rectified flow, z_t = (1 - t) eps + t z0, t in [0, 1]); `predict` may also be "z0" (x0-prediction) for the ablation.

Sampling (`sample`): Euler (or Heun) integration from t = 0 (pure noise) to 1 in S steps; `from_t` with a given z_t
starts mid-way (noisy-oracle reconstruction).  Nothing here reads token logits of the 4B: conditioning is on hidden
states only; the output latents are decoded by codec.decode (nearest codebook row).
Parameters: in/out projections (d_lat <-> d_model), slot embeddings [K, d_model], time MLP, cond projection
(d_cond -> d_model) + LayerNorm, n_layers x (self-attn, cross-attn, FFN 4x).  d_model 1024 / 8 layers ~ 120 M.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LatentDenoiser", "flow_interpolate", "sample"]


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * 1000.0 * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


class Block(nn.Module):
    def __init__(self, d, heads, d_cond_proj):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True, kdim=d_cond_proj, vdim=d_cond_proj)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, cond, cond_pad):
        h = self.ln1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        if cond is not None:
            h = self.ln2(x)
            x = x + self.cross_attn(h, cond, cond, key_padding_mask=cond_pad, need_weights=False)[0]
        return x + self.ff(self.ln3(x))


class LatentDenoiser(nn.Module):
    def __init__(self, K=16, d_lat=256, d_model=1024, n_layers=8, n_heads=16, d_cond=2560, predict="v", use_cond=True):
        super().__init__()
        assert predict in ("v", "z0")
        self.K, self.d_lat, self.d_model, self.predict, self.use_cond = K, d_lat, d_model, predict, use_cond
        self.in_proj = nn.Linear(d_lat, d_model)
        self.slot = nn.Parameter(torch.randn(K, d_model) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(256, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.cond_proj = nn.Sequential(nn.Linear(d_cond, d_model), nn.LayerNorm(d_model)) if use_cond else None
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_model) for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_lat)
        nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)
        self.hparams = dict(K=K, d_lat=d_lat, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_cond=d_cond,
                            predict=predict, use_cond=use_cond)

    def forward(self, z_t, t, cond=None, cond_pad=None):
        """z_t [B, K, d_lat], t [B] in [0,1], cond [B, Lc, d_cond] (float), cond_pad [B, Lc] bool True = PAD."""
        x = self.in_proj(z_t.float()) + self.slot[None] + self.time_mlp(timestep_embedding(t, 256))[:, None]
        c = self.cond_proj(cond.float()) if (self.use_cond and cond is not None) else None
        for blk in self.blocks:
            x = blk(x, c, cond_pad)
        return self.out_proj(self.ln_out(x))

    def velocity(self, z_t, t, cond=None, cond_pad=None):
        out = self.forward(z_t, t, cond, cond_pad)
        if self.predict == "v":
            return out
        # z0-prediction -> velocity: v = (z0_hat - z_t) / (1 - t)
        return (out - z_t.float()) / (1.0 - t.float()).clamp_min(0.05)[:, None, None]


def flow_interpolate(z0, eps, t):
    """z_t = (1 - t) eps + t z0, target velocity z0 - eps."""
    tt = t.float()[:, None, None]
    return (1 - tt) * eps + tt * z0, z0 - eps


@torch.no_grad()
def sample(denoiser, cond, cond_pad, K, d_lat, steps=16, method="euler", z_t=None, from_t=0.0, generator=None, device=None):
    """Integrate the flow from t = from_t to 1 in `steps` steps.  z_t: starting latent (noise if None)."""
    B = cond.shape[0] if cond is not None else 1
    device = device or (cond.device if cond is not None else "cpu")
    if z_t is None:
        z_t = torch.randn(B, K, d_lat, generator=generator, device="cpu").to(device)
    z = z_t.float()
    ts = torch.linspace(from_t, 1.0, steps + 1, device=device)
    for i in range(steps):
        t0, t1 = ts[i], ts[i + 1]
        dt = (t1 - t0)
        tb = t0.expand(B)
        v = denoiser.velocity(z, tb, cond, cond_pad)
        if method == "heun" and i < steps - 1:
            z_e = z + dt * v
            v2 = denoiser.velocity(z_e, t1.expand(B), cond, cond_pad)
            z = z + dt * 0.5 * (v + v2)
        else:
            z = z + dt * v
    return z
