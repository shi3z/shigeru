"""Token <-> continuous latent codec for SHIGERU-LDM (Phase H1, option A: token-embedding projection).

z(x) = standardize(E[x] @ P), with E the frozen input embedding of the 4B (V x D, D = 2560) and P a fixed random
orthonormal projection D -> d (seeded, saved with the checkpoint).  Standardization (per latent dim, over the vocabulary)
keeps the latents at unit scale so that Gaussian noise of unit variance is a meaningful corruption.  Decoding is the
nearest codebook row (squared euclidean) — no parameters, so "Phase H1 reconstruction" measures only how invertible the
projection is at dimension d (reported by bench/eval_ldm.py: 16-slot exact match of oracle latents).  logits(z, tau)
gives -||z - C||^2 / tau over the vocabulary for a token cross-entropy on predicted latents (the optional token loss).
The [MASK] row (id `exclude`) is never decoded.
"""
import torch
import torch.nn as nn

__all__ = ["EmbedCodec"]


class EmbedCodec(nn.Module):
    def __init__(self, embed_weight, d_lat=256, seed=0, exclude=None):
        super().__init__()
        E = embed_weight.detach().float()
        V, D = E.shape
        g = torch.Generator().manual_seed(seed)
        P = torch.linalg.qr(torch.randn(D, d_lat, generator=g))[0]           # D x d orthonormal columns
        C = E @ P
        mu, sd = C.mean(0, keepdim=True), C.std(0, keepdim=True).clamp_min(1e-6)
        self.register_buffer("P", P)
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        self.register_buffer("codebook", ((C - mu) / sd).contiguous())      # V x d, unit scale per dim
        self.d_lat, self.V, self.exclude = d_lat, V, exclude
        self.register_buffer("cb_sq", (self.codebook ** 2).sum(-1))         # V

    def encode(self, ids):
        """ids [...] long -> z [..., d] (float32)."""
        return self.codebook[ids]

    def logits(self, z, tau=1.0):
        """z [..., d] -> [..., V] = -||z - C||^2 / tau (float32)."""
        zf = z.float()
        sq = (zf ** 2).sum(-1, keepdim=True)
        lg = -(sq - 2 * zf @ self.codebook.T + self.cb_sq) / tau
        if self.exclude is not None:
            lg[..., self.exclude] = float("-inf")
        return lg

    def decode(self, z):
        """z [..., d] -> token ids [...]: nearest codebook row."""
        return self.logits(z).argmax(-1)

    def extra_state(self):
        return dict(d_lat=self.d_lat, exclude=self.exclude)
