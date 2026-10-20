"""Open Mercury engine — our own implementation of the LLaDA-8B masked-diffusion transformer with a block-diffusion KV cache.

Architecture (from the checkpoint config, Llama-style, bidirectional attention): d_model 4096, 32 layers, 32 heads (no GQA),
head_dim 128, SwiGLU with gate=ff_proj / up=up_proj / down=ff_out, RMSNorm(eps 1e-5, affine, no bias), RoPE theta 5e5 (full
precision, rotate-half), untied LM head (transformer.ff_out), vocab 126464, [MASK] id 126336, EOS 126081.

Cache design: per layer K and V for ALL sequence positions (prompt + every generation block). A full forward refreshes the
whole cache; a block forward computes q/k/v only for the active block's positions, writes its k/v into the cache in place and
attends over the complete cache (prefix, the block itself and the still-masked future blocks) — the "dual cache" idea of
Fast-dLLM, reimplemented here from the paper's description. Nothing in this file is copied from the reference code.
"""
import json, math, os, glob
import torch, torch.nn as nn, torch.nn.functional as F
from safetensors.torch import load_file

MASK_ID, EOS_ID = 126336, 126081

class Config:
    def __init__(self, path):
        c = json.load(open(os.path.join(path, "config.json")))
        self.d_model, self.n_layers, self.n_heads = c["d_model"], c["n_layers"], c["n_heads"]
        self.n_kv_heads = c.get("n_kv_heads") or c["n_heads"]
        self.head_dim = self.d_model // self.n_heads
        self.mlp_hidden, self.vocab = c["mlp_hidden_size"], c["vocab_size"]
        self.eps, self.rope_theta = c.get("rms_norm_eps", 1e-5), c.get("rope_theta", 500000.0)
        self.max_seq = c.get("max_sequence_length", 4096)
        assert c.get("layer_norm_type", "rms") == "rms" and c.get("block_type") == "llama" and c.get("activation_type") == "silu"

class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight

def rope_cos_sin(cfg, device):
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, device=device).float() / cfg.head_dim))
    t = torch.arange(cfg.max_seq, device=device).float()
    f = torch.outer(t, inv); emb = torch.cat([f, f], -1)
    return emb.cos(), emb.sin()  # [max_seq, head_dim] fp32

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], -1)

def apply_rope(x, cos, sin):
    # x [B,H,T,hd] (bf16), cos/sin [T,hd] fp32 -> full-precision rotary like the reference (rope_full_precision)
    xf = x.float()
    return (xf * cos[None, None] + rotate_half(xf) * sin[None, None]).to(x.dtype)

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.attn_norm, self.ff_norm = RMSNorm(d, cfg.eps), RMSNorm(d, cfg.eps)
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.attn_out = nn.Linear(d, d, bias=False)
        self.ff_proj, self.up_proj = nn.Linear(d, cfg.mlp_hidden, bias=False), nn.Linear(d, cfg.mlp_hidden, bias=False)
        self.ff_out = nn.Linear(cfg.mlp_hidden, d, bias=False)
        self.cfg = cfg

    def forward(self, x, cos, sin, cache=None, start=0):
        """x [B,T,d] for sequence positions [start, start+T). cache: (K,V) tensors [B,H,L,hd] for the whole sequence or None.
        With a cache the block's k/v are written in place and attention spans all L positions."""
        B, T, _ = x.shape; H, hd = self.cfg.n_heads, self.cfg.head_dim
        h = self.attn_norm(x)
        q = self.q_proj(h).view(B, T, H, hd).transpose(1, 2)
        k = self.k_proj(h).view(B, T, H, hd).transpose(1, 2)
        v = self.v_proj(h).view(B, T, H, hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is None:
            K, V = k, v
        else:
            K, V = cache
            K[:, :, start:start + T] = k; V[:, :, start:start + T] = v
        a = F.scaled_dot_product_attention(q, K, V, is_causal=False)  # bidirectional
        x = x + self.attn_out(a.transpose(1, 2).reshape(B, T, -1))
        h = self.ff_norm(x)
        x = x + self.ff_out(F.silu(self.ff_proj(h)) * self.up_proj(h))
        return x

class LLaDA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.d_model, cfg.eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.register_buffer("cos", torch.zeros(1), persistent=False); self.register_buffer("sin", torch.zeros(1), persistent=False)

    @classmethod
    def from_pretrained(cls, path, device="cuda", dtype=torch.bfloat16):
        cfg = Config(path)
        with torch.device("meta"):
            model = cls(cfg)
        sd = {}
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            sd.update(load_file(f))
        def take(name):
            return sd.pop(name).to(dtype)
        model.wte.weight = nn.Parameter(take("model.transformer.wte.weight"), requires_grad=False)
        model.ln_f.weight = nn.Parameter(take("model.transformer.ln_f.weight"), requires_grad=False)
        model.lm_head.weight = nn.Parameter(take("model.transformer.ff_out.weight"), requires_grad=False)
        for i, b in enumerate(model.blocks):
            p = f"model.transformer.blocks.{i}."
            for mod, key in (("attn_norm", "attn_norm"), ("ff_norm", "ff_norm"), ("q_proj", "q_proj"), ("k_proj", "k_proj"), ("v_proj", "v_proj"),
                             ("attn_out", "attn_out"), ("ff_proj", "ff_proj"), ("up_proj", "up_proj"), ("ff_out", "ff_out")):
                getattr(b, mod).weight = nn.Parameter(take(p + key + ".weight"), requires_grad=False)
        assert not sd, f"unused weights: {list(sd)[:5]}"
        model = model.to(device)
        model.cos, model.sin = rope_cos_sin(cfg, device)
        return model.eval()

    def new_cache(self, B, L, device="cuda", dtype=torch.bfloat16):
        H, hd = self.cfg.n_heads, self.cfg.head_dim
        return [(torch.empty(B, H, L, hd, device=device, dtype=dtype), torch.empty(B, H, L, hd, device=device, dtype=dtype)) for _ in self.blocks]

    @torch.no_grad()
    def forward(self, tokens, cache=None, start=0):
        """tokens [B,T] at positions [start, start+T). Returns logits [B,T,V] (float32).
        cache=None: plain full forward (no cache). cache given: write this span's K/V and attend over the full cache."""
        T = tokens.shape[1]
        cos, sin = self.cos[start:start + T], self.sin[start:start + T]
        x = self.wte(tokens)
        for i, b in enumerate(self.blocks):
            x = b(x, cos, sin, cache[i] if cache is not None else None, start)
        return self.lm_head(self.ln_f(x)).float()
