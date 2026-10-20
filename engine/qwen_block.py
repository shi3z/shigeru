"""Open Mercury — block-diffusion Qwen2.5 (the Fast-dLLM v2 architecture) with a frozen-prefix / mutable-block KV cache.

Architecture (Qwen2.5-7B-Instruct base): hidden 3584, 28 layers, 28 query heads / 4 KV heads (GQA), head_dim 128, q/k/v
biases, SwiGLU 18944, RMSNorm eps 1e-6, RoPE theta 1e6 (rotate-half), untied LM head, vocab 152064, [MASK] id 151665,
stop <|im_end|> 151645, block size 32. Inference attention is *block-causal*: a query at absolute position i attends to
every key j with j // block_size <= i // block_size (bidirectional inside the block, causal across blocks), and the model
is autoregressively shifted: the logits at position i predict the token at i+1.

Cache model (SHIGERU Phase 1): `PrefixCache` holds frozen K/V of committed tokens (prompt and finished blocks) in a
pre-allocated buffer [layers][2][1,Hkv,max_len,hd]; `BlockCache` holds the active block's K/V [layers][2][1,Hkv,block,hd]
and is rewritten in place per sub-block step. Positions are explicit everywhere. This is our own implementation written from
the published description (Fast-dLLM v2 paper/README) and the checkpoint's config; no reference code is copied.
"""
import glob, json, os, math
import torch, torch.nn as nn, torch.nn.functional as F
from safetensors.torch import load_file

MASK_ID, STOP_ID = 151665, 151645

class Config:
    def __init__(self, path):
        c = json.load(open(os.path.join(path, "config.json")))
        self.hidden, self.n_layers = c["hidden_size"], c["num_hidden_layers"]
        self.n_heads, self.n_kv = c["num_attention_heads"], c["num_key_value_heads"]
        self.head_dim = self.hidden // self.n_heads
        self.inter, self.vocab, self.eps = c["intermediate_size"], c["vocab_size"], c["rms_norm_eps"]
        self.rope_theta = c.get("rope_theta", 1e6); self.max_pos = c.get("max_position_embeddings", 32768)
        self.block = c.get("block_size", 32); self.mask_id = c.get("mask_token_id", MASK_ID)

class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight

def rotate_half(x):
    h = x.shape[-1] // 2; return torch.cat([-x[..., h:], x[..., :h]], -1)

class Rope:
    """cos/sin tables indexed by absolute position; `scale` hooks for Phase 4 (position interpolation / YaRN)."""
    def __init__(self, cfg, device, max_len=None, scaling=None):
        self.cfg, self.device, self.scaling = cfg, device, scaling
        self.len = 0; self.cos = self.sin = None
        self.extend(max_len or cfg.max_pos)
    def extend(self, n):
        if n <= self.len: return
        hd, theta = self.cfg.head_dim, self.cfg.rope_theta
        inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=self.device).float() / hd))
        if self.scaling and self.scaling.get("type") == "linear":   # position interpolation
            inv = inv / float(self.scaling["factor"])
        t = torch.arange(n, device=self.device).float()
        f = torch.outer(t, inv); emb = torch.cat([f, f], -1)
        self.cos, self.sin, self.len = emb.cos(), emb.sin(), n
    def get(self, pos):
        if int(pos.max()) >= self.len: self.extend(int(pos.max()) + 4096)
        return self.cos.index_select(0, pos), self.sin.index_select(0, pos)

def apply_rope(x, cos, sin):   # x [B,H,T,hd] bf16, cos/sin [T,hd] fp32 (Qwen applies in the activation dtype after casting tables)
    c, s = cos.to(x.dtype)[None, None], sin.to(x.dtype)[None, None]
    return x * c + rotate_half(x) * s

class Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__(); d = cfg.hidden; hk = cfg.n_kv * cfg.head_dim
        self.input_layernorm, self.post_attention_layernorm = RMSNorm(d, cfg.eps), RMSNorm(d, cfg.eps)
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(d, d, bias=True), nn.Linear(d, hk, bias=True), nn.Linear(d, hk, bias=True)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.gate_proj, self.up_proj, self.down_proj = nn.Linear(d, cfg.inter, bias=False), nn.Linear(d, cfg.inter, bias=False), nn.Linear(cfg.inter, d, bias=False)
        self.cfg = cfg

    def forward(self, x, cos, sin, keys, values, write, attn_mask):
        """x [1,T,d]; keys/values: list of (K,V) tensors that form the attention context in order (prefix slice, block);
        write: (K_buf, V_buf, pos) to scatter this span's k/v into before attention, or None; attn_mask [1,1,T,L] bool or None."""
        B, T, _ = x.shape; H, Hk, hd = self.cfg.n_heads, self.cfg.n_kv, self.cfg.head_dim
        h = self.input_layernorm(x)
        q = self.q_proj(h).view(B, T, H, hd).transpose(1, 2)
        k = self.k_proj(h).view(B, T, Hk, hd).transpose(1, 2)
        v = self.v_proj(h).view(B, T, Hk, hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if write is not None:
            Kb, Vb, pos = write
            Kb.index_copy_(2, pos, k); Vb.index_copy_(2, pos, v)
        K = torch.cat(keys + [k], 2) if keys else k
        V = torch.cat(values + [v], 2) if values else v
        a = F.scaled_dot_product_attention(q, K, V, attn_mask=attn_mask, is_causal=False, enable_gqa=True)
        x = x + self.o_proj(a.transpose(1, 2).reshape(B, T, -1))
        h = self.post_attention_layernorm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))

class BlockQwen(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab, cfg.hidden)
        self.layers = nn.ModuleList([Layer(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden, cfg.eps); self.lm_head = nn.Linear(cfg.hidden, cfg.vocab, bias=False)
        self.rope = None

    @classmethod
    def from_pretrained(cls, path, device="cuda", dtype=torch.bfloat16, rope_scaling=None):
        cfg = Config(path)
        with torch.device("meta"):
            m = cls(cfg)
        sd = {}
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))): sd.update(load_file(f))
        def take(n): return nn.Parameter(sd.pop(n).to(dtype), requires_grad=False)
        m.embed_tokens.weight = take("model.embed_tokens.weight"); m.norm.weight = take("model.norm.weight"); m.lm_head.weight = take("lm_head.weight")
        for i, L in enumerate(m.layers):
            p = f"model.layers.{i}."
            for mod, key in (("input_layernorm", "input_layernorm"), ("post_attention_layernorm", "post_attention_layernorm"),
                             ("q_proj", "self_attn.q_proj"), ("k_proj", "self_attn.k_proj"), ("v_proj", "self_attn.v_proj"), ("o_proj", "self_attn.o_proj"),
                             ("gate_proj", "mlp.gate_proj"), ("up_proj", "mlp.up_proj"), ("down_proj", "mlp.down_proj")):
                getattr(L, mod).weight = take(p + key + ".weight")
                if key.endswith(("q_proj", "k_proj", "v_proj")): getattr(L, mod).bias = take(p + key + ".bias")
        assert not sd, f"unused weights: {list(sd)[:5]}"
        m = m.to(device).eval(); m.rope = Rope(cfg, device, scaling=rope_scaling)
        return m

    def new_kv(self, n, device="cuda", dtype=torch.bfloat16):
        c = self.cfg
        return [(torch.empty(1, c.n_kv, n, c.head_dim, device=device, dtype=dtype), torch.empty(1, c.n_kv, n, c.head_dim, device=device, dtype=dtype)) for _ in self.layers]

    @staticmethod
    def block_causal_mask(q_pos, k_pos, block):
        """bool [1,1,T,L]: query block index >= key block index (absolute positions)"""
        return ((q_pos[:, None] // block) >= (k_pos[None, :] // block))[None, None]

    @torch.no_grad()
    def forward(self, tokens, pos, prefix=None, prefix_len=0, block_kv=None, block_pos=None, logits_positions=None):
        """tokens [1,T] at absolute positions pos [T].
        prefix: frozen K/V buffers (list per layer) of which the first prefix_len positions (absolute 0..prefix_len-1) are valid;
        block_kv/block_pos: the mutable block buffers [1,Hkv,block,hd] and their absolute positions [block] — this span's k/v are
        written into block_kv at the slots matching `pos` and the attention context is prefix[:prefix_len] + block_kv.
        With prefix=None and block_kv=None: self-contained forward over `tokens` (prefill of a chunk; k/v written to `write_to`
        is not needed here — use prefill()). Returns logits [1,T',V] fp32 for logits_positions (indices into T) or all T."""
        T = tokens.shape[1]; cos, sin = self.rope.get(pos)
        x = self.embed_tokens(tokens)
        if block_kv is not None:
            # slots of this span inside the block buffer
            slot = torch.searchsorted(block_pos, pos)
            k_pos = torch.cat([torch.arange(prefix_len, device=pos.device), block_pos])
        else:
            k_pos = pos
        mask = self.block_causal_mask(pos, k_pos, self.cfg.block)
        for i, L in enumerate(self.layers):
            if block_kv is not None:
                Kb, Vb = block_kv[i]
                keys = [prefix[i][0][:, :, :prefix_len]] if prefix_len else []
                vals = [prefix[i][1][:, :, :prefix_len]] if prefix_len else []
                # write this span into the block buffer, then attend over prefix + the whole block buffer
                h = L.input_layernorm(x); B_, T_, _ = x.shape; c = self.cfg
                q = L.q_proj(h).view(1, T_, c.n_heads, c.head_dim).transpose(1, 2)
                k = L.k_proj(h).view(1, T_, c.n_kv, c.head_dim).transpose(1, 2)
                v = L.v_proj(h).view(1, T_, c.n_kv, c.head_dim).transpose(1, 2)
                q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
                Kb.index_copy_(2, slot, k); Vb.index_copy_(2, slot, v)
                K = torch.cat(keys + [Kb], 2); V = torch.cat(vals + [Vb], 2)
                a = F.scaled_dot_product_attention(q, K, V, attn_mask=mask, is_causal=False, enable_gqa=True)
                x = x + L.o_proj(a.transpose(1, 2).reshape(1, T_, -1))
                h = L.post_attention_layernorm(x); x = x + L.down_proj(F.silu(L.gate_proj(h)) * L.up_proj(h))
            else:
                x = L(x, cos, sin, [], [], None, mask)
        if logits_positions is not None:
            x = x.index_select(1, logits_positions)
        return self.lm_head(self.norm(x)).float()

    @torch.no_grad()
    def prefill(self, tokens, pos, prefix, prefix_len, chunk=4096):
        """Write frozen K/V of `tokens` (absolute positions pos, all beyond prefix_len) into the prefix buffers in chunks;
        returns the logits of the last position (needed for the first token of the next block). Chunks attend to the
        committed prefix (block-causal), never to later chunks."""
        T = tokens.shape[1]; last_logits = None
        for s in range(0, T, chunk):
            e = min(T, s + chunk); tp = pos[s:e]; cos, sin = self.rope.get(tp)
            x = self.embed_tokens(tokens[:, s:e])
            k_pos = torch.cat([torch.arange(prefix_len, device=pos.device), tp])
            mask = self.block_causal_mask(tp, k_pos, self.cfg.block)
            for i, L in enumerate(self.layers):
                Kp, Vp = prefix[i]
                keys = [Kp[:, :, :prefix_len]] if prefix_len else []
                vals = [Vp[:, :, :prefix_len]] if prefix_len else []
                x = L(x, cos, sin, keys, vals, (Kp, Vp, tp), mask)
            last_logits = self.lm_head(self.norm(x[:, -1:])).float()
            prefix_len = e if False else prefix_len + (e - s)
        return last_logits, prefix_len
