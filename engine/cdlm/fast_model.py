"""Self-contained Qwen3 decoder for the causal-diffusion (WeDLM-style streaming window) decoder — SHIGERU Phase E.

Our own port of the Qwen3 architecture (RMSNorm, per-head q/k RMSNorm before RoPE, GQA, rotate-half RoPE with explicit
position ids, SwiGLU, tied or untied LM head), written in the style of engine/llada.py and engine/qwen_block.py and
checked against transformers' Qwen3ForCausalLM (tests/test_fast.py).  No reference code is copied.

Cache semantics (the thing decoder.py gets from DynamicCache.crop, done here without any copy)
  The model owns ONE preallocated K/V cache per layer, [layers, B=1, Hkv, max_len, hd], and the caller owns an integer
  `cache_len`: slots [0, cache_len) are the committed prefix.  forward(tokens[T], pos_ids[T], cache_len, write=True)
  rotates the T window rows with their own position ids, writes their K/V at slots cache_len .. cache_len+T-1 and
  attends over [cache[:cache_len] | window rows] with the causal mask "row i sees the prefix and rows <= i".  Because
  the decoder forwards the window observed-first, the leftmost filled run occupies rows 0..m-1, and those rows attended
  exactly the prefix plus the run tokens before them: committing the run is `cache_len += m` — nothing is copied or
  recomputed; the slots of the rows >= m are simply overwritten by the next step (DynamicCache.crop(-(T-m)) in
  decoder.py removes those rows; here they are just beyond cache_len).  write=False attends over the same context
  without touching the cache (the window K/V are concatenated on the fly).

CUDA-graph friendliness (engine/graphs.py style; capture is wired but gated on `use_graph and cuda`, never on CPU)
  forward_static(tokens[Wb], pos[Wb], slots[Wb], key_mask[Wb, Cb], rows[Wb]) has no Python-side dependence on the
  data: the window is padded to a fixed bucket Wb in BUCKETS, the context is the first Cb cache slots (a prefix view
  of the cache, Cb = the CACHE_BUCKET multiple covering cache_len + Wb, at most max_len) and the validity of every key
  is in the boolean key_mask, the write slots and the logits rows are index tensors, and the output is a static
  [Wb, V] buffer.  One graph per (Wb, Cb): the graphs are captured lazily (or all at once by capture_graphs) and share
  one memory pool; every graph reads fixed prefix views of ONE set of static buffers (tokens/pos/slots/rows [4, 32],
  key_mask [32, max_len]), so a step costs one pinned host->device copy of the four index rows, three small kernels
  for the key mask and one graph launch.  `run(...)` pads to the bucket, fills the static buffers and either calls
  forward_static eagerly (static=True on any device) or replays the captured graph of that (Wb, Cb) (use_graph=True,
  CUDA only).  Invalid cache slots are zero (never NaN), so masked-out keys contribute exactly 0 and padded rows are
  finite; the padded rows write K/V beyond cache_len+T, which the next step overwrites.  The non-static forward()
  attends over exactly cache_len+T keys and is what the CPU tests compare against HF; attending over the Cb-slot
  prefix instead of max_len is what makes the static/graph path faster than it (docs/distill_design.md §10.2 measured
  the whole-cache version slower than the plain path).  cache_bucket=None restores the whole-cache context.

Numerics follow transformers 5.20 modeling_qwen3.py: RMSNorm in float32 then cast and scaled by the weight; RoPE cos/sin
computed in float32 from the explicit position ids and cast to the activation dtype before q*cos + rotate_half(q)*sin;
q_norm/k_norm on the [T, H, hd] view before the transpose; SDPA with scale hd**-0.5, GQA materialised with repeat_kv
as HF does whenever a mask is given (enable_gqa + mask falls back to the math kernel on CUDA); logits in the model
dtype, returned as float32 for the requested rows only.
"""
import glob
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

__all__ = ["BUCKETS", "CACHE_BUCKET", "bucket_for", "cache_bucket_for", "Config", "FastQwen3", "KVCache"]

BUCKETS = (1, 4, 8, 16, 32)
CACHE_BUCKET = 256


def bucket_for(T):
    for b in BUCKETS:
        if T <= b:
            return b
    raise ValueError(f"window of {T} rows exceeds the largest bucket {BUCKETS[-1]}")


def cache_bucket_for(ctx, max_len, bucket=CACHE_BUCKET):
    """Context length of the static step: the smallest multiple of `bucket` >= ctx (= cache_len + Wb), capped at
    max_len; bucket=None means the whole cache (max_len)."""
    assert 0 <= ctx <= max_len, (ctx, max_len)
    if bucket is None:
        return max_len
    return min(max_len, -(-ctx // int(bucket)) * int(bucket))


class Config:
    """The subset of a Qwen3 config.json this model needs (HF 4.x 'rope_theta' and 5.x 'rope_parameters' both read)."""

    def __init__(self, path=None, d=None):
        c = d if d is not None else json.load(open(os.path.join(path, "config.json")))
        assert c.get("model_type", "qwen3") == "qwen3", c.get("model_type")
        self.hidden, self.n_layers = c["hidden_size"], c["num_hidden_layers"]
        self.n_heads, self.n_kv = c["num_attention_heads"], c["num_key_value_heads"]
        self.head_dim = c.get("head_dim") or self.hidden // self.n_heads
        self.inter, self.vocab, self.eps = c["intermediate_size"], c["vocab_size"], c["rms_norm_eps"]
        rp = c.get("rope_parameters") or {}
        self.rope_theta = float(rp.get("rope_theta", c.get("rope_theta", 1e6)))
        assert rp.get("rope_type", "default") == "default", rp
        self.max_pos = c.get("max_position_embeddings", 40960)
        self.tie = bool(c.get("tie_word_embeddings", False))
        self.attn_bias = bool(c.get("attention_bias", False))
        self.mask_id = c.get("cdlm_mask_token_id")
        assert c.get("hidden_act", "silu") == "silu"
        lt = c.get("layer_types") or []
        assert not c.get("use_sliding_window", False) and "sliding_attention" not in lt, "sliding window layers unsupported"


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        xf = x.float()
        return self.weight * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)


def apply_rope(x, cos, sin):
    """x [1, H, T, hd]; cos/sin [T, hd] already in x.dtype (HF casts the float32 tables before applying)."""
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


def repeat_kv(x, n):
    """[1, Hkv, L, hd] -> [1, Hkv*n, L, hd] with each kv head repeated n times consecutively (HF repeat_kv order)."""
    if n == 1:
        return x
    B, Hk, L, hd = x.shape
    return x[:, :, None].expand(B, Hk, n, L, hd).reshape(B, Hk * n, L, hd)


class Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, hq, hk = cfg.hidden, cfg.n_heads * cfg.head_dim, cfg.n_kv * cfg.head_dim
        self.input_layernorm, self.post_attention_layernorm = RMSNorm(d, cfg.eps), RMSNorm(d, cfg.eps)
        self.q_proj, self.k_proj, self.v_proj = (nn.Linear(d, hq, bias=cfg.attn_bias), nn.Linear(d, hk, bias=cfg.attn_bias),
                                                 nn.Linear(d, hk, bias=cfg.attn_bias))
        self.o_proj = nn.Linear(hq, d, bias=cfg.attn_bias)
        self.q_norm, self.k_norm = RMSNorm(cfg.head_dim, cfg.eps), RMSNorm(cfg.head_dim, cfg.eps)
        self.gate_proj, self.up_proj = nn.Linear(d, cfg.inter, bias=False), nn.Linear(d, cfg.inter, bias=False)
        self.down_proj = nn.Linear(cfg.inter, d, bias=False)
        self.cfg = cfg

    def qkv(self, x, cos, sin):
        B, T, _ = x.shape
        c = self.cfg
        h = self.input_layernorm(x)
        q = self.q_norm(self.q_proj(h).view(B, T, c.n_heads, c.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(h).view(B, T, c.n_kv, c.head_dim)).transpose(1, 2)
        v = self.v_proj(h).view(B, T, c.n_kv, c.head_dim).transpose(1, 2)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def finish(self, x, a):
        """residual + o_proj, then the SwiGLU block.  a: [1, H, T, hd] attention output."""
        B, H, T, hd = a.shape
        x = x + self.o_proj(a.transpose(1, 2).reshape(B, T, H * hd))
        h = self.post_attention_layernorm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))

    def attend(self, q, K, V, mask):
        n = self.cfg.n_heads // self.cfg.n_kv
        return F.scaled_dot_product_attention(q, repeat_kv(K, n), repeat_kv(V, n), attn_mask=mask, is_causal=False,
                                              scale=self.cfg.head_dim ** -0.5)


class KVCache:
    """K and V for every layer, [layers, 1, Hkv, max_len, hd], zero-initialised (masked slots must be finite)."""

    def __init__(self, cfg, max_len, device, dtype):
        self.max_len = max_len
        shape = (cfg.n_layers, 1, cfg.n_kv, max_len, cfg.head_dim)
        self.K = torch.zeros(shape, device=device, dtype=dtype)
        self.V = torch.zeros(shape, device=device, dtype=dtype)

    def zero_(self):
        self.K.zero_(); self.V.zero_()


class FastQwen3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab, cfg.hidden)
        self.layers = nn.ModuleList([Layer(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden, cfg.eps)
        self.lm_head = nn.Linear(cfg.hidden, cfg.vocab, bias=False)
        self.cos = self.sin = None       # float32 rotary tables [max_pos, hd] (plain tensors, built on the real device)
        self.kv = None                   # KVCache
        self.graphs = {}                 # (Wb, Cb) -> (graph, static output)
        self.static = None               # static input buffers for forward_static (allocated lazily)
        self.cache_bucket = CACHE_BUCKET  # context granularity of the static step (None = whole cache)
        self.pool = None                 # shared CUDA-graph memory pool

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, path, device="cpu", dtype=torch.float32, max_len=4096):
        """Load an HF Qwen3 directory (config.json + *.safetensors).  A tied checkpoint has no lm_head.weight."""
        cfg = Config(path)
        sd = {}
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            sd.update(load_file(f))
        assert sd, f"no *.safetensors in {path}"
        return cls._build(cfg, sd, device, dtype, max_len)

    @classmethod
    def from_hf(cls, hf_model, device="cpu", dtype=torch.float32, max_len=4096):
        """Build from an in-memory transformers Qwen3ForCausalLM (same key names as the safetensors files)."""
        cfg = Config(d=hf_model.config.to_dict())
        sd = {k: v.detach() for k, v in hf_model.state_dict().items()}
        if cfg.tie:
            sd.pop("lm_head.weight", None)
        return cls._build(cfg, sd, device, dtype, max_len)

    @classmethod
    def _build(cls, cfg, sd, device, dtype, max_len):
        with torch.device("meta"):
            m = cls(cfg)

        def take(n):
            return nn.Parameter(sd.pop(n).to(dtype), requires_grad=False)

        m.embed_tokens.weight = take("model.embed_tokens.weight")
        m.norm.weight = take("model.norm.weight")
        if cfg.tie:
            assert "lm_head.weight" not in sd, "tie_word_embeddings=True but the checkpoint carries lm_head.weight"
            m.lm_head.weight = m.embed_tokens.weight
        else:
            m.lm_head.weight = take("lm_head.weight")
        for i, L in enumerate(m.layers):
            p = f"model.layers.{i}."
            for mod, key in (("input_layernorm", "input_layernorm"), ("post_attention_layernorm", "post_attention_layernorm"),
                             ("q_proj", "self_attn.q_proj"), ("k_proj", "self_attn.k_proj"), ("v_proj", "self_attn.v_proj"),
                             ("o_proj", "self_attn.o_proj"), ("q_norm", "self_attn.q_norm"), ("k_norm", "self_attn.k_norm"),
                             ("gate_proj", "mlp.gate_proj"), ("up_proj", "mlp.up_proj"), ("down_proj", "mlp.down_proj")):
                getattr(L, mod).weight = take(p + key + ".weight")
                if cfg.attn_bias and key.endswith(("q_proj", "k_proj", "v_proj", "o_proj")):
                    getattr(L, mod).bias = take(p + key + ".bias")
        sd = {k: v for k, v in sd.items() if not k.endswith(("rotary_emb.inv_freq",))}
        assert not sd, f"unused weights: {list(sd)[:5]}"
        m = m.to(device).eval()
        m.cos, m.sin = m._rope_tables(max(cfg.max_pos, max_len), device)
        m.new_cache(max_len)
        return m

    def _rope_tables(self, n, device):
        c = self.cfg
        inv = 1.0 / (c.rope_theta ** (torch.arange(0, c.head_dim, 2, device=device, dtype=torch.float) / c.head_dim))
        f = torch.arange(n, device=device, dtype=torch.float)[:, None] * inv[None, :]   # == HF position_ids.float()*inv_freq
        emb = torch.cat([f, f], -1)
        return emb.cos(), emb.sin()

    @property
    def device(self):
        return self.embed_tokens.weight.device

    @property
    def dtype(self):
        return self.embed_tokens.weight.dtype

    def new_cache(self, max_len):
        self.kv = KVCache(self.cfg, max_len, self.device, self.dtype)
        self.graphs, self.static, self.pool = {}, None, None
        return self.kv

    # ------------------------------------------------------------------ masks
    @staticmethod
    def window_mask(cache_len, T, ctx_len, device):
        """bool [T, ctx_len]: window row i may attend key slot j iff j <= cache_len + i (prefix + rows <= i)."""
        q = torch.arange(T, device=device)[:, None] + cache_len
        k = torch.arange(ctx_len, device=device)[None, :]
        return k <= q

    # ------------------------------------------------------------------ exact (data-shaped) forward
    @torch.no_grad()
    def forward(self, tokens, pos_ids, cache_len=0, write=True, rows=None, key_mask=None):
        """tokens [T] long, pos_ids [T] long (explicit RoPE positions).  Attends over [cache[:cache_len] | window] with
        the causal window mask (or key_mask, bool [T, cache_len+T], verbatim).  write=True stores the window K/V at
        slots cache_len..cache_len+T-1.  Returns float32 logits [len(rows), V] (rows: indices into the window; None =
        all T rows)."""
        T = int(tokens.numel())
        assert cache_len + T <= self.kv.max_len, (cache_len, T, self.kv.max_len)
        ctx = cache_len + T
        mask = self.window_mask(cache_len, T, ctx, tokens.device) if key_mask is None else key_mask
        assert mask.shape == (T, ctx), (mask.shape, (T, ctx))
        mask = mask[None, None]
        cos, sin = self.cos.index_select(0, pos_ids).to(self.dtype), self.sin.index_select(0, pos_ids).to(self.dtype)
        x = self.embed_tokens(tokens[None])
        for i, L in enumerate(self.layers):
            q, k, v = L.qkv(x, cos, sin)
            if write:
                self.kv.K[i, :, :, cache_len:ctx] = k
                self.kv.V[i, :, :, cache_len:ctx] = v
                K, V = self.kv.K[i, :, :, :ctx], self.kv.V[i, :, :, :ctx]
            else:
                K = torch.cat([self.kv.K[i, :, :, :cache_len], k], 2)
                V = torch.cat([self.kv.V[i, :, :, :cache_len], v], 2)
            x = L.finish(x, L.attend(q, K, V, mask))
        if rows is not None:
            x = x.index_select(1, torch.as_tensor(rows, dtype=torch.long, device=x.device))
        return self.lm_head(self.norm(x))[0].float()

    @torch.no_grad()
    def prefill(self, prompt_ids, cache_len=0, chunk=512):
        """Write the prompt into the cache in chunks starting at slot/position cache_len; returns (next-token logits of
        the last prompt token [V] float32, new cache_len)."""
        ids = torch.as_tensor(prompt_ids, dtype=torch.long, device=self.device).flatten()
        n = int(ids.numel())
        assert n > 0
        logits = None
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            pos = torch.arange(cache_len + s, cache_len + e, device=self.device)
            logits = self.forward(ids[s:e], pos, cache_len + s, write=True, rows=[e - s - 1])[0]
        return logits, cache_len + n

    # ------------------------------------------------------------------ static-shape forward (graph capturable)
    def _alloc_static(self):
        dev, L = self.device, self.kv.max_len
        Wb = BUCKETS[-1]
        cuda = dev.type == "cuda"
        self.static = dict(idx=torch.zeros(4, Wb, dtype=torch.long, device=dev),          # rows: tokens, pos, slots, rows
                           stage=torch.zeros(4, Wb, dtype=torch.long, pin_memory=cuda),   # host staging for one copy
                           key_mask=torch.zeros(Wb, L, dtype=torch.bool, device=dev),
                           qpos=torch.arange(Wb, device=dev)[:, None], kpos=torch.arange(L, device=dev)[None, :],
                           event=torch.cuda.Event() if cuda else None)

    def _static_args(self, Wb, Cb):
        st = self.static
        return (st["idx"][0, :Wb], st["idx"][1, :Wb], st["idx"][2, :Wb], st["key_mask"][:Wb, :Cb], st["idx"][3, :Wb])

    @torch.no_grad()
    def forward_static(self, tokens, pos_ids, slots, key_mask, rows):
        """All inputs are bucket-shaped device tensors: tokens/pos/slots/rows [Wb], key_mask [Wb, Cb].  Writes the
        window K/V at `slots` (index_copy_), attends over the first Cb cache slots under key_mask, returns float32
        logits [Wb, V] for the window rows `rows` (gathered by index).  No shape depends on the data: graph-capturable
        per (Wb, Cb)."""
        Cb = key_mask.shape[1]
        mask = key_mask[None, None]
        cos, sin = self.cos.index_select(0, pos_ids).to(self.dtype), self.sin.index_select(0, pos_ids).to(self.dtype)
        x = self.embed_tokens(tokens[None])
        for i, L in enumerate(self.layers):
            q, k, v = L.qkv(x, cos, sin)
            self.kv.K[i].index_copy_(2, slots, k)
            self.kv.V[i].index_copy_(2, slots, v)
            x = L.finish(x, L.attend(q, self.kv.K[i][:, :, :Cb], self.kv.V[i][:, :, :Cb], mask))
        x = x.index_select(1, rows)
        return self.lm_head(self.norm(x))[0].float()

    def _capture(self, Wb, Cb):
        """CUDA-graph capture of forward_static for one (window bucket, cache bucket) (engine/graphs.py recipe; all
        graphs share one memory pool, which is safe because they are replayed one at a time and every graph's output
        buffer stays allocated).  CUDA only."""
        assert self.device.type == "cuda", "graph capture is CUDA-only"
        if self.static is None:
            self._alloc_static()
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        args = self._static_args(Wb, Cb)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.forward_static(*args)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            out = self.forward_static(*args)
        self.graphs[(Wb, Cb)] = (g, out)
        return self.graphs[(Wb, Cb)]

    def capture_graphs(self, max_ctx, window_buckets=BUCKETS):
        """Capture (once, up front) every graph a decode whose cache never exceeds max_ctx slots can need: window
        buckets x cache buckets up to cache_bucket_for(max_ctx).  Returns the list of (Wb, Cb) captured now."""
        assert self.device.type == "cuda"
        step = self.cache_bucket or self.kv.max_len
        top = cache_bucket_for(min(max_ctx, self.kv.max_len), self.kv.max_len, self.cache_bucket)
        done = []
        for Wb in window_buckets:
            for Cb in sorted(set(min(self.kv.max_len, c) for c in range(step, top + 1, step)) | {top}):
                if (Wb, Cb) not in self.graphs and Wb <= Cb:
                    self._capture(Wb, Cb)
                    done.append((Wb, Cb))
        return done

    @torch.no_grad()
    def run(self, tokens, pos_ids, cache_len, rows, static=False, use_graph=False):
        """The decoder's step.  tokens/pos_ids: [T] long on any device (CPU tensors are cheapest: the static path
        stages them through pinned memory, the plain path moves them in one copy).  static=False: exact forward()
        over cache_len+T keys.  static=True: pad the window to its bucket Wb, fill the static buffers and run
        forward_static over the Cb = cache_bucket_for(cache_len + Wb) first cache slots (eagerly, or as the replayed
        CUDA graph of (Wb, Cb) when use_graph and the model is on CUDA — never on CPU).  Returns float32 logits
        [len(rows), V] (static path: a view of a reusable buffer — clone if kept across steps)."""
        T = int(tokens.numel())
        dev = self.device
        if not static and not use_graph:
            if tokens.device != dev or pos_ids.device != dev:
                tp = torch.stack([tokens, pos_ids]).to(dev, non_blocking=False)
                tokens, pos_ids = tp[0], tp[1]
            return self.forward(tokens, pos_ids, cache_len, write=True, rows=rows)
        Wb = bucket_for(T)
        assert cache_len + Wb <= self.kv.max_len, (cache_len, Wb, self.kv.max_len)
        Cb = cache_bucket_for(cache_len + Wb, self.kv.max_len, self.cache_bucket)
        if self.static is None:
            self._alloc_static()
        st = self.static
        rows = list(rows) if rows is not None else list(range(T))
        assert len(rows) <= Wb                      # rows may be empty: the step still writes the window K/V
        Wmax = BUCKETS[-1]                          # the four index rows are always copied in full (one contiguous copy)
        tl, pl = tokens.tolist(), pos_ids.tolist()
        host = [tl + [0] * (Wmax - T), pl + [pl[-1]] * (Wmax - T), list(range(cache_len, cache_len + Wmax)),
                rows + [rows[0] if rows else 0] * (Wmax - len(rows))]
        if st["event"] is not None:
            st["event"].synchronize()               # the previous step's staged copy has landed: the buffer is free
            st["stage"].copy_(torch.tensor(host, dtype=torch.long))
            st["idx"].copy_(st["stage"], non_blocking=True)
            st["event"].record()
        else:
            st["idx"].copy_(torch.tensor(host, dtype=torch.long))
        st["key_mask"][:Wb, :Cb].copy_(st["kpos"][:, :Cb] <= st["qpos"][:Wb] + cache_len)
        if use_graph and dev.type == "cuda":
            if (Wb, Cb) not in self.graphs:
                self._capture(Wb, Cb)
            g, out = self.graphs[(Wb, Cb)]
            g.replay()
        else:
            out = self.forward_static(*self._static_args(Wb, Cb))
        return out[:len(rows)]
