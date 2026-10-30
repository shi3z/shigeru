"""Phase B13 — auxiliary-conditioned drafter + representation alignment (docs/b13_auxiliary_drafter.md).

PLM (arXiv 2610.26693, "Parallel Language Models") replaces the uninformative [MASK] input of the parallel decoder by an
AUXILIARY token that carries a coarse, sampled summary of the AR model's predictive distribution at that position, and
adds a representation-alignment loss between the decoder's and the encoder's hidden states.  The paper's auxiliaries are
Voronoi cells over the output embeddings; here the first version is the "quantile" family in its simplest exact form:

  aux(t) = RANK of the true token x_t under Q_t = p_AR(. | x_<t), the model's own AR distribution (computed on the x0
  stream of the dual-stream example, no grad, same forward pass family as the training step).  rank 0 = "x_t is the AR
  argmax", rank k < n_aux = "x_t is the (k+1)-th choice", rank >= n_aux = the plain [MASK] id (no information).

The xt (masked) rows are forwarded with aux ids instead of [MASK]; labels, weights and losses are unchanged.  For an
exactly-greedy verified decoder (engine/cdlm/decoder.py generate_verified) every mask row is then fed rank 0 at
inference: "every token of this block is the greedy continuation" — the conditioning the AR verifier will check.  The
drafter at position t+1 can read x_t from the aux row at t (argmax of that row's own prediction) instead of marginalising
over it, which is where the mask-trained drafter loses its drafts (Clover study: 83 % of rejections have p_AR < 0.01).

Alignment (PLM §3.3): cos similarity between the decoder's hidden state at layer dec_layer on a mask row (position t)
and the encoder's hidden state at layer enc_layer on the x0 row t-1 (the AR row that predicts x_t), through a learned
linear map on the decoder side, stop-grad on the encoder side; loss = mean(1 - cos).
"""
import torch

from .data import IGNORE

__all__ = ["add_aux_tokens", "aux_ids_of", "assign_aux", "AlignHooks", "align_loss"]


@torch.no_grad()
def add_aux_tokens(model, mask_id, n_aux, base=None, seed=0):
    """Make ids base..base+n_aux-1 usable auxiliary rows (default base = mask_id + 1; resize only when needed — Qwen3
    has 151936 rows for 151669 tokens + [MASK], so 128 auxiliaries fit without a resize).  Init: mean row + a random
    unit direction scaled to the median row norm (distinct, token-like magnitude; trained like every other row).
    Records config.cdlm_aux_base / cdlm_aux_n.  Returns the info dict."""
    emb = model.get_input_embeddings()
    old = emb.weight.shape[0]
    base = mask_id + 1 if base is None else int(base)
    n_aux = int(n_aux)
    assert n_aux >= 1 and base > mask_id, (base, mask_id, n_aux)
    if base + n_aux > old:
        model.resize_token_embeddings(base + n_aux, mean_resizing=False)
        emb = model.get_input_embeddings()
    w = emb.weight
    ref = w[: min(old, mask_id)].float()
    mean, norm = ref.mean(0), ref.norm(dim=-1).median()
    g = torch.Generator(device="cpu").manual_seed(seed)
    d = torch.randn(n_aux, w.shape[1], generator=g)
    d = d / d.norm(dim=-1, keepdim=True)
    w[base: base + n_aux] = (mean[None].cpu() + norm.cpu() * d).to(w.dtype).to(w.device)
    head = model.get_output_embeddings()
    if head is not None and head.weight.data_ptr() != w.data_ptr():
        head.weight[base: base + n_aux] = w[base: base + n_aux]
    model.config.cdlm_aux_base, model.config.cdlm_aux_n = base, n_aux
    return {"aux_base": base, "aux_n": n_aux, "old_rows": old, "new_rows": emb.weight.shape[0]}


def aux_ids_of(config):
    """(base, n) from a config carrying cdlm_aux_base/cdlm_aux_n, else None."""
    base, n = getattr(config, "cdlm_aux_base", None), getattr(config, "cdlm_aux_n", None)
    return (int(base), int(n)) if base is not None and n else None


@torch.no_grad()
def assign_aux(model, batch, mask_id, base, n_aux, device, chunk=256):
    """Replace the [MASK] input of every scored xt row by its auxiliary id (module docstring).

    One no-grad causal forward of the x0 rows (input_ids[:, :L] per example — the first max(seq_lens) columns of the
    batch; the xt rows are never read) gives Q_q = p(. | x_<=q) at every x0 row q; the rank of x_{q+1} under Q_q is the
    auxiliary of position q+1.  A mask row at position p (position_ids) with label x_p gets base + rank if rank < n_aux,
    else keeps mask_id.  Position 0 (no AR row before it) keeps mask_id.  Returns (input_ids on `device`, stats)."""
    ids = batch["input_ids"].to(device)
    pos = batch["position_ids"].to(device)
    lab = batch["mlm_labels"].to(device)
    seq = batch["seq_lens"].to(device)
    N = ids.shape[0]
    L = int(seq.max())
    x0 = ids[:, :L]
    model.eval()
    try:
        h = model.model(input_ids=x0, use_cache=False).last_hidden_state          # [N, L, H], causal
    finally:
        model.train()
    head = model.get_output_embeddings()
    # rank of the next token under each x0 row: rank_next[b, q] for q < L-1
    rank_next = torch.full((N, L), n_aux, dtype=torch.long, device=device)
    nxt = x0[:, 1:]
    for s in range(0, L - 1, chunk):
        e = min(L - 1, s + chunk)
        lg = head(h[:, s:e]).float()                                                 # [N, c, V]
        tgt = lg.gather(-1, nxt[:, s:e, None])
        rank_next[:, s:e] = (lg > tgt).sum(-1)
    rows = (lab != IGNORE) & (ids == mask_id) & (pos >= 1)
    b_idx, r_idx = rows.nonzero(as_tuple=True)
    r = rank_next[b_idx, pos[b_idx, r_idx] - 1]
    new = torch.where(r < n_aux, base + r, torch.full_like(r, mask_id))
    out = ids.clone()
    out[b_idx, r_idx] = new
    n = max(1, int(b_idx.numel()))
    stats = {"aux_rows": int(b_idx.numel()), "aux_rank0_frac": round(float((r == 0).sum()) / n, 4),
             "aux_mask_frac": round(float((r >= n_aux).sum()) / n, 4)}
    return out, stats


class AlignHooks:
    """Forward hooks on two decoder layers of a HF model capturing their output hidden states for align_loss."""

    def __init__(self, model, dec_layer, enc_layer):
        self.layers = model.model.layers
        self.dec_layer, self.enc_layer = int(dec_layer), int(enc_layer)
        self.h = {}
        self.handles = []

    def _hook(self, key):
        def f(mod, args, out):
            self.h[key] = out[0] if isinstance(out, (tuple, list)) else out
        return f

    def __enter__(self):
        self.h = {}
        self.handles = [self.layers[self.dec_layer].register_forward_hook(self._hook("dec")),
                        self.layers[self.enc_layer].register_forward_hook(self._hook("enc"))]
        return self

    def __exit__(self, *exc):
        for hd in self.handles:
            hd.remove()
        self.handles = []
        return False


def align_loss(hooks, proj, batch, device):
    """mean(1 - cos(proj(h_dec[mask row at position p]), sg(h_enc[x0 row p-1]))) over the scored xt rows with p >= 1
    (x0 rows are the first L rows of each example, row index = position)."""
    lab = batch["mlm_labels"].to(device)
    pos = batch["position_ids"].to(device)
    rows = (lab != IGNORE) & (pos >= 1)
    b_idx, r_idx = rows.nonzero(as_tuple=True)
    if b_idx.numel() == 0:
        return hooks.h["dec"].sum() * 0.0
    hd = hooks.h["dec"][b_idx, r_idx]
    he = hooks.h["enc"][b_idx, pos[b_idx, r_idx] - 1].detach()
    cos = torch.nn.functional.cosine_similarity(proj(hd).float(), he.float(), dim=-1)
    return (1.0 - cos).mean()
