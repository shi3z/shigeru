"""Trust estimation for B-series decoding (docs/trust_aware_diffusion.md §2).

Per-position features computed from the SAME network call the sampler already makes (no extra forward), for a block
state x (K tokens, [MASK] or placed) and the logits of the current step:
  conf      max softmax probability of the current prediction
  margin    p(top-1) - p(top-2)
  entropy   Shannon entropy of the predictive distribution (nats)
  p_tok     probability the model gives to the token currently placed at a visible position (1 - "disagreement")
  agree     1 if the argmax equals the placed token
  stable    1 if the argmax equals the argmax of the previous step at the same position
  age       steps since the token was placed (visible positions)
  conf_at   confidence the token had when it was placed
  dconf     conf_at - p_tok (how much the model's belief in the placed token dropped since placement)
  changes   number of times the argmax at the position changed across steps so far
  nb_changed 1 if a neighbouring position (distance 1) was placed or re-masked in the previous step (dependency proxy)
A trust score is a logistic combination of these features (weights fitted on teacher-forced trajectories of the
development items, bench/diag_trust.py); `trust_score` applies saved weights.  Metrics (AUROC, AUPRC, Brier, ECE) are
implemented here without sklearn.
"""
import json

import torch

__all__ = ["FEATURES", "step_features", "TrustLR", "auroc", "auprc", "brier", "ece"]

FEATURES = ("conf", "margin", "entropy", "p_tok", "agree", "stable", "age", "conf_at", "dconf", "changes", "nb_changed")


def step_features(probs, x, mask_id, state):
    """probs [K, V] float (mask id already excluded), x [K] current tokens, state: dict of running tensors
    (prev_pred [K] or None, placed_step [K] (-1 = masked), conf_at [K], changes [K], nb_changed [K] bool) and 'step'.
    Returns (feat [K, F] float, pred [K], conf [K])."""
    conf, pred = probs.max(-1)                  # same tie-breaking as the original sampler (first index): bf16 logits
    top2 = probs.topk(2, dim=-1).values         # tie at the top often enough that topk's order changed 20 % of blocks
    margin = conf - top2[:, 1]
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(-1)
    is_mask = x == mask_id
    safe = torch.where(is_mask, pred, x)
    p_tok = probs.gather(-1, safe[:, None]).squeeze(-1)
    p_tok = torch.where(is_mask, conf, p_tok)
    agree = torch.where(is_mask, torch.ones_like(conf), (pred == x).float())
    prev = state.get("prev_pred")
    stable = torch.ones_like(conf) if prev is None else (pred == prev).float()
    placed = state["placed_step"]
    age = torch.where(placed >= 0, (state["step"] - placed).float(), torch.zeros_like(conf))
    conf_at = torch.where(placed >= 0, state["conf_at"], conf)
    dconf = conf_at - p_tok
    changes = state["changes"].float()
    nb = state["nb_changed"].float()
    feat = torch.stack([conf, margin, entropy, p_tok, agree, stable, age, conf_at, dconf, changes, nb], -1)
    return feat, pred, conf


class TrustLR:
    """Logistic regression trust score P(token correct | features) with standardised inputs."""

    def __init__(self, w, b, mean, std, features=FEATURES):
        self.w = torch.tensor(w, dtype=torch.float32)
        self.b = float(b)
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.features = tuple(features)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(d["w"], d["b"], d["mean"], d["std"], d.get("features", FEATURES))

    def save(self, path, extra=None):
        json.dump(dict(w=self.w.tolist(), b=self.b, mean=self.mean.tolist(), std=self.std.tolist(), features=list(self.features), **(extra or {})), open(path, "w"), indent=1)

    def __call__(self, feat):
        z = ((feat.cpu().float() - self.mean) / self.std) @ self.w + self.b
        return torch.sigmoid(z).to(feat.device)

    @classmethod
    def fit(cls, X, y, l2=1e-2, steps=2000, lr=0.1, features=FEATURES):
        """X [N, F] float, y [N] {0,1} (1 = correct).  Full-batch L-BFGS-free gradient descent (small problems)."""
        X = X.float(); y = y.float()
        mean = X.mean(0); std = X.std(0).clamp_min(1e-6)
        Z = (X - mean) / std
        w = torch.zeros(Z.shape[1], requires_grad=True); b = torch.zeros((), requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            logit = Z @ w + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y) + l2 * (w * w).sum()
            loss.backward(); opt.step()
        return cls(w.detach().tolist(), b.item(), mean.tolist(), std.tolist(), features)


def auroc(score, label):
    """score: higher = more likely positive; label {0,1}. Rank-based (ties averaged)."""
    s = torch.as_tensor(score, dtype=torch.float64); y = torch.as_tensor(label, dtype=torch.float64)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = s.argsort()
    ranks = torch.empty_like(s)
    ss = s[order]
    i = 0
    r = torch.arange(1, len(s) + 1, dtype=torch.float64)
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = r[i:j + 1].mean()
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(score, label):
    s = torch.as_tensor(score, dtype=torch.float64); y = torch.as_tensor(label, dtype=torch.float64)
    if y.sum() == 0:
        return float("nan")
    order = s.argsort(descending=True)
    ys = y[order]
    tp = ys.cumsum(0)
    prec = tp / torch.arange(1, len(ys) + 1, dtype=torch.float64)
    return float((prec * ys).sum() / y.sum())


def brier(prob, label):
    p = torch.as_tensor(prob, dtype=torch.float64); y = torch.as_tensor(label, dtype=torch.float64)
    return float(((p - y) ** 2).mean())


def ece(prob, label, bins=10):
    p = torch.as_tensor(prob, dtype=torch.float64); y = torch.as_tensor(label, dtype=torch.float64)
    e = 0.0
    edges = torch.linspace(0, 1, bins + 1, dtype=torch.float64)
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i > 0 else (p >= edges[i]) & (p <= edges[i + 1])
        if m.any():
            e += float(m.sum()) / len(p) * abs(float(p[m].mean() - y[m].mean()))
    return e
