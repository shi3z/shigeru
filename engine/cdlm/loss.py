"""Losses (port of third_party/WeDLM/finetune/src/loss.py; combination as trainer.py:207).

Normalisation chosen for `mlm_loss` ("wedlm", the default): WeDLM computes per-token CE on the masked rows, weights
w_i = 1 / (gamma_i + eps) and *normalises the weights to sum to one over all masked rows of the batch*:
loss = sum_i w_i CE_i / sum_i w_i.  This is a self-normalised importance-weighted mean, so the loss stays on the scale
of a per-token cross-entropy regardless of how many rows are masked (unlike LLaDA's sum_i CE_i / (gamma_i * L), which
scales with the mask fraction), and gradient accumulation / data-parallel averaging of per-microbatch losses behaves
like the reference (accelerate averages microbatch losses).  "tokens" (sum_i w_i CE_i / n_masked) is provided for
experiments; it keeps the per-block 1/gamma reweighting (low-gamma blocks count more) but is not what WeDLM trains with.
`ar_loss` is the plain mean CE over labelled rows (ar_labels from data.py are already next-token aligned, so no shift).
`total_loss` = (mlm + w * ar) / (1 + w), w = ar_loss_weight = 1.0 in WeDLM.
"""
import torch
import torch.nn.functional as F

IGNORE = -100
__all__ = ["mlm_loss", "ar_loss", "total_loss"]


def _flat(logits, labels):
    V = logits.shape[-1]
    return logits.reshape(-1, V), labels.reshape(-1)


def mlm_loss(logits, labels, weights, normalize="wedlm", eps=1e-8):
    """1/gamma-weighted CE on rows with labels != -100.  logits [..., V], labels [...], weights [...] (1/gamma or 0)."""
    logits, labels = _flat(logits, labels)
    weights = weights.reshape(-1).to(logits.dtype)
    sel = labels != IGNORE
    if not bool(sel.any()):
        return logits.sum() * 0.0
    ce = F.cross_entropy(logits[sel].float(), labels[sel], reduction="none")
    w = weights[sel].float()
    if normalize == "wedlm":
        return (ce * w).sum() / (w.sum() + eps)
    if normalize == "tokens":
        return (ce * w).sum() / sel.sum()
    raise ValueError(normalize)


def ar_loss(logits, labels):
    """Mean next-token CE over rows with labels != -100 (labels already shifted by data.py)."""
    logits, labels = _flat(logits, labels)
    sel = labels != IGNORE
    if not bool(sel.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[sel].float(), labels[sel], reduction="mean")


def total_loss(mlm, ar, ar_weight=1.0):
    return (mlm + ar_weight * ar) / (1.0 + ar_weight) if ar_weight > 0 else mlm
