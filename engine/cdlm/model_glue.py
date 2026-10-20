"""Run a HF Qwen3ForCausalLM (transformers 5.20) on the dual stream with our 4-D mask and explicit position ids.

How transformers 5.20 accepts a custom 4-D mask (checked in modeling_qwen3.py / masking_utils.py and exercised by the
tests with both attention implementations):
  * `Qwen3Model.forward` does `if not isinstance(causal_mask_mapping := attention_mask, dict): ... create_causal_mask(...)`
    — passing `attention_mask={"full_attention": m, "sliding_attention": m}` skips mask creation entirely and `m` is
    handed verbatim to every layer's attention interface.  This is the path used here (`model_attention_mask`).
  * "sdpa": `m` may be a bool [N, 1, T, T] (True = attend) or an additive float mask; it goes straight to
    F.scaled_dot_product_attention(attn_mask=m) with is_causal=False (sdpa_attention.py:136 drops is_causal when a mask
    is given).  "eager": `m` is *added* to the scores (modeling_qwen3.py:192), so it must be the additive float form
    (0 / dtype-min) in the model dtype; a bool mask would silently add 1.0 — hence the conversion below.
  * A plain 4-D tensor (not a dict) also works for sdpa/eager: _preprocess_mask_arguments returns a 4-D mask as-is
    before any mask creation (masking_utils.py:796-798, "already prepared, or custom"); the dict is used because it is
    explicit and independent of that passthrough.  "flash_attention_2" cannot take a 4-D mask (needs the FlexAttention
    BlockMask route via mask.make_mask_mod + attn_implementation="flex_attention", not wired here).
  * position_ids [N, T] are used verbatim for RoPE (rotary_emb(hidden, position_ids)); nothing is recomputed from the
    physical index, so the shared x0/xt positions of data.py are honoured.
Logits are produced only for the requested rows: hidden states are gathered by (batch, row) index before lm_head,
which avoids materialising [N, 2L, V] (152k-wide for Qwen3).
"""
import torch

from .mask import to_additive

IGNORE = -100
__all__ = ["model_attention_mask", "forward_rows", "forward_batch", "add_mask_token"]


def model_attention_mask(model, mask_bool):
    """Convert a bool [N, 1, T, T] mask into what the model's attention implementation expects, as the dict
    Qwen3Model.forward takes without touching it."""
    impl = model.config._attn_implementation
    if impl == "sdpa":
        m = mask_bool
    elif impl == "eager":
        m = to_additive(mask_bool, model.get_input_embeddings().weight.dtype)
    else:
        raise ValueError(f"attn_implementation={impl!r}: only sdpa/eager take a dense 4-D mask (flex: use mask.make_mask_mod)")
    return {"full_attention": m, "sliding_attention": m}


def forward_rows(model, input_ids, position_ids, attention_mask, rows):
    """Logits [R, V] for the rows (batch_idx, pos_idx) only.  attention_mask: bool [N, 1, T, T] or the dict from
    model_attention_mask."""
    if not isinstance(attention_mask, dict):
        attention_mask = model_attention_mask(model, attention_mask)
    out = model.model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, use_cache=False)
    b, p = rows
    return model.lm_head(out.last_hidden_state[b, p])


def forward_batch(model, batch):
    """One forward on a collate() batch; returns mlm_logits/mlm_labels/mlm_weights and ar_logits/ar_labels (row lists)
    ready for loss.mlm_loss / loss.ar_loss."""
    dev = model.get_input_embeddings().weight.device
    mlm_rows = (batch["mlm_labels"] != IGNORE).nonzero(as_tuple=True)
    ar_rows = (batch["ar_labels"] != IGNORE).nonzero(as_tuple=True)
    rows = (torch.cat([mlm_rows[0], ar_rows[0]]).to(dev), torch.cat([mlm_rows[1], ar_rows[1]]).to(dev))
    logits = forward_rows(model, batch["input_ids"].to(dev), batch["position_ids"].to(dev),
                          batch["attention_mask"].to(dev), rows)
    n = mlm_rows[0].numel()
    return dict(
        mlm_logits=logits[:n], mlm_labels=batch["mlm_labels"][mlm_rows].to(dev),
        mlm_weights=batch["mlm_weights"][mlm_rows].to(dev),
        ar_logits=logits[n:], ar_labels=batch["ar_labels"][ar_rows].to(dev),
    )


@torch.no_grad()
def add_mask_token(model, mask_id=None):
    """Make `mask_id` a usable [MASK] row (default: first id beyond the current embedding matrix).

    If mask_id >= current rows, resize embeddings + lm_head to mask_id + 1 (HF resize_token_embeddings).  The mask row is
    then set to the mean of the pre-existing rows (input embedding and, if untied, lm_head) — the initialisation
    docs/wedlm_analysis.md section 11 recommends.  Returns {"mask_id", "old_rows", "new_rows", "added": [ids added]}
    and records mask_id in model.config.cdlm_mask_token_id.
    """
    emb = model.get_input_embeddings()
    old = emb.weight.shape[0]
    mask_id = old if mask_id is None else int(mask_id)
    if mask_id >= old:
        model.resize_token_embeddings(mask_id + 1, mean_resizing=False)
        emb = model.get_input_embeddings()
    emb.weight[mask_id] = emb.weight[:old].mean(0)
    head = model.get_output_embeddings()
    if head is not None and head.weight.data_ptr() != emb.weight.data_ptr():
        head.weight[mask_id] = head.weight[:old].mean(0)
    model.config.cdlm_mask_token_id = mask_id
    return {"mask_id": mask_id, "old_rows": old, "new_rows": emb.weight.shape[0], "added": list(range(old, emb.weight.shape[0]))}
