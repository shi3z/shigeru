"""HF Qwen3 wrapper for the B-series: block-bidirectional 4D mask, optional time conditioning (a learned embedding of
the mask ratio added to the block rows' input embeddings — brief §4 option A, 16 buckets x hidden, ~41 k parameters),
LoRA adaptation (peft) so the base weights are never modified.  `forward_block` runs [prefix | block] in one pass
(training) and `denoise` runs the K block rows on a prefix cache (inference, prefix K/V computed once per block and
reused across denoising steps; the block rows are never appended to the cache).
"""
import torch
import torch.nn as nn

from .masks import block_mask, train_mask

__all__ = ["TimeEmbed", "BDiffModel"]


class TimeEmbed(nn.Module):
    def __init__(self, hidden, n_buckets=16):
        super().__init__()
        self.emb = nn.Embedding(n_buckets, hidden)
        nn.init.zeros_(self.emb.weight)
        self.n = n_buckets

    def forward(self, g):
        """g: float tensor [B] mask ratio in (0, 1] -> [B, hidden]."""
        idx = (g.clamp(0, 1) * (self.n - 1)).round().long()
        return self.emb(idx)


class BDiffModel(nn.Module):
    def __init__(self, lm, mask_id, time_cond=True, n_buckets=16):
        super().__init__()
        self.lm = lm
        self.mask_id = mask_id
        hidden = lm.config.hidden_size
        self.time = TimeEmbed(hidden, n_buckets) if time_cond else None

    @property
    def device(self):
        return self.lm.get_input_embeddings().weight.device

    def forward_block(self, prefix_ids, block_ids, g):
        """Training forward.  prefix_ids [B, P] (right-padded prefixes are NOT supported: use equal-length batches or
        batch 1 per prefix length), block_ids [B, K], g [B] mask ratio.  Returns logits of the block rows [B, K, V]."""
        B, P = prefix_ids.shape
        K = block_ids.shape[1]
        ids = torch.cat([prefix_ids, block_ids], 1)
        emb = self.lm.get_input_embeddings()(ids)
        if self.time is not None:
            te = self.time(g.to(emb.device)).to(emb.dtype)
            emb = torch.cat([emb[:, :P], emb[:, P:] + te[:, None]], 1)
        mask = train_mask(P, K, device=emb.device)[None, None]
        pos = torch.arange(P + K, device=emb.device)[None].expand(B, -1)
        out = self.lm(inputs_embeds=emb, attention_mask=mask, position_ids=pos)
        return out.logits[:, P:]

    @torch.no_grad()
    def prefix_cache(self, prefix_ids):
        from transformers import DynamicCache
        cache = DynamicCache(config=self.lm.config)
        self.lm(input_ids=prefix_ids, past_key_values=cache, use_cache=True)
        return cache

    @torch.no_grad()
    def denoise(self, cache, P, block_ids, g):
        """One reverse step's network call: block_ids [1, K] (masks and visible tokens), g mask ratio -> logits [K, V].
        The prefix cache is reused and cropped back to P afterwards (block K/V are never kept)."""
        K = block_ids.shape[1]
        emb = self.lm.get_input_embeddings()(block_ids)
        if self.time is not None:
            emb = emb + self.time(torch.tensor([g], device=emb.device)).to(emb.dtype)[:, None]
        mask = block_mask(P, K, device=emb.device)[None, None]
        pos = torch.arange(P, P + K, device=emb.device)[None]
        out = self.lm(inputs_embeds=emb, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True)
        extra = cache.get_seq_length() - P
        if extra > 0:
            cache.crop(-extra)
        return out.logits[0]
