"""CUDA-graph execution of the Open Mercury block forward.

A request of prompt length P and generation length G uses a cache padded to L_bucket = roundup(P+G, 64); padded slots are
excluded from attention by a key mask. Two graph kinds are captured per (shape) key and replayed with static input buffers:
  * block graph:  tokens [1,T_block] + pos [T_block]  -> logits [1,T_block,V]   (one per (T_block, L_bucket))
  * full graph:   tokens [1,L_bucket] + pos            -> logits [1,L_bucket,V] (one per L_bucket; padded tokens are EOS)
Replay cost is one graph launch instead of ~300 kernel launches for the 32-layer block forward.
"""
import torch
from .llada import EOS_ID

BUCKET = 64

def bucket(n):
    return ((n + BUCKET - 1) // BUCKET) * BUCKET

class GraphRunner:
    def __init__(self, model, max_len=4096):
        self.model, self.graphs = model, {}
        self.cache = None; self.cache_len = 0

    def get_cache(self, L):
        if self.cache is None or self.cache_len != L:
            self.cache = self.model.new_cache(1, L); self.cache_len = L
            self.graphs = {}   # graphs refer to the cache tensors: recapture after a resize
        return self.cache

    def _capture(self, key, T, L, key_mask):
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        tokens = torch.full((1, T), EOS_ID, dtype=torch.long, device="cuda")
        pos = torch.arange(T, device="cuda")
        with torch.cuda.stream(s):
            for _ in range(2):   # warm up (allocator, cuBLAS, SDPA kernel selection)
                out = self.model(tokens, cache=self.cache, pos=pos, key_mask=key_mask)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self.model(tokens, cache=self.cache, pos=pos, key_mask=key_mask)
        self.graphs[key] = (g, tokens, pos, out)
        return self.graphs[key]

    def run(self, tokens, pos, key_mask, use_graph=True):
        """tokens [1,T] (long), pos [T] (long), key_mask [L] (bool) -> logits [1,T,V] (a static buffer: clone if kept)"""
        T, L = tokens.shape[1], key_mask.shape[0]
        if not use_graph:
            return self.model(tokens, cache=self.cache, pos=pos, key_mask=key_mask)
        key = (T, L)
        if key not in self.graphs:
            self._capture(key, T, L, key_mask)
        g, t_in, p_in, out = self.graphs[key]
        t_in.copy_(tokens); p_in.copy_(pos)
        g.replay()
        return out
