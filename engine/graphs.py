"""CUDA-graph execution of the Open Mercury block forward.

The runner owns ONE cache sized for `max_len` positions (allocated once) and a static key mask [max_len]; a request of
length L (prompt + generation) uses positions 0..L-1 and masks the rest. Graphs are captured per span length T (the
32-token block forward, and one per bucketed full-forward length) with static token/position buffers and replayed;
the key mask is copied into its static buffer before every replay, so graphs stay valid across requests.
Replay cost is one graph launch instead of ~300 kernel launches for the 32-layer block forward.
"""
import torch
from .llada import EOS_ID

BUCKET = 64

def bucket(n):
    return ((n + BUCKET - 1) // BUCKET) * BUCKET

class GraphRunner:
    def __init__(self, model, max_len=4096):
        self.model, self.graphs, self.max_len = model, {}, max_len
        self.cache = model.new_cache(1, max_len)
        self.key_mask = torch.zeros(max_len, dtype=torch.bool, device="cuda")

    def get_cache(self, L):
        assert L <= self.max_len, f"request length {L} exceeds the runner's max_len {self.max_len}"
        return self.cache

    def set_valid(self, L):
        self.key_mask.zero_(); self.key_mask[:L] = True

    def _capture(self, T):
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        tokens = torch.full((1, T), EOS_ID, dtype=torch.long, device="cuda")
        pos = torch.arange(T, device="cuda")
        with torch.cuda.stream(s):
            for _ in range(2):   # warm up (allocator, cuBLAS, SDPA kernel selection)
                self.model(tokens, cache=self.cache, pos=pos, key_mask=self.key_mask)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self.model(tokens, cache=self.cache, pos=pos, key_mask=self.key_mask)
        self.graphs[T] = (g, tokens, pos, out)
        return self.graphs[T]

    def run(self, tokens, pos, L, use_graph=True):
        """tokens [1,T] (long), pos [T] (long), L = valid length of the request -> logits [1,T,V] (static buffer: clone if kept)"""
        self.set_valid(L)
        if not use_graph:
            return self.model(tokens, cache=self.cache, pos=pos, key_mask=self.key_mask)
        T = tokens.shape[1]
        if T not in self.graphs:
            self._capture(T)
        g, t_in, p_in, out = self.graphs[T]
        t_in.copy_(tokens); p_in.copy_(pos)
        g.replay()
        return out
