#!/usr/bin/env python3
"""Check our LLaDA implementation against the Hugging Face reference: logits on a prompt+masked span, full forward vs HF,
and the cached block forward vs a full forward (should be identical up to bf16 noise)."""
import sys, os, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from engine.llada import LLaDA, MASK_ID
from transformers import AutoTokenizer, AutoModel
path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/ssd/models/hf/LLaDA-8B-Instruct"
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
text = tok.apply_chat_template([{"role": "user", "content": "Explain how a hash table works in one paragraph."}], add_generation_prompt=True, tokenize=False)
ids = tok(text, return_tensors="pt").input_ids.cuda()
x = torch.cat([ids, torch.full((1, 32), MASK_ID, device="cuda")], 1)
ours = LLaDA.from_pretrained(path)
with torch.no_grad():
    l_full = ours(x)
    cache = ours.new_cache(1, x.shape[1])
    l_cached_full = ours(x, cache=cache, start=0)
    l_blk = ours(x[:, ids.shape[1]:], cache=cache, start=ids.shape[1])
print("ours: cached full vs plain full max|diff| =", (l_full - l_cached_full).abs().max().item())
print("ours: block forward vs full forward on the block max|diff| =", (l_full[:, ids.shape[1]:] - l_blk).abs().max().item(),
      "argmax agree:", (l_full[:, ids.shape[1]:].argmax(-1) == l_blk.argmax(-1)).float().mean().item())
del ours; torch.cuda.empty_cache()
ref = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
with torch.no_grad():
    l_ref = ref(x).logits.float()
print("ours vs HF reference max|diff| =", (l_full - l_ref).abs().max().item(), "mean|diff| =", (l_full - l_ref).abs().mean().item(),
      "argmax agree:", (l_full.argmax(-1) == l_ref.argmax(-1)).float().mean().item())
