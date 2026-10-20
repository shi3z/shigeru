"""Uniform text-generation adapters for the quality benchmarks: load(name) -> (generate(prompt_text) -> text, info dict).
Names: ours[:threshold[:graph]], llada[:steps], fastdllm_v1[:threshold], fastdllm_v2[:threshold], llada2[:threshold:edit], hf:<model path>"""
import os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
LLADA = "/mnt/ssd/models/hf/LLaDA-8B-Instruct"

def load(name, gen_length=512, block_length=32):
    from transformers import AutoTokenizer
    parts = name.split(":")
    kind = parts[0]
    if kind == "ours":
        from engine.llada import LLaDA, EOS_ID
        from engine.sampler import generate
        from engine.graphs import GraphRunner
        thr = float(parts[1]) if len(parts) > 1 else 0.9; graph = len(parts) > 2 and parts[2] == "graph"
        tok = AutoTokenizer.from_pretrained(LLADA, trust_remote_code=True); model = LLaDA.from_pretrained(LLADA); runner = GraphRunner(model) if graph else None
        def gen(text):
            ids = tok(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False), return_tensors="pt").input_ids.cuda()
            x = generate(model, ids, gen_length=gen_length, block_length=block_length, mode="threshold", threshold=thr, runner=runner, use_graph=graph, stop_on_eos=True)
            seq = x[0, ids.shape[1]:].tolist(); eff = seq.index(EOS_ID) if EOS_ID in seq else len(seq)
            return tok.decode(seq[:eff], skip_special_tokens=True)
        return gen, {"engine": name, "model": LLADA}
    if kind == "llada":
        from transformers import AutoModel
        sys.path.insert(0, os.path.join(ROOT, "third_party", "LLaDA")); import generate as og
        steps = int(parts[1]) if len(parts) > 1 else gen_length
        tok = AutoTokenizer.from_pretrained(LLADA, trust_remote_code=True); model = AutoModel.from_pretrained(LLADA, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
        def gen(text):
            ids = tok(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False), return_tensors="pt").input_ids.cuda()
            x = og.generate(model, ids, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0.0, remasking="low_confidence")
            seq = x[0, ids.shape[1]:].tolist(); eff = seq.index(126081) if 126081 in seq else len(seq)
            return tok.decode(seq[:eff], skip_special_tokens=True)
        return gen, {"engine": name, "model": LLADA}
    if kind == "fastdllm_v1":
        sys.path.insert(0, os.path.join(ROOT, "third_party", "Fast-dLLM", "v1", "llada"))
        from model.modeling_llada import LLaDAModelLM; import generate as fd
        thr = float(parts[1]) if len(parts) > 1 else 0.9
        tok = AutoTokenizer.from_pretrained(LLADA, trust_remote_code=True); model = LLaDAModelLM.from_pretrained(LLADA, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
        def gen(text):
            ids = tok(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False), return_tensors="pt").input_ids.cuda()
            with torch.inference_mode():
                out = fd.generate_with_dual_cache(model, ids, steps=gen_length, gen_length=gen_length, block_length=block_length, temperature=0.0, remasking="low_confidence", threshold=thr)
            x = out[0] if isinstance(out, tuple) else out; seq = x[0, ids.shape[1]:].tolist(); eff = seq.index(126081) if 126081 in seq else len(seq)
            return tok.decode(seq[:eff], skip_special_tokens=True)
        return gen, {"engine": name, "model": LLADA}
    if kind == "fastdllm_v2":
        from transformers import AutoModelForCausalLM
        path = "/mnt/ssd/models/hf/Fast_dLLM_v2_7B"; thr = float(parts[1]) if len(parts) > 1 else 0.9
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True); model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
        def gen(text):
            ids = tok([tok.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)], return_tensors="pt").input_ids.cuda()
            with torch.inference_mode():
                out = model.generate(ids, tokenizer=tok, max_new_tokens=gen_length, threshold=thr, small_block_size=8, block_size=32, temperature=0)
            seq = out[0, ids.shape[1]:].tolist(); eff = seq.index(151645) if 151645 in seq else len(seq)
            return tok.decode(seq[:eff], skip_special_tokens=True)
        return gen, {"engine": name, "model": path}
    if kind == "llada2":
        from transformers import AutoModelForCausalLM
        path = "/mnt/ssd/models/hf/LLaDA2.1-mini"; thr = float(parts[1]) if len(parts) > 1 else 0.7; edit = float(parts[2]) if len(parts) > 2 else 0.5
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True); model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
        def gen(text):
            ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True, return_tensors="pt").cuda()
            with torch.inference_mode():
                out = model.generate(inputs=ids, eos_early_stop=True, gen_length=gen_length, block_length=block_length, threshold=thr, editing_threshold=edit, temperature=0.0)
            seq = out[0].tolist()
            if len(seq) > ids.shape[1] and seq[:ids.shape[1]] == ids[0].tolist(): seq = seq[ids.shape[1]:]
            eos = tok.eos_token_id; eff = seq.index(eos) if eos in seq else len(seq)
            return tok.decode(seq[:eff], skip_special_tokens=True)
        return gen, {"engine": name, "model": path}
    if kind == "hf":
        from transformers import AutoModelForCausalLM
        path = parts[1]
        tok = AutoTokenizer.from_pretrained(path); model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
        def gen(text):
            enc = tok(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False, enable_thinking=False), return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=gen_length, do_sample=False)
            return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        return gen, {"engine": name, "model": path}
    raise ValueError(name)
