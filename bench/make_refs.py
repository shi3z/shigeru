"""Reference answers = a model's own greedy AR responses on a prompt file (HF generate), cached as
{"model", "max_new_tokens", "refs": {id: {response_ids, text, correct}}} — the format every teacher-forced B-series
diagnostic reads (--refs).  Used for the B7 trust diagnostics with the B-series BASE model (runs/zeta_sft_final_e5/final),
so that "correct token" means "the token the base model itself would have written" rather than another model's wording.
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. .venv-train/bin/python bench/make_refs.py --model runs/zeta_sft_final_e5/final \
      --prompts bench/prompts_gsm8k_dev.json --n 50 --out results/pure/dev_refs_e5.json"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts
from bench.diag_mask_recovery import make_refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    prompts = load_prompts(a.prompts)[a.skip: a.skip + a.n]
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        p["ids"] = ids[0].tolist() if torch.is_tensor(ids) else list(ids["input_ids"][0])
    stop_ids = {tok.eos_token_id} | ({tok.convert_tokens_to_ids("<|im_end|>")} if "<|im_end|>" in tok.get_vocab() else set())
    refs = make_refs(model, tok, prompts, a.max_new_tokens, sorted(stop_ids), a.out)
    print(f"=== refs {len(refs)} correct {sum(bool(v['correct']) for v in refs.values())}")


if __name__ == "__main__":
    main()
