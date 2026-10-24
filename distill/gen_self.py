"""On-policy self-distillation data (docs/pure_diffusion_research.md section 2.4, hypothesis H1): re-generate the responses of a
stage-2 JSONL with the greedy AR path of a converted checkpoint itself (vLLM, batched), keeping every other field.  The prompt is
built exactly as distill/train_cdlm.py does (chat template of context + "\\n\\n" + instruction, add_generation_prompt,
enable_thinking=False).  Output records: same keys, `response` = the model's greedy response, `response_teacher` = the original,
`self_finish` = "stop" | "length".  Items whose prompt exceeds --max-prompt tokens are dropped (they would not fit max_len 2048
with a response anyway).  Runs in the vLLM venv:
  CUDA_VISIBLE_DEVICES=4 /mnt/ssdraid/project/qwen-server/.venv/bin/python distill/gen_self.py --model runs/phaseC_sft3_4b/final \
      --data data/distill/phaseA_stage2c_train.jsonl --out data/distill/pure_self_sft3_stage2c.jsonl"""
import argparse, json, os, sys, time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--max-prompt", type=int, default=1600)
    ap.add_argument("--max-model-len", type=int, default=3072)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model)
    items = [json.loads(l) for l in open(a.data)]
    if a.limit:
        items = items[: a.limit]
    prompts, keep, dropped = [], [], 0
    for ex in items:
        text = tok.apply_chat_template([{"role": "user", "content": ((ex["context"] + "\n\n") if ex.get("context") else "") + ex["instruction"]}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        if n > a.max_prompt:
            dropped += 1
            continue
        prompts.append(text)
        keep.append(ex)
    print(f"{len(keep)} prompts ({dropped} dropped for length > {a.max_prompt})", file=sys.stderr, flush=True)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util,
              enable_prefix_caching=True, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_new, stop_token_ids=[tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|im_end|>")] if "<|im_end|>" in tok.get_vocab() else []))
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    n_tok = 0
    with open(a.out, "w") as f:
        for ex, o in zip(keep, outs):
            c = o.outputs[0]
            n_tok += len(c.token_ids)
            rec = dict(ex)
            rec["response_teacher"] = ex.get("response")
            rec["response"] = c.text
            rec["self_finish"] = "length" if c.finish_reason == "length" else "stop"
            rec["self_model"] = a.model
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    dt = time.time() - t0
    print(f"=== gen_self {len(keep)} items, {n_tok} tokens in {dt:.0f}s = {n_tok / dt:.0f} tok/s -> {a.out}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
