"""B-series data generators (brief B3 / B6).

--mode states : on-policy denoising states of the B-sampler itself.  For training prefixes (cuts inside the responses of
  the stage-2 JSONL), run engine/bdiff/sampler.sample_block from all masks with the given adapter and record a few
  intermediate states (tokens with -1 for masks) -> JSONL rows {id, tokens (prefix + GOLD block), prompt_len, start, state,
  n_wrong}; train_bdiff.py consumes them with --states (the target is the gold block, the input the recorded state).
--mode samples: teacher samples for step distillation.  Run the S-step sampler on training prefixes and emit
  {prefix_ids, x0 = the sampled 16-token block} rows (train_bdiff --blocks); the student then learns to reach the
  teacher's 16-step output in fewer steps (trained with --t-min 0.5 to emphasise high-noise inputs).
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv-train/bin/python distill/gen_bstates.py --mode states --model runs/phaseC_sft3_4b/final \
      --adapter runs/bdiff_sft3_b1/adapter --time-embed runs/bdiff_sft3_b1/time_embed.pt --data data/distill/phaseA_stage2c_train.jsonl \
      --limit 3000 --out data/distill/bstates_sft3_b1.jsonl"""
import argparse, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from distill.train_cdlm import load_jsonl, tokenize_examples
from engine.bdiff.model import BDiffModel
from engine.bdiff.sampler import sample_block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["states", "samples"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--time-embed", default=None)
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--cuts", type=int, default=2)
    ap.add_argument("--states-per-block", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    prng = random.Random(a.seed)
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    mask_id = getattr(lm.config, "cdlm_mask_token_id", None) or tok.convert_tokens_to_ids("<|mask|>")
    if a.adapter:
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, a.adapter).eval().merge_and_unload()
    model = BDiffModel(lm, mask_id, time_cond=not a.no_time).to(dev).eval()
    if model.time is not None and a.time_embed:
        model.time.load_state_dict(torch.load(a.time_embed, map_location="cpu")); model.time.to(dev)
    rows = load_jsonl(a.data)
    prng.shuffle(rows)
    rows = rows[: a.limit]
    items, dropped, trunc = tokenize_examples(rows, tok, a.max_len)
    K = a.K
    t0 = time.time()
    n_out = n_wrong_states = 0
    with open(a.out, "w") as f, torch.no_grad():
        for n, ((t, P), row) in enumerate(zip(items, rows)):
            L = t.numel()
            if L - P < K:
                continue
            ids = t.tolist()
            for _ in range(a.cuts):
                c = prng.randint(P, L - K)
                cache = model.prefix_cache(torch.tensor([ids[:c]], device=dev))
                gold = ids[c:c + K]
                toks, info = sample_block(model, cache, c, K, a.steps, mask_id, order="confidence", trace=True,
                                          generator=torch.Generator().manual_seed(a.seed + n))
                if a.mode == "samples":
                    f.write(json.dumps({"id": row.get("id", n), "prefix_ids": ids[:c], "x0": toks, "gold": gold}) + "\n"); n_out += 1
                else:
                    tr = info["trace"]
                    picks = sorted({max(0, round(i * (len(tr) - 1) / max(1, a.states_per_block))) for i in range(1, a.states_per_block + 1)})
                    for k in picks:
                        st = [-1 if x == mask_id else x for x in tr[k]["tokens"]]
                        if all(x == -1 for x in st):
                            continue
                        wrong = sum(1 for i in range(K) if st[i] != -1 and st[i] != gold[i])
                        f.write(json.dumps({"id": row.get("id", n), "tokens": ids[:c + K], "prompt_len": P, "start": c, "state": st, "n_wrong": wrong}) + "\n")
                        n_out += 1; n_wrong_states += int(wrong > 0)
            if n % 100 == 0:
                print(f"{n}/{len(items)} rows -> {n_out} ({n_wrong_states} with errors) {time.time() - t0:.0f}s", flush=True)
    print(f"=== gen_bstates {a.mode} {n_out} rows ({n_wrong_states} with errors) from {len(items)} items in {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
