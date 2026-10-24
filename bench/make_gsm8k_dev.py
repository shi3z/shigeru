"""Development set for the pure-diffusion work (docs/pure_diffusion_research.md): GSM8K *train* questions that do not
appear (verbatim or as a substring, first 120 chars) in any training JSONL under data/distill/, in the exact format of
bench/prompts_gsm8k_full.json (id, lang, prompt with the same instruction suffix, numeric answer).  GSM8K test is
reserved for final evaluation and is never used for selecting sampling strategies or hyper-parameters.
  .venv/bin/python bench/make_gsm8k_dev.py --n 200 --seed 0 --out bench/prompts_gsm8k_dev.json"""
import argparse, glob, json, os, random, re, sys

import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUFFIX = "\nSolve step by step, then give the final answer on the last line as '#### <number>'."


def used_texts():
    used = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data/distill/*.jsonl"))):
        for line in open(f):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            # only the prompt-side fields (the question can only be there; scanning responses too was 100x slower)
            for k in ("instruction", "prompt", "question"):
                if isinstance(d.get(k), str):
                    used.append(d[k])
            for m in d.get("messages", []) if isinstance(d.get("messages"), list) else []:
                if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                    used.append(m["content"])
    return "\n".join(used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "bench/prompts_gsm8k_dev.json"))
    a = ap.parse_args()
    rows = pq.read_table(os.path.join(ROOT, "data/sources/gsm8k_train.parquet")).to_pylist()
    blob = used_texts()
    clean = [r for r in rows if r["question"].strip()[:120] not in blob]
    print(f"gsm8k train {len(rows)}, not in any training file: {len(clean)}", file=sys.stderr)
    rng = random.Random(a.seed)
    rng.shuffle(clean)
    out = []
    for i, r in enumerate(clean[: a.n]):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        assert re.fullmatch(r"-?\d+(\.\d+)?", gold), gold
        out.append({"id": f"gsm8k_dev_{i:04d}", "lang": "en", "prompt": r["question"].strip() + SUFFIX, "answer": gold,
                    "source": "openai/gsm8k:train", "selection": {"seed": a.seed, "excluded_if_in": "data/distill/*.jsonl"}})
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {len(out)} -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
