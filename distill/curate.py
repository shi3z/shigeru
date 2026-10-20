"""Phase A curation: quality filter, dedup, evaluation-set contamination check, train/val/test split.
Usage: .venv/bin/python -m distill.curate data/distill/phaseA_small.jsonl --out data/distill/phaseA_small [--min-score 4]
Writes <out>_{train,val,test}.jsonl, <out>_rejected.jsonl and <out>_stats.json."""
import argparse, collections, gzip, json, os, random, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACCEPT = {"verified_correct", "verified_pass", "judged"}


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def ngrams(s, n):
    toks = norm(s).split()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def eval_texts():
    """Texts of every evaluation set used by this repo; any training item overlapping them is dropped."""
    out = []
    for l in open(os.path.join(ROOT, "data/gsm8k_test.jsonl")):
        out.append(json.loads(l)["question"])
    for l in open(os.path.join(ROOT, "data/HumanEval.jsonl")):
        out.append(json.loads(l)["prompt"])
    for t in json.load(open(os.path.join(ROOT, "data/mbpp_sanitized.json"))):
        out.append(t.get("prompt") or t.get("text") or "")
    for p in json.load(open(os.path.join(ROOT, "bench/prompts.json"))):
        out.append(p["prompt"] if isinstance(p, dict) else str(p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-score", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--test", type=float, default=0.05)
    ap.add_argument("--ngram", type=int, default=13)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.inp)]
    stats = collections.Counter()
    stats["input"] = len(rows)
    ev = set()
    for t in eval_texts():
        ev |= ngrams(t, a.ngram)
    kept, rejected = [], []
    seen_exact, seen_sh = set(), []
    for r in rows:
        why = None
        if r["verification_status"] not in ACCEPT:
            why = "unverified:" + r["verification_status"]
        elif (r.get("quality_score") or 0) < a.min_score:
            why = f"low_score:{r.get('quality_score')}"
        elif r.get("finish_reason") == "length":
            why = "truncated"
        elif not r["instruction"].strip() or not r["response"].strip():
            why = "empty"
        elif ngrams(r["instruction"], a.ngram) & ev or ngrams(r["response"], a.ngram) & ev:
            why = "eval_contamination"
        else:
            key = norm(r["instruction"])
            if key in seen_exact:
                why = "dup_exact"
            else:
                sh = ngrams(r["instruction"], 8)
                for other in seen_sh:
                    if sh and other and len(sh & other) / len(sh | other) > 0.7:
                        why = "dup_near"
                        break
                if why is None:
                    seen_exact.add(key)
                    seen_sh.append(sh)
        if why:
            r["reject_reason"] = why
            rejected.append(r)
            stats["reject:" + why.split(":")[0]] += 1
        else:
            kept.append(r)
    rng = random.Random(a.seed)
    rng.shuffle(kept)
    n = len(kept)
    nt, nv = int(n * a.test), int(n * a.val)
    splits = {"test": kept[:nt], "val": kept[nt:nt + nv], "train": kept[nt + nv:]}
    for name, rs in splits.items():
        with open(f"{a.out}_{name}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[name] = len(rs)
    with open(f"{a.out}_rejected.jsonl", "w") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["kept"] = n
    stats["by_category"] = dict(collections.Counter(r["task_category"] for r in kept))
    stats["response_tokens_total"] = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in kept)
    json.dump(stats, open(f"{a.out}_stats.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
