"""Phase F1 diagnostic — Position-Free Future Prediction (docs/future_token_set_research.md §3-4).

Teacher-forced: the clean prefix (prompt + response so far) is in the cache; K [MASK] rows at the next K positions are
predicted in one forward (observed-first layout with nothing observed = the all-mask first step of every block decoder).
No AR logits.  For each block we score the K top-1 predictions against the K reference tokens:
  position accuracy      pred_i == gold_i, also by distance i
  multiset P / R / F1    multiset intersection of {pred} and {gold}; with K predictions and K golds P = R = F1
  optimal assignment     maximum bipartite matching pred_i -> gold_j with pred_i == gold_j (Hungarian on 0/1 cost ==
                         multiset intersection; computed explicitly as a check and reported)
  top-k future recall    maximum matching where position i may offer its top-k candidates (each position matches at
                         most one gold occurrence, each gold occurrence at most one position) -> count-aware, no credit
                         for duplicates; k = 1, 5, 10, 20.  Also fixed-candidate-count recall: the N (position, token)
                         pairs with the highest probability over the whole block, N = K and 2K, matched the same way
                         (so the number of candidates does not grow with k).
  duplicate rate         1 - unique(top-1 preds) / K
  count error            sum_t |count_pred(t) - count_gold(t)| / K
  block exact            all positions right
Genres: gsm8k (the model's own greedy references, --refs), and held-out teacher/human responses by task category from
--genre-files (en_general + reasoning -> "en", ja_general -> "ja", code + code_completion -> "code").  Blocks of
K = 4, 8, 16, 32 at stride K over the response, at most --max-blocks per item and K.
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. .venv-train/bin/python bench/diag_pffp.py --model runs/phaseC_sft3_4b/final \
      --refs results/pure/dev_refs_sft3.json --n 50 --out results/pure/diag_pffp_sft3.json"""
import argparse, collections, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts, gpu_info, save_json
from engine.cdlm.fast_model import FastQwen3

KS = (4, 8, 16, 32)
TOPK = (1, 5, 10, 20)
GENRE_OF = {"en_general": "en", "reasoning": "en", "ja_general": "ja", "code": "code", "code_completion": "code"}


def max_matching(cands, gold):
    """Maximum bipartite matching between positions (each offering a list of candidate tokens) and gold occurrences
    (list of tokens); an edge when the token is among the position's candidates.  Augmenting paths (K <= 32)."""
    K = len(gold)
    adj = [[j for j in range(K) if gold[j] in set(c)] for c in cands]
    match_g = [-1] * K

    def try_(i, seen):
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if match_g[j] < 0 or try_(match_g[j], seen):
                match_g[j] = i
                return True
        return False

    return sum(try_(i, set()) for i in range(len(cands)))


def multiset_inter(a, b):
    ca, cb = collections.Counter(a), collections.Counter(b)
    return sum((ca & cb).values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="bench/prompts_gsm8k_dev.json")
    ap.add_argument("--refs", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--genre-files", nargs="*", default=["data/distill/phaseA_medium_test.jsonl", "data/distill/phaseA_medium_val.jsonl",
                                                          "data/distill/phaseA_codecomp_test.jsonl"])
    ap.add_argument("--max-blocks", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    mask_id = getattr(hf.config, "cdlm_mask_token_id", None) or tok.convert_tokens_to_ids("<|mask|>")
    model = FastQwen3.from_hf(hf, device="cuda", dtype=torch.bfloat16, max_len=4096)
    del hf
    torch.cuda.empty_cache()
    # items: (genre, prompt_ids, response_ids)
    items = []
    prompts = load_prompts(a.prompts)[: a.n]
    refs = json.load(open(a.refs))["refs"]
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True,
                                      enable_thinking=False, return_tensors="pt")
        ids = ids[0].tolist() if torch.is_tensor(ids) else list(ids["input_ids"][0])
        items.append(("gsm8k", ids, refs[p["id"]]["response_ids"]))
    for f in a.genre_files:
        for line in open(f):
            ex = json.loads(line)
            g = GENRE_OF.get(ex.get("task_category"))
            if not g:
                continue
            text = tok.apply_chat_template([{"role": "user", "content": ((ex["context"] + "\n\n") if ex.get("context") else "") + ex["instruction"]}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False)
            p_ids = tok(text, add_special_tokens=False)["input_ids"]
            r_ids = tok(ex["response"] + tok.eos_token, add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(r_ids) > a.max_len or len(r_ids) < 8:
                continue
            items.append((g, p_ids, r_ids))
    print("items per genre:", collections.Counter(g for g, _, _ in items), file=sys.stderr, flush=True)
    agg = {}

    def A(genre, K):
        return agg.setdefault(genre, {}).setdefault(K, dict(blocks=0, pos_ok=[0] * K, inter=0, assign=0, exact=0, dup=0.0, cnt_err=0,
                                                              topk={k: 0 for k in TOPK}, fixedN={N: 0 for N in (1, 2)}, n_tok=0))
    t0 = time.time()
    n_fwd = 0
    for n, (genre, p_ids, r_ids) in enumerate(items):
        full = torch.tensor(p_ids + r_ids, dtype=torch.long, device="cuda")
        P = len(p_ids)
        model.kv.zero_()
        model.prefill(full, 0)
        for K in KS:
            starts = list(range(P, P + len(r_ids) - K + 1, K))[: a.max_blocks]
            for s in starts:
                gold = full[s:s + K].tolist()
                ids = torch.full((K,), mask_id, dtype=torch.long, device="cuda")
                pos = torch.arange(s, s + K, device="cuda")
                logits = model.forward(ids, pos, cache_len=s, write=False)
                n_fwd += 1
                lg = logits.float()
                lg[:, mask_id] = float("-inf")
                probs = torch.softmax(lg, -1)
                topv, topi = probs.topk(max(TOPK), dim=-1)
                topi, topv = topi.tolist(), topv.tolist()
                pred = [t[0] for t in topi]
                acc = A(genre, K)
                acc["blocks"] += 1
                acc["n_tok"] += K
                for i in range(K):
                    acc["pos_ok"][i] += int(pred[i] == gold[i])
                acc["inter"] += multiset_inter(pred, gold)
                acc["assign"] += max_matching([[t] for t in pred], gold)
                acc["exact"] += int(pred == gold)
                acc["dup"] += 1.0 - len(set(pred)) / K
                cp, cg = collections.Counter(pred), collections.Counter(gold)
                acc["cnt_err"] += sum(abs(cp[t] - cg[t]) for t in set(cp) | set(cg))
                for k in TOPK:
                    acc["topk"][k] += max_matching([t[:k] for t in topi], gold)
                # fixed candidate count N*K over the block: top (position, token) pairs by probability
                pairs = sorted(((topv[i][r], i, topi[i][r]) for i in range(K) for r in range(max(TOPK))), reverse=True)
                for N in (1, 2):
                    chosen = pairs[: N * K]
                    cands = [[] for _ in range(K)]
                    for _, i, t in chosen:
                        cands[i].append(t)
                    acc["fixedN"][N] += max_matching(cands, gold)
        if n % 25 == 0:
            print(f"{n}/{len(items)} {genre} forwards {n_fwd} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    summary = {}
    for genre, per_k in agg.items():
        summary[genre] = {}
        for K, c in sorted(per_k.items()):
            nt = c["n_tok"]
            summary[genre][str(K)] = dict(
                blocks=c["blocks"], position_acc=round(sum(c["pos_ok"]) / nt, 4),
                position_acc_by_distance=[round(x / c["blocks"], 3) for x in c["pos_ok"]],
                multiset_f1=round(c["inter"] / nt, 4), optimal_assignment_acc=round(c["assign"] / nt, 4),
                block_exact=round(c["exact"] / c["blocks"], 4), duplicate_rate=round(c["dup"] / c["blocks"], 4),
                count_error_per_token=round(c["cnt_err"] / nt, 4),
                topk_recall={str(k): round(v / nt, 4) for k, v in c["topk"].items()},
                fixed_candidates_recall={f"{N}K": round(v / nt, 4) for N, v in c["fixedN"].items()})
    print("=== pffp", json.dumps(summary), flush=True)
    save_json(a.out, {"model": a.model, "refs": a.refs, "genre_files": a.genre_files, "n_items": collections.Counter(g for g, _, _ in items),
                      "gpu": gpu_info(), "forwards": n_fwd, "seconds": round(time.time() - t0, 1), "summary": summary})


if __name__ == "__main__":
    main()
