"""Phase B10.5 §3–§4 — syntax-collapse and structural-dependency analysis of B-series code generation
(docs/distillation_ablation_structural_consistency.md).

GEN mode (free generation on HumanEval prompts, S in --steps, left-to-right, exactly S forwards per block, traced):
  * every program is classified with ast.parse + the HumanEval tests: pass / test_fail (AssertionError) /
    runtime_error / timeout / IndentationError / SyntaxError / incomplete (hit the length budget, no closing fence)
  * for syntax errors the ast error line/column is mapped back to the generated token, its block, its position in the
    block and the denoising step at which that token was placed (from the sampler trace)
  * token categories over the whole output: newline, indent (whitespace-only), colon, bracket, quote, other; and the
    categories of the two tokens each step places together (pair types)
  * S = 8 vs S = 16 on the same prompt: index and category of the first token where the two outputs diverge
TF mode (--refs: the base model's own greedy answers): left-to-right S-step trajectories on the reference blocks; token
accuracy by category, accuracy right after a newline and at indentation positions, pair accuracy when the pair crosses
a syntactic boundary (newline / colon / bracket in either token) vs not — on HumanEval and, for comparison, GSM8K.
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv-train/bin/python bench/diag_code_structure.py --model runs/zeta_sft_final_e5/final \
      --adapter runs/bdiff_e5_lr1e4/adapter --adapter2 runs/consist_e/adapter --steps 8,16 --n 164 --out results/pure/code_struct_e.json"""
import argparse, ast, collections, json, os, re, subprocess, sys, tempfile, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench.common import load_prompts, gpu_info, save_json
from engine.bdiff.model import BDiffModel
from engine.bdiff.sampler2 import sample_block_fixed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def category(tok_text):
    if "\n" in tok_text:
        return "newline"
    if tok_text.strip() == "" and tok_text != "":
        return "indent"
    if ":" in tok_text:
        return "colon"
    if any(c in tok_text for c in "()[]{}"):
        return "bracket"
    if any(c in tok_text for c in "\"'"):
        return "quote"
    return "other"


def extract_code(text):
    m = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (max(m, key=len) if m else text), bool(m)


def run_tests(program, timeout=15):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program); path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout, text=True)
        return p.returncode == 0, (p.stderr or "")[-400:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(path)


def classify(program, finish, fenced):
    try:
        ast.parse(program)
    except IndentationError as e:
        return "IndentationError", e.lineno, e.offset
    except SyntaxError as e:
        if finish == "length" or not fenced:
            return "incomplete", e.lineno, e.offset
        return "SyntaxError", e.lineno, e.offset
    ok, err = run_tests(program)
    if ok:
        return "pass", None, None
    if "timeout" in err:
        return "timeout", None, None
    if "AssertionError" in err:
        return "test_fail", None, None
    return "runtime_error", None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--adapter2", default=None)
    ap.add_argument("--prompts", default="bench/prompts_humaneval.json")
    ap.add_argument("--refs", default=None, help="TF mode: reference answers (humaneval_refs_e5.json / dev_refs_e5.json)")
    ap.add_argument("--n", type=int, default=164)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--steps", default="8,16")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-blocks", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    mask_id = getattr(lm.config, "cdlm_mask_token_id", None) or tok.convert_tokens_to_ids("<|mask|>")
    from peft import PeftModel
    for ad in (a.adapter, a.adapter2):
        if ad:
            lm = PeftModel.from_pretrained(lm, ad).eval().merge_and_unload()
    model = BDiffModel(lm, mask_id, time_cond=False).to(dev).eval()
    K = a.K
    stop_ids = {tok.eos_token_id} | ({tok.convert_tokens_to_ids("<|im_end|>")} if "<|im_end|>" in tok.get_vocab() else set())
    prompts = load_prompts(a.prompts)[: a.n]
    he = {}
    if "humaneval" in a.prompts:
        for it in (json.loads(l) for l in open(os.path.join(ROOT, "data", "HumanEval.jsonl"))):
            he[it["task_id"].replace("/", "_")] = it
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        p["ids"] = [int(t) for t in (ids[0] if torch.is_tensor(ids) else ids["input_ids"][0])]
    t0 = time.time()
    out = dict(model=a.model, adapter=a.adapter, adapter2=a.adapter2, prompts=a.prompts, n=len(prompts), gpu=gpu_info())
    steps_list = [int(x) for x in a.steps.split(",")]
    if a.refs:
        # ---------------- TF mode: accuracy by category on reference blocks
        refs = json.load(open(a.refs))["refs"]
        res = {}
        for S in steps_list:
            cat_acc = collections.defaultdict(lambda: [0, 0]); after_nl = [0, 0]; indent_pos = [0, 0]
            pair_cross = [0, 0]; pair_plain = [0, 0]; pair_type = collections.defaultdict(lambda: [0, 0])
            for p in prompts:
                if p["id"] not in refs:
                    continue
                resp = refs[p["id"]]["response_ids"]; full = p["ids"] + resp; P0 = len(p["ids"])
                for s0 in list(range(P0, P0 + len(resp) - K + 1, K))[: a.max_blocks]:
                    gold = full[s0:s0 + K]
                    gcat = [category(tok.decode([g])) for g in gold]
                    cache = model.prefix_cache(torch.tensor([full[:s0]], device=dev))
                    toks, info = sample_block_fixed(model, cache, s0, K, S, mask_id, unmask="left", remask="none", early_stop=False, trace=True)
                    prev = [mask_id] * K
                    for st in info["trace"]:
                        placed = [q for q in range(K) if prev[q] == mask_id and st["tokens"][q] != mask_id]
                        prev = st["tokens"]
                        if len(placed) == 2:
                            i, j = placed
                            cross = any(c in ("newline", "colon", "bracket", "indent") for c in (gcat[i], gcat[j]))
                            ok = toks[i] == gold[i] and toks[j] == gold[j]
                            (pair_cross if cross else pair_plain)[0] += int(ok); (pair_cross if cross else pair_plain)[1] += 1
                            pt = pair_type[f"{gcat[i]}+{gcat[j]}"]; pt[0] += int(ok); pt[1] += 1
                    for q in range(K):
                        ok = int(toks[q] == gold[q])
                        cat_acc[gcat[q]][0] += ok; cat_acc[gcat[q]][1] += 1
                        if q > 0 and gcat[q - 1] == "newline":
                            after_nl[0] += ok; after_nl[1] += 1
                        if gcat[q] == "indent":
                            indent_pos[0] += ok; indent_pos[1] += 1
            res[f"S{S}"] = dict(token_acc_by_category={c: dict(acc=round(v[0] / max(1, v[1]), 4), n=v[1]) for c, v in sorted(cat_acc.items())},
                                acc_after_newline=dict(acc=round(after_nl[0] / max(1, after_nl[1]), 4), n=after_nl[1]),
                                acc_at_indent=dict(acc=round(indent_pos[0] / max(1, indent_pos[1]), 4), n=indent_pos[1]),
                                pair_acc_crossing_boundary=dict(acc=round(pair_cross[0] / max(1, pair_cross[1]), 4), n=pair_cross[1]),
                                pair_acc_plain=dict(acc=round(pair_plain[0] / max(1, pair_plain[1]), 4), n=pair_plain[1]),
                                pair_acc_by_type={k: dict(acc=round(v[0] / v[1], 4), n=v[1]) for k, v in sorted(pair_type.items(), key=lambda kv: -kv[1][1])[:14]})
            print("=== tf", S, json.dumps({k: v for k, v in res[f"S{S}"].items() if k != "pair_acc_by_type"}), flush=True)
        out["tf"] = res
    else:
        # ---------------- GEN mode
        gens = {}
        for S in steps_list:
            rows = []
            cls = collections.Counter(); err_step = collections.Counter(); err_pos = collections.Counter(); all_step = collections.Counter()
            cat_count = collections.Counter(); pair_types = collections.Counter(); err_cat = collections.Counter()
            for p in prompts:
                ids = list(p["ids"]); outt = []; finish = None; placed_step = []; placed_pos = []; blk = 0
                while finish is None:
                    Kb = min(K, a.max_new_tokens - len(outt))
                    cache = model.prefix_cache(torch.tensor([ids + outt], device=dev))
                    toks, info = sample_block_fixed(model, cache, len(ids) + len(outt), Kb, S, mask_id, unmask="left", remask="none", trace=True)
                    step_of = [None] * Kb; prev = [mask_id] * Kb
                    for st in info["trace"]:
                        pl = [q for q in range(Kb) if prev[q] == mask_id and st["tokens"][q] != mask_id]
                        for q in pl:
                            step_of[q] = st["step"]
                        prev = st["tokens"]
                        if len(pl) == 2:
                            pair_types[category(tok.decode([toks[pl[0]]])) + "+" + category(tok.decode([toks[pl[1]]]))] += 1
                    keep = Kb
                    for q in range(Kb):
                        if toks[q] in stop_ids:
                            keep, finish = q + 1, "stop"; break
                    outt += toks[:keep]; placed_step += step_of[:keep]; placed_pos += list(range(keep)); blk += 1
                    if len(outt) >= a.max_new_tokens:
                        finish = finish or "length"
                text = tok.decode(outt, skip_special_tokens=True)
                code, fenced = extract_code(text)
                it = he.get(p["id"])
                if it is not None:
                    header = it["prompt"]; entry = it["entry_point"]
                    body = code if f"def {entry}" in code else header + code
                    program = body + "\n\n" + it["test"] + f"\n\ncheck({entry})\n"
                else:
                    program = code
                label, lineno, offset = classify(program, finish, fenced)
                cls[label] += 1
                # token categories of the output
                pieces = [tok.decode([t]) for t in outt]
                for pc in pieces:
                    cat_count[category(pc)] += 1
                for s_ in placed_step:
                    all_step[s_] += 1
                err_info = None
                if lineno is not None and it is not None:
                    # map the error line in `body` to a character offset in the generated text
                    body_lines = body.split("\n")
                    if lineno - 1 < len(body_lines):
                        char = sum(len(l) + 1 for l in body_lines[: lineno - 1]) + (offset or 1) - 1
                        # position of `body` text inside the generated text: body starts with header (not generated) if header was prepended
                        gen_char = char - (len(header) if body.startswith(header) and f"def {entry}" not in code else 0) + (text.find(code) if code in text else 0)
                        cum = 0; tidx = None
                        for n_, pc in enumerate(pieces):
                            cum += len(pc)
                            if cum > gen_char:
                                tidx = n_; break
                        if tidx is not None:
                            err_step[placed_step[tidx]] += 1; err_pos[placed_pos[tidx]] += 1; err_cat[category(pieces[tidx])] += 1
                            err_info = dict(token_index=tidx, block=tidx // K, pos_in_block=placed_pos[tidx], step=placed_step[tidx], token=pieces[tidx], context=text[max(0, gen_char - 40): gen_char + 20])
                rows.append(dict(id=p["id"], label=label, finish=finish, n_tokens=len(outt), err=err_info, text=text))
                print(f"[S{S}] {p['id']} {label} {len(outt)} tok", flush=True)
            n = len(rows)
            gens[f"S{S}"] = dict(classification={k: v for k, v in cls.most_common()}, classification_rate={k: round(v / n, 4) for k, v in cls.most_common()},
                                 placement_step_of_error_token={str(k): v for k, v in sorted(err_step.items())}, placement_step_of_all_tokens={str(k): v for k, v in sorted(all_step.items())},
                                 error_token_position_in_block={str(k): v for k, v in sorted(err_pos.items())}, error_token_category={k: v for k, v in err_cat.most_common()},
                                 token_category_rate={k: round(v / max(1, sum(cat_count.values())), 4) for k, v in cat_count.most_common()},
                                 pair_types={k: v for k, v in pair_types.most_common(12)}, rows=rows)
            print("=== gen", S, json.dumps({k: v for k, v in gens[f"S{S}"].items() if k not in ("rows",)}), flush=True)
        # S8 vs S16 divergence on the same prompt
        if len(steps_list) >= 2:
            s_a, s_b = f"S{steps_list[0]}", f"S{steps_list[1]}"
            div = collections.Counter(); div_idx = []
            ra = {r["id"]: r for r in gens[s_a]["rows"]}; rb = {r["id"]: r for r in gens[s_b]["rows"]}
            for pid in ra:
                ta = tok(ra[pid]["text"])["input_ids"]; tb = tok(rb[pid]["text"])["input_ids"]
                k = next((q for q in range(min(len(ta), len(tb))) if ta[q] != tb[q]), min(len(ta), len(tb)))
                div_idx.append(k)
                if k < len(ta):
                    div[category(tok.decode([ta[k]]))] += 1
            gens["divergence"] = dict(first_divergence_token_index_mean=round(sum(div_idx) / max(1, len(div_idx)), 1), first_divergence_category={k: v for k, v in div.most_common()},
                                      agree_rate_first_32_tokens=round(sum(1 for k in div_idx if k >= 32) / max(1, len(div_idx)), 4))
            print("=== divergence", json.dumps(gens["divergence"]), flush=True)
        out["gen"] = gens
    out["seconds"] = round(time.time() - t0, 1)
    save_json(a.out, out)


if __name__ == "__main__":
    main()
