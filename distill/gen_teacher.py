"""Phase A: generate verified teacher data from DeepSeek V4.1 Flash (OpenAI-compatible, read-only use).

Output: JSONL, one record per item:
  id, task_category, source, instruction, context, response, teacher_model, generation_parameters,
  verification_status, verification_detail, quality_score, usage, latency_s, created_at, git_commit
Resumable: ids already present in the output file are skipped. Rate-limited via Teacher(concurrency, min_interval).
Usage: .venv/bin/python -m distill.gen_teacher distill/configs/phaseA_small.yaml [--only CAT] [--limit N]
"""
import argparse, datetime, hashlib, json, os, random, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from distill.teacher import Teacher
from distill.verify import verify_math, verify_code, verify_json, extract_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CODE_TOPICS = ["string processing", "list/dict manipulation", "sorting and searching", "simple math/number theory",
               "date and time arithmetic (no external libs)", "parsing a small text format", "matrix/grid operations",
               "recursion and dynamic programming", "stack/queue simulation", "simple geometry", "bit manipulation",
               "set operations and counting", "string formatting and validation", "simple graph traversal (BFS/DFS)",
               "intervals and ranges", "simple state machines"]
JSON_TOPICS = ["a product listing", "a weather report", "a restaurant menu item", "a job posting", "a book summary",
               "a travel itinerary for one day", "a sports match result", "a support ticket", "a recipe",
               "a software bug report", "a calendar event", "a scientific paper abstract summary", "a city profile"]
EN_TOPICS = ["everyday science explanations", "history", "personal finance basics", "cooking", "health and fitness (general)",
             "technology explained simply", "writing and editing", "travel planning", "study tips", "career advice",
             "environment and nature", "music and art", "productivity", "language learning", "simple business questions"]
JA_TOPICS = ["日常の科学", "日本の歴史", "料理とレシピ", "ビジネスメールの書き方", "旅行の計画", "健康と運動", "テクノロジーの解説",
             "日本語の文法や敬語", "読書と文学", "勉強法", "仕事のコミュニケーション", "家庭の節約術", "季節の行事", "趣味の始め方",
             "プログラミング入門の質問"]
REASON_TOPICS = ["logic puzzle about ordering/seating", "truth-tellers and liars", "scheduling constraints",
                 "counting and probability (simple)", "unit conversion and rates", "a short deductive puzzle",
                 "comparing quantities with constraints", "calendar/weekday arithmetic", "set/Venn reasoning",
                 "simple algebra word problem", "pattern completion with justification", "estimation with explicit assumptions"]

JUDGE_RUBRIC = """You are grading an answer to an instruction. Score 1-5:
5 = correct, complete, well organized, appropriate language and length; 4 = minor issues; 3 = partially correct or
incomplete; 2 = mostly wrong or unhelpful; 1 = wrong, harmful, or off-topic. Reply with JSON only: {"score": <int>, "reason": "<short>"}"""


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def item_id(cat, key):
    return cat + "-" + hashlib.sha1(key.encode()).hexdigest()[:12]


def load_gsm8k_train(n, rng):
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(ROOT, "data/sources/gsm8k_train.parquet")).to_pylist()
    rng.shuffle(t)
    return t[:n]


class Gen:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        t = cfg["teacher"]
        self.teacher = Teacher(t["base_url"], t["model"], t.get("concurrency", 2), t.get("min_interval_s", 0.5), t.get("timeout_s", 900))
        self.gen = cfg["generation"]
        self.rng = random.Random(cfg["seed"])
        self.out = os.path.join(ROOT, cfg["out"])
        os.makedirs(os.path.dirname(self.out), exist_ok=True)
        self.done = set()
        if os.path.exists(self.out):
            for line in open(self.out):
                try:
                    self.done.add(json.loads(line)["id"])
                except Exception:
                    pass
        self.lock = threading.Lock()
        self.commit = git_commit()
        self.teacher_model = t["model"]

    def ask(self, instruction, system=None, seed=None, max_tokens=None, response_format=None, thinking=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": instruction}]
        g = self.gen
        return self.teacher.chat(msgs, max_tokens=max_tokens or g["max_tokens"], temperature=g["temperature"], top_p=g["top_p"],
                                 seed=seed, thinking=g["thinking"] if thinking is None else thinking, response_format=response_format)

    def judge(self, instruction, response, seed):
        r = self.ask(f"INSTRUCTION:\n{instruction}\n\nANSWER:\n{response}", system=JUDGE_RUBRIC, seed=seed, max_tokens=200,
                     thinking=False)
        obj = extract_json(r["content"]) or {}
        try:
            return int(obj.get("score")), obj.get("reason")
        except (TypeError, ValueError):
            return None, r["content"][:200]

    def record(self, rec):
        rec.update({"teacher_model": f"{self.teacher_model} (served: DeepSeek-V4.1-Flash-1017-abliterated, tsuginosuke:8000)",
                    "generation_parameters": dict(self.gen, seed=rec.pop("seed", None)),
                    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "git_commit": self.commit})
        with self.lock:
            with open(self.out, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.done.add(rec["id"])
        return rec

    # ---- category workers (each returns a record dict or None) ----
    def do_math(self, row, seed):
        iid = item_id("math_gsm8k", row["question"])
        if iid in self.done:
            return None
        gold = row["answer"].split("####")[-1].strip()
        instr = row["question"] + "\n\nSolve the problem step by step, then give the final numeric answer on the last line as '#### <number>'."
        r = self.ask(instr, seed=seed)
        st, det = verify_math(r["content"], gold)
        return self.record({"id": iid, "task_category": "math", "source": "openai/gsm8k:train", "instruction": instr, "context": "",
                            "response": r["content"], "verification_status": st, "verification_detail": det,
                            "quality_score": 5 if st == "verified_correct" else 1, "usage": r["usage"], "latency_s": r["latency_s"],
                            "finish_reason": r["finish_reason"], "seed": seed, "gold": gold})

    def do_code(self, k, seed):
        topic = CODE_TOPICS[k % len(CODE_TOPICS)]
        diff = ["easy", "medium", "medium", "hard"][(k // len(CODE_TOPICS)) % 4]
        iid = item_id("code_synth", f"{topic}|{diff}|{seed}")
        if iid in self.done:
            return None
        spec = self.ask(f"Invent one self-contained Python programming problem about {topic} (difficulty: {diff}). Variation seed: {seed}.\n"
                        "The solution must be a single function using only the standard library.\n"
                        "Reply with JSON only: {\"problem\": \"<full problem statement including the exact function name and signature>\", "
                        "\"entry_point\": \"<function name>\", \"reference_solution\": \"<complete Python code of a correct solution>\", "
                        "\"tests\": [\"assert <call> == <expected>\", ... 4-6 asserts]}",
                        seed=seed, max_tokens=2000, thinking=False)
        obj = extract_json(spec["content"])
        if obj and isinstance(obj.get("tests"), list) and obj.get("reference_solution"):
            # keep only the tests that the spec writer's own reference solution passes (guards against wrong expected values)
            obj["tests"] = [t for t in obj["tests"] if verify_code("```python\n" + obj["reference_solution"] + "\n```", [t])[0] == "verified_pass"]
        if not obj or not isinstance(obj.get("tests"), list) or len(obj["tests"]) < 3 or not obj.get("problem"):
            return self.record({"id": iid, "task_category": "code", "source": "teacher_synth", "instruction": "", "context": "",
                                "response": "", "verification_status": "spec_invalid", "verification_detail": spec["content"][:300],
                                "quality_score": 0, "usage": spec["usage"], "latency_s": spec["latency_s"], "seed": seed})
        instr = obj["problem"] + "\n\nWrite the Python function. Return the complete code in a single ```python code block."
        r = self.ask(instr, seed=seed + 1)
        st, det = verify_code(r["content"], obj["tests"])
        return self.record({"id": iid, "task_category": "code", "source": "teacher_synth", "instruction": instr, "context": "",
                            "response": r["content"], "verification_status": st, "verification_detail": det[:500],
                            "quality_score": 5 if st == "verified_pass" else 1, "usage": r["usage"], "latency_s": r["latency_s"],
                            "finish_reason": r["finish_reason"], "seed": seed, "tests": obj["tests"], "entry_point": obj.get("entry_point")})

    def do_json(self, k, seed):
        topic = JSON_TOPICS[k % len(JSON_TOPICS)]
        iid = item_id("json_synth", f"{topic}|{seed}")
        if iid in self.done:
            return None
        rng = random.Random(seed)
        keys = ["name", "summary", "tags", "rating", "details"]
        extra = rng.sample(["price", "date", "location", "author", "duration_minutes", "category", "url", "language"], 3)
        keys = keys + extra
        lang = "Japanese" if rng.random() < 0.4 else "English"
        instr = (f"Create a realistic, specific example of {topic} as a JSON object with exactly these keys: {', '.join(keys)}. "
                 f"'tags' must be a list of strings, 'rating' a number from 1 to 5, 'details' a nested object. Text values in {lang}. "
                 "Keep it compact (under 40 lines). Output only the JSON, no prose.")
        r = self.ask(instr, seed=seed, max_tokens=1500)
        st, det = verify_json(r["content"], keys)
        return self.record({"id": iid, "task_category": "json", "source": "teacher_synth", "instruction": instr, "context": "",
                            "response": r["content"], "verification_status": st, "verification_detail": det,
                            "quality_score": 5 if st == "verified_pass" else 1, "usage": r["usage"], "latency_s": r["latency_s"],
                            "finish_reason": r["finish_reason"], "seed": seed})

    def do_selfinstruct(self, cat, k, seed):
        topics, lang = {"en_general": (EN_TOPICS, "English"), "ja_general": (JA_TOPICS, "Japanese"),
                        "reasoning": (REASON_TOPICS, "English")}[cat]
        topic = topics[k % len(topics)]
        if cat == "reasoning" and random.Random(seed).random() < 0.3:
            lang = "Japanese"
        iid = item_id(cat, f"{topic}|{seed}")
        if iid in self.done:
            return None
        if cat == "reasoning":
            ask_q = (f"Write one original, self-contained {topic} question in {lang} with a single unambiguous correct answer. "
                     f"Variation seed {seed}. Reply with JSON only: {{\"question\": \"...\", \"answer\": \"<short final answer>\"}}")
        else:
            ask_q = (f"Write one realistic user request in {lang} on the topic '{topic}' that a helpful assistant could answer in "
                     f"under 400 words. Make it specific and different from typical examples (variation seed {seed}). "
                     "Reply with JSON only: {\"question\": \"...\"}")
        q = self.ask(ask_q, seed=seed, max_tokens=400, thinking=False)
        obj = extract_json(q["content"]) or {}
        question = obj.get("question")
        if not question:
            return self.record({"id": iid, "task_category": cat, "source": "teacher_selfinstruct", "instruction": "", "context": "",
                                "response": "", "verification_status": "spec_invalid", "verification_detail": q["content"][:300],
                                "quality_score": 0, "usage": q["usage"], "latency_s": q["latency_s"], "seed": seed})
        instr = question + ("\n\nThink step by step, then state the final answer clearly on the last line." if cat == "reasoning" else "")
        r = self.ask(instr, seed=seed + 1)
        rec = {"id": iid, "task_category": cat, "source": "teacher_selfinstruct", "instruction": instr, "context": "",
               "response": r["content"], "usage": r["usage"], "latency_s": r["latency_s"], "finish_reason": r["finish_reason"], "seed": seed}
        if cat == "reasoning" and obj.get("answer"):
            # self-consistency check: does the answer match the question-writer's answer? (judged, not a proof)
            rec["gold"] = str(obj["answer"])
            score, reason = self.judge(instr + f"\n\n(Reference answer from the question author: {obj['answer']})", r["content"], seed + 2)
        else:
            score, reason = self.judge(instr, r["content"], seed + 2)
        rec.update({"verification_status": "judged", "verification_detail": reason, "quality_score": score})
        return self.record(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--only")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    g = Gen(cfg, a)
    print("teacher models:", g.teacher.probe(), flush=True)
    base = cfg["seed"]
    jobs = []
    for cat, c in cfg["categories"].items():
        if a.only and cat != a.only:
            continue
        n = min(c["n"], a.limit) if a.limit else c["n"]
        if cat == "math_gsm8k":
            for i, row in enumerate(load_gsm8k_train(n, random.Random(base))):
                jobs.append((g.do_math, (row, base + i)))
        elif cat == "code_synth":
            jobs += [(g.do_code, (k, base + 1000 + 10 * k)) for k in range(n)]
        elif cat == "json_synth":
            jobs += [(g.do_json, (k, base + 2000 + 10 * k)) for k in range(n)]
        else:
            jobs += [(g.do_selfinstruct, (cat, k, base + 3000 + 10 * k + 1000 * list(cfg["categories"]).index(cat))) for k in range(n)]
    print(f"{len(jobs)} jobs, {len(g.done)} already done -> {g.out}", flush=True)
    t0 = time.time()
    n_ok = 0
    with ThreadPoolExecutor(max_workers=cfg["teacher"].get("concurrency", 2)) as ex:
        futs = [ex.submit(fn, *args) for fn, args in jobs]
        for i, f in enumerate(as_completed(futs)):
            try:
                rec = f.result()
            except Exception as e:
                print(f"[{i}] ERROR {type(e).__name__}: {e}", flush=True)
                continue
            if rec:
                n_ok += 1
                print(f"[{i}] {rec['task_category']:<10} {rec['verification_status']:<16} q={rec.get('quality_score')} "
                      f"{rec['latency_s']}s {rec.get('usage', {}).get('completion_tokens', 0)} tok", flush=True)
    st = g.teacher.stats()
    print(f"done: {n_ok} new records in {time.time() - t0:.0f}s; teacher calls {st}", flush=True)


if __name__ == "__main__":
    main()
