#!/usr/bin/env python3
"""Generate docs/phase_f.md: the SHIGERU (DeepSeek-distilled causal diffusion LM) results tables.

Reads every top-level results/*.json (sub-directories such as results/invalid_rope/ are ignored) and writes
  1. the Phase F table — one row per checkpoint x decoding, with the R0–R7 row ids of docs/distill_design.md §4,
  2. the diffusion-decoding table — every configuration of every bench/run_cdlm.py result file,
  3. an "other" section for result files that do not belong to a Phase F row (so nothing is silently dropped),
  4. a provenance list (file → where its numbers went) and the generation timestamp / git commit.
Idempotent: re-running only refreshes the timestamp. Robust to missing files (rows appear as "–").

File formats (see bench/run_ar.py, bench/run_code_eval.py, bench/run_cdlm.py):
  run_ar.py       : engine, model, config{prompts,max_new_tokens,...}, gpu{cuda_visible_devices}, runs[], accuracy{n,accuracy}|None,
                    summary{tok_s_effective{mean,...},...}
  run_code_eval.py: task, engine, model, n, pass@1
  run_cdlm.py     : model, engine (hf|fast|fast_static|fast_graph; missing on the oldest file), args{n,tau,window,lam,verify,
                    no_adjacent,fill_mode,margin_tau,...}, gpu, summary{<config>: {accuracy,tokens_per_step,tok_s_mean,
                    draft_acceptance_rate?,n}, tau_none_equals_ar?}

Usage: .venv/bin/python bench/summarize_distill.py [--results results] [--out docs/phase_f.md] [--quiet]
"""
import argparse, datetime, glob, json, os, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ----------------------------------------------------------------------------------------------------------------
# Row mapping (docs/distill_design.md §4). Key = checkpoint name derived from the result's model path (runs/<name>/...)
# or, for released models, from the model directory name. Order = position in the Phase F table.
# ----------------------------------------------------------------------------------------------------------------
ROWS = {
    #  key                        row id                model                       training data
    "qwen3_8b":                  ("R0",                 "Qwen3-8B",                 "released"),
    "qwen3_4b":                  ("R1",                 "Qwen3-4B",                 "released"),
    "phaseB_pilot_4b":           ("R2 (pilot)",         "Qwen3-4B + LoRA SFT",      "pilot 185"),
    "phaseB_medium_4b":          ("R2 (medium)",        "Qwen3-4B + LoRA SFT",      "medium 1714"),
    "phaseB_medium_4b_lr5e5_e1": ("R2 ablation 1",      "Qwen3-4B + LoRA SFT",      "medium 1714, lr 5e-5, 1 epoch"),
    "phaseB_medium_4b_nocode":   ("R2 ablation 2",      "Qwen3-4B + LoRA SFT",      "medium no-code (1431)"),
    "phaseB_pilot_8b":           ("R3 (pilot)",         "Qwen3-8B + LoRA SFT",      "pilot 185"),
    "phaseB_medium_8b":          ("R3 (medium)",        "Qwen3-8B + LoRA SFT",      "medium 1714"),
    "phaseC_pilot_4b":           ("R4 shakedown",       "Shigeru-4B-Diffusion (shakedown)", "pilot 185 (B) → C on 185 × 4 ep"),
    "phaseC_cpt_4b":             ("R4/R5 (CPT)",        "Shigeru-4B-Diffusion (CPT)",       "pilot 185 (B) → CPT 60 M tok"),
    "phaseC_sft_4b":             ("R4/R5 (stage 2)",    "Shigeru-4B-Diffusion (stage 2)",   "CPT → stage 2 medium 1714 × 3 ep"),
    "phaseC_cpt2_4b":            ("R4/R5 (CPT×2)",      "Shigeru-4B-Diffusion (CPT×2)",     "CPT 60 M → CPT 60 M tok"),
    "phaseC_sft2_4b":            ("R4/R5 (CPT×2 + stage 2)", "Shigeru-4B-Diffusion (CPT×2, stage 2)", "CPT×2 → stage 2 medium 1714"),
}
ROW_ORDER = {k: i for i, k in enumerate(ROWS)}
RELEASED = {"qwen3-4b": "qwen3_4b", "qwen3-8b": "qwen3_8b"}

# Speed runs that are known to be contended although their GPU id does not say so (file basename → reason).
CONTENDED_FILES = {"qwen3_4b_idle_256.json": "overlapped with another run on GPU 6"}
SHARED_GPUS = {"5"}            # GPU 5 was shared with other tenants all day: every speed number from it is contended.
SPEED_PROMPTS = "prompts.json"  # the 10 EN/JA prompts of the speed runs (bench/prompts.json, 256 new tokens)

ENGINE_LABEL = {"hf": "HF-cache decoder", "fast": "own engine",
                "fast_static": "own engine, static shapes", "fast_graph": "own engine, CUDA graphs"}


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:  # unreadable / partial file: report, never crash
        return {"__error__": str(e)}


def ckpt_key(model, fname):
    """Checkpoint identity from the model path; falls back to the file name."""
    if model:
        m = str(model).rstrip("/")
        parts = m.split("/")
        if "runs" in parts and parts.index("runs") + 1 < len(parts):
            return parts[parts.index("runs") + 1]
        base = os.path.basename(m).lower()
        if base in RELEASED:
            return RELEASED[base]
        return base
    stem = os.path.splitext(fname)[0]
    for p in ("gsm8k_full_", "gsm8k50_", "humaneval_", "cdlm_"):
        if stem.startswith(p):
            stem = stem[len(p):]
    return stem


def row_info(key):
    if key in ROWS:
        return ROWS[key]
    for k, v in ROWS.items():   # prefix match (qwen3_4b* etc.)
        if key.startswith(k):
            return v
    return None


def classify(d, fname):
    if "__error__" in d:
        return "error"
    if d.get("task") == "humaneval" and "pass@1" in d:
        return "humaneval"
    if "args" in d and isinstance(d.get("summary"), dict) and "runs" in d:
        return "cdlm"
    if isinstance(d.get("summary"), dict) and "runs" in d and "config" in d:
        return "ar"
    return "skip"


def pct(x, n=None):
    if x is None:
        return "–"
    s = f"{100 * x:.1f} %"
    return s if n is None else f"{s} ({n})"


def f1(x):
    return "–" if x is None else f"{x:.1f}"


def git_commit():
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()
        return h + (" (+ uncommitted changes)" if dirty else "")
    except Exception:
        return "unknown"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(out)


class Row:
    def __init__(self, key, decoding_key, decoding, order2):
        info = row_info(key)
        self.key, self.decoding_key, self.decoding, self.order2 = key, decoding_key, decoding, order2
        self.row_id, self.model, self.data = info if info else ("?", key, "?")
        if key.startswith("phaseC"):   # converted checkpoints: the decoding decides the row (ref = AR rows of the same weights)
            base, _, stage = self.row_id.partition(" ")
            self.row_id = ({0: "R4/R5 ref", 1: "R4", 2: "R5"}[order2] if base == "R4/R5" else ("R4 ref" if order2 == 0 else base)) + (" " + stage if stage else "")
        self.gsm50 = self.gsm50_n = self.gsm_full = self.humaneval = self.tok_s = None
        self.gsm50_n_best = -1
        self.tok_s_contended = None     # None = clean
        self.tok_s_file = None
        self.notes, self.alt_speed = [], []

    @property
    def order(self):
        return (ROW_ORDER.get(self.key, 10 ** 6), self.key, self.order2, self.decoding)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "phase_f.md"))
    ap.add_argument("--quiet", action="store_true", help="do not print the tables to stdout")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.results, "*.json")))
    ignored_dirs = {os.path.basename(d): len(glob.glob(os.path.join(d, "*.json")))
                    for d in sorted(glob.glob(os.path.join(a.results, "*"))) if os.path.isdir(d)}
    rows = {}                 # (ckpt key, decoding key) -> Row
    provenance = {}           # file -> [usage, ...]
    diffusion = []            # diffusion table rows (ckpt order, file, ...)
    other = []                # files that are results but not Phase F rows
    skipped = []

    def use(fname, what):
        provenance.setdefault(fname, []).append(what)

    def get_row(key, dkey, decoding, order2):
        if (key, dkey) not in rows:
            rows[(key, dkey)] = Row(key, dkey, decoding, order2)
        return rows[(key, dkey)]

    for path in files:
        fname = os.path.basename(path)
        d = load(path)
        kind = classify(d, fname)
        if kind == "error":
            skipped.append((fname, "unreadable: " + d["__error__"]))
            continue
        if kind == "skip":
            skipped.append((fname, "not a benchmark result"))
            continue
        key = ckpt_key(d.get("model"), fname)
        mapped = row_info(key) is not None
        gpu = str((d.get("gpu") or {}).get("cuda_visible_devices", "?"))

        if kind == "ar":
            cfg, s = d.get("config") or {}, d.get("summary") or {}
            tok_s = ((s.get("tok_s_effective") or {}).get("mean"))
            acc = d.get("accuracy") or {}
            prompts = os.path.basename(str(cfg.get("prompts", "")))
            if d.get("engine") != "hf-transformers-generate" or not mapped:
                other.append((fname, d.get("engine"), d.get("model"), acc.get("n"), pct(acc.get("accuracy")) if acc else "–", f1(tok_s), gpu))
                use(fname, "other (not a Phase F row)")
                continue
            r = get_row(key, "ar", "AR greedy (HF generate)", 0)
            contended = SHARED_GPUS & {gpu} and f"shared GPU {gpu}" or CONTENDED_FILES.get(fname)
            if prompts == "prompts_gsm8k_full.json" and acc:
                r.gsm_full = acc["accuracy"]
                use(fname, f"{r.row_id}: GSM8K-1319 ({pct(acc['accuracy'])}, n={acc['n']}, GPU {gpu})")
            elif prompts == "prompts_gsm8k50.json" and acc:
                prefer = (not contended, os.path.getmtime(path))
                if r.gsm50 is None or prefer > r._gsm50_pref:
                    if r.gsm50 is not None:
                        r.notes.append(f"GSM8K-50 also {pct(r.gsm50)} ({r._gsm50_file})")
                    r.gsm50, r.gsm50_n, r._gsm50_pref, r._gsm50_file = acc["accuracy"], acc["n"], prefer, fname
                else:
                    r.notes.append(f"GSM8K-50 also {pct(acc['accuracy'])} ({fname})")
                use(fname, f"{r.row_id}: GSM8K-50 ({pct(acc['accuracy'])}, n={acc['n']}, GPU {gpu}" + (", contended" if contended else "") + ")")
            elif prompts == SPEED_PROMPTS and tok_s is not None:
                mnt = cfg.get("max_new_tokens")
                prefer = (not contended, mnt == 256, os.path.getmtime(path))
                cand = (prefer, tok_s, fname, contended, mnt)
                if r.tok_s is None or prefer > r._tok_pref:
                    if r.tok_s is not None:
                        r.alt_speed.append((r.tok_s, r.tok_s_file, r.tok_s_contended))
                    r.tok_s, r.tok_s_file, r.tok_s_contended, r._tok_pref = tok_s, fname, contended, prefer
                    if mnt != 256:
                        r.notes.append(f"speed run with {mnt} new tokens")
                else:
                    r.alt_speed.append((tok_s, fname, contended))
                use(fname, f"{r.row_id}: tok/s ({f1(tok_s)}, GPU {gpu}" + (f", contended: {contended}" if contended else ", clean") + ")")
            else:
                other.append((fname, d.get("engine"), d.get("model"), acc.get("n"), pct(acc.get("accuracy")) if acc else "–", f1(tok_s), gpu))
                use(fname, f"other (prompts {prompts!r} not a Phase F benchmark)")

        elif kind == "humaneval":
            if not mapped:
                other.append((fname, d.get("engine"), d.get("model"), d.get("n"), pct(d.get("pass@1")), "–", gpu))
                use(fname, "other (not a Phase F row)")
                continue
            r = get_row(key, "ar", "AR greedy (HF generate)", 0)
            if key.startswith("phaseC"):
                r.decoding = "AR rows (HF greedy)"
                r.notes.append("HumanEval via the AR rows of the converted model (R5 code)")
            if r.humaneval is not None:
                r.notes.append(f"HumanEval also {pct(d['pass@1'])} ({fname})")
            else:
                r.humaneval = d["pass@1"]
            use(fname, f"{r.row_id}: HumanEval ({pct(d['pass@1'])}, n={d.get('n')})")

        elif kind == "cdlm":
            args, s = d.get("args") or {}, d.get("summary") or {}
            engine = d.get("engine") or "hf"   # the oldest file predates the --engine option (HF-cache decoder)
            W, lam = args.get("window"), args.get("lam")
            verify = bool(args.get("verify", False))
            fill = []
            if args.get("no_adjacent"):
                fill.append("no_adjacent")
            if args.get("fill_mode") not in (None, "all"):
                fill.append(f"fill_mode={args['fill_mode']}")
            if args.get("margin_tau") is not None:
                fill.append(f"margin≥{args['margin_tau']}")
            standard = not fill and (lam in (None, 0.02)) and engine in ("hf", "fast")   # static/graph variants: timing only
            for cfg_name, v in s.items():
                if not isinstance(v, dict):
                    continue
                n, acc, tps, tok_s, accept = v.get("n"), v.get("accuracy"), v.get("tokens_per_step"), v.get("tok_s_mean"), v.get("draft_acceptance_rate")
                is_ar = cfg_name == "ar"
                tau = None if is_ar else cfg_name.split("=", 1)[-1]
                diffusion.append((ROW_ORDER.get(key, 10 ** 6), key, fname, ENGINE_LABEL.get(engine, engine), "AR rows" if is_ar else tau,
                                  "–" if is_ar else W, "–" if is_ar else lam, "–" if is_ar else (", ".join(fill) or "all"),
                                  "–" if is_ar else ("yes" if verify else "no"), n, pct(acc), "–" if is_ar else f"{tps:.2f}" if tps is not None else "–",
                                  pct(accept) if accept is not None else "–", f1(tok_s), gpu))
                if not mapped:
                    continue
                # Phase F rows: AR rows, τ=None (R4) and verified parallel decoding with default fill options (R5).
                if is_ar:
                    dkey, decoding, order2 = "ar", "AR rows (HF greedy)", 0
                elif tau == "none" and standard:
                    dkey, decoding, order2 = f"cdlm:none:{engine}", f"diffusion τ=None (1 tok/step), {ENGINE_LABEL.get(engine, engine)}", 1
                elif verify and standard:
                    dkey, decoding, order2 = f"cdlm:verify:{engine}:{tau}:{W}", f"diffusion verified τ={tau} W={W}, {ENGINE_LABEL.get(engine, engine)}", 2
                else:
                    use(fname, f"diffusion table only ({cfg_name}: {pct(acc)} n={n})")
                    continue
                r = get_row(key, dkey, decoding, order2)
                if n == 1319:
                    r.gsm_full = acc
                    r.tok_s, r.tok_s_file, r.tok_s_contended = tok_s, fname, None
                    r.notes.append("tok/s from the 1319-item run")
                    col = "GSM8K-1319"
                elif n is not None and n <= 50 and n > r.gsm50_n_best:
                    r.gsm50, r.gsm50_n, r.gsm50_n_best = acc, n, n
                    if r.gsm_full is None:
                        r.tok_s, r.tok_s_file, r.tok_s_contended = tok_s, fname, None
                    col = f"GSM8K-50 column (n={n})"
                else:
                    r.notes.append(f"also {pct(acc, n)} ({fname})")
                    col = "note"
                if not is_ar and tps is not None:
                    r.notes.append(f"{tps:.2f} tok/step" + (f", acceptance {pct(accept)}" if accept is not None else "") + f" (n={n})")
                use(fname, f"{r.row_id} [{decoding}]: {col} ({pct(acc)}, {f1(tok_s)} tok/s, GPU {gpu})")

    # ------------------------------------------------------------------------------------------------ render
    phase_f = []
    for r in sorted(rows.values(), key=lambda r: r.order):
        tok = f1(r.tok_s)
        if r.tok_s is not None and r.tok_s_contended:
            tok += f" (contended: {r.tok_s_contended})"
        for v, f, c in r.alt_speed:
            r.notes.append(f"tok/s also {f1(v)} ({f}" + (f", contended: {c}" if c else ", clean") + ")")
        gsm50 = pct(r.gsm50) if r.gsm50_n in (None, 50) else pct(r.gsm50, r.gsm50_n)
        notes = sorted(set(r.notes), key=r.notes.index)
        phase_f.append([r.row_id, r.model, r.data, r.decoding, gsm50, pct(r.gsm_full), pct(r.humaneval), tok, "; ".join(notes)])

    diffusion.sort(key=lambda t: (t[0], t[1], t[2]))
    diff_rows = [[t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12], t[13], t[14]] for t in diffusion]

    now = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    commit = git_commit()
    H = ["Row", "Model", "Training data", "Decoding", "GSM8K-50", "GSM8K-1319", "HumanEval", "tok/s", "Notes"]
    DH = ["Checkpoint", "File", "Engine", "τ", "W", "λ", "Fill rule", "Verify", "n", "Accuracy", "tok/step", "Acceptance", "tok/s", "GPU"]
    OH = ["File", "Engine", "Model", "n", "Accuracy / pass@1", "tok/s", "GPU"]
    doc = []
    doc.append("# Phase F results (SHIGERU: DeepSeek-distilled causal diffusion LM)\n")
    doc.append(f"Generated by `bench/summarize_distill.py` on {now} at commit {commit}. Do not edit by hand: re-run the script.\n")
    doc.append("Row ids follow `docs/distill_design.md` §4 (R0 released 8B, R1 released 4B, R2/R3 Phase B SFT students, "
               "R4 diffusion conversion decoded one token per step, R5 parallel (verified) diffusion decoding; \"ref\" = the AR rows of the "
               "converted checkpoint decoded with HF greedy, i.e. the quality the diffusion decoding is compared against). Accuracy columns: "
               "GSM8K greedy, thinking off, 384 new tokens (`bench/run_ar.py` on `bench/prompts_gsm8k50.json` / "
               "`bench/prompts_gsm8k_full.json`; the converted models through `bench/run_cdlm.py`, whose short runs use the first n "
               "prompts of the 50-item subset, shown as \"acc (n)\"); HumanEval pass@1 (`bench/run_code_eval.py`, 164 problems). "
               "tok/s: AR rows from the 10-prompt 256-token speed runs (`bench/prompts.json`) on an idle A100, batch 1, bf16 — "
               "contended runs (shared GPU 5, or overlapped) are marked; the converted models' tok/s come from the `run_cdlm.py` "
               "run that supplied the accuracy (idle GPUs).\n")
    doc.append("## 1. Phase F table\n")
    doc.append(md_table(H, phase_f) + "\n")
    doc.append("## 2. Diffusion decoding (all `bench/run_cdlm.py` results)\n")
    doc.append("τ = confidence threshold for filling masks ahead of the frontier (\"none\" = frontier mask only); W = window; "
               "λ = distance penalty; Verify = AR-exact speculative acceptance of the drafted tokens; Acceptance = draft acceptance rate. "
               "\"AR rows\" = HF greedy decoding of the same weights.\n")
    doc.append(md_table(DH, diff_rows) if diff_rows else "_no cdlm_*.json files_")
    doc.append("")
    doc.append("## 3. Other result files (not Phase F rows)\n")
    doc.append(md_table(OH, other) if other else "_none_")
    doc.append("")
    doc.append("## 4. Provenance (file → use)\n")
    for f in sorted(provenance):
        doc.append(f"* `{f}` → " + "; ".join(provenance[f]))
    for f, why in skipped:
        doc.append(f"* `{f}` → skipped ({why})")
    for dname, cnt in ignored_dirs.items():
        doc.append(f"* `{dname}/` → ignored directory ({cnt} JSON files" + (", wrong-RoPE runs quarantined per docs/distill_design.md §10" if dname == "invalid_rope" else "") + ")")
    doc.append("")
    doc.append(f"_Generated {now}, commit {commit}._\n")
    text = "\n".join(doc)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(text)
    if not a.quiet:
        print(md_table(H, phase_f))
        print()
        print(md_table(DH, diff_rows) if diff_rows else "(no cdlm results)")
        print(f"\nwrote {a.out} ({len(phase_f)} Phase F rows, {len(diff_rows)} diffusion rows, {len(other)} other, {len(files)} files scanned)")


if __name__ == "__main__":
    sys.exit(main())
