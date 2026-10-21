"""Convert public SFT datasets (ShareGPT / messages / SWE-smith trajectories / UltraChat) into the project's JSONL format
(id, task_category, instruction, context, response, source). Multi-turn conversations become ONE example per assistant turn
whose instruction is the preceding conversation rendered as plain text turns ("User: ...\\nAssistant: ...") — the trainers
apply the chat template to a single user turn, so earlier turns live inside the instruction. Tool/function calls are kept
verbatim in the assistant text. Usage:
  .venv/bin/python -m distill.convert_sft --kind sharegpt --src /mnt/raid6/data/sft/glaive-function-calling-v2-sharegpt-japanese --out data/distill/sft_glaive_ja.jsonl --max 20000 --category tool_ja
  .venv/bin/python -m distill.convert_sft --kind messages --src /mnt/raid6/data/sft/ultrachat_200k --out data/distill/sft_ultrachat.jsonl --max 20000 --category chat_en
  .venv/bin/python -m distill.convert_sft --kind swesmith --src /mnt/raid6/data/sft/SWE-smith-trajectories --out data/distill/sft_swesmith.jsonl --max 5000 --category swe"""
import argparse, glob, hashlib, json, os, random, re

ROLE = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system", "tool": "tool",
        "function": "tool", "function_call": "assistant", "observation": "tool"}


def rows(src):
    import pyarrow.parquet as pq
    for p in sorted(glob.glob(os.path.join(src, "*.parquet"))):
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            for r in pf.read_row_group(rg).to_pylist():
                yield r


def to_messages(kind, r):
    if kind == "sharegpt":
        conv = r.get("conversations") or r.get("conversation") or []
        return [{"role": ROLE.get(str(m.get("from", m.get("role", ""))).lower(), "user"), "content": normalize(str(m.get("value", m.get("content", ""))))} for m in conv]
    if kind == "messages":
        msgs = r.get("messages") or []
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        return [{"role": ROLE.get(str(m.get("role", "")).lower(), "user"), "content": str(m.get("content", ""))} for m in msgs]
    if kind == "swesmith":
        if str(r.get("resolved", "True")).lower() == "false":      # keep only trajectories that solved the task
            return []
        msgs = r.get("messages") or r.get("trajectory") or r.get("conversations") or []
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        out = []
        for m in msgs:
            role = ROLE.get(str(m.get("role", m.get("from", ""))).lower(), "user")
            c = m.get("content", m.get("value", ""))
            if isinstance(c, list):
                c = "\n".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in c)
            if m.get("tool_calls"):
                c = (str(c) + "\n" + json.dumps(m["tool_calls"], ensure_ascii=False)).strip()
            out.append({"role": role, "content": str(c)})
        return out
    raise ValueError(kind)


FC = re.compile(r"<functioncall>\s*(\{.*\})\s*$", re.S)


def normalize(text):
    """Glaive-style artefacts -> Qwen/Hermes tool-call format: strip <|endoftext|>, turn
    <functioncall> {"name": X, "arguments": '{...}'} into <tool_call>{"name": X, "arguments": {...}}</tool_call>."""
    text = text.replace("<|endoftext|>", "").strip()
    m = FC.search(text)
    if m:
        body = m.group(1)
        obj = None
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            mn = re.search(r'"name"\s*:\s*"([^"]+)"', body)
            ma = re.search(r'"arguments"\s*:\s*\'(.*)\'\s*\}\s*$', body, re.S)
            if mn:
                args = ma.group(1) if ma else "{}"
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
                obj = {"name": mn.group(1), "arguments": args}
        if obj is not None:
            if isinstance(obj.get("arguments"), str):
                try:
                    obj["arguments"] = json.loads(obj["arguments"])
                except json.JSONDecodeError:
                    pass
            head = text[: m.start()].rstrip()
            text = (head + "\n" if head else "") + "<tool_call>\n" + json.dumps(obj, ensure_ascii=False) + "\n</tool_call>"
    return text.replace("FUNCTION RESPONSE:", "<tool_response>").strip()


def render(prefix):
    parts = []
    for m in prefix:
        tag = {"user": "User", "assistant": "Assistant", "system": "System", "tool": "Tool"}[m["role"]]
        parts.append(f"{tag}: {m['content']}")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["sharegpt", "messages", "swesmith"], required=True)
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=20000); ap.add_argument("--category", required=True)
    ap.add_argument("--max-chars", type=int, default=6000, help="skip examples whose instruction+response exceed this")
    ap.add_argument("--last-turn-only", action="store_true", help="one example per conversation (final assistant turn)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    cands = []
    for r in rows(a.src):
        msgs = to_messages(a.kind, r)
        if not msgs:
            continue
        idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i > 0]
        if a.last_turn_only and idx:
            idx = idx[-1:]
        for i in idx:
            instr = render(msgs[:i]); resp = msgs[i]["content"].strip()
            if not resp or len(instr) + len(resp) > a.max_chars:
                continue
            # the final user turn is the instruction proper; earlier turns are context
            if msgs[i - 1]["role"] == "user" and i - 1 == 0:
                instruction, context = msgs[0]["content"], ""
            else:
                instruction, context = msgs[i - 1]["content"] if msgs[i - 1]["role"] == "user" else "(continue)", render(msgs[: i - 1]) if i > 1 else ""
            cands.append({"id": a.category + "-" + hashlib.sha1((instr + resp).encode()).hexdigest()[:12], "task_category": a.category,
                          "instruction": instruction, "context": context, "response": resp, "source": os.path.basename(a.src.rstrip("/"))})
        if len(cands) > a.max * 20:
            break
    rng.shuffle(cands)
    seen, out = set(), []
    for c in cands:
        if c["id"] in seen:
            continue
        seen.add(c["id"]); out.append(c)
        if len(out) >= a.max:
            break
    with open(a.out, "w") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{a.out}: {len(out)} examples from {len(cands)} candidates")


if __name__ == "__main__":
    main()
