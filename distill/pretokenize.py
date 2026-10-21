"""Pre-tokenize text corpora for the conversion trainer: parquet/JSONL documents -> one uint32 token stream with an EOS
separator after each document, per-file token caps, plus <out>.json metadata. The trainer memory-maps the stream
(config `data_tokens`) and chunks it into max_len windows, so DDP ranks do not re-tokenize at start-up.
Usage: .venv-train/bin/python -m distill.pretokenize --tokenizer /mnt/raid6/models/hf/Qwen3-4B --out data/cpt/fineweb_1b \
         --cap 250000000 /mnt/raid6/data/fineweb-edu/000_00000.parquet /mnt/raid6/data/fineweb2-ja/000_00000.parquet ..."""
import argparse, json, os, time

import numpy as np


def docs(path, text_key="text", batch=256):
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            col = pf.read_row_group(rg, columns=[text_key]).column(text_key).to_pylist()
            for i in range(0, len(col), batch):
                yield [str(x) for x in col[i:i + batch]]
    else:
        buf = []
        for l in open(path):
            if l.strip():
                buf.append(json.loads(l)[text_key])
                if len(buf) >= batch:
                    yield buf; buf = []
        if buf:
            yield buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True, help="prefix: writes <out>.bin (uint32) and <out>.json")
    ap.add_argument("--cap", type=int, default=0, help="max tokens per file (0 = all)")
    ap.add_argument("--min-doc", type=int, default=32)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    eos = tok.eos_token_id
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    f = open(a.out + ".bin", "wb")
    meta = {"tokenizer": a.tokenizer, "eos": eos, "files": [], "total": 0}
    t0 = time.time()
    for path in a.paths:
        n_tok = n_doc = 0
        for batch in docs(path):
            ids = tok(batch, add_special_tokens=False)["input_ids"]
            arr = np.concatenate([np.asarray(x + [eos], dtype=np.uint32) for x in ids if len(x) >= a.min_doc]) if any(len(x) >= a.min_doc for x in ids) else np.zeros(0, np.uint32)
            if a.cap and n_tok + len(arr) > a.cap:
                arr = arr[: a.cap - n_tok]
            f.write(arr.tobytes()); n_tok += len(arr); n_doc += sum(len(x) >= a.min_doc for x in ids)
            if a.cap and n_tok >= a.cap:
                break
        meta["files"].append({"path": path, "tokens": n_tok, "docs": n_doc}); meta["total"] += n_tok
        print(f"{path}: {n_tok/1e6:.1f}M tokens, {n_doc} docs, {time.time()-t0:.0f}s", flush=True)
    f.close()
    json.dump(meta, open(a.out + ".json", "w"), indent=1)
    print(f"total {meta['total']/1e6:.1f}M tokens -> {a.out}.bin", flush=True)


if __name__ == "__main__":
    main()
