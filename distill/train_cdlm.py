"""Phase C: convert a Qwen3 causal LM into a WeDLM-style causal diffusion LM (full fine-tune, dual-stream objective of
engine/cdlm; docs/distill_design.md section 7 steps 1-4).  Runs in .venv-train (torch 2.6, transformers 5.x).

Usage (single GPU):
  CUDA_VISIBLE_DEVICES=5 .venv-train/bin/python -m distill.train_cdlm distill/configs/phaseC_pilot_4b.yaml
DDP (one process per GPU, torch.distributed, NCCL; gloo on CPU):
  CUDA_VISIBLE_DEVICES=4,5,6,7 .venv-train/bin/torchrun --nproc_per_node=4 -m distill.train_cdlm <config.yaml>
Flags: --max-steps N (stop after N optimizer steps; also sets the schedule length), --exit-after-step N (debug: hard-exit
right after the checkpoint at step N is written, to exercise the resume path).

Config keys (distill/configs/phaseC_pilot_4b.yaml documents each): model, data_train, out_dir, mask_token, block, max_len,
batch, grad_accum, lr, epochs, warmup_ratio | warmup_steps, min_lr_ratio, weight_decay, betas, grad_clip, ar_weight,
mlm_normalize, gamma {min, max}, ar_on_prompt, mask_mode, seed, save_steps, save_total_limit, logging_steps, attn,
grad_ckpt, master_fp32, num_workers.

What one example becomes.  prompt = chat template of instruction (+ "\n\n" + context) with add_generation_prompt and
enable_thinking=False (as in train_sft.py), response = response + eos.  tokens = prompt_ids + response_ids, truncated to
max_len LOGICAL tokens (the prompt is kept whole; an example whose prompt alone fills max_len is dropped and counted).
engine.cdlm.data.build_example turns the L tokens into the 2L physical rows [x0 clean | xt block-masked] (prompt tokens
never masked, gamma ~ U(gamma.min, gamma.max] per block of `block` tokens, fresh draw every epoch), so the model forward
sees sequences of up to 2*max_len rows: attention cost is 4x and activation memory 2x that of AR SFT at the same max_len.
mask_mode (default "random") selects WHICH round(n*gamma) positions of a block are masked: "random" = a uniform subset
(WeDLM), "suffix" = the last ones (a contiguous masked suffix with no observed right context, the layout the streaming
decoder poses), "mixed" = 50/50 per example (engine/cdlm/data.py docstring, engine/cdlm/README.md "mask_mode").

Loss.  mlm = 1/gamma self-normalised CE on the [MASK] rows, ar = next-token CE on the x0 rows (same forward),
total = (mlm + ar_weight * ar) / (1 + ar_weight) as in WeDLM's trainer (engine/cdlm/loss.py: total_loss); both parts are
logged.  Each optimizer step covers batch * grad_accum examples per rank; the step loss is the mean of the micro-batch
losses (accelerate semantics, as the reference).

Precision.  Default: the model is bf16 and AdamW keeps its states in bf16 (no autocast needed; ~32 GiB of weights +
grads + states for a 4B model).  Caveat: bf16 weights have ~3 significant digits, so updates of relative size below
~1e-3 (lr 1e-5 on O(1e-2) weights) are largely rounded away; `master_fp32: true` keeps fp32 weights and optimizer states
and runs the forward under bf16 autocast (the standard mixed-precision recipe, ~64 GiB of states for 4B — see
docs/distill_design.md section 7).

OOM.  A torch.cuda.OutOfMemoryError inside a micro-batch is caught: the partial gradients of that optimizer step are
dropped (zero_grad), the step is skipped (its examples are not seen; the step counter and the LR schedule still advance so
that resume positions stay consistent), the event is logged to log.jsonl and run.json, and the micro-batch size is halved
for the rest of the run (down to 1; a loader batch of `batch` examples is forwarded in chunks of `micro_bs`, so the
effective batch and the optimizer semantics do not change).  Under DDP an OOM is fatal (re-raised after logging): the
other ranks would hang in the gradient all-reduce.

Checkpoints.  out_dir/checkpoint-<step>/state.pt = model (unwrapped) + optimizer + scheduler + step + micro_bs + RNG
states + OOM log, every `save_steps` steps and at the end, keeping the last `save_total_limit`; the latest one under
out_dir is resumed automatically (same config, same data order: the sampler is seeded by seed + epoch and the
micro-batches already consumed in the resume epoch are skipped).  out_dir/log.jsonl: one record per logged step (step,
epoch, loss, loss_mlm, loss_ar, lr, grad_norm, tokens/s, rows/s, n_masked, micro_bs, peak VRAM) plus resume/oom events.
out_dir/run.json at the end (config, git commit, GPU config, peak VRAM, timings, log history, mask token id).
out_dir/final: save_pretrained (bf16 weights, config.cdlm_mask_token_id) + the tokenizer with the mask token registered.
"""
import argparse, datetime, glob, json, math, os, random, shutil, subprocess, sys, time

import yaml

from distill.gpu_guard import check_env, check_runtime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


# ----------------------------------------------------------------------------------------------- distributed helpers
class Dist:
    """World/rank bookkeeping; initialised only when launched by torchrun (WORLD_SIZE > 1)."""

    def __init__(self, torch):
        self.torch = torch
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.enabled = self.world > 1
        if self.enabled:
            import torch.distributed as dist
            self.dist = dist
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, timeout=datetime.timedelta(minutes=30))
        self.device = torch.device(f"cuda:{self.local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)

    @property
    def main(self):
        return self.rank == 0

    def all_reduce_sum(self, t):
        if self.enabled:
            self.dist.all_reduce(t, op=self.dist.ReduceOp.SUM)
        return t

    def barrier(self):
        if self.enabled:
            self.dist.barrier()

    def finish(self):
        if self.enabled:
            self.dist.destroy_process_group()


# ---------------------------------------------------------------------------------------------------------- data
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_text_corpus(paths, tok, max_len, max_tokens=None, text_key="text", seed=0, min_doc_tokens=32):
    """Raw-text (continued-pretraining) mode for the conversion objective, as WeDLM's CPT stage: parquet/JSONL documents are
    tokenized, each document gets a leading BOS-like separator (eos token, Qwen has no BOS), documents are concatenated and
    chunked into fixed max_len windows (prompt_len = 0: every token is maskable, AR loss everywhere).  max_tokens caps the
    corpus; the budget is split equally over the files (so EN and JA shards both contribute); chunks are shuffled with seed."""
    import torch, random
    rng = random.Random(seed)
    out = []
    per_file = (max_tokens // len(paths)) if max_tokens else None
    for path in paths:
        buf, total = [], 0
        if path.endswith(".parquet"):
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            docs_iter = (str(d) for rg in range(pf.num_row_groups) for d in pf.read_row_group(rg, columns=[text_key]).column(text_key).to_pylist())
        else:
            docs_iter = (json.loads(l)[text_key] for l in open(path) if l.strip())
        for doc in docs_iter:
            ids = tok(doc, add_special_tokens=False)["input_ids"]
            if len(ids) < min_doc_tokens:
                continue
            buf += ids + [tok.eos_token_id]
            while len(buf) >= max_len:
                out.append((torch.tensor(buf[:max_len], dtype=torch.long), 0))
                buf = buf[max_len:]
                total += max_len
            if per_file and total >= per_file:
                break
    rng.shuffle(out)
    return out, 0, 0


def tokenize_examples(rows, tok, max_len):
    """-> list of (tokens LongTensor [L], prompt_len), n_dropped, n_truncated.  Same prompt/completion split as train_sft."""
    import torch
    out, dropped, truncated = [], 0, 0
    for ex in rows:
        prompt = tok.apply_chat_template([{"role": "user", "content": ((ex["context"] + "\n\n") if ex.get("context") else "") + ex["instruction"]}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tok(ex["response"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        if len(p_ids) >= max_len:                       # nothing of the response would survive: no training signal
            dropped += 1
            continue
        ids = (p_ids + r_ids)[:max_len]
        truncated += int(len(p_ids) + len(r_ids) > max_len)
        out.append((torch.tensor(ids, dtype=torch.long), len(p_ids)))
    return out, dropped, truncated


class TokenStreamItems:
    """Lazy list of (tokens[max_len], prompt_len=0) windows over a memory-mapped uint32 token stream (distill/pretokenize.py).
    Window order is a seeded permutation so DDP ranks and resumed runs see the same order."""

    def __init__(self, path, max_len, seed, max_tokens=None):
        import numpy as np, torch
        self.arr = np.memmap(path, dtype=np.uint32, mode="r")
        n = len(self.arr) // max_len
        if max_tokens:
            n = min(n, int(max_tokens) // max_len)
        g = torch.Generator().manual_seed(int(seed))
        self.order = torch.randperm(n, generator=g).tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.order)

    def __getitem__(self, i):
        import numpy as np, torch
        j = self.order[i]
        return torch.from_numpy(np.array(self.arr[j * self.max_len:(j + 1) * self.max_len], dtype=np.int64)), 0


class CDLMDataset:
    """Tokenized examples -> dual-stream examples; the gamma/mask draw is deterministic in (seed, epoch, index)."""

    def __init__(self, items, mask_id, block, gamma_min, gamma_max, seed, ar_on_prompt=False, mask_mode="random"):
        import torch
        from engine.cdlm import MASK_MODES, build_example
        if mask_mode not in MASK_MODES:
            raise ValueError(f"mask_mode={mask_mode!r}, expected one of {MASK_MODES}")
        self.items, self.mask_id, self.block, self.seed = items, mask_id, block, seed
        self.ar_on_prompt, self.mask_mode, self.epoch = ar_on_prompt, mask_mode, 0
        lo, hi = float(gamma_min), float(gamma_max)
        assert 0.0 <= lo < hi <= 1.0, (lo, hi)

        def gamma_sampler(nblk, rng):           # U(lo, hi], clamped away from 0 like data.uniform_gamma
            return (lo + (hi - lo) * torch.rand(nblk, generator=rng)).clamp_min(1e-8)
        self.gamma_sampler, self.build_example = gamma_sampler, build_example

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch
        tokens, prompt_len = self.items[i]
        rng = torch.Generator().manual_seed((self.seed * 1_000_003 + self.epoch) * 1_000_003 + i)
        return self.build_example(tokens, prompt_len, self.mask_id, block=self.block, gamma_sampler=self.gamma_sampler,
                                  rng=rng, ar_on_prompt=self.ar_on_prompt, mask_mode=self.mask_mode)


def make_loader(ds, batch, epoch, seed, dist, num_workers):
    """A DataLoader over lists of examples (collation happens per micro-batch in the step), epoch-seeded order."""
    import torch
    from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
    ds.set_epoch(epoch)
    if dist.enabled:
        sampler = DistributedSampler(ds, num_replicas=dist.world, rank=dist.rank, shuffle=True, seed=seed, drop_last=False)
        sampler.set_epoch(epoch)
    else:
        sampler = RandomSampler(ds, generator=torch.Generator().manual_seed(seed + epoch))
    return DataLoader(ds, batch_size=batch, sampler=sampler, collate_fn=lambda xs: xs, num_workers=num_workers, drop_last=False)


# --------------------------------------------------------------------------------------------------------- model
def build_model(cfg, mask_id, dist):
    import torch
    from transformers import AutoModelForCausalLM
    from engine.cdlm import add_mask_token
    dtype = torch.float32 if cfg.get("master_fp32", False) else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype, attn_implementation=cfg.get("attn", "sdpa"))
    info = add_mask_token(model, mask_id)                 # new row (resize if beyond the matrix), mean init, config.cdlm_mask_token_id
    model.config.use_cache = False
    if cfg.get("grad_ckpt", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train().to(dist.device)
    return model, info


class DualStreamForward:
    """Callable forward on a collate() batch returning (mlm_logits, mlm_labels, mlm_weights, ar_logits, ar_labels).

    Wrapped in an nn.Module so that DDP's forward hook sees every forward (forward_rows calls model.model/lm_head
    directly, which DDP would not intercept)."""

    def __init__(self, model, dist):
        import torch
        from engine.cdlm import IGNORE, forward_rows

        class Inner(torch.nn.Module):
            def __init__(self, lm):
                super().__init__()
                self.lm = lm

            def forward(self, input_ids, position_ids, attention_mask, rows):
                return forward_rows(self.lm, input_ids, position_ids, attention_mask, rows)
        self.module = Inner(model)
        self.IGNORE = IGNORE
        if dist.enabled:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.module = DDP(self.module, device_ids=[dist.local_rank] if dist.device.type == "cuda" else None)
        self.dist = dist

    def no_sync(self):
        import contextlib
        return self.module.no_sync() if self.dist.enabled else contextlib.nullcontext()

    def __call__(self, batch, device):
        import torch
        mlm_rows = (batch["mlm_labels"] != self.IGNORE).nonzero(as_tuple=True)
        ar_rows = (batch["ar_labels"] != self.IGNORE).nonzero(as_tuple=True)
        rows = (torch.cat([mlm_rows[0], ar_rows[0]]).to(device), torch.cat([mlm_rows[1], ar_rows[1]]).to(device))
        logits = self.module(batch["input_ids"].to(device), batch["position_ids"].to(device),
                             batch["attention_mask"].to(device), rows)
        n = mlm_rows[0].numel()
        return (logits[:n], batch["mlm_labels"][mlm_rows].to(device), batch["mlm_weights"][mlm_rows].to(device),
                logits[n:], batch["ar_labels"][ar_rows].to(device))



# -------------------------------------------------------------------------------------------------- optim/sched
def cosine_with_warmup(step, warmup, total, min_ratio):
    """LR multiplier for optimizer step `step` (0-based): linear warmup (step+1)/warmup — the first step is not wasted at
    LR 0 — then cosine from 1 to min_ratio over the remaining steps."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if total <= warmup:
        return 1.0
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))


def build_optim(model, cfg, total_steps, warmup_steps, device):
    import torch
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "norm" in n else decay).append(p)
    wd = float(cfg.get("weight_decay", 0.0))
    betas = tuple(float(b) for b in cfg.get("betas", (0.9, 0.95)))
    groups = [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=float(cfg["lr"]), betas=betas, eps=float(cfg.get("adam_eps", 1e-8)),
                            fused=(device.type == "cuda"))
    min_ratio = float(cfg.get("min_lr_ratio", 0.1))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_with_warmup(s, warmup_steps, total_steps, min_ratio))
    return opt, sched


# ----------------------------------------------------------------------------------------------- checkpointing
def rng_state(torch):
    st = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(torch, st):
    random.setstate(st["python"])
    torch.set_rng_state(st["torch"])
    if torch.cuda.is_available() and "cuda" in st:
        try:
            torch.cuda.set_rng_state_all(st["cuda"])
        except Exception as e:                                    # different GPU count than at save time
            print(f"[resume] cuda rng not restored: {e}", flush=True)


def latest_checkpoint(out_dir):
    cks = sorted(glob.glob(os.path.join(out_dir, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    cks = [c for c in cks if os.path.exists(os.path.join(c, "state.pt"))]
    return cks[-1] if cks else None


def save_checkpoint(out_dir, step, model, opt, sched, state, limit, torch):
    d = os.path.join(out_dir, f"checkpoint-{step}")
    tmp = d + ".tmp"
    # disk guard: a full-model checkpoint (weights + Adam states) is ~6x the bf16 weights; the root filesystem filled up on
    # 2026-10-20 and killed a run mid-save. Skip the checkpoint (keep training) when free space is below MIN_FREE_GIB.
    need_gib = 6 * sum(p.numel() for p in model.parameters()) * 2 / 2**30
    free_gib = shutil.disk_usage(out_dir).free / 2**30
    if free_gib < need_gib + float(os.environ.get("CDLM_MIN_FREE_GIB", "20")):
        print(f"[train_cdlm] WARNING: {free_gib:.0f} GiB free < {need_gib:.0f} GiB needed: checkpoint-{step} skipped", flush=True)
        return
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "step": step,
                "rng": rng_state(torch), **state}, os.path.join(tmp, "state.pt"))
    shutil.rmtree(d, ignore_errors=True)
    os.rename(tmp, d)                                             # atomic: a half-written checkpoint is never resumed
    cks = sorted(glob.glob(os.path.join(out_dir, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    for old in cks[:-limit] if limit and limit > 0 else []:
        shutil.rmtree(old, ignore_errors=True)
    return d


# ---------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after N optimizer steps (also the schedule length)")
    ap.add_argument("--exit-after-step", type=int, default=-1, help="debug: os._exit right after the checkpoint at step N")
    a = ap.parse_args()
    visible = check_env()
    cfg = yaml.safe_load(open(a.config))
    import torch
    from transformers import AutoTokenizer, set_seed
    from engine.cdlm import MASK_MODES, ar_loss, collate, mlm_loss, total_loss
    if cfg.get("mask_mode", "random") not in MASK_MODES:        # fail before the model is loaded
        raise SystemExit(f"config mask_mode={cfg.get('mask_mode')!r}, expected one of {MASK_MODES}")

    dist = Dist(torch)
    seed = int(cfg.get("seed", 0))
    set_seed(seed + dist.rank)
    out_dir = os.path.join(ROOT, cfg["out_dir"]) if not os.path.isabs(cfg["out_dir"]) else cfg["out_dir"]
    if dist.main:
        os.makedirs(out_dir, exist_ok=True)
    dist.barrier()
    log_path = os.path.join(out_dir, "log.jsonl")

    def log(rec):
        if dist.main:
            with open(log_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(json.dumps(rec, ensure_ascii=False), flush=True)

    # --- tokenizer + mask token (registered in the tokenizer, saved with final/)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mask_token = cfg.get("mask_token", "<|mask|>")
    if tok.convert_tokens_to_ids(mask_token) in (None, tok.unk_token_id):
        tok.add_special_tokens({"additional_special_tokens": [mask_token]})      # Qwen3: hands out 151669 (free padded row)
    tok.mask_token = mask_token
    mask_id = int(tok.convert_tokens_to_ids(mask_token))
    assert mask_id == tok.mask_token_id and mask_id != tok.unk_token_id and mask_id != tok.pad_token_id, mask_id

    # --- data
    max_len, block = int(cfg["max_len"]), int(cfg.get("block", 32))
    if cfg.get("data_tokens"):   # pre-tokenized uint32 stream (distill/pretokenize.py); fastest start-up, exact resume
        data_path = cfg["data_tokens"] if os.path.isabs(cfg["data_tokens"]) else os.path.join(ROOT, cfg["data_tokens"])
        items, n_dropped, n_truncated = TokenStreamItems(data_path, max_len, seed, cfg.get("text_max_tokens")), 0, 0
    elif cfg.get("data_text"):     # raw-text CPT mode (list of parquet/JSONL paths); data_train is ignored
        paths = [p if os.path.isabs(p) else os.path.join(ROOT, p) for p in cfg["data_text"]]
        items, n_dropped, n_truncated = tokenize_text_corpus(paths, tok, max_len, cfg.get("text_max_tokens"), cfg.get("text_key", "text"), seed)
        data_path = ",".join(paths)
    else:
        data_path = cfg["data_train"] if os.path.isabs(cfg["data_train"]) else os.path.join(ROOT, cfg["data_train"])
        items, n_dropped, n_truncated = tokenize_examples(load_jsonl(data_path), tok, max_len)
    assert items, f"no trainable examples in {data_path}"
    gamma = cfg.get("gamma") or {}
    ds = CDLMDataset(items, mask_id, block, gamma.get("min", 0.0), gamma.get("max", 1.0), seed, cfg.get("ar_on_prompt", False),
                     mask_mode=cfg.get("mask_mode", "random"))
    batch, grad_accum, epochs = int(cfg["batch"]), int(cfg["grad_accum"]), float(cfg["epochs"])
    n_local = math.ceil(len(ds) / dist.world) if dist.enabled else len(ds)
    micro_per_epoch = math.ceil(n_local / batch)
    steps_per_epoch = math.ceil(micro_per_epoch / grad_accum)
    total_steps = a.max_steps if a.max_steps > 0 else int(steps_per_epoch * epochs)
    warmup_steps = int(cfg["warmup_steps"]) if "warmup_steps" in cfg else int(float(cfg.get("warmup_ratio", 0.03)) * total_steps)

    # --- model, optimizer, schedule
    model, mask_info = build_model(cfg, mask_id, dist)
    fwd = DualStreamForward(model, dist)
    opt, sched = build_optim(model, cfg, total_steps, warmup_steps, dist.device)
    use_autocast = bool(cfg.get("master_fp32", False))
    ar_weight, mlm_norm = float(cfg.get("ar_weight", 1.0)), cfg.get("mlm_normalize", "wedlm")
    grad_clip = float(cfg.get("grad_clip", 1.0))
    save_steps, save_limit, logging_steps = int(cfg.get("save_steps", 100)), int(cfg.get("save_total_limit", 2)), int(cfg.get("logging_steps", 1))

    # --- resume
    step, micro_bs, oom_events, resumed_from = 0, batch, [], None
    ck = latest_checkpoint(out_dir)
    if ck:
        st = torch.load(os.path.join(ck, "state.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        step, micro_bs, oom_events = int(st["step"]), int(st.get("micro_bs", batch)), list(st.get("oom_events", []))
        if dist.main:
            set_rng_state(torch, st["rng"])
        resumed_from = ck
        del st
        log({"event": "resume", "from": ck, "step": step, "micro_bs": micro_bs, "time": datetime.datetime.now().isoformat(timespec="seconds")})
    used = check_runtime()
    if dist.main:
        print(f"[gpu_guard] CUDA_VISIBLE_DEVICES={visible} physical GPUs in use by this pid: {used}", flush=True)
        print(f"[train_cdlm] n_train={len(ds)} (dropped {n_dropped}, truncated {n_truncated}) world={dist.world} batch={batch} "
              f"grad_accum={grad_accum} steps/epoch={steps_per_epoch} total_steps={total_steps} warmup={warmup_steps} "
              f"mask_id={mask_id} {mask_info} dtype={'fp32 master + bf16 autocast' if use_autocast else 'bf16'}", flush=True)

    # --- train
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    state = {"step": step, "micro_bs": micro_bs, "done": step >= total_steps}
    skip_micro = (step % steps_per_epoch) * grad_accum       # micro-batches already consumed in the resume epoch

    def optimizer_step(pending, epoch):
        """One optimizer step over the pending loader batches; logging, OOM policy, checkpoint, stop condition."""
        res = run_step(pending, fwd, opt, sched, dist, state["micro_bs"], use_autocast, ar_weight, mlm_norm, grad_clip,
                       collate, mlm_loss, ar_loss, total_loss, tok.pad_token_id, torch)
        state["step"] += 1
        step = state["step"]
        if res.get("oom"):
            if dist.enabled:
                raise res["oom"]
            state["micro_bs"] = max(1, state["micro_bs"] // 2)
            ev = {"event": "oom", "step": step, "micro_bs_after": state["micro_bs"], "msg": str(res["oom"])[:200],
                  "time": datetime.datetime.now().isoformat(timespec="seconds")}
            oom_events.append(ev)
            log(ev)
        elif step % logging_steps == 0 or step == total_steps:
            log({"step": step, "epoch": round(epoch + min(1.0, (step - epoch * steps_per_epoch) / steps_per_epoch), 3), **res,
                 "micro_bs": state["micro_bs"],
                 "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0})
        if step % save_steps == 0 or step >= total_steps:
            if dist.main:
                save_checkpoint(out_dir, step, model, opt, sched,
                                {"micro_bs": state["micro_bs"], "oom_events": oom_events, "epoch": epoch}, save_limit, torch)
                snap_every = int(cfg.get("hf_snapshot_every", 0))
                if snap_every and (step % snap_every == 0 or step >= total_steps):
                    # HF-format snapshot (bf16 weights + tokenizer, config repaired for transformers 4.57) so the checkpoint can be
                    # evaluated while training continues; keeps the last `hf_snapshot_keep` (default 2).
                    _lm = model.module if hasattr(model, "module") else model
                    snap = os.path.join(out_dir, f"snapshot-{step}")
                    _lm.save_pretrained(snap, safe_serialization=True); tok.save_pretrained(snap)
                    from distill.fix_config import fix as _fix_cfg; _fix_cfg(snap)
                    snaps = sorted(glob.glob(os.path.join(out_dir, "snapshot-*")), key=lambda d: int(d.rsplit("-", 1)[1]))
                    for old in snaps[:-int(cfg.get("hf_snapshot_keep", 2))]:
                        shutil.rmtree(old, ignore_errors=True)
                    print(f"[train_cdlm] snapshot-{step} written", flush=True)
            dist.barrier()
            if a.exit_after_step > 0 and step >= a.exit_after_step:
                print(f"[train_cdlm] --exit-after-step {a.exit_after_step}: hard exit after checkpoint-{step}", flush=True)
                os._exit(3)
        state["done"] = step >= total_steps

    epoch = step // steps_per_epoch
    while not state["done"] and epoch < math.ceil(epochs):
        it = iter(make_loader(ds, batch, epoch, seed, dist, int(cfg.get("num_workers", 0))))
        for _ in range(skip_micro):
            next(it, None)
        skip_micro = 0
        pending = []                                             # loader batches of the current optimizer step
        for lb in it:
            pending.append(lb)
            if len(pending) == grad_accum:
                optimizer_step(pending, epoch)
                pending = []
                if state["done"]:
                    break
        if pending and not state["done"]:                        # epoch tail shorter than grad_accum micro-batches
            optimizer_step(pending, epoch)
        epoch += 1
    step, micro_bs = state["step"], state["micro_bs"]
    train_s = time.time() - t_start

    # --- final save (rank 0) + run.json
    if dist.main:
        final = os.path.join(out_dir, "final")
        lm = model
        lm.config.cdlm_mask_token_id = mask_id
        lm.config.cdlm_block = block
        if use_autocast:
            lm = lm.to(torch.bfloat16)
        lm.save_pretrained(final, safe_serialization=True)
        tok.save_pretrained(final)
        from distill.fix_config import fix as _fix_cfg; _fix_cfg(final)   # transformers-5 config -> also correct under 4.57
        info = {"config": cfg, "config_path": a.config, "git_commit": git_commit(), "cuda_visible_devices": visible,
                "physical_gpus_used": used, "world_size": dist.world,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "n_train": len(ds), "n_dropped": n_dropped, "n_truncated": n_truncated, "steps": step, "total_steps": total_steps,
                "steps_per_epoch": steps_per_epoch, "warmup_steps": warmup_steps, "resumed_from": resumed_from,
                "mask_token": mask_token, "mask_token_id": mask_id, "mask_info": mask_info, "micro_bs_final": micro_bs,
                "oom_events": oom_events, "train_seconds": round(train_s, 1),
                "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0,
                "log_history": [json.loads(l) for l in open(log_path) if l.strip()], "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "total_params": sum(p.numel() for p in model.parameters()), "final_dir": final}
        json.dump(info, open(os.path.join(out_dir, "run.json"), "w"), indent=1, ensure_ascii=False)
        print(json.dumps({k: v for k, v in info.items() if k != "log_history"}, indent=1, ensure_ascii=False, default=str))
    dist.barrier()
    dist.finish()


def run_step(loader_batches, fwd, opt, sched, dist, micro_bs, use_autocast, ar_weight, mlm_norm, grad_clip,
             collate, mlm_loss_fn, ar_loss_fn, total_loss_fn, pad_id, torch):
    """One optimizer step over the examples of `loader_batches`, forwarded in chunks of micro_bs."""
    import contextlib
    examples = [e for lb in loader_batches for e in lb]
    chunks = [examples[i:i + micro_bs] for i in range(0, len(examples), micro_bs)]
    device = dist.device
    t0 = time.time()
    sums = torch.zeros(5, dtype=torch.float64)                   # mlm, ar, tokens, rows, n_masked
    try:
        for ci, chunk in enumerate(chunks):
            batch = collate(chunk, pad_id)
            sync = fwd.no_sync() if ci < len(chunks) - 1 else contextlib.nullcontext()
            ac = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if use_autocast else contextlib.nullcontext()
            with sync, ac:
                mlm_logits, mlm_labels, mlm_w, ar_logits, ar_labels = fwd(batch, device)
                mlm = mlm_loss_fn(mlm_logits, mlm_labels, mlm_w, normalize=mlm_norm)
                ar = ar_loss_fn(ar_logits, ar_labels)
                loss = total_loss_fn(mlm, ar, ar_weight) / len(chunks)
            loss.backward()
            sums += torch.tensor([float(mlm.detach()), float(ar.detach()), int(batch["seq_lens"].sum()),
                                  batch["input_ids"].numel(), int(mlm_labels.numel())], dtype=torch.float64)
            del mlm_logits, ar_logits, mlm, ar, loss, batch
    except torch.cuda.OutOfMemoryError as e:
        opt.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sched.step()
        return {"oom": e}
    params = [p for g in opt.param_groups for p in g["params"]]
    gnorm = torch.nn.utils.clip_grad_norm_(params, grad_clip) if grad_clip and grad_clip > 0 else torch.tensor(0.0)
    lr_used = sched.get_last_lr()[0]
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    dt = time.time() - t0
    stats = torch.cat([sums[:2] / len(chunks), sums[2:]])
    stats = dist.all_reduce_sum(stats.to(device) if dist.enabled else stats).cpu()
    lm, la = float(stats[0]) / dist.world, float(stats[1]) / dist.world
    return {"loss": float(total_loss_fn(lm, la, ar_weight)), "loss_mlm": round(lm, 5), "loss_ar": round(la, 5), "lr": lr_used,
            "grad_norm": round(float(gnorm), 4), "tokens_s": round(float(stats[2]) / dt, 1), "rows_s": round(float(stats[3]) / dt, 1),
            "n_masked": int(stats[4]), "n_examples": len(examples) * dist.world, "step_s": round(dt, 3)}


if __name__ == "__main__":
    main()
