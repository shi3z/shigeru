"""CPU smoke test for distill/train_cdlm.py (plain asserts, tiny random Qwen3 + char-level tokenizer, no downloads).

Run from the repo root:  CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -m distill.tests.test_train_cdlm
Each trainer run is a real subprocess (python -m distill.train_cdlm / torchrun), so resume and DDP are exercised the way
they happen in production.
"""
import json
import os
import shutil
import socket
import string
import subprocess
import sys
import tempfile
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": ROOT, "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"}
MASK = "<|mask|>"
TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n"
            "{% if enable_thinking is defined and not enable_thinking %}<think>\n\n</think>\n\n{% endif %}{% endif %}")


def make_tokenizer():
    """Character-level fast tokenizer with Qwen-like special tokens and a chat template that honours enable_thinking."""
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab = {c: i for i, c in enumerate(["[UNK]", "<|endoftext|>", "<|im_start|>", "<|im_end|>"] + list(string.printable))}
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), "isolated")
    return PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]", eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                   additional_special_tokens=["<|im_start|>"], chat_template=TEMPLATE)


def make_model_dir(d):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    tok = make_tokenizer()
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=512, tie_word_embeddings=False)
    Qwen3ForCausalLM(cfg).to(torch.bfloat16).save_pretrained(d, safe_serialization=True)
    tok.save_pretrained(d)
    return len(tok)                                           # the mask token will get this id (resize path)


def make_data(path):
    rows = [{"instruction": f"What is {i} + {i + 1}?", "context": "" if i % 2 else f"Hint number {i}.",
             "response": f"{i} + {i + 1} = {2 * i + 1}. The answer is {2 * i + 1}."} for i in range(6)]
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def write_cfg(path, model_dir, data, out_dir, **over):
    cfg = dict(model=model_dir, data_train=data, out_dir=out_dir, mask_token=MASK, block=8, max_len=160, batch=2, grad_accum=1,
               lr=1e-3, epochs=2, warmup_steps=1, save_steps=2, save_total_limit=5, logging_steps=1, seed=0, attn="sdpa",
               grad_ckpt=True, master_fp32=False, ar_weight=1.0)
    cfg.update(over)
    yaml.safe_dump(cfg, open(path, "w"))
    return path


def run(args, expect=0):
    p = subprocess.run([PY, "-m", "distill.train_cdlm", *args], cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert p.returncode == expect, f"rc={p.returncode} (expected {expect})\nSTDOUT:\n{p.stdout[-3000:]}\nSTDERR:\n{p.stderr[-3000:]}"
    return p


def read_log(out_dir):
    return [json.loads(l) for l in open(os.path.join(out_dir, "log.jsonl")) if l.strip()]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


TMP = tempfile.mkdtemp(prefix="cdlm_train_")
MODEL_DIR = os.path.join(TMP, "model")
DATA = os.path.join(TMP, "train.jsonl")
N_VOCAB = make_model_dir(MODEL_DIR)
make_data(DATA)


def test_train_resume_and_final():
    """4 steps, checkpoint every 2, hard exit after checkpoint-2, restart: the step counter continues at 3."""
    out = os.path.join(TMP, "run1")
    cfg = write_cfg(os.path.join(TMP, "cfg1.yaml"), MODEL_DIR, DATA, out)
    p = run([cfg, "--max-steps", "4", "--exit-after-step", "2"], expect=3)
    assert "hard exit after checkpoint-2" in p.stdout, p.stdout[-500:]
    assert os.path.exists(os.path.join(out, "checkpoint-2", "state.pt")) and not os.path.exists(os.path.join(out, "final"))
    log1 = read_log(out)
    assert [r.get("step") for r in log1] == [1, 2], log1
    for r in log1:
        for k in ("loss", "loss_mlm", "loss_ar", "lr", "tokens_s", "rows_s", "peak_vram_gib", "grad_norm", "micro_bs", "n_masked"):
            assert k in r, (k, r)
        assert all(map(lambda v: v == v, (r["loss"], r["loss_mlm"], r["loss_ar"]))) and r["n_masked"] > 0 and r["tokens_s"] > 0
    assert log1[0]["lr"] == 1e-3 and log1[1]["lr"] == 1e-3                      # LR used by the step: warmup 1 step, peak at the boundary

    p = run([cfg, "--max-steps", "4"])
    log2 = read_log(out)
    assert log2[:2] == log1, "log.jsonl must be appended to, not rewritten"
    ev = log2[2]
    assert ev.get("event") == "resume" and ev["step"] == 2 and ev["from"].endswith("checkpoint-2"), ev
    assert [r.get("step") for r in log2[3:]] == [3, 4], log2[3:]
    assert 1e-4 < log2[4]["lr"] < log2[3]["lr"] < 1e-3                          # cosine decay towards min_lr_ratio 0.1 (reached after the last step)
    assert os.path.exists(os.path.join(out, "checkpoint-4", "state.pt"))
    st = torch.load(os.path.join(out, "checkpoint-4", "state.pt"), map_location="cpu", weights_only=False)
    assert st["step"] == 4 and st["scheduler"]["last_epoch"] == 4 and "rng" in st and "optimizer" in st
    info = json.load(open(os.path.join(out, "run.json")))
    assert info["steps"] == 4 and info["total_steps"] == 4 and info["resumed_from"].endswith("checkpoint-2"), info["steps"]
    assert info["n_train"] == 6 and info["n_dropped"] == 0 and info["world_size"] == 1 and info["oom_events"] == []
    assert info["mask_token_id"] == N_VOCAB and info["mask_info"]["added"] == [N_VOCAB]
    assert len(info["log_history"]) == 5 and info["log_history"][2]["event"] == "resume"   # whole log.jsonl: 1,2,resume,3,4
    assert info["git_commit"] and "peak_vram_gib" in info and info["cuda_visible_devices"] == []

    # final/ loads as a plain HF causal LM with the mask token in config and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(os.path.join(out, "final"), dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(os.path.join(out, "final"))
    assert m.config.cdlm_mask_token_id == N_VOCAB and m.config.cdlm_block == 8 and m.config.vocab_size == N_VOCAB + 1
    assert tok.convert_tokens_to_ids(MASK) == N_VOCAB and tok.mask_token == MASK and tok.mask_token_id == N_VOCAB
    assert len(tok) == N_VOCAB + 1 and m.get_input_embeddings().weight.shape[0] == N_VOCAB + 1
    assert m.dtype == torch.bfloat16
    with torch.no_grad():
        out_logits = m(input_ids=torch.tensor([[1, 2, N_VOCAB, 3]])).logits
    assert out_logits.shape == (1, 4, N_VOCAB + 1) and torch.isfinite(out_logits.float()).all()
    # the trained weights differ from the base (something was learned) and the checkpoint weights match final/
    base = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    assert not torch.equal(base.model.layers[0].mlp.up_proj.weight, m.model.layers[0].mlp.up_proj.weight)
    assert torch.equal(st["model"]["model.layers.0.mlp.up_proj.weight"].to(torch.bfloat16), m.model.layers[0].mlp.up_proj.weight)


def test_schedule_covers_epochs_and_tail():
    """epochs=2 with 6 examples, batch 2, grad_accum 2: 3 micro/epoch -> 2 steps/epoch (one tail step), 4 steps total."""
    out = os.path.join(TMP, "run2")
    cfg = write_cfg(os.path.join(TMP, "cfg2.yaml"), MODEL_DIR, DATA, out, grad_accum=2, save_steps=100, master_fp32=True)
    run([cfg])
    info = json.load(open(os.path.join(out, "run.json")))
    assert info["steps_per_epoch"] == 2 and info["total_steps"] == 4 and info["steps"] == 4
    log = read_log(out)
    assert [r["n_examples"] for r in log] == [4, 2, 4, 2], [r["n_examples"] for r in log]
    assert [r["epoch"] for r in log] == [0.5, 1.0, 1.5, 2.0], [r["epoch"] for r in log]
    assert os.path.exists(os.path.join(out, "checkpoint-4")) and not os.path.exists(os.path.join(out, "checkpoint-2"))
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(os.path.join(out, "final"))
    assert m.dtype == torch.bfloat16, "master_fp32 run must still save bf16 weights"


def test_oom_step_is_skipped_and_grads_dropped():
    """run_step's OOM branch in-process: a forward raising torch.cuda.OutOfMemoryError yields {'oom'} with no gradients
    left behind and the schedule advanced (the halving itself is 1 line in the loop; CUDA OOM cannot be raised on CPU)."""
    from distill import train_cdlm as T
    from engine.cdlm import ar_loss, collate, mlm_loss, total_loss
    os.environ.pop("WORLD_SIZE", None)
    dist = T.Dist(torch)
    model = torch.nn.Linear(4, 4)
    opt, sched = T.build_optim(model, {"lr": 1e-3}, 10, 1, dist.device)

    class Boom:
        def no_sync(self):
            import contextlib
            return contextlib.nullcontext()

        def __call__(self, batch, device):
            model.weight.grad = torch.ones_like(model.weight)       # partial gradient that must be dropped
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
    ex = {"input_ids": torch.zeros(4, dtype=torch.long), "position_ids": torch.zeros(4, dtype=torch.long),
          "mlm_labels": torch.zeros(4, dtype=torch.long), "mlm_weights": torch.zeros(4), "ar_labels": torch.zeros(4, dtype=torch.long),
          "seq_len": 2, "block": 8, "is_xt": torch.zeros(4, dtype=torch.bool)}
    res = T.run_step([[ex, ex]], Boom(), opt, sched, dist, 1, False, 1.0, "wedlm", 1.0, collate, mlm_loss, ar_loss, total_loss, 0, torch)
    assert isinstance(res.get("oom"), torch.cuda.OutOfMemoryError) and model.weight.grad is None
    assert sched.last_epoch == 1


def test_ddp_cpu_gloo():
    """torchrun --nproc_per_node=2 on CPU (gloo): 2 steps, n_examples summed over ranks, run.json world_size 2."""
    out = os.path.join(TMP, "run_ddp")
    cfg = write_cfg(os.path.join(TMP, "cfg_ddp.yaml"), MODEL_DIR, DATA, out, save_steps=100)
    cmd = [PY, "-m", "torch.distributed.run", "--nproc_per_node=2", "--master_addr=127.0.0.1", f"--master_port={free_port()}",
           "-m", "distill.train_cdlm", cfg, "--max-steps", "2"]
    p = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert p.returncode == 0, f"rc={p.returncode}\nSTDOUT:\n{p.stdout[-3000:]}\nSTDERR:\n{p.stderr[-4000:]}"
    info = json.load(open(os.path.join(out, "run.json")))
    assert info["world_size"] == 2 and info["steps"] == 2 and info["steps_per_epoch"] == 2    # 3 local examples -> 2 micro/epoch
    log = read_log(out)
    assert [r["step"] for r in log] == [1, 2] and [r["n_examples"] for r in log] == [4, 2], log   # 3 local -> micro 2 + 1 per rank
    assert all(r["loss_mlm"] == r["loss_mlm"] and r["loss"] > 0 for r in log)
    assert os.path.exists(os.path.join(out, "final", "config.json")) and os.path.exists(os.path.join(out, "checkpoint-2", "state.pt"))
    assert len(os.listdir(os.path.join(out))) >= 4                                         # log, run.json, final, checkpoint


def test_mask_mode_plumbing():
    """mask_mode reaches build_example: the dataset builds suffix / random / mixed examples (same gamma draw per (seed,
    epoch, index) in every mode), an unknown mode is rejected before the model loads, and a 1-step run with
    mask_mode: suffix trains on masked rows."""
    from distill.train_cdlm import CDLMDataset
    items = [(torch.arange(10 + 7 * i), 3 + i) for i in range(6)]            # (tokens, prompt_len), several blocks each
    mask_id, Bk = 1000, 8
    dss = {m: CDLMDataset(items, mask_id, Bk, 0.0, 1.0, seed=5, mask_mode=m) for m in ("random", "suffix", "mixed")}
    seen = set()
    for i in range(len(items)):
        ex = {m: ds[i] for m, ds in dss.items()}
        assert ex["random"]["mask_mode"] == "random" and ex["suffix"]["mask_mode"] == "suffix"
        assert ex["mixed"]["mask_mode"] in ("random", "suffix"); seen.add(ex["mixed"]["mask_mode"])
        assert torch.equal(ex["random"]["gamma"], ex["suffix"]["gamma"])        # the gamma draw does not depend on the mode
        L, pl = ex["suffix"]["seq_len"], ex["suffix"]["prompt_len"]
        ids, pos = ex["suffix"]["input_ids"][L:], ex["suffix"]["position_ids"][L:]
        assert torch.equal(pos, torch.arange(L))                                # suffix layout == original order
        for blk in range((L + Bk - 1) // Bk):
            s_, t_ = blk * Bk, min(L, (blk + 1) * Bk)
            is_m = ids[s_:t_] == mask_id
            n = int(round(max(0, t_ - max(s_, pl)) * float(ex["suffix"]["gamma"][blk])))
            assert torch.equal(is_m, torch.arange(s_, t_) >= t_ - n) and (pos[s_:t_][is_m] >= pl).all()
    assert seen == {"random", "suffix"}, seen
    try:
        CDLMDataset(items, mask_id, Bk, 0.0, 1.0, seed=0, mask_mode="prefix"); assert False
    except ValueError as err:
        assert "prefix" in str(err)
    out = os.path.join(TMP, "run_mm")
    cfg = write_cfg(os.path.join(TMP, "cfg_mm_bad.yaml"), MODEL_DIR, DATA, out, mask_mode="prefix")
    p = run([cfg, "--max-steps", "1"], expect=1)
    assert "mask_mode='prefix'" in p.stderr and not os.path.exists(out), p.stderr[-500:]
    cfg = write_cfg(os.path.join(TMP, "cfg_mm.yaml"), MODEL_DIR, DATA, out, mask_mode="suffix")
    run([cfg, "--max-steps", "1"])
    log = read_log(out)
    assert log and log[0]["n_masked"] > 0 and log[0]["loss_mlm"] == log[0]["loss_mlm"]
    assert json.load(open(os.path.join(out, "run.json")))["config"]["mask_mode"] == "suffix"


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = failed = 0
    try:
        for t in tests:
            try:
                t(); passed += 1; print(f"PASS {t.__name__}")
            except Exception:
                failed += 1; print(f"FAIL {t.__name__}"); traceback.print_exc()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
