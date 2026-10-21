"""Phase B: autoregressive SFT distillation of a Qwen3 student on verified teacher data (LoRA by default).
Runs in .venv-train (transformers 5.x, peft, trl). Usage:
  CUDA_VISIBLE_DEVICES=5 .venv-train/bin/python -m distill.train_sft distill/configs/phaseB_pilot_4b.yaml
Config keys: model, data_train, data_val, out_dir, lora{r,alpha,dropout,targets}|null, max_len, lr, epochs, batch, grad_accum,
seed, save_steps, eval_steps, warmup_ratio, merge (bool: write merged bf16 weights to out_dir/merged).
Writes out_dir/run.json (config, git commit, GPU config, timings, peak VRAM, final losses) and the Trainer's JSON log history.
Resume: a checkpoint-* under out_dir is resumed automatically."""
import argparse, datetime, glob, json, os, subprocess, sys, time

import yaml

from distill.gpu_guard import check_env, check_runtime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--max-steps", type=int, default=-1, help="debug: stop after N optimizer steps")
    a = ap.parse_args()
    visible = check_env()
    cfg = yaml.safe_load(open(a.config))
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    from trl import SFTConfig, SFTTrainer
    set_seed(cfg.get("seed", 0))
    out_dir = os.path.join(ROOT, cfg["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def to_pc(ex):
        # Qwen3 chat template; thinking disabled so the student learns the visible answer only (teacher data has no reasoning)
        prompt = tok.apply_chat_template([{"role": "user", "content": ((ex["context"] + "\n\n") if ex.get("context") else "") + ex["instruction"]}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return {"prompt": prompt, "completion": ex["response"] + tok.eos_token}

    files = {"train": os.path.join(ROOT, cfg["data_train"])}
    if cfg.get("data_val"):
        files["validation"] = os.path.join(ROOT, cfg["data_val"])
    ds = load_dataset("json", data_files=files)
    cols = ds["train"].column_names
    ds = ds.map(to_pc, remove_columns=cols)
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16, attn_implementation=cfg.get("attn", "sdpa"))
    peft_config = None
    if cfg.get("lora"):
        from peft import LoraConfig
        l = cfg["lora"]
        peft_config = LoraConfig(r=l["r"], lora_alpha=l["alpha"], lora_dropout=l.get("dropout", 0.0), task_type="CAUSAL_LM",
                                 target_modules=l.get("targets", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    steps_per_epoch = max(1, len(ds["train"]) // (cfg["batch"] * cfg["grad_accum"]))
    total_steps = a.max_steps if a.max_steps > 0 else int(steps_per_epoch * cfg["epochs"])
    warmup_steps = int(cfg.get("warmup_ratio", 0.03) * total_steps)
    args = SFTConfig(output_dir=out_dir, per_device_train_batch_size=cfg["batch"], gradient_accumulation_steps=cfg["grad_accum"],
                     learning_rate=float(cfg["lr"]), num_train_epochs=cfg["epochs"], max_steps=a.max_steps, bf16=True,
                     logging_steps=cfg.get("logging_steps", 5), save_steps=cfg.get("save_steps", 200), save_total_limit=3,
                     eval_strategy="steps" if "validation" in ds else "no", eval_steps=cfg.get("eval_steps", 100),
                     warmup_steps=warmup_steps, lr_scheduler_type=cfg.get("scheduler", "cosine"),
                     max_length=cfg["max_len"], gradient_checkpointing=cfg.get("grad_ckpt", True), completion_only_loss=True,
                     report_to=[], seed=cfg.get("seed", 0), dataloader_num_workers=2)
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds.get("validation"),
                         processing_class=tok, peft_config=peft_config)
    used = check_runtime()
    print(f"[gpu_guard] CUDA_VISIBLE_DEVICES={visible} physical GPUs in use by this pid: {used}", flush=True)
    resume = sorted(glob.glob(os.path.join(out_dir, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=resume[-1] if resume else None)
    train_s = time.time() - t0
    trainer.save_model(os.path.join(out_dir, "adapter" if peft_config else "final"))
    info = {"config": cfg, "config_path": a.config, "git_commit": git_commit(), "cuda_visible_devices": visible, "physical_gpus_used": used,
            "gpu_name": torch.cuda.get_device_name(0), "n_train": len(ds["train"]), "n_val": len(ds["validation"]) if "validation" in ds else 0,
            "train_seconds": round(train_s, 1), "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "log_history": trainer.state.log_history, "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in model.parameters())}
    if peft_config and cfg.get("merge", True):
        merged = trainer.model.merge_and_unload()
        mdir = os.path.join(out_dir, "merged")
        merged.save_pretrained(mdir, safe_serialization=True)
        # copy the base model's tokenizer files verbatim (files written by transformers 5 do not load in the 4.57 bench venv)
        import shutil
        for fn in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "generation_config.json"]:
            src = os.path.join(cfg["model"], fn)
            if os.path.exists(src):
                shutil.copy(src, mdir)
        from distill.fix_config import fix as _fix_cfg; _fix_cfg(mdir)   # transformers-5 config -> also correct under 4.57
        info["merged_dir"] = mdir
    json.dump(info, open(os.path.join(out_dir, "run.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in info.items() if k != "log_history"}, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
