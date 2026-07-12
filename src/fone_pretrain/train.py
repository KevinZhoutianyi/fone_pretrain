"""DDP pretraining loop for the three-variant FoNE comparison.

Launch (inside an sbatch/srun allocation, never on the login node):
  torchrun --standalone --nproc_per_node=8 -m fone_pretrain.train configs/125m_fone.yaml
  torchrun ... -m fone_pretrain.train configs/125m_fone.yaml --smoke   # 20 steps, tiny batch

Config is YAML; every hyperparameter that affects results is printed at start
(experiments/CLAUDE.md §6). Metrics stream to stdout and metrics.jsonl in the run
dir; checkpoints go to the run dir (a background sbatch loop syncs them to S3).
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

from .data import PackedDataset
from .model import FonePretrainModel, ModelConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--smoke", action="store_true", help="20 steps at tiny scale, full code path")
    ap.add_argument("--resume", default=None, help="checkpoint path to resume from")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    # === DDP setup ===
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        torch.cuda.set_device(device)
    else:
        rank, world, device = 0, 1, "cuda"
    master = rank == 0
    torch.manual_seed(cfg["seed"] + rank)

    if args.smoke:
        cfg.update(max_steps=20, batch_size=4, grad_accum=1, warmup_steps=5,
                   log_every=1, ckpt_every=10, eval_every=10, eval_iters=2)

    # === data ===
    train_ds = PackedDataset(cfg["data_dir"], cfg["seq_len"], "train")
    val_ds = PackedDataset(cfg["data_dir"], cfg["seq_len"], "val")
    man = train_ds.manifest
    rng = np.random.default_rng(cfg["seed"] * 1000 + rank)

    # === model ===
    mcfg = ModelConfig(
        vocab_size=man["vocab_size"], n_layer=cfg["n_layer"], n_head=cfg["n_head"],
        d_model=cfg["d_model"], d_ff=cfg["d_ff"], max_seq_len=cfg["seq_len"],
        embed_mode=cfg["embed_mode"], n_learned_freq=cfg.get("n_learned_freq", 20),
    )
    # number-chunk maps (is_number_token, token_value) are precomputed by prepare_data.py
    # and saved next to the shards; baseline does not need them.
    is_num = tok_value = None
    if cfg["embed_mode"] != "baseline":
        nm = np.load(Path(cfg["data_dir"]) / "number_map.npz")
        is_num = torch.from_numpy(nm["is_number_token"])
        tok_value = torch.from_numpy(nm["token_value"])
    raw = FonePretrainModel(mcfg, is_num, tok_value).to(device)   # uncompiled ref: clean state_dict keys
    model = torch.compile(raw)                 # params are shared, not copied
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model)

    # log_periods (fone_learned only) are the learnable Fourier periods; weight decay
    # would drag them toward log-period 0 (period 1), collapsing every chunk's code to a
    # constant and destroying the number signal. Exclude from decay; everything else
    # keeps the original rate.
    no_decay = [p for n, p in raw.named_parameters() if n.endswith("log_periods")]
    decay = [p for n, p in raw.named_parameters() if not n.endswith("log_periods")]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=(0.9, 0.95), fused=True)

    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        raw.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_step = state["step"]

    def lr_at(step):  # linear warmup then cosine to 10%
        if step < cfg["warmup_steps"]:
            return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
        t = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"])
        return cfg["lr"] * (0.1 + 0.45 * (1 + math.cos(math.pi * t)))

    run_dir = Path(cfg["run_dir"])
    if master:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.dump(cfg))
        tokens_per_step = cfg["batch_size"] * cfg["grad_accum"] * world * cfg["seq_len"]
        print("=== Training ===")
        for k in ["embed_mode", "n_layer", "d_model", "n_head", "d_ff", "seq_len",
                  "lr", "batch_size", "grad_accum", "max_steps", "warmup_steps", "seed", "data_dir"]:
            print(f"{k:16s} {cfg[k]}")
        print(f"{'world_size':16s} {world}")
        print(f"{'tokens/step':16s} {tokens_per_step:,}")
        print(f"{'total tokens':16s} {tokens_per_step * cfg['max_steps']:,}", flush=True)
        metrics_f = open(run_dir / "metrics.jsonl", "a")

    @torch.no_grad()
    def evaluate():
        model.eval()
        agg = {"lm_loss": 0.0}
        for _ in range(cfg["eval_iters"]):
            batch = val_ds.sample_batch(cfg["batch_size"], rng, device)
            with torch.autocast("cuda", torch.bfloat16):
                out = raw(**batch)
            for k in agg:
                agg[k] += out[k].item() / cfg["eval_iters"]
        model.train()
        return agg

    # === train loop ===
    model.train()
    t0, tokens_seen = time.time(), 0
    for step in range(start_step, cfg["max_steps"]):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        for micro in range(cfg["grad_accum"]):
            batch = train_ds.sample_batch(cfg["batch_size"], rng, device)
            if ddp:  # only sync grads on the last micro-batch
                model.require_backward_grad_sync = micro == cfg["grad_accum"] - 1
            with torch.autocast("cuda", torch.bfloat16):
                out = model(**batch)
            (out["loss"] / cfg["grad_accum"]).backward()
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        tokens_seen += cfg["batch_size"] * cfg["grad_accum"] * world * cfg["seq_len"]

        if master and (step % cfg["log_every"] == 0 or step == cfg["max_steps"] - 1):
            dt = time.time() - t0
            rec = {"step": step, "loss": out["loss"].item(), "lm_loss": out["lm_loss"].item(),
                   "lr": lr_at(step), "tok_per_s": int(tokens_seen / dt)}
            print(json.dumps(rec), flush=True)
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()

        if master and step > 0 and step % cfg["eval_every"] == 0:
            ev = evaluate()
            print(json.dumps({"step": step, "val": ev}), flush=True)
            metrics_f.write(json.dumps({"step": step, "val": ev}) + "\n")
            metrics_f.flush()

        if master and (step + 1) % cfg["ckpt_every"] == 0:
            torch.save({"model": raw.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1, "cfg": cfg},
                       run_dir / "ckpt_latest.pt")
            print(f"saved ckpt at step {step + 1}", flush=True)

    if master:
        torch.save({"model": raw.state_dict(), "opt": opt.state_dict(),
                    "step": cfg["max_steps"], "cfg": cfg}, run_dir / "ckpt_final.pt")
        ev = evaluate()
        print(json.dumps({"final_val": ev}), flush=True)
        metrics_f.close()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
