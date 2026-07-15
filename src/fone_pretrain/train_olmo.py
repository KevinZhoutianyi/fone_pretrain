"""DDP stage-2 (mid-training) loop for the OLMo-2 FoNE surgery experiment.

Starts from a pretrained OLMo-2 stage-1 checkpoint (HF Olmo2ForCausalLM), applies FoNE
row surgery (surgery.py), and continues training on the Dolmino stage-2 mix. Three arms
(baseline / unfreeze_ctrl / fone) selected by cfg["arm"]. Reuses PackedDataset and the
DDP/metrics/decay patterns from train.py; kept separate so train.py's from-scratch line
is untouched.

Launch (inside an sbatch/srun allocation, never the login node):
  torchrun --standalone --nproc_per_node=8 -m fone_pretrain.train_olmo configs/olmo_stage2/fone_s1337.yaml
  torchrun ... -m fone_pretrain.train_olmo <config> --smoke   # 20 steps, tiny batch

LR schedule is a linear anneal from lr to lr_end over the run (WSD decay phase, matching
OLMo-2 mid-training), with a short warmup; all values are config fields.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

from .data import PackedDataset
from .surgery import SurgeredLM


def build_model(cfg, device):
    """Load the stage-1 HF checkpoint and wrap with the arm's surgery."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ckpt = cfg["init_checkpoint"]
    hf = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.float32)
    hf.config.use_cache = False
    if cfg.get("grad_checkpoint", True):
        hf.gradient_checkpointing_enable()   # 1B full-backprop at seq 4096 needs this
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = SurgeredLM(hf, tok, cfg["arm"]).to(device)
    return model


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
        cfg.update(max_steps=20, batch_size=1, grad_accum=1, warmup_steps=2,
                   log_every=1, ckpt_every=10, eval_every=10, eval_iters=2)

    # === data ===
    train_ds = PackedDataset(cfg["data_dir"], cfg["seq_len"], "train")
    val_ds = PackedDataset(cfg["data_dir"], cfg["seq_len"], "val")
    rng = np.random.default_rng(cfg["seed"] * 1000 + rank)

    # === model (pretrained OLMo-2 + arm surgery) ===
    raw = build_model(cfg, device)
    model = torch.nn.parallel.DistributedDataParallel(raw) if ddp else raw

    # num_code params (code scale + log_periods) define the number signal, not ordinary
    # weights: exclude from weight decay (decay would shrink scale to 0 / collapse
    # periods). Everything else (transformer body, glue, non-number embedding rows)
    # keeps the standard rate. Frozen params (HF emb/head weight) have requires_grad
    # False and are skipped automatically.
    no_decay, decay = [], []
    for n, p in raw.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if ".num_code." in n else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.get("weight_decay", 0.1)},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8, fused=True)

    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        raw.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_step = state["step"]

    lr_end = cfg.get("lr_end", 0.0)
    def lr_at(step):  # linear warmup then linear anneal to lr_end (WSD decay phase)
        if step < cfg["warmup_steps"]:
            return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
        t = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"])
        return cfg["lr"] + (lr_end - cfg["lr"]) * t

    run_dir = Path(cfg["run_dir"])
    if master:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.dump(cfg))
        tokens_per_step = cfg["batch_size"] * cfg["grad_accum"] * world * cfg["seq_len"]
        n_train = sum(p.numel() for p in raw.parameters() if p.requires_grad)
        print("=== Stage-2 training ===")
        for k in ["arm", "init_checkpoint", "seq_len", "lr", "lr_end", "batch_size",
                  "grad_accum", "max_steps", "warmup_steps", "seed", "data_dir"]:
            print(f"{k:16s} {cfg.get(k)}")
        print(f"{'world_size':16s} {world}")
        print(f"{'trainable':16s} {n_train/1e6:.1f}M")
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

    # code diagnostics: track that the FoNE code stays alive (scale not collapsing)
    def code_stats():
        if cfg["arm"] != "fone":
            return {}
        return {"emb_scale": raw.emb_surgery.num_code.scale.item(),
                "head_scale": raw.head_surgery.num_code.scale.item()}

    # === train loop ===
    model.train()
    t0, tokens_seen = time.time(), 0
    for step in range(start_step, cfg["max_steps"]):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        for micro in range(cfg["grad_accum"]):
            batch = train_ds.sample_batch(cfg["batch_size"], rng, device)
            if ddp:
                model.require_backward_grad_sync = micro == cfg["grad_accum"] - 1
            with torch.autocast("cuda", torch.bfloat16):
                out = model(**batch)
            (out["loss"] / cfg["grad_accum"]).backward()
        torch.nn.utils.clip_grad_norm_((p for p in raw.parameters() if p.requires_grad), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        tokens_seen += cfg["batch_size"] * cfg["grad_accum"] * world * cfg["seq_len"]

        if master and (step % cfg["log_every"] == 0 or step == cfg["max_steps"] - 1):
            dt = time.time() - t0
            rec = {"step": step, "loss": out["loss"].item(), "lm_loss": out["lm_loss"].item(),
                   "lr": lr_at(step), "tok_per_s": int(tokens_seen / dt), **code_stats()}
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
                        "step": step + 1, "cfg": cfg}, run_dir / "ckpt_latest.pt")
            print(f"saved ckpt at step {step + 1}", flush=True)

    if master:
        torch.save({"model": raw.state_dict(), "step": cfg["max_steps"], "cfg": cfg},
                   run_dir / "ckpt_final.pt")
        ev = evaluate()
        print(json.dumps({"final_val": ev, **code_stats()}), flush=True)
        metrics_f.close()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
