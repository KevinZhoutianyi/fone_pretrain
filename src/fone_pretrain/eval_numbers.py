"""Number-task eval suite: the same tasks run on our checkpoints and, later, on
frontier models (paper.md §3). Tasks are generated fresh from a seed so every model
sees identical examples.

Tasks:
  add / sub / mul  -- few-shot arithmetic, exact match, swept over digit lengths
  compare          -- "is A larger than B", answered by option log-prob

Usage (inside a GPU allocation):
  python -m fone_pretrain.eval_numbers CKPT --tasks add,sub,compare --digits 2,4,6,8,10 \
      --n 200 --out results.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import PackedDataset  # noqa: F401  (manifest reuse)
from .model import FonePretrainModel, ModelConfig
from .number_embed import N_SLOTS, digits_to_slots, extract_numbers, slots_to_string

NUM_TOKEN = "<NUM>"


# === task generation (model-agnostic; same seed -> same examples for every model) ===
def gen_arith(op: str, n_digits: int, n: int, rng: random.Random) -> list[dict]:
    """Few-shot arithmetic. Answers can exceed FoNE's 10-digit input range for large
    multiplications; we cap operands so targets stay in range for all variants."""
    lo, hi = 10 ** (n_digits - 1), 10 ** n_digits - 1
    out = []
    for _ in range(n):
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if op == "add":
            q, ans = f"{a} + {b} =", a + b
        elif op == "sub":
            a, b = max(a, b), min(a, b)  # keep answers non-negative (sign is a text token)
            q, ans = f"{a} - {b} =", a - b
        elif op == "mul" and n_digits <= 5:  # 5x5 digits stays within 10 digits
            q, ans = f"{a} * {b} =", a * b
        else:
            continue
        out.append({"q": q, "ans": str(ans)})
    return out


def gen_compare(n_digits: int, n: int, rng: random.Random) -> list[dict]:
    lo, hi = 10 ** (n_digits - 1), 10 ** n_digits - 1
    out = []
    for _ in range(n):
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if a == b:
            continue
        out.append({"q": f"Between {a} and {b}, the larger number is", "ans": str(max(a, b))})
    return out


FEWSHOT = {  # 4-shot prefixes, digits chosen away from the sweep values
    "add": "371 + 254 = 625\n89 + 33 = 122\n705 + 118 = 823\n46 + 91 = 137\n",
    "sub": "625 - 371 = 254\n122 - 89 = 33\n823 - 705 = 118\n137 - 91 = 46\n",
    "mul": "37 * 25 = 925\n89 * 33 = 2937\n70 * 18 = 1260\n46 * 91 = 4186\n",
    "compare": ("Between 371 and 254, the larger number is 371\n"
                "Between 89 and 133, the larger number is 133\n"
                "Between 705 and 118, the larger number is 705\n"),
}


# === our-model side: tokenize prompts (fone-aware) and generate ===
class CkptRunner:
    def __init__(self, ckpt_path: str, device="cuda"):
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = state["cfg"]
        from transformers import AutoTokenizer
        man = json.loads((Path(cfg["data_dir"]) / "manifest.json").read_text())
        self.tok = AutoTokenizer.from_pretrained(man["tokenizer"])
        self.mode = man["mode"]
        if self.mode == "fone":
            self.tok.add_special_tokens({"additional_special_tokens": [NUM_TOKEN]})
        self.num_id = man["num_token_id"]
        mcfg = ModelConfig(vocab_size=man["vocab_size"], n_layer=cfg["n_layer"],
                           n_head=cfg["n_head"], d_model=cfg["d_model"], d_ff=cfg["d_ff"],
                           max_seq_len=cfg["seq_len"], embed_mode=cfg["embed_mode"],
                           num_token_id=man["num_token_id"],
                           num_loss_weight=cfg.get("num_loss_weight", 1.0))
        self.model = FonePretrainModel(mcfg).to(device).eval()
        self.model.load_state_dict(state["model"])
        self.device = device

    def encode(self, text: str) -> tuple[list[int], list[list[int]]]:
        """Token ids plus per-position digit slots (zeros off <NUM>)."""
        if self.mode == "fone":
            text, numbers = extract_numbers(text, NUM_TOKEN)
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        slots = [[0] * N_SLOTS for _ in ids]
        if self.mode == "fone":
            it = iter(numbers)
            for i, t in enumerate(ids):
                if t == self.num_id:
                    a, b = next(it)
                    slots[i] = digits_to_slots(a, b)
        return ids, slots

    @torch.no_grad()
    def generate(self, prompt: str, max_new: int = 24) -> str:
        """Greedy decode; a generated <NUM> is expanded via the digit head and its
        slots are fed back so the model sees the value it produced."""
        ids, slots = self.encode(prompt)
        pieces = []
        for _ in range(max_new):
            idx = torch.tensor([ids], device=self.device)
            sl = torch.tensor([slots], device=self.device)
            h = self.model(idx, num_slots=sl)[0, -1]
            nxt = int(self.model.lm_head(h).argmax())
            if nxt == self.tok.eos_token_id:
                break
            if self.mode == "fone" and nxt == self.num_id:
                dec = self.model.num_head.decode(h[None])[0].tolist()
                pieces.append(slots_to_string(dec))
                ids.append(nxt)
                slots.append(dec)
            else:
                pieces.append(self.tok.decode([nxt]))
                ids.append(nxt)
                slots.append([0] * N_SLOTS)
            if "\n" in pieces[-1]:
                break
        return "".join(pieces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tasks", default="add,sub,mul,compare")
    ap.add_argument("--digits", default="2,4,6,8,10")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runner = CkptRunner(args.ckpt)
    results = {}
    for task in args.tasks.split(","):
        for nd in [int(d) for d in args.digits.split(",")]:
            rng = random.Random(args.seed * 1000 + nd)
            examples = (gen_compare if task == "compare" else gen_arith)(
                *((nd, args.n, rng) if task == "compare" else (task, nd, args.n, rng)))
            if not examples:
                continue
            correct = 0
            for k, ex in enumerate(examples):
                gen = runner.generate(FEWSHOT[task] + ex["q"]).strip().split("\n")[0].strip()
                gen = gen.rstrip(".")
                correct += gen == ex["ans"]
                if k < 3:  # log a few triples per config (experiments/CLAUDE.md §6)
                    print(f"[{task}/{nd}d] q={ex['q']!r} gen={gen!r} label={ex['ans']!r}")
            acc = correct / len(examples)
            results[f"{task}_{nd}d"] = {"acc": acc, "n": len(examples)}
            print(f"== {task} {nd}-digit: {acc:.3f} ({correct}/{len(examples)})", flush=True)

    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
