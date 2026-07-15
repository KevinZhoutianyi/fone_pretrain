"""GSM8K 8-shot exact-match eval, own greedy decode (no vLLM).

Evaluates either:
  - a surgery checkpoint from train_olmo.py (--ckpt path/to/ckpt_final.pt), or
  - an HF model id / local snapshot for the external reference (--hf allenai/OLMo-2-0425-1B).

GSM8K answers are the number after "####". We use the standard 8-shot chain-of-thought
prompt (Wei et al.), greedy-decode up to --max-new tokens, take the last number in the
generation as the predicted answer, and compare to the gold final number.

Usage (inside a GPU allocation):
  python scripts/eval_gsm8k.py --ckpt .../ckpt_final.pt --n 200 --out results.json
  python scripts/eval_gsm8k.py --hf allenai/OLMo-2-0425-1B --n 200 --out ref.json
"""
import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("HF_HOME", "/fsx/zhouty/data/hf_cache")

import torch

# Standard 8-shot CoT exemplars (Wei et al. 2022; the canonical GSM8K few-shot set).
FEWSHOT = [
    ("Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
     "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "A: There are 15 trees originally. Then there were 21 trees after some more were planted. "
     "So there must have been 21 - 15 = 6. The answer is 6."),
    ("Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
     "After eating 35, they had 74 - 35 = 39. The answer is 39."),
    ("Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
     "How many lollipops did Jason give to Denny?",
     "A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
     "So he gave Denny 20 - 12 = 8. The answer is 8."),
    ("Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
     "How many toys does he have now?",
     "A: Shawn started with 5 toys. He then got 2 toys each from his mom and dad. "
     "So he got 2 * 2 = 4 more toys. 5 + 4 = 9. The answer is 9."),
    ("Q: There were nine computers in the server room. Five more computers were installed each day, "
     "from monday to thursday. How many computers are now in the server room?",
     "A: There were originally 9 computers. For each of 4 days, 5 more computers were added. "
     "So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29."),
    ("Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
     "How many golf balls did he have at the end of wednesday?",
     "A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
     "After losing 2 more, he had 35 - 2 = 33. The answer is 33."),
    ("Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
     "So she has 23 - 15 = 8 dollars left. The answer is 8."),
]
PREFIX = "\n\n".join(f"{q}\n{a}" for q, a in FEWSHOT) + "\n\n"


def gold_answer(ans_field: str) -> str:
    return ans_field.split("####")[-1].strip().replace(",", "")


def last_number(text: str) -> str:
    nums = re.findall(r"-?\d[\d,]*", text)
    return nums[-1].replace(",", "") if nums else ""


class Model:
    """Wraps either a surgery checkpoint or a plain HF model; exposes greedy generate."""
    def __init__(self, ckpt=None, hf=None, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if ckpt:
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            cfg = state["cfg"]
            from fone_pretrain.surgery import SurgeredLM
            self.tok = AutoTokenizer.from_pretrained(cfg["init_checkpoint"])
            base = AutoModelForCausalLM.from_pretrained(cfg["init_checkpoint"], dtype=torch.bfloat16)
            self.model = SurgeredLM(base, self.tok, cfg["arm"]).to(device).eval()
            self.model.load_state_dict(state["model"])
            self.surgered = cfg["arm"] != "baseline"
        else:
            self.tok = AutoTokenizer.from_pretrained(hf)
            self.model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16).to(device).eval()
            self.surgered = None
        self.device = device
        # precompute effective weights once for surgery models (static during eval)
        self.W_in = self.W_out = None
        if self.surgered:
            self.W_in = self.model.emb_surgery.effective_weight()
            self.W_out = self.model.head_surgery.effective_weight()

    @torch.no_grad()
    def logits_next(self, ids):
        idx = torch.tensor([ids], device=self.device)
        if self.surgered is True:
            import torch.nn.functional as F
            x = F.embedding(idx, self.W_in)
            body = self.model.lm.model(inputs_embeds=x)
            return F.linear(body.last_hidden_state[0, -1], self.W_out)
        elif self.surgered is False:  # our baseline arm wraps HF
            return self.model(idx)[0, -1]
        else:  # plain HF reference
            return self.model(input_ids=idx).logits[0, -1]

    @torch.no_grad()
    def generate(self, prompt, max_new=256):
        ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        newline = self.tok("\n\n", add_special_tokens=False)["input_ids"]
        out = []
        for _ in range(max_new):
            nxt = int(self.logits_next(ids).argmax())
            if nxt == self.tok.eos_token_id:
                break
            ids.append(nxt); out.append(nxt)
            txt = self.tok.decode(out)
            if "\n\n" in txt or txt.count("The answer is") >= 1 and txt.rstrip().endswith("."):
                break
        return self.tok.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--hf", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    assert args.ckpt or args.hf, "need --ckpt or --hf"

    from datasets import load_dataset
    test = load_dataset("openai/gsm8k", "main", split="test")
    m = Model(ckpt=args.ckpt, hf=args.hf)

    correct = 0
    for i in range(min(args.n, len(test))):
        q, a = test[i]["question"], test[i]["answer"]
        gen = m.generate(PREFIX + f"Q: {q}\nA:", args.max_new)
        pred = last_number(gen.split("The answer is")[-1] if "The answer is" in gen else gen)
        gold = gold_answer(a)
        correct += (pred == gold)
        if i < 3:
            print(f"[{i}] gold={gold!r} pred={pred!r} gen={gen[:120]!r}")
    acc = correct / min(args.n, len(test))
    res = {"gsm8k_acc": acc, "n": min(args.n, len(test)),
           "model": args.ckpt or args.hf}
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    main()
