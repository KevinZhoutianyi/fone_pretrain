"""Aggregate multi-seed compare eval into a mean +/- std table per variant.

Reads outputs/eval/ms_<variant>_s<seed>.json (written by eval_numbers) and prints,
for each variant and digit length, the mean and standard deviation of compare exact
match across seeds. This is the table that survives the single-seed variance seen in
the 9-run sweep (fone vs fone_seed2 differed by 0.20 at one seed).

Usage: uv run python scripts/aggregate_compare.py
"""
import json
import os
from collections import defaultdict

E = "outputs/eval"
VARIANTS = ["baseline", "fone", "fone_learned", "fone_12d"]
SEEDS = [1337, 2024, 777]
DIGITS = [2, 4, 6, 8, 10]


def main():
    # variant -> digit -> list of per-seed accs
    acc = defaultdict(lambda: defaultdict(list))
    missing = []
    for v in VARIANTS:
        for s in SEEDS:
            p = f"{E}/ms_{v}_s{s}.json"
            if not os.path.exists(p):
                missing.append(f"ms_{v}_s{s}")
                continue
            d = json.load(open(p))
            for n in DIGITS:
                a = d.get(f"compare_{n}d", {}).get("acc")
                if a is not None:
                    acc[v][n].append(a)
    if missing:
        print(f"WARNING missing {len(missing)} results: {missing}\n")

    def mean(xs): return sum(xs) / len(xs) if xs else float("nan")
    def std(xs):
        if len(xs) < 2: return 0.0
        m = mean(xs); return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    print("compare exact match, mean +/- std over seeds", SEEDS)
    print(f"\n{'variant':16s} " + " ".join(f"{n:>11d}d" for n in DIGITS) + f" {'avg':>7s}")
    summary = {"digits": DIGITS, "seeds": SEEDS, "variants": {}}
    for v in VARIANTS:
        cells, means, stds = [], [], []
        for n in DIGITS:
            xs = acc[v][n]
            m, sd = mean(xs), std(xs)
            means.append(m); stds.append(sd)
            cells.append(f"{m:.2f}+/-{sd:.2f}")
        summary["variants"][v] = {"n_seeds": len(acc[v][DIGITS[0]]), "mean": means, "std": stds}
        print(f"{v:16s} " + " ".join(f"{c:>12s}" for c in cells) + f" {mean(means):7.2f}")

    out = "experiments/chunk_fone/results.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
