# Tracking — experiment status

<!-- HOW TO USE THIS TEMPLATE
This file is the **project-state view**: what's running, what just
landed, what failed, what's next. The argument and the evidence live
in **paper.md**; each row here cross-references the paper.md section
that the experiment serves. See CLAUDE.md §3 for the full two-file
rule.

Update in place — do not append dated entries. When a job finishes:
(1) move its row from Active runs → Recently completed, (2) integrate
the finding into the relevant paper.md § evidence section.
-->

## Active runs

<!-- Jobs currently on the cluster. Remove when the job lands and its
result is integrated into paper.md. -->

| job | exp | status | serves paper.md § | note |
|---|---|---|---|---|
| login-node bg | data prep mix3b (baseline + fone) | ~2.3B / ~1.1B of 3.2B tokens | §1 | CPU tokenization; gates the 125M runs |
| 219 | exp 01 smoke rerun (compile-clean check) | running | §1 | confirms static-shape rewrite removed graph breaks |

---

## Recently completed jobs

<!-- One row per landed job. Keep the result one-line; the full
write-up lives in paper.md §N. Once the finding is integrated into
paper.md, this row can be kept here as the historical record. -->

| job | exp | serves paper.md § | result (one-line) |
|---|---|---|---|
| [job_id] | [Exp Y] | [§N] | [one-line headline finding] |

---

## Recently failed jobs

<!-- MANDATORY. Document each failure with the diagnostic and the
resolution (or "blocked on X"). Otherwise the same broken approach
gets re-attempted blindly. -->

| job | exp | failure mode | resolution |
|---|---|---|---|
| [job_id] | [Exp Z] | [OOM / meta-tensor / path bug] | [fixed in commit X / re-submitted as job Y / blocked on Z] |

---

## Next steps

<!-- Numbered list. Each step names which paper.md § it serves —
otherwise the queue drifts away from the paper's argument. -->

1. **Unit-test number extraction and FoNE features (CPU)** — serves paper.md §1.
   Correctness gate before any GPU spend; pytest on extraction regex, digit slots,
   exact phase computation, decode round-trip.
2. **Prepare 3B-token data, baseline and fone variants** — serves paper.md §1.
   FineWeb-Edu 70% + FineMath 30%, TinyLlama tokenizer; manifest token/number
   counts spot-checked by decoding samples. CPU-only on login node, hours.
3. **125M pipeline-validation runs, all three variants via srun** — serves paper.md §1.
   One node 8×H100 each, ~3B tokens; check loss curves and digit accuracy rise.
4. **350M formal three-variant comparison, 10B+ tokens** — serves paper.md §2.
   The equal-token-budget comparison that isolates the embedding as the cause.
5. **Number eval suite incl. frontier-model comparison** — serves paper.md §3.
   Arithmetic exact match by digit length, number comparison, numeric precision;
   frontier API models need keys from the user (blocked on that).
