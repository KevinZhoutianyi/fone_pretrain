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
| [job_id] | [Exp X] | [ETA tonight] | [§3.4] | [closes Q3 in §3] |

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

1. **[Action item]** — serves paper.md §[N]. [One-line description of
   what this experiment closes; compute estimate.]
2. **[Action item]** — serves paper.md §[N]. [...]
