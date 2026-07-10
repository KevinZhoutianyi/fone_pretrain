# Working Guide

**Re-read this file periodically during long conversations.** After extended implementation work (rounds of code changes, waiting for jobs, debugging), re-read it before the next task so the project goals stay in view.

This file holds universal project rules. Context-specific rules live in subdirectory CLAUDE.md files:

| where | what it governs |
|---|---|
| `doc/CLAUDE.md` | `paper.md`, `proposal.md`, `tracking.md`, the LaTeX paper, related-papers notes. Writing rules, claims calibration, the stranger-read pass. |
| `doc/weekly_updates/CLAUDE.md` | slide decks. |
| `experiments/CLAUDE.md` | code style, correctness, compute, experiment logging. |
| `theory/CLAUDE.md` | theory writing: no obvious theorems, justified assumptions, tightness, proof intuition. |

The LaTeX paper lives at `doc/paper/`; see `doc/paper/README.md`.

---

## 0. Feedback file

The user writes feedback into `doc/paper/feedback.md` after reading the paper or experiments on their own (never during a conversation). At the start of every session: read it, act on every bullet, delete each bullet once handled. Ask before acting on ambiguous bullets. Do not add bullets yourself.

(Distinct from `doc/weekly_updates/YYYY-MM-DD/post_talk_notes.md`, which records advisor feedback after a talk.)

---

## 1. Goal-driven work

Before designing an experiment or building a deck, state the goal in one sentence ("Show that method X beats baseline Y on benchmark Z under condition W"). If one sentence is impossible, the goal is not clear enough yet; clarify first.

Then list the subgoals, and name which subgoal every slide, paragraph, table, plot, and experimental knob serves. No subgoal, no content. Re-check at the end: does the whole thing read straight to the goal? Cut anything that detours, defends, or self-justifies.

This filter comes before any formatting rule.

---

## 2. Tables over prose

Results, configs, and comparisons go in markdown tables, not bullets or paragraphs. Applies to reports, `tracking.md` entries, and printed experiment summaries.

---

## 3. Git: commit and push

After adding an experiment, updating tracking, or any substantive change: commit and push immediately. The user never has to ask or push manually.

---

## 4. Surgical changes

Touch only what the request requires; clean up only your own mess.

- Do not improve adjacent code, comments, or formatting.
- Do not refactor what is not broken. Match existing style.
- Mention pre-existing dead code; do not delete it.
- Do remove imports/variables/functions that YOUR change orphaned.

Test: every changed line traces directly to the user's request.

---

## 5. Verify each step

Turn tasks into verifiable goals and loop until verified. For multi-step tasks, state the plan as steps each paired with a check ("Add validation -> tests for invalid inputs pass"). Weak criteria ("make it work") need clarification before starting.

A task is not done until the output passes the relevant subdirectory CLAUDE.md:

| task type | verify against |
|---|---|
| Doc edits (`paper.md`, `proposal.md`, `main.tex`) | `doc/CLAUDE.md`: stranger-read pass, structure rules, self-verification checklist, render-and-read |
| Experiment code or analysis | `experiments/CLAUDE.md`: code style, correctness, smoke-test, logging |
| Figure code (`make_figures.py`) | `experiments/CLAUDE.md` §3: Okabe-Ito constants, no inline rcParams, variance bands |
| Slide decks | `doc/weekly_updates/CLAUDE.md` |
| `tracking.md` updates | `doc/CLAUDE.md` §5: three-way consistency (experiments, paper claims, tracking) |

Skipping the checklist means the task is not finished. Verification is part of the work.

---

## 6. Reports carry their own context

Result summaries and explanations (chat messages, READMEs, tracking rows, slides) are read by someone who did not watch the work happen. Names coined during the work (a pipeline stage, an eval config, a task subset) mean nothing to that reader.

Before the numbers: state in one or two sentences what question the experiment answers, what was run, and what is being compared. Define every coined term at first use in every report; a definition given in an earlier message does not carry over. Translate each headline number into its concrete meaning ("5 of the 11 tasks that passed without steering now fail", not "5 broke").

Test: can the reader repeat the claim to a third person without asking a follow-up? Any sentence that would trigger "what does that mean" gets expanded before sending.
