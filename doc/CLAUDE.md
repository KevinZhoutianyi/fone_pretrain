# Doc Folder Rules

| file / folder | what it holds |
|---|---|
| `paper.md` | The argument: goal, thesis, outline, per-section evidence, related work, appendices. Changes land here first. |
| `proposal.md` | The research proposal: method, baselines, risks, plan. Same writing rules as `paper.md`; numbers in it must match `tracking.md`. |
| `paper/` | The same argument as LaTeX for Overleaf. Mirrors `paper.md`; lags it. |
| `tracking.md` | The status: active runs, completed, failed, next steps. Each row names the `paper.md` section it serves. |
| `weekly_updates/` | Slide decks. Rules in `weekly_updates/CLAUDE.md`. |
| `related_papers/` | One note per cited paper. |
| `paper/example_papers/` | Reference PDFs whose writing, figure, and section style this paper targets. |

These are living documents. Update in place; never append dated entries.

`paper.md` and `tracking.md` are canonical templates: copy their section structure into new projects instead of inventing parallel structures.

---

## 1. Writing rules (every file in this folder)

### Motivation first

Every paragraph, section, table row, and figure caption leads with why it exists: what question it answers or what gap it closes. A reader asking "why am I reading this?" at any sentence finds the answer in the surrounding prose.

Test for new content: write "Because [previous claim raised question X], we did [new thing], which shows [new claim]." If that sentence does not write itself, the content does not belong yet.

The same rule kills redundancy. Every claim traces to the thesis or a named subgoal. When a claim appears twice, keep the strongest version and demote the other to a pointer. A body table with more than ~5 rows, or one that repeats numbers from the prose, moves to an appendix.

### The stranger-read pass

Run this on every doc and figure before delivering. It exists because repeated review rounds traced to one root cause: text was checked for correctness but never re-read with zero context. Typical failures: a paragraph labeled "Training:" when no training had happened; metric abbreviations used before definition; an abstract category where one concrete instance was needed; figure labels colliding with arrows.

1. No dangling referents. Every label, abbreviation, metric, and run name is defined before first use. Every number says what it counts or what it is a percentage of. Specific checks: (a) every term introduced in the abstract or intro that is not plain English gets a parenthetical gloss at first body use; (b) every named evaluation set (e.g., a "battery" of questions) is described — size, content, construction — at first use; (c) every named direction or axis gets a definition the first time it appears in the body, not deferred to an appendix; (d) every external technique name (e.g., "logit-lens probe") either appears in Key Terms or gets a one-clause gloss inline.
2. Facts as sentences, not labels. "We have not trained anything yet", never a section opener like "Training:".
3. Instance before category. Open an abstract claim with one worked example from the project's own data, then state the general point once.
4. One point per sentence. A qualifier either earns its own sentence or is cut. Emphasizing everything emphasizes nothing.
5. Figures get the same pass: render, then read the render. Trace every arrow tail to head. Check every label for collisions. Ask what each icon looks like at actual size.
6. One logical move per paragraph. A paragraph that presents multiple independent pieces of evidence must signpost each one explicitly ("First... Second... Third...", "Ruling that out, we then show...", "A separate result confirms..."). The reader must never have to infer the logical connection between consecutive sentences. Test: cover every sentence but the last — does the final sentence follow obviously? If not, add a bridge.

### Style

Match the example papers in `paper/example_papers/` across three dimensions: writing voice, figure design, section structure. Before drafting a section opener, designing a figure, or restructuring the paper, open one of those PDFs and check how it handles the same situation. The `related_papers/` notes inform what we cite; the example papers inform how the writing reads.

Structure:
- Introduction has no subsections. Flowing prose, ending with a numbered contributions list.
- Section titles are noun phrases ("Depth Generalization"), not sentences. Same for subsections.
- Related Work is flowing prose, one paragraph per paper or group.
- Mechanism subsection titles state the finding ("Detection forms in the last seven layers"), not a question.
- Caveats go inline after the result they qualify. No standalone "Limitations" subsections inside experiment sections.
- Prose over bullets in the body. Bullets are for contributions lists and procedural steps only.
- Dense 2D evidence (N conditions x M outcomes) goes in a table, not prose.

Language:
- No em dashes, in either `---` or Unicode form. Use commas, colons, parentheses, or split the sentence. Grep before committing.
- No contractions. First-person plural throughout.
- Short declarative sentences. "X reaches 99.7%. Y fails at 63.3%." Not "We observe that X consistently reaches...".
- Cut filler openers ("It is worth noting that"), defensive hedging, grandiose framing, and narration ("Table X shows that").
- Banned words, rewrite every instance: delve, crucial, pivotal, robust, leverage, utilize, showcase, comprehensive, notably, importantly, interestingly, it is worth noting, this allows us to, in summary, in conclusion.
- No implementation details in the body. Library names, file paths, CLI flags, experiment IDs go to `tracking.md` or a run script docstring.
- Every technical claim gets a plain-English gloss within one paragraph. If a non-specialist cannot restate the claim, the gloss is missing.

---

## 2. Claims and evidence

Calibrate claim strength to the data. If 10/15 cases support it, write "10/15", not "the claim holds". Partial results stay partial in the title, abstract, and figure titles: a chart showing N pass and M fail names both numbers. (Caught failure: a title claiming "is X, not Y" over data that was X for 24/39 and Y for 15/39; the fix was "Partial X over a strong Y baseline".)

Classify each piece of evidence as one of:
- Central: theories predict different outcomes and the data picks one.
- Sanity check: all theories predict the same outcome; confirms the setup works.
- Supporting: consistent with the claim but not a discriminator.

Write both theories' predictions down before running a control. If they predict the same thing, it is a sanity check, not a discriminator.

Every prediction carries an explicit falsifier, stated as concretely as the prediction ("if X reaches Y under Z, the claim is challenged"). When data lands, check the falsifier first.

Caveats sit inline next to the result: asymmetric comparisons, single-seed runs, partially completed experiments.

When deleting an experiment, search `tracking.md`, `paper.md`, slides, and `paper/` for citations of its findings. Remove them or demote to "lives in git history at commit `<hash>`".

---

## 3. `paper.md`

Audience: coauthors and the advisor, not venue reviewers. Keep the full argument visible, surface weak spots for discussion, and prefer "We tried X. It failed because Y. So we did Z." over reviewer-style hedging.

Structure:

```
## Goal
  Plain-language setup (3-6 sentences a non-specialist can follow).
  Thesis in one sentence.
  Prior work and what makes the question hard, with a theory-1 vs theory-2 table.

## Outline
  Four-column table per section: Question | What we did | What we showed | Therefore ->
  Plain language only. Each "Therefore" points at the next section or the headline implication.
  Planned rows are marked (planned) with predictions and falsifiers.

## §N Subgoal: <name> (<status>)
  Claim: one sentence stating what this subgoal proves.
  Evidence: experiments tagged Central / Sanity check / Supporting.
  Body keeps headline + plain-language reading + one caveat + appendix pointer.

## Related Work        (table: Paper | Relation to our work; notes live in related_papers/)
## Reframing for slides (one phrase per subgoal; the source of slide headlines)
## Appendix A  Key terms
## Appendix B  How to read the metrics
## Appendix C+ Per-experiment detail dumps (per-bucket tables, top-N lists, sweeps)
```

Section rules the template cannot show:
- The main body tells the positive story only. Ablations, falsified hypotheses, and "what we do NOT claim" lists go to the appendix.
- §2 (controls) includes a discovery arc: observation, naive control, why it was not enough, refined control, discriminator. State what each theory predicts before each result.
- §3 (mechanism) opens with numbered questions, one subsection per question, a one-sentence answer each, then a synthesis table and an explicit what-we-do-NOT-claim list citing what falsifies each over-reading.
- Planned sections are framed as predictions to test, never as results.

Self-verification before declaring a paper edit done:
1. Every new sentence supports a positive claim (else: appendix).
2. Subsection titles are declarative.
3. Multi-condition results are tables.
4. Causal claims have `results.json` backing.
5. `\ref{}` targets are correct.
6. No implementation details in the diff.
7. No em dashes or banned words in the diff.
8. The rendered PDF reads cleanly (overflow, figures, abstract length, references).
9. Numerical consistency: every number in the body matches its source table/appendix. Specifically: (a) pair counts in clustering tables satisfy C(n_pass,2) + n_pass*n_fail + C(n_fail,2) = C(N,2); (b) baseline values stated in multiple places agree; (c) directional claims about a sequence of numbers (e.g., "maintained", "strengthened", "reduced") match the actual sign of the change; (d) approximate rates ("1 in 20", "~30%") are consistent with the exact numbers in the corresponding table.
10. No forward-use of undefined terms: every technical term, named direction, named evaluation set, and external technique name is defined at or before first body use per the stranger-read pass rules above.

---

## 4. `paper/` (LaTeX)

Sync: `paper.md` is the source of truth. Promote a section to `main.tex` only when stable. Structure mirrors `paper.md`; if they drift, the working draft wins. `paper/README.md` maintains the section mapping.

Render and read after every non-trivial edit:

```bash
cd doc/paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
# render pages for the Read tool:
uv run python -c "
import fitz, os
doc = fitz.open('doc/paper/main.pdf')
os.makedirs('/tmp/paper_pages', exist_ok=True)
for i, page in enumerate(doc):
    page.get_pixmap(dpi=110).save(f'/tmp/paper_pages/page_{i+1:02d}.png')"
```

Read each page. Audit: figures cut off or unreadable at print size, empty plots, broken glyphs, `??` references, table overflow (`Overfull \hbox`: fix with `p{width}` columns), abstract over 10 lines (trim to 200 words).

Figures earn their place by answering a question the prose set up, being readable at final size, and carrying structure prose cannot (distributions, comparisons, sweeps). A figure showing one number already in the prose is redundant. Captions open with the question the figure answers. When removing a figure: comment out its call in `make_figures.py` with a one-line reason, keep the function, remove the `\ref{}`s from both `main.tex` and `paper.md`. Figure PDFs are regenerated by `paper/figures/make_figures.py` from `results.json`; never hand-edit them.

Floats: load `placeins`; put `\FloatBarrier` after the Conclusion, before the bibliography. Document order: Conclusion, `\FloatBarrier`, bibliography, `\appendix`, appendices. Use `[t]` placement; never bare `[h]`.

Bibliography: every entry is title + url + year + `author = {TBD}`. Never type author names, even partially remembered ones; the human fills them in from verified metadata. Inline citations go through `\citet`/`\citep` only. "TBD (2025)" rendering in drafts is intentional: it marks unverified entries.

Never declare a human-supplied citation fabricated from an arXiv miss alone. Search OpenReview, ACL Anthology, Semantic Scholar, Google Scholar, and the venue page; fetch any URL the human gave (extract PDF text if needed). After all that, say "could not confirm via X, Y, Z", not "fabricated".

Coauthor comments: `\tynote{}` (blue, author), `\robinnote{}` (red, advisor), `\coauthornote{Name}{}` (green), `\inlinecomment{TY}{}` (purple). Disable all via `\usepackage[disable]{todonotes}`.

Overleaf zip, from `doc/paper/`: `zip <project>_overleaf.zip main.tex references.bib README.md figures/*.pdf`. Place at the repo root (gitignored). Exclude `main.pdf`, aux files, and `make_figures.py`.

---

## 5. `tracking.md`

Four tables, every row cross-referencing the `paper.md` section it serves:

```
## Active runs                job | exp | status | serves paper.md § | note (the note says WHY)
## Recently completed jobs    job | exp | serves paper.md § | result (one line)
## Recently failed jobs       job | exp | failure mode | resolution    <- mandatory; prevents blind re-attempts
## Next steps                 numbered, each naming the paper.md § it serves
```

Three artifacts carry project state: experiment code/results, paper claims, tracking rows. A change to one is incomplete until the other two match:
- New experiment: add a tracking row and update the paper's evidence section.
- New paper claim: point it at a supporting `results.json`, or mark the section (planned) and add a next step.
- Landed result: move the tracking row to completed; update the paper's headline number, reading, and caveat. If `tracking.md` says "15/39 pass" while the paper says "passes the test", the paper is stale.
- Falsified hypothesis: remove it from the paper body (or mark it ruled out), record the verdict in tracking, note the triggered falsifier in the run script docstring.

Job IDs and ETAs never appear in `paper.md`.

---

## 6. `related_papers/`

One `<citekey>.md` per cited paper: a one-paragraph summary, the relation to our work, key numbers. Citekeys match `paper/references.bib`. The `paper.md` Related Work table stays a summary; long notes live here.

---

## 7. `paper/example_papers/`

PDFs of papers whose presentation this paper aims to match. Different from `related_papers/`: those define what we cite; these define how we write, plot, and structure.

Three dimensions to mirror:
- Writing voice: sentence rhythm, how a claim is introduced and qualified, motivation-first openers, declarative section titles.
- Figure design: panel layout, axis labels, color palette, caption form (question first, then plot, then implication).
- Section structure: intro staging, body vs appendix split, how a multi-step argument is laid out across §3/§4.

Before any non-trivial paper edit, ask: would this paragraph, figure, or section structure fit naturally inside one of the example papers? If no, adjust before committing.

Files are named `<short_slug>_<arxiv_id>.pdf`. Update the set when the target venue or paper voice changes.
