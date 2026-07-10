# Paper

<!-- HOW TO USE THIS TEMPLATE
This file is the **argument** — the living paper-in-progress. It holds
the goal, thesis, outline, per-section evidence, related work, slide
reframing, and detail appendices. A reader scanning top-to-bottom should
hit: Goal → Thesis → Prior work → Outline → Evidence → Reframing →
Appendices, with no detours into job IDs or queue state.

The companion file **tracking.md** holds experiment status (active runs,
recently completed, recently failed, next steps). Each row of
tracking.md cross-references the paper.md section it serves. See
CLAUDE.md §3 for the full two-file rule.

The structure below is taken from a real research project and is meant
to generalize. Replace [BRACKETED PLACEHOLDERS] with your content; the
HTML comments explain the *intent* of each block.
-->

## Goal

### Plain-language setup (read this first)

<!-- 3–6 sentences a non-specialist can follow. Name the system / object
of study, say what people thought was happening, and say what your
project is investigating. Link any technical terms to Appendix A. -->

Language models read and write numbers as text fragments: a tokenizer splits
2024.5 into pieces and the model must reassemble the value from the pieces.
FoNE (Fourier Number Embedding) instead gives the model each number as one
token whose embedding encodes the exact value through Fourier features, one
cosine and sine pair per digit. Prior work showed FoNE reaches near-perfect
arithmetic when trained on synthetic arithmetic alone; whether it helps when
mixed into ordinary language pretraining is open. This project pretrains the
same model three ways on the same web plus math corpus (standard tokenizer,
FoNE with fixed frequencies, FoNE with learned frequencies) and asks whether
the FoNE runs win on number tasks.

→ Key terms: Appendix A.
→ How we measure success: Appendix B.

### Thesis

> A small model pretrained from scratch with FoNE on a web plus math corpus
> beats both its standard-tokenizer twin and much larger frontier models on
> multi-digit arithmetic and numeric-precision tasks, at equal pretrain token
> budget and a fraction of the compute.

### Prior work and what makes the question hard

The FoNE paper (arXiv:2502.09741) trained small models on synthetic arithmetic
only and reported near-perfect accuracy with far less data than digit-wise or
subword baselines. Google's TabFM used a learned-frequency variant inside a
tabular foundation model. Neither tested FoNE inside natural-language
pretraining, where numbers are sparse, noisy, and mixed with text, so the
gains could vanish once the embedding must share capacity with language.

| Theory | What it predicts in our pretrain comparison | Discriminating test |
|---|---|---|
| (1) FoNE gains transfer to pretraining | FoNE runs beat the baseline twin on number tasks at equal tokens | equal-budget three-variant comparison, §2 |
| (2) Gains are an artifact of arithmetic-only training | number-task gap shrinks to noise once data is mostly text | same comparison; theory 2 predicts no gap |

---

## Outline — N subgoals and how we achieve each

<!-- KEY PATTERN: the Outline is the paper's argument in compressed form.
Each row reads left-to-right as a self-contained chain:
  Question we asked → What we did to answer it → What we showed →
  Therefore (what it implies + where it leads next).

Rules for cells:
  - Plain language only. No symbol-only shorthand (no L=63, cos(δ, v),
    etc.). If you must use a technical term, define it elsewhere and
    refer by name.
  - Each cell is 1–3 short sentences. Detailed numbers stay in the
    section below; the Outline is the abstract.
  - "Therefore" must point at the *next* section (→ §N) or at the
    headline implication. This is what makes the Outline read as an arc
    instead of a list. -->

| § | Question | What we did | What we showed | Therefore → |
|---|---|---|---|---|
| **§1 Pipeline validity** *(planned)* | Does FoNE-in-pretraining train stably at all: does the mixed loss converge and does per-digit accuracy rise? | (planned) 125M-parameter runs of all three variants on ~3B tokens; watch loss curves and digit accuracy. | *(predicted)* All three train stably; FoNE digit accuracy climbs well above chance early. | The pipeline is sound; any §2 gap is real, not an artifact. → §2 |
| **§2 Equal-budget comparison** *(planned)* | At the same pretrain token budget, does FoNE beat its standard-tokenizer twin on number tasks? | (planned) 350M-parameter runs, 10B+ tokens, identical data and schedule; number eval suite on all three. | *(predicted, theory 1)* FoNE variants win by a wide margin on arithmetic exact match; falsifier: gap within noise means theory 2. | The embedding, not data or scale, causes the gain. → §3 |
| **§3 Frontier comparison** *(planned)* | Does the small FoNE model beat frontier company models on the same number tasks? | (planned) Run the identical eval suite on GPT / Claude / Gemini / open Llama; compare exact match by digit length. | *(predicted)* Frontier models degrade sharply past ~6 digits; the FoNE model stays near ceiling. | A 350M model wins on numeracy at a fraction of the compute: the headline. → §4 |
| **§4 Fixed vs learned frequencies** *(planned)* | Do learned Fourier frequencies (TabFM style) help or hurt relative to fixed powers of ten? | (planned) Same §2 protocol; the two FoNE variants differ only in the input featurizer. | *(predicted)* Parity or small gain for learned; falsifier: instability or a loss on precision tasks. | Guidance for how future models should adopt FoNE. |

---

## §1 Phenomenon ([status])

**Claim.** [One sentence stating what §1 proves — usually "the prior
finding replicates under our fairer measurement".]

### Evidence: [exp name] — [one-line description]

<!-- KEY PATTERN: body shows narrative, not data dumps.
Per the appendix rule in CLAUDE.md, keep only:
  (a) a headline sentence with the bottom-line number,
  (b) the plain-language reading,
  (c) any single caveat that changes the interpretation,
  (d) a pointer "→ See Appendix X" for the detail tables.

Per-bucket pass-rates, top-N rankings, full hyperparameter sweeps,
counter-example tables — ALL of these go in an experiment-specific
appendix, not here. -->

**Classification:** [Central | Sanity check | Supporting].

**Headline.** [One sentence with the bottom-line number, e.g., "15 / 39
[items] pass [threshold]. The other 24 are statistically indistinguishable
from [control]."]

**Plain-language reading.** [Theory 1 predicts X; theory 2 predicts Y.
Observed: [result]. There is a tendency for [pattern], but [within-bucket
counter-examples / caveats] prevent a clean rule.]

**[Caveat name].** [One sentence on the methodological caveat that
changes the obvious reading — e.g., "concrete-bucket failures are
off-distribution, not 'no'." Single caveats only; full diagnostic
numbers go in the appendix.]

→ See **Appendix C** for the per-bucket table, top-N passers, counter-
example pairs, and off-distribution diagnostics.

### Evidence: [exp name] — methodological control

**Classification:** Supporting (rules out [specific rebuttal]).

[1-paragraph result + caveat inline.]

### Context: [prior paper]

[How prior work measured the same phenomenon; why our setup differs.]

---

## §2 Controls — theory 1 vs theory 2 ([status])

**Claim.** [One sentence stating what §2 proves — the discriminator.]

### The discovery arc (read in order)

<!-- This subsection narrates how the controls developed. It is the
*reasoning chain* that turns observations into a discriminator. Format
each step as: observation → question → next experiment. -->

1. [Observation from §1 that motivated the first control.]
2. [Theory 1's prediction; theory 2's prediction.]
3. [The naive control + why it was unfair / inconclusive.]
4. [The refined control that closes the loophole.]
5. [Final discriminating result.]

### Discriminating evidence vs sanity checks

| Evidence | Classification | What it discriminates |
|---|---|---|
| [Exp B/01] | Central | Theory 1 predicts X; theory 2 predicts Y; observed Y. |
| [Exp B/02] | Sanity check | Both theories predict same outcome; confirms setup. |
| [Exp B/03] | Supporting | Robustness across [knob]. |

### Evidence: [exp name] — [central control]

**Classification:** Central.

<!-- This is where the main control result lives. Include:
  - the measurement
  - the prediction under each theory
  - the observed result
  - the verdict on which theory survives
Caveats inline (single seed? one setting? asymmetric comparison?). -->

[Result paragraph with table; verdict.]

### Context: [prior paper]

[How prior work would interpret the same numbers.]

---

## §3 Mechanism — Q&A format ([status])

<!-- KEY PATTERN: §3 (mechanism) should be structured as explicit
questions and answers, not a list of facts. Each question is
mechanistic; each answer is one sentence + citation; each maps to a
subsection.

Why Q&A: a reader scanning §3 should be able to extract the mechanism
claims without reading every sub-experiment. The questions ARE the
mechanism story; the experiments answer them. -->

The §2 controls established **what the trigger is not**. §3 asks the
mechanistic questions that remain. Each subsection answers one.

**Q1. Where in the system does the effect form?**
A: [One-sentence answer with citation.] [Exp C/01].

**Q2. Is [property X] sufficient to predict the effect across all
conditions?**
A: [No / Yes / partially]. [One-sentence finding with the discriminating
numbers.] [Exp C/02].

**Q3. Is [step A] alone enough, or is the [step A → step B] cascade
essential?**
A: [Cascade essential / step A sufficient]. [Direct test of step B alone
gives result X; only the full path triggers.] [Exp C/03].

**Q4. Is the trigger [feature 1] or [feature 2]?**
A: [feature 2]. [Falsification of feature-1 hypothesis with numbers.]
[Exp C/04 + C/05].

**Q5. Does the same circuit answer [related question]?**
A: [No / yes]. [Brief.] [Exp C/06].

**Subsection map (question → evidence):**

| § | Question | Evidence |
|---|---|---|
| §3.1 | Q1: where? | [Exp C/01] |
| §3.2 | (sets up Q2) | [Exp C/02] |
| §3.3 | Q2 | [Exp C/03] |
| §3.4 | Q3 | [Exp C/04] |
| §3.5 | Q4 | [Exp C/05 + C/06] |
| §3.6 | Q5 | [Exp C/07] |

### Evidence: [Exp C/01]

[Detail; classification; caveats inline.]

### Evidence: [Exp C/02]

[Detail.]

... (one per Q)

### §3 Synthesis: answers to Q1–Q5

<!-- Recap the Q&A answers in a single compact table, then ONE prose
paragraph that combines them into the mechanism story. -->

| Q | Answer | Evidence |
|---|---|---|
| Q1: where? | [one-phrase answer] | [Exp] |
| Q2: [property]? | [yes/no + key number] | [Exp] |
| Q3: cascade essential? | [yes/no] | [Exp] |
| Q4: direction or magnitude? | [feature 2] | [Exp] |
| Q5: one circuit or two? | [two/one] | [Exp] |

**Combined picture.** [1 paragraph synthesizing the mechanism. Cite the
subsections, do not re-introduce numbers already in the table.]

**What we explicitly do NOT claim:**

<!-- Calibration: name the natural over-readings of the data and
explicitly disavow them, with the evidence that rules each out. This
is how the document survives review. -->

- **Not** "[over-strong claim 1]." Falsified by [Exp / Q-number].
- **Not** "[over-strong claim 2]." Falsified by [Exp / Q-number].
- **Not** "[over-strong claim 3]." [Reason.]

### Context: [prior paper]

[How prior mechanistic work positions our findings.]

---

## §4 Origin — [research question] (planned)

<!-- §4 is typically forward-looking. Frame it as a *prediction to test*,
not a result. State what each theory predicts, then the experiment that
would discriminate. -->

**Prediction (to test).** [If our theory is right, the observable should
behave like X during training; under the alternative theory, it should
behave like Y.]

### Evidence: [Exp D/01] (planned; scaffold built)

[Method outline; compute cost; status.]

### Context: [prior paper]

[How prior work motivates this prediction.]

---

## Related Work

<!-- All external papers live here, not inline in subgoals. Each row
explains how the paper supports, contrasts, or is orthogonal to our
claims. -->

| Paper | Relation to our work |
|---|---|
| **[Paper A]** (Author, Year) | [Supports §1 by ...] |
| **[Paper B]** (Author, Year) | [Contrasts §2 by claiming X; we show Y.] |
| **[Paper C]** (Author, Year) | [Orthogonal — different measurement; mentioned because reviewers will ask.] |

---

<!-- Experiment status (Active runs / Recently completed / Recently
failed / Next steps) lives in **tracking.md** — this file holds the
argument only. -->

## Reframing for slides

<!-- A terse list of the same claims phrased for a 5-slide weekly
update. Used as raw material for slide decks; lets you check that the
slide story matches the document. -->

- §1: [one phrase — phenomenon claim].
- §2: [one phrase — control finding].
- §3: [one phrase — mechanism claim].
- §4: [one phrase — next thing to measure].
- Headline: [one sentence].

---

## Appendix A — Key terms

<!-- Define every technical term used in the body. The body links to
this appendix rather than defining inline. Keeps the main narrative
readable. -->

- **[Term 1]** — [definition + how it's measured].
- **[Term 2]** — [definition].
- **[Setup name]** — [the canonical experimental configuration (knobs +
  defaults). Refer to "canonical setup" in the body instead of repeating
  the values.]

---

## Appendix B — How to read the metrics

### [Primary metric]

<!-- One-paragraph plain-language explanation. State the formula and
what high/low values mean. -->

[Formula + what range counts as "trigger" / "no effect" / "ambiguous".]

### Common confusion: does [metric] < 0.5 mean [naive interpretation]?

<!-- This is the one section reviewers most often misread. Pre-empt it. -->

[No. Explanation of why the metric's scale is not what it looks like at
first glance.]

### Z-score against the reference distribution

<!-- If you use z-scores anywhere, define them once here, formula
included, with a sentence on what z=2 corresponds to. -->

z = ([observed metric] − [reference distribution mean]) / [reference distribution std].

z = 2 corresponds to the observation being more than 2 SDs above the
reference, i.e., not explained by random chance under the reference
distribution (one-tailed p ≈ 0.025). The reference distribution is built
from [Exp B/02]'s [N] samples.
