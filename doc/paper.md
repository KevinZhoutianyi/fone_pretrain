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

[Plain-language setup paragraph: what's the system, what did prior work
claim, what is the operational question this project answers.]

→ Key terms: [linked to Appendix A].
→ How we measure success: [linked to Appendix B].

### Thesis

<!-- ONE sentence. The claim the whole document argues for. Should be
falsifiable and specific. -->

> [One-sentence thesis the paper argues for.]

### Prior work and what makes the question hard

<!-- Why the obvious experiment doesn't settle the question. What
confound or measurement ambiguity makes prior claims open to alternative
interpretations? Name the two (or more) theories that the same observable
data is consistent with — this sets up §1 phenomenon and §2 controls. -->

[2–4 sentences: prior claim + the measurement that supports it +
the alternative interpretation prior work did not rule out.]

| Theory | What it predicts in the standard experiment | What a discriminating test would look like |
|---|---|---|
| (1) [Strong claim from prior work] | [observable] | [test that would fail under theory 2] |
| (2) [Alternative interpretation] | [same observable, different mechanism] | [test that would fail under theory 1] |

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
| **§1 Phenomenon** | [Does the observation prior work claims actually hold under a fair measurement?] | [Re-ran the canonical experiment with the obvious confounds removed; tested at small scale for robustness.] | [The phenomenon is real but [varies with X]; not explained by [confound A or B].] | [The phenomenon exists, but the variation is what needs explaining. → §2] |
| **§2 Controls** | [What's actually causing the observed effect — theory 1 (concept-specific) or theory 2 (alternative)?] | [First tried the naive control X. When that failed, noticed property Y in the data. Built a refined control matching property Y. Stress-tested.] | [The naive control failed; the refined control reproduced the effect *without* the property theory 1 requires. Theory 1 falsified for the majority of cases.] | [The trigger is [property Y], not [theory 1's mechanism]. Open question: how does property Y produce the output? → §3] |
| **§3 Mechanism** | [N mechanistic questions in order: (1) where? (2) one feature or many? (3) is step A alone enough? (4) feature 1 or feature 2? (5) ...] | [Localization probe; single-axis predictor test; direct injection of the candidate feature; falsification test of a candidate alternative.] | [(1) Localized to [region X]. (2) Multi-dimensional. (3) [Step A] alone is *not* enough — the cascade through [steps B] is essential. (4) [feature 2], not [feature 1].] | [The effect is the system's normal output behavior conditioned on a perturbed state, not a dedicated detector. → §4] |
| **§4 Origin** *(planned)* | [When during training does the [readout-layer geometry / capability / mechanism] form? Is it [post-training-specific] or [already present in pretraining]?] | [(planned) Run §1 + §2 measurements across intermediate training checkpoints; compare emergence trajectory to other capabilities.] | *(predicted)* [The mechanism emerges with general-output capability during pretraining, not as a separate post-training milestone.] | [If predicted: the phenomenon is a side-effect of general capability geometry. If falsified: it's genuinely [post-training-specific]. Either way, sharpens the headline claim.] |

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
