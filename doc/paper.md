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

Language models read and write numbers as text fragments. Modern tokenizers
(Llama-3, GPT-4) already split a number into chunks of up to three digits: 1234567
becomes the tokens 123, 456, 7. Each chunk is one ordinary vocabulary token whose
embedding the model learns from scratch, so nothing tells the model that the token
123 stands for the value one hundred twenty-three. FoNE (Fourier Number Embedding)
supplies that structure: it writes a Fourier code of the chunk's value into the first
few dimensions of that token's embedding, one cosine and sine pair per frequency, and
leaves the remaining dimensions free for the model to learn. Because the model ties its
input embedding and output layer to one weight matrix, the same code serves both when
the model reads a number and when it generates one. Prior work showed FoNE reaches
near-perfect arithmetic when trained on synthetic arithmetic alone; whether it helps
when mixed into ordinary language pretraining is open. This project pretrains the same
model three ways on the same web plus math corpus (plain learned embeddings, FoNE with
fixed frequencies, FoNE with learned frequencies) and asks whether the FoNE runs win on
number tasks.

→ Key terms: Appendix A.
→ How we measure success: Appendix B.

### Thesis

> A small model pretrained from scratch with chunk-based FoNE on a web plus math
> corpus beats both its plain-embedding twin (same tokenizer and data, learned
> number-token embeddings) and much larger frontier models on multi-digit arithmetic
> and numeric-precision tasks, at equal pretrain token budget and a fraction of the
> compute.

### Prior work and what makes the question hard

The FoNE paper (arXiv:2502.09741) trained small models on synthetic arithmetic
only and reported near-perfect accuracy with far less data than digit-wise or
subword baselines. It encoded each whole number in one embedding. We instead attach
the Fourier code to the tokenizer's existing three-digit chunk tokens, so the method
drops into a standard pretraining stack without a custom number token or an
output-side decode head. The code occupies only the first few dimensions of a chunk
token's embedding; the rest stay learned, so the number signal shares capacity with
language rather than replacing it. Google's TabFM used a learned-frequency variant
inside a tabular foundation model. Neither tested FoNE inside natural-language
pretraining, where numbers are sparse, noisy, and mixed with text, so the gains could
vanish once the embedding must share capacity with language.

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
| **§1 Pipeline validity** *(met)* | Does chunk-based FoNE-in-pretraining train stably at all: does the language-model loss converge with the Fourier code injected into number-chunk tokens? | 125M runs of all variants on ~3B tokens, same token stream; watched loss curves. | All variants train stably and reach the same held-out loss (2.86); the code does not destabilize training. | The pipeline is sound; any §2 gap is real, not an artifact. → §2 |
| **§2 Equal-budget comparison** *(partial: comparison task, 125M)* | At the same pretrain token budget and identical data, does FoNE beat its plain-embedding twin on number tasks? | 125M runs, ~3B tokens, one shared tokenization; number eval suite (arithmetic + number-magnitude comparison) across digit length, multi-seed. | *(theory 1)* FoNE beats baseline on comparison, and the gap grows with digit length (6d: 0.34 vs 0.07). Arithmetic is ~0 for all variants (scale floor), so it does not yet discriminate. | The embedding, not data, causes the gain on comparison. Larger-scale and arithmetic remain open. → §3 |
| **§3 Frontier comparison** *(planned)* | Does the small FoNE model beat frontier company models on the same number tasks? | (planned) Run the identical eval suite on GPT / Claude / Gemini / open Llama; compare exact match by digit length. | *(predicted)* Frontier models degrade sharply past ~6 digits; the FoNE model stays near ceiling. | A 350M model wins on numeracy at a fraction of the compute: the headline. → §4 |
| **§4 Fixed vs learned frequencies** *(planned)* | Do learned Fourier frequencies (TabFM style) help or hurt relative to fixed powers of ten? | (planned) Same §2 protocol; the fixed variant uses 3 periods (10, 100, 1000), the learned variant adds 20 more learnable periods. | *(predicted)* Parity or small gain for learned; falsifier: instability or a loss on precision tasks. | Guidance for how future models should adopt FoNE. |

---

## §1 Pipeline validity (met)

**Claim.** Chunk-based FoNE trains stably inside ordinary language pretraining: the
injected Fourier code does not destabilize training, and all variants converge together.

### Evidence: 125M three-variant runs, ~3B tokens

**Classification:** Sanity check (all variants predict the same outcome here; this
confirms the setup, it does not yet discriminate FoNE from baseline).

**Headline.** All variants (plain-embedding baseline, fixed FoNE, learned FoNE, and six
ablations) train to completion and reach the same held-out language-model loss, 2.86.
The learnable code scale holds the variants' starting loss within 0.02 of each other, so
they begin and end together.

**Plain-language reading.** Injecting a fixed Fourier code into the number-chunk token
rows, and sharing it with the output layer through the tied weight, is compatible with
standard pretraining. Nothing diverges or destabilizes.

**Caveat.** Held-out language-model loss is text-dominated, so it is identical across
variants and is not a measure of numeracy. Whether FoNE helps on number tasks is the
subject of §2, measured on a number-task suite rather than on language-model loss.

### Context: FoNE (arXiv:2502.09741)

The FoNE paper established that the Fourier value code trains well on synthetic
arithmetic. §1 confirms the chunk-based form trains well inside natural-language
pretraining, which is the setting the prior work did not test.

---

## §2 Equal-budget comparison (partial: number-magnitude task, 125M)

**Claim.** At equal token budget and identical data, FoNE beats its plain-embedding twin
on number-magnitude understanding, and the gain grows with digit length. Learning the
Fourier frequencies beats fixing them.

### The discovery arc (read in order)

1. §1 established all variants reach the same language-model loss, so that metric cannot
   separate them. We turned to a number-task suite: arithmetic (add, sub) and
   number-magnitude comparison ("between A and B, the larger is"), swept over digit length.
2. Theory 1 (FoNE gains transfer) predicts the FoNE variants beat baseline on number
   tasks. Theory 2 (gains are an arithmetic-only artifact) predicts no gap once data is
   mostly text.
3. Arithmetic exact match is ~0 for every variant at every digit length. A 125M model
   trained on 3B tokens of mostly text cannot generate multi-digit arithmetic answers, so
   arithmetic does not discriminate the theories at this scale (it is a scale floor, not a
   FoNE failure; verified by inspecting raw generations, which are complete wrong numbers,
   not truncations).
4. Comparison does discriminate: it needs only to read the two values and pick the larger,
   which is exactly what the input-side Fourier code should support. A first single-seed
   pass showed a signal but also showed run-to-run variance larger than the between-variant
   gaps (the same config at two seeds scored 0.16 and 0.36 average), so we repeated every
   variant across seeds and report the mean.

### Evidence: number-magnitude comparison across digit length

**Classification:** Central. Theory 1 predicts a FoNE advantage; theory 2 predicts none.
Observed: a FoNE advantage that grows with digit length.

Comparison exact match, mean over 3 seeds (extension to 6 seeds in progress). At 2 to 4
digits every variant is comparable: the baseline learns short numbers on its own. From 6
digits on, where subword-learned number embeddings degrade, the FoNE variants separate:

| variant (125M) | 2d | 4d | 6d | 8d | 10d | avg |
|---|---|---|---|---|---|---|
| baseline (learned embeddings) | 0.55 | 0.30 | 0.07 | 0.09 | 0.07 | 0.21 |
| FoNE, fixed 3 periods | 0.54 | 0.34 | 0.22 | 0.13 | 0.14 | 0.28 |
| FoNE, fixed 6 periods | 0.59 | 0.35 | 0.26 | 0.11 | 0.09 | 0.28 |
| **FoNE, learned frequencies** | 0.55 | 0.35 | **0.34** | **0.29** | **0.23** | **0.35** |

**Plain-language reading.** At 6 digits the learned-frequency FoNE model answers 34 of
100 comparisons correctly against the baseline's 7, and the gap holds at 8 and 10 digits
(0.29 vs 0.09, 0.23 vs 0.07). Theory 1 survives, theory 2 is rejected for this task.
Learning the frequencies is the design that matters: it beats both fixed-frequency
variants (average 0.35 vs 0.28), while adding fixed dimensions (3 to 6 periods) does not
help. This answers §4's question in the same experiment.

**Caveat.** This is one task (comparison) at one scale (125M), and the three-seed bands
are wide (standard deviation 0.10 to 0.22 per cell); the learned-FoNE advantage at 6 to
10 digits exceeds one standard deviation but the extension to six seeds is running to
tighten it. Arithmetic remains at the scale floor and is not yet a discriminator.

### Context: FoNE (arXiv:2502.09741)

Prior work reported the FoNE advantage on arithmetic under synthetic training. Here the
advantage appears on number-magnitude comparison under natural-language pretraining, and
specifically at the long digit lengths where subword number embeddings break down.

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
