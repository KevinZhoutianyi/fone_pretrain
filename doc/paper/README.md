# doc/paper/ — LaTeX version

LaTeX paper draft, sibling to [`../paper.md`](../paper.md) (the living
markdown argument document). `main.tex` mirrors the section structure
of `doc/paper.md`; figures are generated from
`experiments/*/results.json` into this folder.

## Contents

```
doc/paper/
├── main.tex              ← paper body, sections inline (mirrors doc/paper.md)
├── references.bib        ← BibTeX entries; one per paper in doc/related_papers/
├── figures/
│   ├── make_figures.py   ← reads experiments/*/results.json, writes PDF figures
│   ├── fig1_*.pdf        ← generated figures
│   └── ...
└── README.md             ← this file
```

## Building locally

```bash
cd doc/paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

This produces `main.pdf` (gitignored). For the workflow that includes
visually auditing the rendered PDF, see the "Render-and-read workflow"
section in [`../CLAUDE.md`](../CLAUDE.md).

## Importing to Overleaf

From the repo root, bundle a minimal upload archive:

```bash
cd doc/paper
zip <project>_overleaf.zip main.tex references.bib README.md figures/*.pdf
```

Upload the resulting zip via Overleaf's **New Project → Upload Project**.
Compiler: pdfLaTeX (default). Main document: `main.tex`. The zip itself
is gitignored — regenerate as needed.

## Regenerating figures

```bash
uv run python doc/paper/figures/make_figures.py
```

`make_figures.py` reads results from the `experiments/` tree (paths
relative to the repo root) and writes PDFs into `doc/paper/figures/`.
Each `fig*` function is self-contained — you can comment out figures
you don't need to regenerate.

## How the docs relate

| file | purpose |
|---|---|
| `doc/paper.md` | Living markdown argument: thesis, evidence, what we do NOT claim. Updated continuously as experiments land. |
| `doc/paper/main.tex` | Polished LaTeX version for Overleaf / submission. Lags `doc/paper.md` slightly — rebuilds when a section is publication-ready. |
| `doc/CLAUDE.md` | The rule set for both. |

When a finding lands, update `doc/paper.md` first (single source of
truth); promote to `doc/paper/main.tex` once the section is stable.
