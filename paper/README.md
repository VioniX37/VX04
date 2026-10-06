# Paper

LaTeX source of the paper describing this project. Every number in it comes from `backend/evaluation/results/local_large` (large-data study), `evaluation.audit_ablation` and `evaluation.forecasting_benchmark`; see [Results](../docs/research/results.md).

| Path | Content |
|---|---|
| `main.tex` | Preamble and section includes (standard `article` class) |
| `sections/*.tex` | One file per section; all sections are complete |
| `refs.bib` | Bibliography |
| `figures/` | Figures written by `python -m evaluation.analysis evaluation/results/local_large --figures ../paper/figures` (run from `backend/`) |

## Build

- **Overleaf:** upload the `paper/` folder and set `main.tex` as the main document.
- **Locally:** `latexmk -pdf main.tex` (requires TeX Live or MiKTeX).

Figures that have not been generated yet appear as labelled placeholders, so the document always compiles. The pipeline figure is drawn with TikZ.

To submit to a venue, replace the preamble in `main.tex` with the venue's style file. The section files stay unchanged.
