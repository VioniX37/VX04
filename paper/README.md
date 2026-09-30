# Paper

LaTeX source of the paper describing this project.

| Path | Content |
|---|---|
| `main.tex` | Preamble and section includes (standard `article` class) |
| `sections/*.tex` | One file per section; `\todo{...}` marks what remains |
| `refs.bib` | Bibliography |
| `figures/` | Figures written by `python -m evaluation.analysis <results> --figures ../paper/figures` |

## Build

- **Overleaf:** upload the `paper/` folder and set `main.tex` as the main document.
- **Locally:** `latexmk -pdf main.tex` (requires TeX Live or MiKTeX).

Figures that have not been generated yet appear as labelled placeholders, so the document always compiles.

To submit to a venue, replace the preamble in `main.tex` with the venue's style file. The section files stay unchanged.
