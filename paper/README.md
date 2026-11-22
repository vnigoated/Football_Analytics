# Workshop paper draft

**"How Much Can You Trust Broadcast Pressing Data? Visibility Bias in Football Tracking Metrics"**, Springer LNCS format
(fits e.g. the ECML-PKDD MLSA workshop).

## Compile

- **Overleaf (easiest):** New Project → Upload Project → upload `main.tex`, `refs.bib` and the `figures/` folder.
  Overleaf already has `llncs.cls` and `splncs04.bst`. Compile with pdfLaTeX.
- **Locally:** install TeX Live or MiKTeX, then `latexmk -pdf main.tex`.

## Rebuild the figures

From the repo root, after running the experiments in `docs/results/`:

```bash
python paper/make_figures.py
```

## Before submission

- Search the PDF for red **[TODO: …]** markers and resolve each one.
- Check every entry in `refs.bib` marked `TODO verify`.
- Read arXiv 2607.11548 in full and check how we describe it.
- Repeat the Metrica equal-visibility comparison with the fitted camera.
- Fit and validate the bias correction (leave-one-stadium-out), or drop the claim.
- Make the repo public and archive a release (Zenodo DOI); fill in authors and affiliations.
- Check the workshop's page limit, anonymity rules and template version.
