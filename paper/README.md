# Workshop paper draft

**"How Much Can You Trust Broadcast Pressing Data? Visibility Bias in Football Tracking Metrics"**, Springer LNCS format
(fits e.g. the ECML-PKDD MLSA workshop).

## Compile

- **Overleaf (easiest):** New Project → Upload Project → upload `paper_overleaf.zip`
  (build it with `python paper/make_overleaf_zip.py`), or upload `main.tex`, `refs.bib` and `figures/`.
  Overleaf already has `llncs.cls` and `splncs04.bst`. Compile with pdfLaTeX.
- **Locally:** install TeX Live or MiKTeX, then `latexmk -pdf main.tex`.

## Rebuild the figures

From the repo root, after running the experiments in `docs/results/`:

```bash
python paper/make_figures.py
```

## Before submission

- [x] Author and affiliation.
- [x] `refs.bib`: all metadata verified against arXiv and Crossref (2026-11-22).
- [ ] Make the repo public (the paper links to it); optionally archive a release with a Zenodo DOI.
- [ ] Check the workshop's page limit, anonymity rules and template version.
- [x] Read arXiv 2607.11548 in full and describe it accurately (2026-11-22).
- [x] Novelty search: no published work measures pressing on broadcast data (2026-11-22).
- [x] Metrica equal-visibility comparison with the fitted cameras.
- [x] Bias correction, strict leave-stadium-and-match-out evaluation (`pitchlens bias-correction`).

## Version 2: real-video section (2026-11-25)

- New Section 5, "From Real Broadcast Video": the pipeline and its official SoccerNet-GSR result (Table 4),
  counting visible players (B1), the correction with the video count (B2, Fig. 4), and the main-camera filter (B3).
  Every number comes from `docs/results/phase2_*.md`.
- The abstract, introduction (question 4, contribution iv), data, discussion, limitations and conclusion are updated.
  Six new references were checked against arXiv and Crossref.
- Fig. 4: `python paper/make_figures.py real-video`. It needs `outputs/b1_visible/counts_tracks.parquet` (from Drive)
  and `outputs/correction/correction_per_broadcast.csv`.
- [ ] Compile on Overleaf and check the page count against the venue's limit (the paper is now longer).
- [ ] Before an arXiv replacement (v2), make sure v1 has been announced.
