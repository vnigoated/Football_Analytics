# Journal version

**"How much can you trust broadcast pressing data? Visibility bias in team pressing metrics, its drivers, and a
correction validated on real broadcast video"**

- **Target:** *Science and Medicine in Football*.
- **Fallbacks:** *Journal of Quantitative Analysis in Sports*, *International Journal of Computer Science in Sport*.
- **Template:** `main.tex` uses the Taylor & Francis **Interact** layout with APA (author–date) references
  (`apacite`). `interact.cls` is not in the repo; it comes with Overleaf's template "Taylor & Francis LaTeX template
  for authors (Interact layout + American Psychological Association reference style)". `main_article.tex` is the
  same text in the generic `article` class, as a fallback for other journals.
- **Abstract:** 249 words, unstructured.
- The journal's own author-instructions page could not be read automatically (HTTP 403). Third-party summaries
  disagree on the reference style (APA, CSE author–date or Vancouver), so check the official page before submitting.

## What changed from the conference version (`paper/main.tex`)

- **Ground truth:** 10 matches (Metrica game 3 added) and 400 team-broadcasts. Every number is updated: Metrica
  −20.2%, correction 3.0%, video-count propagation 3.0%.
- **New results sections:**
  - robustness to teams (journal step 1);
  - a second league's cameras (step 3, Fig. 5a);
  - the count on an independent dataset (SportsMOT, step 4, Table 5);
  - broadcast composition and leak tolerance (step 5, Fig. 5b).
- **Structure:** Introduction / Methods / Results / Discussion / Conclusion. The Discussion has practical
  implications, a worked example of the correction, and its validity range.
- **New references:** SportsMOT and SoccerNet-v2, checked against arXiv (2026-11-26).

## Build

```bash
python paper/journal/make_journal_figures.py   # figures 2-5 from outputs/, figure 1 copied from paper/figures
python paper/journal/make_overleaf_zip.py      # -> paper/journal/smf_overleaf.zip
```

To compile on Overleaf:

1. Open the T&F Interact APA template.
2. Delete its sample `.tex` and `.bib` files.
3. Upload the zip contents.
4. Set `main.tex` as the main document and compile with pdfLaTeX.

The steps are also in `README_OVERLEAF.txt` inside the zip.

## To do before submission

- [ ] Step 6: replay and close-up recall of the main-camera filter on full broadcasts (SoccerNet_raw_HQ). It fills the
      two red TODOs in Results and Limitations.
- [ ] Check the journal's author guidelines: word limit, abstract length and format, reference style,
      figure/table limits, title page, anonymised manuscript, and whether a highlights or practical-applications box
      is required.
- [ ] Funding statement.
- [ ] Zenodo release of the code with a DOI, cited in the data availability statement.
- [ ] Read the whole text once more against `docs/PAPER_V2_NOTES.md` rows 1–25.
- [ ] Optional: a co-author with football practice (a club analyst or a supervisor).
