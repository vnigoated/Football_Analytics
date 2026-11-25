"""Bundle the journal version for Overleaf.

    python paper/journal/make_overleaf_zip.py

-> paper/journal/smf_overleaf.zip: the Science and Medicine in Football manuscript (Taylor & Francis Interact
layout, APA references): main.tex, refs.bib, figures/ and README_OVERLEAF.txt. interact.cls is not included; it
comes with Overleaf's Taylor & Francis template (see the README inside the zip).
"""

import zipfile
from pathlib import Path

README = """How to compile (Overleaf)
=========================

1. Overleaf -> New Project -> Templates -> search "Taylor & Francis Interact APA" ->
   "Taylor & Francis LaTeX template for authors (Interact layout + American Psychological Association reference
   style)" -> Open as Template.
2. In that project, delete interactapasample.tex and interactapasample.bib (keep interact.cls and the other
   class/style files).
3. Upload -> select every file of this zip (main.tex, refs.bib, README_OVERLEAF.txt) and the figures folder
   (or upload the zip and let Overleaf extract it).
4. Menu -> Main document: main.tex. Compiler: pdfLaTeX. Recompile.

Red [TODO: ...] marks are still open: journal step 6 (replay filter), funding statement, Zenodo DOI.
Line numbers are for review; remove \\linenumbers only for a reading copy.
"""

root = Path(__file__).parent
out = root / "smf_overleaf.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(root / "main.tex", "main.tex")
    z.write(root.parent / "refs.bib", "refs.bib")
    z.writestr("README_OVERLEAF.txt", README)
    for p in sorted((root / "figures").glob("*.pdf")):
        z.write(p, f"figures/{p.name}")
print("Saved", out)
