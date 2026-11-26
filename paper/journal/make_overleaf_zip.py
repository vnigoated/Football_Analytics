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
3. Unzip this file on your computer first (Overleaf does not unpack a zip uploaded into an existing project).
   Upload main.tex (replace the template's main.tex) and refs.bib. Create a folder "figures" in the file tree and
   upload the 5 PDFs into it.
4. Menu -> Main document: main.tex. Compiler: pdfLaTeX. Recompile.

No TODOs are left. Line numbers are off; uncomment \\linenumbers in main.tex if the journal asks for them.
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
