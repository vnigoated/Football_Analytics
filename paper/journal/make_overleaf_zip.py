"""Bundle the journal version for Overleaf: python paper/journal/make_overleaf_zip.py -> paper/journal/journal_overleaf.zip"""

import zipfile
from pathlib import Path

root = Path(__file__).parent
with zipfile.ZipFile(root / "journal_overleaf.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(root / "main.tex", "main.tex")
    z.write(root.parent / "refs.bib", "refs.bib")
    for p in sorted((root / "figures").glob("*.pdf")):
        z.write(p, f"figures/{p.name}")
print("Saved", root / "journal_overleaf.zip")
