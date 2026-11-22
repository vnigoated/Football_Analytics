"""Bundle the paper for Overleaf: python paper/make_overleaf_zip.py -> paper/paper_overleaf.zip"""

import zipfile
from pathlib import Path

root = Path(__file__).parent
with zipfile.ZipFile(root / "paper_overleaf.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in ["main.tex", "refs.bib", *sorted(str(p.relative_to(root)) for p in (root / "figures").glob("*.pdf"))]:
        z.write(root / f, f)
print("Saved", root / "paper_overleaf.zip")
