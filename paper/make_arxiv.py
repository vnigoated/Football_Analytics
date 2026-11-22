"""Build the arXiv submission zip: python paper/make_arxiv.py [path/to/llncs.cls]

arXiv compiles the LaTeX source itself but does not run BibTeX, so the bibliography is
shipped as ``main.bbl``. This script writes it from ``refs.bib`` in citation order, in
the style of Springer's ``splncs04`` (initials, ``Author: Title. Venue (year)``).

Output: paper/arxiv_submission.zip with main.tex, main.bbl, figures/*.pdf and, if given,
llncs.cls (Springer LNCS class, CC BY 4.0, from https://ctan.org/pkg/llncs).
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent


def parse_bib(text: str) -> dict[str, dict[str, str]]:
    """Minimal BibTeX parser (enough for refs.bib): handles nested braces in values."""
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        kind, key, i = m.group(1).lower(), m.group(2), m.end()
        fields, depth, start = {"_type": kind}, 1, i
        # find the end of the entry
        j = i
        while depth and j < len(text):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        body = text[start : j - 1]
        for fm in re.finditer(r"(\w+)\s*=\s*\{", body):
            k, p, d = fm.group(1).lower(), fm.end(), 1
            q = p
            while d:
                d += {"{": 1, "}": -1}.get(body[q], 0)
                q += 1
            fields[k] = " ".join(body[p : q - 1].split())
        entries[key] = fields
    return entries


def _strip_protect(s: str) -> str:
    """Remove BibTeX case-protection braces but keep LaTeX commands like \\url{...}."""
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\":
            m = re.match(r"\\[a-zA-Z]+\{[^}]*\}", s[i:])
            if m:
                out.append(m.group(0))
                i += len(m.group(0))
                continue
        if s[i] not in "{}":
            out.append(s[i])
        i += 1
    return "".join(out)


def format_name(name: str) -> str:
    name = name.strip()
    if name.startswith("{") and name.endswith("}"):
        return name[1:-1]  # corporate author
    if name == "others":
        return "et al."
    last, _, given = name.partition(",")
    initials = "".join(f"{p[0]}." for p in re.split(r"[\s~]+", given.strip()) if p)
    last = last.strip().replace(" ", "~")
    return f"{last}, {initials}" if initials else last


def format_authors(field: str) -> str:
    names = [format_name(n) for n in re.split(r"\s+and\s+", field)]
    if names and names[-1] == "et al.":
        return ", ".join(names[:-1]) + " et al."
    return ", ".join(names)


def format_entry(key: str, e: dict[str, str]) -> str:
    authors = format_authors(e["author"]) if "author" in e else ""
    title = _strip_protect(e.get("title", ""))
    year = e.get("year", "")
    parts = [f"{authors}: {title}." if authors else f"{title}."]
    kind = e["_type"]
    if kind == "article":
        venue = _strip_protect(e["journal"])
        vol = f" \\textbf{{{e['volume']}}}" if "volume" in e else ""
        num = f"({e['number']})" if "number" in e else ""
        pages = f", {e['pages']}" if "pages" in e else ""
        parts.append(f"{venue}{vol}{num}{pages} ({year})")
    elif kind == "inproceedings":
        parts.append(f"In: {_strip_protect(e['booktitle'])} ({year})")
        if "eprint" in e:
            parts[-1] += f", arXiv:{e['eprint']}"
    elif "eprint" in e:
        parts.append(f"arXiv preprint arXiv:{e['eprint']} ({year})")
    else:
        how = _strip_protect(e.get("howpublished", ""))
        parts.append(f"{how}" + (f" ({year})" if year else ""))
    text = " ".join(parts)
    if "doi" in e and not ("eprint" in e and "arXiv" in e["doi"]):
        text += f", \\doi{{{e['doi']}}}"
    if "note" in e:
        text += f", {_strip_protect(e['note'])}"
    return f"\\bibitem{{{key}}}\n{text}\n"


def cited_keys(tex: str) -> list[str]:
    keys = []
    for grp in re.findall(r"\\cite\{([^}]+)\}", tex):
        for k in grp.split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    return keys


def main() -> None:
    tex = (ROOT / "main.tex").read_text(encoding="utf-8")
    bib = parse_bib((ROOT / "refs.bib").read_text(encoding="utf-8"))
    keys = cited_keys(tex)
    missing = [k for k in keys if k not in bib]
    if missing:
        raise SystemExit(f"Cited but not in refs.bib: {missing}")
    bbl = [
        f"\\begin{{thebibliography}}{{{len(keys)}}}",
        "\\providecommand{\\url}[1]{\\texttt{#1}}",
        "\\providecommand{\\urlprefix}{URL }",
        "\\providecommand{\\doi}[1]{\\url{https://doi.org/#1}}",
        "",
    ]
    bbl += [format_entry(k, bib[k]) for k in keys]
    bbl.append("\\end{thebibliography}\n")
    (ROOT / "main.bbl").write_text("\n".join(bbl), encoding="utf-8")

    out = ROOT / "arxiv_submission.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "main.tex", "main.tex")
        z.write(ROOT / "main.bbl", "main.bbl")
        for f in sorted((ROOT / "figures").glob("*.pdf")):
            z.write(f, f"figures/{f.name}")
        if len(sys.argv) > 1:
            z.write(sys.argv[1], "llncs.cls")
    print(f"{len(keys)} references -> {ROOT / 'main.bbl'}")
    print("Saved", out, zipfile.ZipFile(out).namelist())


if __name__ == "__main__":
    main()
