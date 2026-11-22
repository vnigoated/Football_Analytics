"""Every Python cell in the Colab notebooks must be valid Python (IPython magics excluded)."""

import ast
import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted(Path(__file__).parents[1].glob("colab/*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_cells_parse(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if src.lstrip().startswith("%%"):  # whole-cell magic, e.g. %%bash
            continue
        # replace line magics / shell lines with `pass`, keeping indentation
        lines = [
            ln[: len(ln) - len(ln.lstrip())] + "pass" if ln.lstrip().startswith(("%", "!")) else ln
            for ln in src.splitlines()
        ]
        try:
            ast.parse("\n".join(lines))
        except SyntaxError as e:
            pytest.fail(f"{path.name} cell {i}: {e}")


def test_notebook_has_no_outputs():
    for path in NOTEBOOKS:
        nb = json.loads(path.read_text(encoding="utf-8"))
        assert not any(c.get("outputs") for c in nb["cells"]), f"{path.name}: clear outputs"
