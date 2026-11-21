"""Every module must import. Catches files missing from git (e.g. caught by .gitignore)."""

import importlib
import pkgutil

import pitchlens


def test_all_modules_import():
    names = [m.name for m in pkgutil.walk_packages(pitchlens.__path__, "pitchlens.")]
    for name in names:
        importlib.import_module(name)
    # modules the CLI relies on must exist
    for required in ("pitchlens.data.datasets", "pitchlens.cli", "pitchlens.simulation.occlusion"):
        assert required in names, f"{required} missing: is it tracked in git?"
