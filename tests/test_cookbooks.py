"""Every cookbook must run top to bottom: they are the documentation users follow.

Skip with ``pytest -m "not cookbook"`` for a quick run.
"""
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

COOKBOOKS = sorted((Path(__file__).resolve().parents[1] / "cookbooks").glob("*.ipynb"))


def test_cookbooks_exist():
    assert len(COOKBOOKS) >= 8


@pytest.mark.cookbook
@pytest.mark.parametrize("path", COOKBOOKS, ids=[p.stem for p in COOKBOOKS])
def test_cookbook_runs(path: Path, tmp_path: Path):
    nb = nbformat.read(path, as_version=4)
    # run in a temporary folder so files the notebook saves don't land in the repo
    client = nbclient.NotebookClient(nb, timeout=600, kernel_name="python3",
                                     resources={"metadata": {"path": str(tmp_path)}})
    client.execute()
    errors = [o for c in nb.cells if c.cell_type == "code"
              for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors, errors
