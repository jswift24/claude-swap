from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_readme_matches_current_cli_surface() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "cbridge run" in readme
    assert "cbridge up" in readme
    assert "cbridge down" in readme
    assert "cbridge restart" in readme
    assert "cbridge doctor" in readme
    assert "cbridge health" in readme
    assert "cbridge config" in readme
    assert "~/.config/cbridge/config.yaml" in readme

    assert "--start" not in readme
    assert "--stop" not in readme
    assert "--status" not in readme
    assert "--logs" not in readme
