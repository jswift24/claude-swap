from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_readme_matches_current_cli_surface() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "kimicc run" in readme
    assert "kimicc up" in readme
    assert "kimicc down" in readme
    assert "kimicc restart" in readme
    assert "kimicc doctor" in readme
    assert "kimicc health" in readme
    assert "kimicc config" in readme
    assert "~/.config/kimicc/config.yaml" in readme

    assert "--start" not in readme
    assert "--stop" not in readme
    assert "--status" not in readme
    assert "--logs" not in readme
