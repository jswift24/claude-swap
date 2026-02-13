from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_readme_matches_current_cli_surface() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "claude-swap run" in readme
    assert "claude-swap up" in readme
    assert "claude-swap down" in readme
    assert "claude-swap restart" in readme
    assert "claude-swap doctor" in readme
    assert "claude-swap health" in readme
    assert "claude-swap config" in readme
    assert "~/.config/claude-swap/config.yaml" in readme

    assert "--start" not in readme
    assert "--stop" not in readme
    assert "--status" not in readme
    assert "--logs" not in readme
