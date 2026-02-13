from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_github_actions_ci_workflow_exists() -> None:
    workflow = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert workflow.exists()
    content = workflow.read_text(encoding="utf-8")
    assert "pytest" in content
    assert "3.10" in content
    assert "3.11" in content
    assert "3.12" in content
