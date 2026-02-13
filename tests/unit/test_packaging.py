from __future__ import annotations

from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]


def _project_table() -> dict:
    pyproject = REPO_ROOT / "pyproject.toml"
    return tomllib.loads(pyproject.read_text())


def test_runtime_dependencies_are_bundled_for_end_users() -> None:
    data = _project_table()
    deps = data["project"]["dependencies"]
    dep_blob = "\n".join(deps).lower()

    assert "litellm" in dep_blob
    assert "platformdirs" in dep_blob
    assert "pyyaml" in dep_blob


def test_version_is_single_sourced_from_package() -> None:
    data = _project_table()
    project = data["project"]

    assert "version" not in project
    assert "dynamic" in project
    assert "version" in project["dynamic"]
    assert data["tool"]["hatch"]["version"]["path"] == "src/claude_swap/__init__.py"


def test_packaged_runtime_assets_exist() -> None:
    assert (REPO_ROOT / "src" / "claude_swap" / "data" / "litellm.yaml").exists()
    assert (REPO_ROOT / "LICENSE").exists()
