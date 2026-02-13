from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_built_wheel_contains_runtime_assets(tmp_path: Path) -> None:
    out_dir = tmp_path / "wheel"
    out_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", str(REPO_ROOT), "--no-deps", "-w", str(out_dir)],
        check=True,
        capture_output=True,
        text=True,
    )

    wheel_files = list(out_dir.glob("*.whl"))
    assert wheel_files, "wheel build did not produce a wheel"

    with zipfile.ZipFile(wheel_files[0]) as archive:
        names = set(archive.namelist())

    assert "kimicc/cli.py" in names
    assert "kimicc/shim.py" in names
    assert "kimicc/data/litellm.yaml" in names
    assert any(name.endswith("dist-info/licenses/LICENSE") for name in names)


def test_python_build_wheel_from_sdist_contains_package_code(tmp_path: Path) -> None:
    out_dir = tmp_path / "dist"
    out_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(out_dir)],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )

    wheel_files = list(out_dir.glob("*.whl"))
    assert wheel_files, "build did not produce a wheel"

    with zipfile.ZipFile(wheel_files[0]) as archive:
        names = set(archive.namelist())

    assert "kimicc/cli.py" in names
    assert "kimicc/shim.py" in names
