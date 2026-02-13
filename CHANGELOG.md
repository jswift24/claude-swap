# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-02-13

### Added
- `cbridge health` runtime health checks.
- `cbridge config path|show|init|edit` configuration management commands.
- GitHub Actions CI for Python 3.10, 3.11, 3.12.
- Release workflow for building and validating distribution artifacts on tags.
- Developer extras (`ruff`, `mypy`, `build`, `twine`, `types-PyYAML`).

### Changed
- Improved `cbridge doctor` handling when cbridge services already own configured ports.
- Added actionable failure suggestions in `cbridge doctor` output.
- Improved service shutdown with graceful wait and force-stop fallback.
- Tightened dependency version ranges in `pyproject.toml`.
- Switched runtime dependency from `uvicorn[standard]` to `uvicorn`.
- Updated README command and contributor setup instructions.
