from __future__ import annotations

from pathlib import Path

import pytest

from kimicc import cli


def test_build_parser_uses_explicit_subcommands() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["run", "--", "--dangerously-skip-permissions"])
    assert args.command == "run"
    assert args.claude_args == ["--dangerously-skip-permissions"]


def test_build_parser_includes_health_and_config_commands() -> None:
    parser = cli.build_parser()
    health_args = parser.parse_args(["health"])
    config_args = parser.parse_args(["config", "path"])
    config_edit_args = parser.parse_args(["config", "edit"])

    assert health_args.command == "health"
    assert config_args.command == "config"
    assert config_args.config_command == "path"
    assert config_edit_args.config_command == "edit"


def test_down_rejects_unused_host_flag() -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["down", "--host", "127.0.0.1"])


def test_load_user_config_creates_default_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    config = cli.load_user_config()

    assert config["model"] == "kimi-k2.5"
    assert config["ports"]["litellm"] == 4000
    assert config["ports"]["shim"] == 4001
    assert (tmp_path / "config" / "kimicc" / "config.yaml").exists()


def test_runtime_paths_are_user_scoped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    paths = cli.runtime_paths()

    assert str(paths.base_dir).startswith(str(tmp_path / "data" / "kimicc"))
    assert paths.litellm_pid.parent == paths.base_dir
    assert paths.shim_log.parent == paths.base_dir


def test_restart_restarts_background_services_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, list[str] | None]] = []

    monkeypatch.setattr(cli, "stop_services", lambda *_args, **_kwargs: calls.append(("stop", None)))
    monkeypatch.setattr(cli, "start_services", lambda *_args, **_kwargs: calls.append(("start", None)))
    monkeypatch.setattr(cli, "launch_claude", lambda _settings, args: calls.append(("launch", list(args))))

    exit_code = cli.main(["restart"])

    assert exit_code == 0
    assert calls == [("stop", None), ("start", None)]


def test_run_starts_services_and_launches_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, list[str] | None]] = []

    monkeypatch.setattr(cli, "stop_services", lambda *_args, **_kwargs: calls.append(("stop", None)))
    monkeypatch.setattr(cli, "start_services", lambda *_args, **_kwargs: calls.append(("start", None)))
    monkeypatch.setattr(cli, "launch_claude", lambda _settings, args: calls.append(("launch", list(args))))

    exit_code = cli.main(["run", "--", "--dangerously-skip-permissions"])

    assert exit_code == 0
    assert calls == [
        ("start", None),
        ("launch", ["--dangerously-skip-permissions"]),
    ]


def test_down_stops_services(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(cli, "stop_services", lambda *_args, **_kwargs: calls.append("stop"))

    exit_code = cli.main(["down"])

    assert exit_code == 0
    assert calls == ["stop"]


def test_doctor_accepts_ports_in_use_when_service_is_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    config = cli.load_user_config()
    settings = cli._settings_from(config, cli.build_parser().parse_args(["doctor"]))
    paths = cli.runtime_paths()

    monkeypatch.setattr(cli, "_check_command", lambda _name: (True, "/bin/fake"))
    monkeypatch.setattr(cli, "ensure_litellm_config", lambda: tmp_path / "config" / "kimicc" / "litellm.yaml")
    monkeypatch.setattr(cli, "_aws_signal", lambda _settings: (True, "ok"))

    def fake_is_running(pid_file: Path, _host: str, port: int) -> bool:
        return pid_file == paths.litellm_pid and port == settings.litellm_port

    def fake_port_available(_host: str, _port: int) -> tuple[bool, str]:
        return False, "Address already in use"

    monkeypatch.setattr(cli, "_is_running", fake_is_running)
    monkeypatch.setattr(cli, "_port_available", fake_port_available)

    assert cli.run_doctor(settings, paths) is False
    output = capsys.readouterr().out
    assert "[OK] port 4000:" in output
    assert "[FAIL] port 4001:" in output
