"""CLI entry point for cbridge."""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

import httpx
import yaml
from platformdirs import user_config_path, user_data_path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_MODEL = "kimi-k2.5"
DEFAULT_LITELLM_PORT = 4000
DEFAULT_SHIM_PORT = 4001
DEFAULT_LOG_LINES = 50


@dataclass(frozen=True)
class RuntimePaths:
    base_dir: Path
    litellm_pid: Path
    shim_pid: Path
    litellm_log: Path
    shim_log: Path


@dataclass(frozen=True)
class Settings:
    host: str
    model: str
    litellm_port: int
    shim_port: int
    aws_profile: str | None
    default_claude_args: list[str]
    log_lines: int


def _default_config() -> dict[str, Any]:
    return {
        "host": DEFAULT_HOST,
        "model": DEFAULT_MODEL,
        "aws_profile": None,
        "ports": {
            "litellm": DEFAULT_LITELLM_PORT,
            "shim": DEFAULT_SHIM_PORT,
        },
        "claude_args": [],
        "logs": {"lines": DEFAULT_LOG_LINES},
    }


def _config_dir() -> Path:
    return Path(user_config_path("cbridge", appauthor=False))


def _config_file() -> Path:
    return _config_dir() / "config.yaml"


def _merge_dict(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_user_config() -> dict[str, Any]:
    """Load user config and create defaults on first run."""
    config_path = _config_file()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    defaults = _default_config()

    if not config_path.exists():
        config_path.write_text(yaml.safe_dump(defaults, sort_keys=False), encoding="utf-8")
        return defaults

    raw = config_path.read_text(encoding="utf-8").strip()
    if not raw:
        config_path.write_text(yaml.safe_dump(defaults, sort_keys=False), encoding="utf-8")
        return defaults

    parsed = yaml.safe_load(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Invalid config file format: {config_path}")

    merged = _merge_dict(defaults, parsed)
    return merged


def runtime_paths() -> RuntimePaths:
    """Return user-scoped runtime file paths."""
    base_dir = Path(user_data_path("cbridge", appauthor=False))
    base_dir.mkdir(parents=True, exist_ok=True)
    return RuntimePaths(
        base_dir=base_dir,
        litellm_pid=base_dir / "litellm.pid",
        shim_pid=base_dir / "shim.pid",
        litellm_log=base_dir / "litellm.log",
        shim_log=base_dir / "shim.log",
    )


def _packaged_litellm_template() -> str:
    return resources.files("cbridge.data").joinpath("litellm.yaml").read_text(encoding="utf-8")


def ensure_litellm_config() -> Path:
    """Ensure user-local litellm config exists and return its path."""
    litellm_config = _config_dir() / "litellm.yaml"
    litellm_config.parent.mkdir(parents=True, exist_ok=True)

    if not litellm_config.exists():
        litellm_config.write_text(_packaged_litellm_template(), encoding="utf-8")

    return litellm_config


def _port_listening(host: str, port: int, timeout_s: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        return sock.connect_ex((host, port)) == 0


def _is_running(pid_file: Path, host: str, port: int) -> bool:
    if not pid_file.exists():
        return False

    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    except (OSError, ProcessLookupError, ValueError):
        return False

    return _port_listening(host, port, timeout_s=1.5)


def _wait_for_ready(pid_file: Path, host: str, port: int, timeout_s: float, proc: subprocess.Popen[Any]) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _is_running(pid_file, host, port):
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.5)
    return False


def _tail_file(path: Path, lines: int) -> str:
    if not path.exists():
        return "(no log)"

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        buf = deque(handle, maxlen=lines)
    return "".join(buf).rstrip() or "(empty log)"


def _check_command(name: str) -> tuple[bool, str]:
    found = shutil.which(name)
    if found:
        return True, found
    return False, "not found in PATH"


def _port_available(host: str, port: int) -> tuple[bool, str]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            return False, str(exc)
    return True, "available"


def _http_ok(url: str, timeout_s: float = 2.0) -> tuple[bool, str]:
    try:
        response = httpx.get(url, timeout=timeout_s)
    except httpx.HTTPError as exc:
        return False, str(exc)

    if response.status_code >= 400:
        return False, f"HTTP {response.status_code}"
    return True, f"HTTP {response.status_code}"


def _aws_signal(settings: Settings) -> tuple[bool, str]:
    if settings.aws_profile:
        return True, f"using profile '{settings.aws_profile}'"

    env_profile = os.getenv("AWS_PROFILE")
    if env_profile:
        return True, f"AWS_PROFILE={env_profile}"

    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    if access_key and secret:
        return True, "using AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY"

    return False, "no AWS profile or keypair detected"


def _doctor_port_check(
    pid_file: Path, service_name: str, host: str, port: int
) -> tuple[bool, str]:
    if _is_running(pid_file, host, port):
        return True, f"in use by running {service_name}"
    return _port_available(host, port)


def emit_env(settings: Settings) -> str:
    """Emit environment variables for Claude."""
    return (
        f'export ANTHROPIC_BASE_URL="http://{settings.host}:{settings.shim_port}"\n'
        f'export ANTHROPIC_MODEL="{settings.model}"\n'
        'export ANTHROPIC_API_KEY="sk-litellm-local"\n'
        "unset CLAUDE_CODE_USE_BEDROCK\n"
        "unset CLAUDE_CODE_USE_VERTEX\n"
        "unset CLAUDE_CODE_USE_FOUNDRY\n"
        "export CLAUDE_CODE_SKIP_BEDROCK_AUTH=1\n"
        "export CLAUDE_CODE_SKIP_VERTEX_AUTH=1\n"
        "export CLAUDE_CODE_SKIP_FOUNDRY_AUTH=1\n"
        "export CLAUDE_CODE_DISABLE_FINE_GRAINED_TOOL_STREAMING=1\n"
    )


def _apply_env_exports(settings: Settings) -> None:
    os.environ["ANTHROPIC_BASE_URL"] = f"http://{settings.host}:{settings.shim_port}"
    os.environ["ANTHROPIC_MODEL"] = settings.model
    os.environ["ANTHROPIC_API_KEY"] = "sk-litellm-local"
    os.environ.pop("CLAUDE_CODE_USE_BEDROCK", None)
    os.environ.pop("CLAUDE_CODE_USE_VERTEX", None)
    os.environ.pop("CLAUDE_CODE_USE_FOUNDRY", None)
    os.environ["CLAUDE_CODE_SKIP_BEDROCK_AUTH"] = "1"
    os.environ["CLAUDE_CODE_SKIP_VERTEX_AUTH"] = "1"
    os.environ["CLAUDE_CODE_SKIP_FOUNDRY_AUTH"] = "1"
    os.environ["CLAUDE_CODE_DISABLE_FINE_GRAINED_TOOL_STREAMING"] = "1"


def launch_claude(settings: Settings, claude_args: Iterable[str]) -> None:
    """Launch Claude with the configured environment."""
    _apply_env_exports(settings)
    args = list(claude_args)

    print("\nServices ready. Launching Claude with cbridge backend...")
    if args:
        print(f"  Claude arguments: {' '.join(args)}")

    os.execvp("claude", ["claude", *args])


def start_litellm(settings: Settings, paths: RuntimePaths) -> None:
    """Start LiteLLM if needed."""
    if _is_running(paths.litellm_pid, settings.host, settings.litellm_port):
        pid = int(paths.litellm_pid.read_text(encoding="utf-8").strip())
        print(f"LiteLLM already running (pid {pid})")
        return

    litellm_bin = shutil.which("litellm")
    if not litellm_bin:
        raise RuntimeError("litellm executable not found in PATH")

    config_path = ensure_litellm_config()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if settings.aws_profile:
        env["AWS_PROFILE"] = settings.aws_profile

    log_handle = paths.litellm_log.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        [
            litellm_bin,
            "--config",
            str(config_path),
            "--host",
            settings.host,
            "--port",
            str(settings.litellm_port),
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
    )

    paths.litellm_pid.write_text(str(proc.pid), encoding="utf-8")

    if not _wait_for_ready(paths.litellm_pid, settings.host, settings.litellm_port, 30.0, proc):
        raise RuntimeError(f"LiteLLM failed to start. Check {paths.litellm_log}")

    print(f"LiteLLM ready (pid {proc.pid})")


def start_shim(settings: Settings, paths: RuntimePaths) -> None:
    """Start shim if needed."""
    if _is_running(paths.shim_pid, settings.host, settings.shim_port):
        pid = int(paths.shim_pid.read_text(encoding="utf-8").strip())
        print(f"Shim already running (pid {pid})")
        return

    env = os.environ.copy()
    env["LITELLM_BASE_URL"] = f"http://{settings.host}:{settings.litellm_port}"
    env["SHIM_HOST"] = settings.host
    env["SHIM_PORT"] = str(settings.shim_port)
    env["SHIM_DEFAULT_MODEL"] = settings.model
    env["PYTHONUNBUFFERED"] = "1"

    log_handle = paths.shim_log.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "cbridge.shim:app",
            "--host",
            settings.host,
            "--port",
            str(settings.shim_port),
            "--log-level",
            "warning",
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
    )

    paths.shim_pid.write_text(str(proc.pid), encoding="utf-8")

    if not _wait_for_ready(paths.shim_pid, settings.host, settings.shim_port, 20.0, proc):
        raise RuntimeError(f"Shim failed to start. Check {paths.shim_log}")

    print(f"Shim ready (pid {proc.pid})")


def start_services(settings: Settings, paths: RuntimePaths | None = None) -> None:
    """Start all background services."""
    paths = paths or runtime_paths()
    start_litellm(settings, paths)
    start_shim(settings, paths)


def _stop_pid(pid_file: Path, name: str) -> None:
    if not pid_file.exists():
        return

    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        pid_file.unlink(missing_ok=True)
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        return

    deadline = time.time() + 3.0
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            print(f"Stopped {name} (pid {pid})")
            pid_file.unlink(missing_ok=True)
            return
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
        print(f"Force-stopped {name} (pid {pid})")
    except ProcessLookupError:
        print(f"Stopped {name} (pid {pid})")
    finally:
        pid_file.unlink(missing_ok=True)


def stop_services(paths: RuntimePaths | None = None) -> None:
    """Stop all background services."""
    paths = paths or runtime_paths()
    _stop_pid(paths.shim_pid, "Shim")
    _stop_pid(paths.litellm_pid, "LiteLLM")


def show_status(settings: Settings, paths: RuntimePaths | None = None) -> None:
    """Print service status."""
    paths = paths or runtime_paths()

    litellm_running = _is_running(paths.litellm_pid, settings.host, settings.litellm_port)
    shim_running = _is_running(paths.shim_pid, settings.host, settings.shim_port)

    if litellm_running:
        litellm_pid = paths.litellm_pid.read_text(encoding="utf-8").strip()
        print(f"LiteLLM: running (pid {litellm_pid})")
    else:
        print("LiteLLM: stopped")

    if shim_running:
        shim_pid = paths.shim_pid.read_text(encoding="utf-8").strip()
        print(f"Shim:    running (pid {shim_pid})")
    else:
        print("Shim:    stopped")


def show_logs(paths: RuntimePaths | None = None, lines: int = DEFAULT_LOG_LINES) -> None:
    """Print tail of service logs."""
    paths = paths or runtime_paths()

    print("=== LiteLLM ===")
    print(_tail_file(paths.litellm_log, lines))
    print("\n=== Shim ===")
    print(_tail_file(paths.shim_log, lines))


def run_doctor(settings: Settings, paths: RuntimePaths | None = None) -> bool:
    """Run environment checks and return success status."""
    paths = paths or runtime_paths()

    checks: list[tuple[str, bool, str]] = []

    litellm_ok, litellm_msg = _check_command("litellm")
    checks.append(("litellm executable", litellm_ok, litellm_msg))

    claude_ok, claude_msg = _check_command("claude")
    checks.append(("claude executable", claude_ok, claude_msg))

    config_path = _config_file()
    checks.append(("config file", config_path.exists(), str(config_path)))

    template_ok = True
    try:
        ensure_litellm_config()
        template_msg = str(_config_dir() / "litellm.yaml")
    except Exception as exc:
        template_ok = False
        template_msg = str(exc)
    checks.append(("litellm config", template_ok, template_msg))

    data_dir_ok = paths.base_dir.exists() and os.access(paths.base_dir, os.W_OK)
    checks.append(("runtime dir writable", data_dir_ok, str(paths.base_dir)))

    port1_ok, port1_msg = _doctor_port_check(
        paths.litellm_pid, "LiteLLM", settings.host, settings.litellm_port
    )
    checks.append((f"port {settings.litellm_port}", port1_ok, port1_msg))

    port2_ok, port2_msg = _doctor_port_check(
        paths.shim_pid, "shim", settings.host, settings.shim_port
    )
    checks.append((f"port {settings.shim_port}", port2_ok, port2_msg))

    aws_ok, aws_msg = _aws_signal(settings)
    checks.append(("AWS credential signal", aws_ok, aws_msg))

    all_good = True
    for name, ok, detail in checks:
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not ok:
            all_good = False

    if not all_good:
        print("\nSuggested fixes:")
        if not litellm_ok:
            print("- Install cbridge in a venv or pipx so `litellm` is on PATH.")
        if not claude_ok:
            print("- Install Claude Code CLI and confirm `claude` is on PATH.")
        if not port1_ok or not port2_ok:
            print("- Run `cbridge status` and `cbridge down`, or change configured ports.")
        if not aws_ok:
            print("- Set `AWS_PROFILE` or `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`.")

    return all_good


def run_health(settings: Settings, paths: RuntimePaths | None = None) -> bool:
    """Run runtime health checks against the local services."""
    paths = paths or runtime_paths()
    checks: list[tuple[str, bool, str]] = []

    litellm_running = _is_running(paths.litellm_pid, settings.host, settings.litellm_port)
    checks.append(("LiteLLM process", litellm_running, "running" if litellm_running else "stopped"))

    litellm_port_ok = _port_listening(settings.host, settings.litellm_port)
    checks.append(("LiteLLM port", litellm_port_ok, f"{settings.host}:{settings.litellm_port}"))

    shim_running = _is_running(paths.shim_pid, settings.host, settings.shim_port)
    checks.append(("Shim process", shim_running, "running" if shim_running else "stopped"))

    shim_health_ok, shim_health_msg = _http_ok(f"http://{settings.host}:{settings.shim_port}/health")
    checks.append(("Shim health endpoint", shim_health_ok, shim_health_msg))

    all_good = True
    for name, ok, detail in checks:
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not ok:
            all_good = False

    return all_good


def config_command(command: str) -> int:
    """Run config subcommands."""
    config_path = _config_file()
    if command == "path":
        print(config_path)
        return 0

    if command == "show":
        cfg = load_user_config()
        print(yaml.safe_dump(cfg, sort_keys=False).rstrip())
        return 0

    if command == "init":
        cfg = _default_config()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        ensure_litellm_config()
        print(f"Initialized config at {config_path}")
        return 0

    if command == "edit":
        load_user_config()
        editor = os.getenv("EDITOR") or os.getenv("VISUAL")
        if not editor:
            print("ERROR: set EDITOR or VISUAL to use `cbridge config edit`", file=sys.stderr)
            return 1
        cmd = [*shlex.split(editor), str(config_path)]
        try:
            return subprocess.run(cmd, check=False).returncode
        except FileNotFoundError:
            print(f"ERROR: editor not found: {editor}", file=sys.stderr)
            return 1

    print(f"ERROR: unknown config subcommand '{command}'", file=sys.stderr)
    return 2


def _split_default_claude_args(config_args: list[str]) -> list[str]:
    env_default = os.getenv("CLAUDE_CODE_ARGS", "")
    from_env = shlex.split(env_default) if env_default else []
    return [*config_args, *from_env]


def _settings_from(config: dict[str, Any], args: argparse.Namespace) -> Settings:
    host = getattr(args, "host", None) or config.get("host", DEFAULT_HOST)
    model = getattr(args, "model", None) or config.get("model", DEFAULT_MODEL)

    config_ports = config.get("ports") or {}
    litellm_port = getattr(args, "litellm_port", None) or int(config_ports.get("litellm", DEFAULT_LITELLM_PORT))
    shim_port = getattr(args, "shim_port", None) or int(config_ports.get("shim", DEFAULT_SHIM_PORT))

    aws_profile = (
        getattr(args, "aws_profile", None)
        if getattr(args, "aws_profile", None) is not None
        else config.get("aws_profile")
    )

    logs_cfg = config.get("logs") or {}
    log_lines = int(getattr(args, "lines", None) or logs_cfg.get("lines", DEFAULT_LOG_LINES))

    config_claude_args = config.get("claude_args", [])
    if not isinstance(config_claude_args, list):
        config_claude_args = []

    default_claude_args = _split_default_claude_args([str(x) for x in config_claude_args])

    return Settings(
        host=str(host),
        model=str(model),
        litellm_port=litellm_port,
        shim_port=shim_port,
        aws_profile=str(aws_profile) if aws_profile else None,
        default_claude_args=default_claude_args,
        log_lines=log_lines,
    )


def _add_network_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default=None, help="Host for LiteLLM and shim (default from config)")
    parser.add_argument("--model", default=None, help="Model name exposed to Claude (default from config)")
    parser.add_argument("--litellm-port", type=int, default=None, help="LiteLLM port override")
    parser.add_argument("--shim-port", type=int, default=None, help="Shim port override")


def _add_aws_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--aws-profile", default=None, help="AWS profile override")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser with explicit subcommands."""
    parser = argparse.ArgumentParser(
        prog="cbridge",
        description="Run Claude Code with a local Kimi-compatible shim backend",
    )

    subparsers = parser.add_subparsers(dest="command")

    run_cmd = subparsers.add_parser("run", help="Start services and launch Claude")
    _add_network_flags(run_cmd)
    _add_aws_flag(run_cmd)
    run_cmd.add_argument("claude_args", nargs="*", help="Arguments forwarded to claude after '--'")

    up_cmd = subparsers.add_parser("up", help="Start background services")
    _add_network_flags(up_cmd)
    _add_aws_flag(up_cmd)
    up_cmd.add_argument("--wait", action="store_true", help="Wait for service readiness checks before exiting")
    up_cmd.add_argument("--timeout", type=float, default=10.0, help="Readiness timeout for --wait")

    subparsers.add_parser("down", help="Stop background services")

    restart_cmd = subparsers.add_parser("restart", help="Restart background services only")
    _add_network_flags(restart_cmd)
    _add_aws_flag(restart_cmd)

    status_cmd = subparsers.add_parser("status", help="Show service status")
    _add_network_flags(status_cmd)

    logs_cmd = subparsers.add_parser("logs", help="Show service logs")
    logs_cmd.add_argument("--lines", type=int, default=None, help="Number of lines per log")

    env_cmd = subparsers.add_parser("env", help="Print environment exports for manual Claude launch")
    env_cmd.add_argument("--host", default=None, help="Shim host override")
    env_cmd.add_argument("--model", default=None, help="Model override")
    env_cmd.add_argument("--shim-port", type=int, default=None, help="Shim port override")

    doctor_cmd = subparsers.add_parser("doctor", help="Check local cbridge readiness")
    _add_network_flags(doctor_cmd)
    _add_aws_flag(doctor_cmd)

    health_cmd = subparsers.add_parser("health", help="Check runtime health of running services")
    _add_network_flags(health_cmd)

    config_cmd = subparsers.add_parser("config", help="Manage cbridge configuration")
    config_subparsers = config_cmd.add_subparsers(dest="config_command")
    config_subparsers.add_parser("path", help="Print config file path")
    config_subparsers.add_parser("show", help="Print merged config")
    config_subparsers.add_parser("init", help="Write default config and bundled templates")
    config_subparsers.add_parser("edit", help="Open config.yaml in $EDITOR")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    config = load_user_config()
    settings = _settings_from(config, args)
    paths = runtime_paths()

    try:
        if args.command == "down":
            stop_services(paths)
            return 0

        if args.command == "status":
            show_status(settings, paths)
            return 0

        if args.command == "logs":
            show_logs(paths, settings.log_lines)
            return 0

        if args.command == "env":
            print(emit_env(settings))
            return 0

        if args.command == "doctor":
            return 0 if run_doctor(settings, paths) else 1

        if args.command == "health":
            return 0 if run_health(settings, paths) else 1

        if args.command == "config":
            if not args.config_command:
                parser.error("config command requires one of: path, show, init, edit")
            return config_command(args.config_command)

        if args.command == "up":
            start_services(settings, paths)
            if args.wait:
                deadline = time.time() + max(0.1, float(args.timeout))
                while time.time() < deadline:
                    if run_health(settings, paths):
                        break
                    time.sleep(0.5)
            print("\nServices are running. Use `cbridge run` to launch Claude.")
            return 0

        if args.command == "restart":
            stop_services(paths)
            time.sleep(0.3)
            start_services(settings, paths)
            return 0

        if args.command == "run":
            start_services(settings, paths)
            passthrough_args = list(args.claude_args)
            if passthrough_args and passthrough_args[0] == "--":
                passthrough_args = passthrough_args[1:]
            cli_args = [*settings.default_claude_args, *passthrough_args]
            launch_claude(settings, cli_args)
            return 0

        parser.error(f"Unknown command: {args.command}")
        return 2

    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"ERROR: required executable not found: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
