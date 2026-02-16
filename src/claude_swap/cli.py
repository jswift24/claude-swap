"""CLI entry point for claude-swap.

claude-swap is a drop-in replacement for the `claude` command that routes
Claude Code through a local shim to use Kimi K2.5 via AWS Bedrock.

Usage:
    claude-swap [CLAUDE_ARGS...]     # Run Claude with Kimi backend
    claude-swap up                   # Start background services
    claude-swap down                 # Stop background services
    claude-swap status               # Check service status
    claude-swap logs                 # View service logs
    claude-swap config init          # Initialize configuration
"""
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
from typing import Any

import httpx
import yaml
from platformdirs import user_config_path, user_data_path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_MODEL = "kimi-k2.5"
DEFAULT_LITELLM_PORT = 4000
DEFAULT_SHIM_PORT = 4001
DEFAULT_LOG_LINES = 50

# Subcommands that claude-swap handles itself (not passed to Claude)
SERVICE_COMMANDS = {"up", "down", "status", "logs", "restart", "env", "doctor", "health", "config", "help"}


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
        "logs": {"lines": DEFAULT_LOG_LINES},
    }


def _config_dir() -> Path:
    return Path(user_config_path("claude-swap", appauthor=False))


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
    base_dir = Path(user_data_path("claude-swap", appauthor=False))
    base_dir.mkdir(parents=True, exist_ok=True)
    return RuntimePaths(
        base_dir=base_dir,
        litellm_pid=base_dir / "litellm.pid",
        shim_pid=base_dir / "shim.pid",
        litellm_log=base_dir / "litellm.log",
        shim_log=base_dir / "shim.log",
    )


def _packaged_litellm_template() -> str:
    return resources.files("claude_swap.data").joinpath("litellm.yaml").read_text(encoding="utf-8")


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


def launch_claude(claude_args: list[str]) -> None:
    """Launch Claude with the configured environment."""
    if not shutil.which("claude"):
        print("ERROR: 'claude' not found in PATH. Install Claude Code CLI first.", file=sys.stderr)
        sys.exit(1)

    os.execvp("claude", ["claude", *claude_args])


def start_litellm(settings: Settings, paths: RuntimePaths, verbose: bool = True) -> None:
    """Start LiteLLM if needed."""
    if _is_running(paths.litellm_pid, settings.host, settings.litellm_port):
        if verbose:
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

    if verbose:
        print(f"LiteLLM ready (pid {proc.pid})")


def start_shim(settings: Settings, paths: RuntimePaths, verbose: bool = True) -> None:
    """Start shim if needed."""
    if _is_running(paths.shim_pid, settings.host, settings.shim_port):
        if verbose:
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
            "claude_swap.shim:app",
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

    if verbose:
        print(f"Shim ready (pid {proc.pid})")


def start_services(settings: Settings, paths: RuntimePaths | None = None, verbose: bool = True) -> None:
    """Start all background services."""
    paths = paths or runtime_paths()
    start_litellm(settings, paths, verbose=verbose)
    start_shim(settings, paths, verbose=verbose)


def ensure_services_running(settings: Settings, paths: RuntimePaths | None = None) -> None:
    """Ensure services are running (silent if already running, starts them if not)."""
    paths = paths or runtime_paths()
    litellm_was_running = _is_running(paths.litellm_pid, settings.host, settings.litellm_port)
    shim_was_running = _is_running(paths.shim_pid, settings.host, settings.shim_port)

    if litellm_was_running and shim_was_running:
        return  # Everything already running

    # Start what's needed
    start_services(settings, paths, verbose=not (litellm_was_running and shim_was_running))


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
            print("- Install claude-swap in a venv or pipx so `litellm` is on PATH.")
        if not claude_ok:
            print("- Install Claude Code CLI and confirm `claude` is on PATH.")
        if not port1_ok or not port2_ok:
            print("- Run `claude-swap status` and `claude-swap down`, or change configured ports.")
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
            print("ERROR: set EDITOR or VISUAL to use `claude-swap config edit`", file=sys.stderr)
            return 1
        cmd = [*shlex.split(editor), str(config_path)]
        try:
            return subprocess.run(cmd, check=False).returncode
        except FileNotFoundError:
            print(f"ERROR: editor not found: {editor}", file=sys.stderr)
            return 1

    print(f"ERROR: unknown config subcommand '{command}'", file=sys.stderr)
    return 2


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

    return Settings(
        host=str(host),
        model=str(model),
        litellm_port=litellm_port,
        shim_port=shim_port,
        aws_profile=str(aws_profile) if aws_profile else None,
        log_lines=log_lines,
    )


def build_service_parser() -> argparse.ArgumentParser:
    """Build parser for service management subcommands."""
    parser = argparse.ArgumentParser(
        prog="claude-swap",
        description="Run Claude Code with Kimi K2.5 backend via AWS Bedrock",
        add_help=False,
    )

    subparsers = parser.add_subparsers(dest="command")

    # Service management subcommands
    up_cmd = subparsers.add_parser("up", help="Start background services")
    up_cmd.add_argument("--host", default=None)
    up_cmd.add_argument("--model", default=None)
    up_cmd.add_argument("--litellm-port", type=int, default=None)
    up_cmd.add_argument("--shim-port", type=int, default=None)
    up_cmd.add_argument("--aws-profile", default=None)
    up_cmd.add_argument("--wait", action="store_true")
    up_cmd.add_argument("--timeout", type=float, default=10.0)

    subparsers.add_parser("down", help="Stop background services")

    restart_cmd = subparsers.add_parser("restart", help="Restart background services")
    restart_cmd.add_argument("--host", default=None)
    restart_cmd.add_argument("--model", default=None)
    restart_cmd.add_argument("--litellm-port", type=int, default=None)
    restart_cmd.add_argument("--shim-port", type=int, default=None)
    restart_cmd.add_argument("--aws-profile", default=None)

    status_cmd = subparsers.add_parser("status", help="Show service status")
    status_cmd.add_argument("--host", default=None)
    status_cmd.add_argument("--litellm-port", type=int, default=None)
    status_cmd.add_argument("--shim-port", type=int, default=None)

    logs_cmd = subparsers.add_parser("logs", help="Show service logs")
    logs_cmd.add_argument("--lines", type=int, default=None)

    env_cmd = subparsers.add_parser("env", help="Print environment exports")
    env_cmd.add_argument("--host", default=None)
    env_cmd.add_argument("--model", default=None)
    env_cmd.add_argument("--shim-port", type=int, default=None)

    doctor_cmd = subparsers.add_parser("doctor", help="Check readiness")
    doctor_cmd.add_argument("--host", default=None)
    doctor_cmd.add_argument("--model", default=None)
    doctor_cmd.add_argument("--litellm-port", type=int, default=None)
    doctor_cmd.add_argument("--shim-port", type=int, default=None)
    doctor_cmd.add_argument("--aws-profile", default=None)

    health_cmd = subparsers.add_parser("health", help="Check service health")
    health_cmd.add_argument("--host", default=None)
    health_cmd.add_argument("--litellm-port", type=int, default=None)
    health_cmd.add_argument("--shim-port", type=int, default=None)

    config_cmd = subparsers.add_parser("config", help="Manage configuration")
    config_subparsers = config_cmd.add_subparsers(dest="config_command")
    config_subparsers.add_parser("path")
    config_subparsers.add_parser("show")
    config_subparsers.add_parser("init")
    config_subparsers.add_parser("edit")

    # Help subcommand
    subparsers.add_parser("help", help="Show this help message")

    return parser


def show_help() -> None:
    """Show usage help."""
    print("""Usage: claude-swap [CLAUDE_ARGS...] | [COMMAND]

Run Claude Code with Kimi K2.5 backend via AWS Bedrock.

Pass-through mode (default):
    claude-swap --dangerously-skip-permissions
    claude-swap --resume last
    claude-swap --help               # Shows Claude's help

Service management:
    claude-swap up                   # Start background services
    claude-swap down                 # Stop background services
    claude-swap restart              # Restart background services
    claude-swap status               # Check service status
    claude-swap logs                 # View service logs
    claude-swap env                  # Print environment exports
    claude-swap doctor               # Check readiness
    claude-swap health               # Check service health
    claude-swap config init          # Initialize configuration
    claude-swap help                 # Show this help

Environment variables:
    ANTHROPIC_BASE_URL               # Set by claude-swap (shim endpoint)
    ANTHROPIC_MODEL                  # Set to kimi-k2.5
    ANTHROPIC_API_KEY                # Set to sk-litellm-local
    AWS_PROFILE                      # AWS profile for Bedrock access
""")


def main(argv: list[str] | None = None) -> int:
    """Main entry point.

    If first arg is a service command, handle it.
    Otherwise, ensure services are running and exec Claude with all args.
    """
    args_list = argv if argv is not None else sys.argv[1:]

    # Check if this is a service management command
    if args_list and args_list[0] in SERVICE_COMMANDS:
        return _handle_service_command(args_list)

    # Otherwise, run in passthrough mode: ensure services and exec Claude
    return _run_passthrough(args_list)


def _handle_service_command(args_list: list[str]) -> int:
    """Handle service management subcommands."""
    parser = build_service_parser()
    args = parser.parse_args(args_list)

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
                print("ERROR: config requires one of: path, show, init, edit", file=sys.stderr)
                return 2
            return config_command(args.config_command)

        if args.command == "up":
            start_services(settings, paths)
            if args.wait:
                deadline = time.time() + max(0.1, float(args.timeout))
                while time.time() < deadline:
                    if run_health(settings, paths):
                        break
                    time.sleep(0.5)
            print("\nServices are running. Claude Code can be launched via 'claude-swap'.")
            return 0

        if args.command == "restart":
            stop_services(paths)
            time.sleep(0.3)
            start_services(settings, paths)
            return 0

        if args.command == "help":
            show_help()
            return 0

        return 0

    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"ERROR: required executable not found: {exc}", file=sys.stderr)
        return 1


def _run_passthrough(claude_args: list[str]) -> int:
    """Ensure services are running and exec Claude with the given args."""
    config = load_user_config()
    # Parse minimal args for settings (no CLI parsing, just config)
    dummy_args = argparse.Namespace(
        host=None,
        model=None,
        litellm_port=None,
        shim_port=None,
        aws_profile=None,
        lines=None,
    )
    settings = _settings_from(config, dummy_args)
    paths = runtime_paths()

    try:
        # Start services if not already running
        ensure_services_running(settings, paths)

        # Set up environment for Claude
        _apply_env_exports(settings)

        # Exec Claude with user's args
        launch_claude(claude_args)
        return 0  # Never reached due to exec

    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"ERROR: required executable not found: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
