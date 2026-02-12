"""CLI entry point for kimicc."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

# Default configuration
DEFAULT_LITELLM_PORT = 4000
DEFAULT_SHIM_PORT = 4001
DEFAULT_HOST = "127.0.0.1"
DEFAULT_MODEL = "kimi-k2.5"

# PID files
PIDFILE_LITELLM = Path("/tmp/kimicc-litellm.pid")
PIDFILE_SHIM = Path("/tmp/kimicc-shim.pid")
LOGFILE_LITELLM = Path("/tmp/kimicc-litellm.log")
LOGFILE_SHIM = Path("/tmp/kimicc-shim.log")


def get_kimicc_home() -> Path:
    """Get the kimicc installation directory."""
    if "KIMICC_HOME" in os.environ:
        return Path(os.environ["KIMICC_HOME"])

    # Try to find from package location
    try:
        import kimicc
        return Path(kimicc.__file__).parent.parent
    except Exception:
        pass

    # Fallback to script location
    script_dir = Path(__file__).parent.parent.parent
    if (script_dir / "config" / "litellm.yaml").exists():
        return script_dir

    raise RuntimeError("Cannot find kimicc installation. Set KIMICC_HOME environment variable.")


def is_running(pidfile: Path, port: int) -> bool:
    """Check if a service is running by PID and port test."""
    if pidfile.exists():
        try:
            pid = int(pidfile.read_text().strip())
            os.kill(pid, 0)  # Check if process exists
            # Also verify the port is listening
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                if s.connect_ex((DEFAULT_HOST, port)) == 0:
                    return True
        except (OSError, ProcessLookupError, ValueError):
            pass
    return False


def start_litellm(kimicc_home: Path, host: str, port: int, aws_profile: str | None = None) -> None:
    """Start the LiteLLM proxy service."""
    if is_running(PIDFILE_LITELLM, port):
        pid = int(PIDFILE_LITELLM.read_text().strip())
        print(f"LiteLLM already running (pid {pid})")
        return

    print(f"Starting LiteLLM on port {port}...")

    config_path = kimicc_home / "config" / "litellm.yaml"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if aws_profile:
        env["AWS_PROFILE"] = aws_profile

    # Check if litellm is available
    try:
        subprocess.run(["litellm", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: litellm not found. Install it with: pip install litellm")
        sys.exit(1)

    # Start LiteLLM process
    log = open(LOGFILE_LITELLM, "a")
    proc = subprocess.Popen(
        [
            "litellm",
            "--config", str(config_path),
            "--host", host,
            "--port", str(port),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )

    PIDFILE_LITELLM.write_text(str(proc.pid))

    # Wait for it to be ready
    for i in range(30):
        time.sleep(1)
        if is_running(PIDFILE_LITELLM, port):
            print(f"LiteLLM ready (pid {proc.pid})")
            return
        if proc.poll() is not None:
            print(f"ERROR: LiteLLM failed to start. Check {LOGFILE_LITELLM}")
            sys.exit(1)

    print(f"ERROR: LiteLLM failed to start in time. Check {LOGFILE_LITELLM}")
    sys.exit(1)


def start_shim(kimicc_home: Path, host: str, port: int, litellm_port: int) -> None:
    """Start the shim service."""
    if is_running(PIDFILE_SHIM, port):
        pid = int(PIDFILE_SHIM.read_text().strip())
        print(f"Shim already running (pid {pid})")
        return

    print(f"Starting shim on port {port}...")

    env = os.environ.copy()
    env["LITELLM_BASE_URL"] = f"http://{host}:{litellm_port}"
    env["SHIM_HOST"] = host
    env["SHIM_PORT"] = str(port)
    env["PYTHONUNBUFFERED"] = "1"

    log = open(LOGFILE_SHIM, "a")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "kimicc.shim:app",
            "--host", host,
            "--port", str(port),
            "--log-level", "warning",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )

    PIDFILE_SHIM.write_text(str(proc.pid))

    # Wait for it to be ready
    for i in range(15):
        time.sleep(1)
        if is_running(PIDFILE_SHIM, port):
            print(f"Shim ready (pid {proc.pid})")
            return
        if proc.poll() is not None:
            print(f"ERROR: Shim failed to start. Check {LOGFILE_SHIM}")
            sys.exit(1)

    print(f"ERROR: Shim failed to start in time. Check {LOGFILE_SHIM}")
    sys.exit(1)


def stop_services() -> None:
    """Stop all running services."""
    for pidfile in [PIDFILE_SHIM, PIDFILE_LITELLM]:
        if pidfile.exists():
            try:
                pid = int(pidfile.read_text().strip())
                os.kill(pid, 15)  # SIGTERM
                print(f"Stopped pid {pid}")
            except (OSError, ProcessLookupError, ValueError):
                pass
            pidfile.unlink(missing_ok=True)
    print("Services stopped.")


def show_status() -> None:
    """Show service status."""
    litellm_status = "running" if is_running(PIDFILE_LITELLM, DEFAULT_LITELLM_PORT) else "stopped"
    shim_status = "running" if is_running(PIDFILE_SHIM, DEFAULT_SHIM_PORT) else "stopped"

    if litellm_status == "running":
        pid = int(PIDFILE_LITELLM.read_text().strip())
        print(f"LiteLLM: running (pid {pid})")
    else:
        print("LiteLLM: stopped")

    if shim_status == "running":
        pid = int(PIDFILE_SHIM.read_text().strip())
        print(f"Shim:    running (pid {pid})")
    else:
        print("Shim:    stopped")


def show_logs() -> None:
    """Show service logs."""
    print("=== LiteLLM (last 20 lines) ===")
    if LOGFILE_LITELLM.exists():
        subprocess.run(["tail", "-20", str(LOGFILE_LITELLM)])
    else:
        print("(no log)")

    print()
    print("=== Shim (last 20 lines) ===")
    if LOGFILE_SHIM.exists():
        subprocess.run(["tail", "-20", str(LOGFILE_SHIM)])
    else:
        print("(no log)")


def emit_env() -> str:
    """Emit environment variables for Claude."""
    return f"""export ANTHROPIC_BASE_URL="http://{DEFAULT_HOST}:{DEFAULT_SHIM_PORT}"
export ANTHROPIC_MODEL="{DEFAULT_MODEL}"
export ANTHROPIC_API_KEY="sk-litellm-local"
unset CLAUDE_CODE_USE_BEDROCK
unset CLAUDE_CODE_USE_VERTEX
unset CLAUDE_CODE_USE_FOUNDRY
export CLAUDE_CODE_SKIP_BEDROCK_AUTH=1
export CLAUDE_CODE_SKIP_VERTEX_AUTH=1
export CLAUDE_CODE_SKIP_FOUNDRY_AUTH=1
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
export CLAUDE_CODE_DISABLE_FINE_GRAINED_TOOL_STREAMING=1
"""


def print_env_help(kimicc_home: Path) -> None:
    """Print environment setup help."""
    print()
    print("To use Kimi K2.5 with Claude Code, run:")
    print()
    print(f"  eval \"$(kimicc --env)\"")
    print("  claude")
    print()


def launch_claude(claude_args: list) -> None:
    """Launch Claude with the configured environment."""
    env_setup = emit_env()
    for line in env_setup.strip().split("\n"):
        if line.startswith("export "):
            key, val = line[7:].split("=", 1)
            os.environ[key] = val.strip('"')
        elif line.startswith("unset "):
            key = line[6:]
            os.environ.pop(key, None)

    print()
    print("✓ Services ready. Launching Claude with kimi-k2.5 backend...")
    if claude_args:
        print(f"  Claude arguments: {' '.join(claude_args)}")
    print()

    # Replace current process with claude
    if claude_args:
        os.execvp("claude", ["claude"] + claude_args)
    else:
        os.execvp("claude", ["claude"])


def show_quick_help() -> None:
    """Show quick help message."""
    print("""kimicc - Claude Code with Kimi K2.5 backend

Usage: kimicc [OPTIONS] [-- CLAUDE_ARGS...]

Kimi Options:
  --start              Start LiteLLM + Shim services
  --stop               Stop services
  --status             Check if services are running
  --logs               Show service logs
  --restart            Restart services
  --env                Print environment variables for Claude
  --aws-profile NAME   Use AWS profile (default: from env/AWS_DEFAULT_PROFILE)
  -h, --help           Show this help message

Claude Arguments:
  All arguments after "--" are passed to Claude Code:
    kimicc -- --dangerously-skip-permissions
  Or pass directly:
    kimicc --dangerously-skip-permissions

Examples:
  kimicc                           # Start services, launch Claude
  kimicc -- --help                 # Show Claude's help
  kimicc --stop                    # Stop background services
  kimicc --aws-profile bedrock     # Use specific AWS profile

Full help: kimicc --help
""")


def show_full_help() -> None:
    """Show full help message."""
    print("""kimicc - Claude Code with Kimi K2.5 backend

Usage: kimicc [OPTIONS] [-- CLAUDE_ARGS...]

Kimi Options:
  --start              Start LiteLLM + Shim services only
  --stop               Stop services
  --status             Check if services are running
  --logs               Show service logs
  --restart            Restart services
  --env                Print environment setup (eval with: eval "$(kimicc --env)")
  --aws-profile NAME   Use AWS profile for Bedrock authentication
                       (can also use AWS_PROFILE env var)
  -h, --help           Show this help message

Claude Arguments:
  Pass arguments after "--":
    kimicc -- --dangerously-skip-permissions
    kimicc -- --dangerously-skip-permissions --no-session-persistence

  Or pass directly:
    kimicc --dangerously-skip-permissions

Examples:
  # Start services and launch Claude
  kimicc

  # Start services only (for manual claude launch)
  kimicc --start

  # Stop services
  kimicc --stop

  # Check status
  kimicc --status

  # View logs
  kimicc --logs

  # Use specific AWS profile
  kimicc --aws-profile bedrock-kimi

  # Pass arguments to Claude
  kimicc -- --dangerously-skip-permissions

  # Launch in specific directory
  kimicc -- -d /path/to/project

Architecture:
  Claude Code → Kimicc Shim → LiteLLM → AWS Bedrock (Kimi K2.5)

Environment Variables:
  KIMICC_HOME          Override installation directory
  AWS_PROFILE          AWS profile for Bedrock access
  CLAUDE_CODE_ARGS     Default arguments passed to Claude

For more information: https://github.com/alonb/kimicc
""")


def main() -> None:
    """Main entry point."""
    # Parse arguments manually to handle -- separator
    args = sys.argv[1:]
    claude_args: list[str] = []
    kimicc_args: list[str] = []

    # Check for -- separator
    if "--" in args:
        sep_idx = args.index("--")
        kimicc_args = args[:sep_idx]
        claude_args = args[sep_idx + 1:]
    else:
        # No separator - try to distinguish kimicc args from claude args
        kimicc_flags = {"--start", "--stop", "--status", "--logs", "--restart", "--env", "--help", "-h", "--aws-profile"}
        i = 0
        while i < len(args):
            if args[i] in kimicc_flags:
                kimicc_args.append(args[i])
                i += 1
                if args[i-1] == "--aws-profile" and i < len(args) and not args[i].startswith("-"):
                    kimicc_args.append(args[i])
                    i += 1
            elif args[i].startswith("--"):
                # Unknown flag - could be claude arg
                claude_args.extend(args[i:])
                break
            else:
                # Positional arg - pass to claude
                claude_args.extend(args[i:])
                break

    # Also check CLAUDE_CODE_ARGS env var
    env_claude_args = os.environ.get("CLAUDE_CODE_ARGS", "")
    if env_claude_args:
        claude_args = env_claude_args.split() + claude_args

    # Parse kimicc arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--logs", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--env", action="store_true")
    parser.add_argument("--aws-profile", default=None)
    parser.add_argument("--help", "-h", action="store_true")

    parsed, _ = parser.parse_known_args(kimicc_args)

    # Handle help
    if parsed.help or "-h" in kimicc_args:
        if len(kimicc_args) == 1 and (kimicc_args[0] == "-h" or kimicc_args[0] == "--help"):
            show_quick_help()
        else:
            show_full_help()
        return

    # Get kimicc home
    kimicc_home = get_kimicc_home()

    aws_profile = parsed.aws_profile or os.environ.get("AWS_PROFILE")

    # Handle commands
    if parsed.stop:
        stop_services()
        return

    if parsed.status:
        show_status()
        return

    if parsed.logs:
        show_logs()
        return

    if parsed.env:
        print(emit_env())
        return

    if parsed.restart:
        stop_services()
        time.sleep(1)
        start_litellm(kimicc_home, DEFAULT_HOST, DEFAULT_LITELLM_PORT, aws_profile)
        start_shim(kimicc_home, DEFAULT_HOST, DEFAULT_SHIM_PORT, DEFAULT_LITELLM_PORT)
        print_env_help(kimicc_home)
        return

    # Start services
    start_litellm(kimicc_home, DEFAULT_HOST, DEFAULT_LITELLM_PORT, aws_profile)
    start_shim(kimicc_home, DEFAULT_HOST, DEFAULT_SHIM_PORT, DEFAULT_LITELLM_PORT)

    if parsed.start:
        # Just start services, don't launch claude
        print_env_help(kimicc_home)
        return

    # Launch claude with all collected arguments
    launch_claude(claude_args)


if __name__ == "__main__":
    main()
