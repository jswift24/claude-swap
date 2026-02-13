# claude-swap

Use Claude Code with Kimi K2.5 on AWS Bedrock through a local compatibility shim.

claude-swap runs two local services:
1. LiteLLM proxy (Bedrock backend)
2. Anthropic-compatible shim for Claude Code

Then it launches Claude Code pointed at that shim.

## Install

### Recommended (end users)

```bash
pipx install git+https://github.com/jswift24/claude-swap
```

When released to PyPI, installation is:

```bash
pipx install claude-swap
```

### Development install

```bash
git clone https://github.com/jswift24/claude-swap
cd claude-swap
python -m venv .venv
source .venv/bin/activate
pip install -e .[test,dev]
```

`litellm` is bundled as a runtime dependency. No separate install step is required.

## Quick Start

```bash
# Configure AWS credentials (one option)
export AWS_PROFILE=bedrock-kimi

# Validate environment
claude-swap doctor

# Start services + launch Claude
claude-swap run

# Pass args to Claude
claude-swap run -- --dangerously-skip-permissions
```

## Commands

```bash
claude-swap run [-- <claude args...>]   # Start services and launch Claude
claude-swap up                          # Start background services only
claude-swap down                        # Stop background services
claude-swap restart                     # Restart background services only
claude-swap status                      # Show service status
claude-swap logs                        # Show service logs
claude-swap env                         # Print exports for manual Claude launch
claude-swap doctor                      # Run local environment checks
claude-swap health                      # Check live runtime health
claude-swap config path|show|init|edit  # Discover/manage config files
```

`claude-swap restart` does not launch Claude.

## Configuration

claude-swap auto-creates user config on first run:

- `~/.config/claude-swap/config.yaml`
- `~/.config/claude-swap/litellm.yaml`

Default `config.yaml`:

```yaml
host: 127.0.0.1
model: kimi-k2.5
aws_profile: null
ports:
  litellm: 4000
  shim: 4001
claude_args: []
logs:
  lines: 50
```

Runtime files are user-scoped (not global `/tmp`):

- `~/.local/share/claude-swap/litellm.pid`
- `~/.local/share/claude-swap/shim.pid`
- `~/.local/share/claude-swap/litellm.log`
- `~/.local/share/claude-swap/shim.log`

## AWS Authentication

claude-swap relies on the standard AWS credential chain used by LiteLLM/Bedrock.

Common options:

```bash
# Profile
export AWS_PROFILE=bedrock-kimi

# Access key pair
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

If startup fails, run:

```bash
claude-swap doctor
```

For runtime connectivity checks once services are up:

```bash
claude-swap health
```

## Architecture

```text
Claude Code -> claude-swap Shim (4001) -> LiteLLM (4000) -> AWS Bedrock (Kimi K2.5)
```

The shim normalizes compatibility details Claude Code expects, including:
- tool name casing
- tool ID formatting
- Anthropic-style SSE framing
- stop-reason normalization
- server-side web search metadata mapping

## Troubleshooting

1. Run `claude-swap doctor` and fix any `FAIL` lines.
2. Check logs with `claude-swap logs`.
3. Confirm your AWS credentials can access Bedrock in your target region.
4. Ensure ports `4000` and `4001` are free or override them in config.

## Development

```bash
source .venv/bin/activate
ruff check .
mypy src
pytest -q
```

## License

MIT (see `LICENSE`)
