# cbridge

Use Claude Code with Kimi K2.5 on AWS Bedrock through a local compatibility shim.

cbridge runs two local services:
1. LiteLLM proxy (Bedrock backend)
2. Anthropic-compatible shim for Claude Code

Then it launches Claude Code pointed at that shim.

## Install

### Recommended (end users)

```bash
pipx install git+https://github.com/jswift24/cbridge
```

When released to PyPI, installation is:

```bash
pipx install cbridge
```

### Development install

```bash
git clone https://github.com/jswift24/cbridge
cd cbridge
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
cbridge doctor

# Start services + launch Claude
cbridge run

# Pass args to Claude
cbridge run -- --dangerously-skip-permissions
```

## Commands

```bash
cbridge run [-- <claude args...>]   # Start services and launch Claude
cbridge up                          # Start background services only
cbridge down                        # Stop background services
cbridge restart                     # Restart background services only
cbridge status                      # Show service status
cbridge logs                        # Show service logs
cbridge env                         # Print exports for manual Claude launch
cbridge doctor                      # Run local environment checks
cbridge health                      # Check live runtime health
cbridge config path|show|init|edit  # Discover/manage config files
```

`cbridge restart` does not launch Claude.

## Configuration

cbridge auto-creates user config on first run:

- `~/.config/cbridge/config.yaml`
- `~/.config/cbridge/litellm.yaml`

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

- `~/.local/share/cbridge/litellm.pid`
- `~/.local/share/cbridge/shim.pid`
- `~/.local/share/cbridge/litellm.log`
- `~/.local/share/cbridge/shim.log`

## AWS Authentication

cbridge relies on the standard AWS credential chain used by LiteLLM/Bedrock.

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
cbridge doctor
```

For runtime connectivity checks once services are up:

```bash
cbridge health
```

## Architecture

```text
Claude Code -> cbridge Shim (4001) -> LiteLLM (4000) -> AWS Bedrock (Kimi K2.5)
```

The shim normalizes compatibility details Claude Code expects, including:
- tool name casing
- tool ID formatting
- Anthropic-style SSE framing
- stop-reason normalization
- server-side web search metadata mapping

## Troubleshooting

1. Run `cbridge doctor` and fix any `FAIL` lines.
2. Check logs with `cbridge logs`.
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
