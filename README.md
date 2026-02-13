# Kimicc

Run Claude Code CLI with a 90% discount by using [Kimi K2.5](https://www.kimi.com/ai-models/kimi-k2-5) on [AWS Bedrock](https://aws.amazon.com/bedrock/) as the AI backend.

## Quick Start

```bash
# Install
pip install git+https://github.com/jswift24/kimicc

# Configure AWS credentials
export AWS_PROFILE=bedrock-kimi
# OR use a bearer token:
# export AWS_ACCESS_KEY_ID=...
# export AWS_SECRET_ACCESS_KEY=...

# Launch Claude with Kimi backend
kimicc

# With Claude arguments
kimicc -- --dangerously-skip-permissions
```

## Setup (Development/Local Install)

If you're working on the kimicc source code or prefer a local install:

### 1. Install with pipx (Recommended)

pipx installs Python CLI tools in isolated environments while making them available globally:

```bash
# Install pipx if needed
sudo apt install pipx
pipx ensurepath

# Install kimicc
cd ~/kimicc
pipx install -e .

# kimicc is now available everywhere
kimicc --help
```

### Alternative: Use Virtual Environment

If you have an existing venv:

```bash
source /path/to/.venv/bin/activate
cd ~/kimicc
pip install -e .
```

Note: You'll need to activate the venv before running kimicc.

### 2. Configure AWS Credentials

Kimi K2.5 requires AWS Bedrock access. Choose one method:

**Option A: AWS Profile (recommended)**

If you've already installed the AWS CLI, you can store your AWS credentials in `~/.aws/credentials` and `~/.aws/config` as follows:

```bash
# Configure a new profile named "bedrock-kimi"
aws configure --profile bedrock-kimi
# Enter your AWS Access Key ID, Secret Access Key, region (e.g., us-east-1), and output format

# Then set the environment variable
export AWS_PROFILE=bedrock-kimi
```

See [AWS docs on creating profiles](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html) for more details.

**Option B: Direct credentials**

To get your AWS Access Key ID and Secret Access Key:

1. Log in to the [AWS IAM Console](https://console.aws.amazon.com/iam/)
2. Navigate to **Users** → select your user → **Security credentials** tab
3. Click **Create access key** (or use an existing one)
4. Copy the Access Key ID and Secret Access Key (the secret is only shown once)

See [AWS docs on access keys](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html) for more information.

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

**Option C: Temporary session token**

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...
```

### 3. Ensure LiteLLM is Available

Kimicc requires `litellm` in your environment. If you have an existing venv (e.g., from cc-kimi):

```bash
# Activate the venv before running kimicc
source /path/to/.venv/bin/activate
kimicc
```

Or install litellm globally:

```bash
pip install litellm
```

### 4. Launch

From any project directory:

```bash
# Start services and launch Claude
kimicc

# Just start services
kimicc --start

# Stop services
kimicc --stop
```

## What is Kimicc?

Kimi K2.5 is a powerful model available through [AWS Bedrock](https://aws.amazon.com/bedrock/). See the [Kimi K2.5 model page](https://platform.moonshot.ai/docs/guide/kimi-k2-5-quickstart) and [documentation](https://www.kimi.com/ai-models/kimi-k2-5) for model details and capabilities. Kimicc creates a shim layer that:

1. Starts a LiteLLM proxy to interface with AWS Bedrock
2. Runs an Anthropic API-compatible shim to translate between Claude Code expectations and Kimi responses
3. Launches Claude Code pointing at the shim

This lets you use Claude Code's excellent tool use and interface with Kimi K2.5's capabilities.

## Installation

### Via pipx (Recommended for most systems)

```bash
# Install pipx first
sudo apt install pipx  # Ubuntu/Debian
# brew install pipx    # macOS

pipx install git+https://github.com/jswift24/kimicc
```

### Manual install (development)

```bash
git clone https://github.com/jswift24/kimicc
cd kimicc
pipx install -e .
# OR use a venv:
# python -m venv venv
# source venv/bin/activate
# pip install -e .
```

## Usage

### Basic Commands

```bash
kimicc                            # Start services, launch Claude
kimicc --start                    # Start services only
kimicc --stop                     # Stop services
kimicc --status                   # Check service status
kimicc --logs                     # View service logs
kimicc --restart                  # Restart services
kimicc --help                     # Show full help
```

### Passing Arguments to Claude

```bash
# Pass arguments after --
kimicc -- --dangerously-skip-permissions
kimicc -- --dangerously-skip-permissions --no-session-persistence

# Or pass directly
kimicc --dangerously-skip-permissions
```

### AWS Authentication

Kimi K2.5 requires AWS Bedrock access. Kimicc supports multiple authentication methods:

1. **AWS Profile (recommended)**:

   ```bash
   kimicc --aws-profile bedrock-kimi
   # Or via environment:
   export AWS_PROFILE=bedrock-kimi
   kimicc
   ```

2. **Direct credentials**:

   ```bash
   export AWS_ACCESS_KEY_ID=...
   export AWS_SECRET_ACCESS_KEY=...
   kimicc
   ```

3. **Temporary token**:
   ```bash
   kimicc --aws-token $AWS_SESSION_TOKEN
   ```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│ Claude Code │────▶│  Kimicc Shim │────▶│   LiteLLM   │────▶│   AWS       │
│  (client)   │◄────│  (port 4001) │◄────│ (port 4000) │◄────│  Bedrock    │
└─────────────┘     └──────────────┘     └─────────────┘     └─────────────┘
                          │
                    Translates Anthropic
                    API ↔ Kimi/OpenAI format
```

### Why Kimicc Exists

Claude Code is an excellent code harness, but Anthropic's models are very expensive. You can *in theory* configure Claude Code to work with any model, but in practice there are lots of problems because CC is built around Anthropic's Messages API contract, which third party models don't support out of the box. This leads to a bunch of gaps, listed below. LiteLLM claims to bridge the gap, but doesn't really.

Kimicc exists to make that combination reliable in real Claude Code workflows (tool use, streaming, and multi-turn sessions), not just basic single-turn completions.

Hopefully, Kimicc will inspire LiteLLM, Moonshot and/or Amazon to support this scenario, so that Kimicc is no longer needed.

### How Kimicc Makes It Work

LiteLLM handles most of the protocol translation, and the shim handles behavior-level compatibility so Claude Code can keep operating normally.

In practice, the shim normalizes and bridges key incompatibilities:

| Problem | Why it breaks Claude Code | What kimicc does |
|---------|---------------------------|------------------|
| Tool name casing differences | Claude tool binding is case-sensitive (`Bash` vs `bash`) | Restores tool names to the original casing from the request |
| Tool ID formatting quirks | Whitespace/malformed IDs can break tool result matching | Normalizes tool IDs before returning to Claude Code |
| Anthropic streaming semantics | Claude expects Anthropic SSE events (`input_json_delta`, etc.) | Emulates Anthropic SSE from non-stream upstream responses when needed |
| Reasoning block format (`thinking`) | Kimi-style thinking blocks may not match Anthropic block shape | Normalizes thinking blocks for Claude-compatible rendering |
| Stop reason mismatch | OpenAI-style stop reasons differ from Anthropic (`stop` vs `end_turn`) | Maps/infers stop reasons to Anthropic-compatible values |
| Model name drift in sub-requests | Claude can send alternate model IDs in internal requests | Rewrites requests to configured `kimi-k2.5` model |
| Server-side web search tool type | Anthropic server tool types are not directly supported upstream | Converts and executes web search in the shim, then returns Anthropic-style search metadata |

This design keeps the developer experience close to native Claude Code, while swapping the backend model to Kimi K2.5.

## License

MIT
