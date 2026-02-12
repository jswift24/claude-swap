# Kimicc

A bridge that lets you use Claude Code CLI with Kimi K2.5 (via AWS Bedrock) as the AI backend.

## Quick Start

```bash
# Install
pip install git+https://github.com/alonb/kimicc

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

### 1. Clone and Install

```bash
cd ~/kimicc
pip install -e .
```

This installs the `kimicc` command globally on your system.

### 2. Configure AWS Credentials

Kimi K2.5 requires AWS Bedrock access. Choose one method:

**Option A: AWS Profile (recommended)**
```bash
export AWS_PROFILE=bedrock-kimi
```

**Option B: Direct credentials**
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

Kimi K2.5 is a powerful model available through AWS Bedrock. Kimicc creates a shim layer that:

1. Starts a LiteLLM proxy to interface with AWS Bedrock
2. Runs an Anthropic API-compatible shim to translate between Claude Code expectations and Kimi responses
3. Launches Claude Code pointing at the shim

This lets you use Claude Code's excellent tool use and interface with Kimi K2.5's capabilities.

## Installation

### Via pip (recommended)

```bash
pip install git+https://github.com/alonb/kimicc
```

### Manual install

```bash
git clone https://github.com/alonb/kimicc
cd kimicc
pip install -e .
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

## License

MIT
