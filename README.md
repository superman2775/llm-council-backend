# LLM Council MCP Server

[![PyPI version](https://badge.fury.io/py/llm-council-mcp.svg)](https://pypi.org/project/llm-council-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server that allows AI assistants to consult a council of multiple LLMs and receive synthesized guidance.

## What is This?

Instead of asking a single LLM for answers, this MCP server:
1. **Stage 1**: Sends your question to multiple LLMs in parallel (GPT, Claude, Gemini, Grok, etc.)
2. **Stage 2**: Each LLM reviews and ranks the other responses (anonymized to prevent bias)
3. **Stage 3**: A Chairman LLM synthesizes all responses into a final, high-quality answer

## Installation

```bash
pip install llm-council-mcp
```

## Setup

### 1. Get an OpenRouter API Key

The council uses [OpenRouter](https://openrouter.ai/) to access multiple LLMs:
1. Sign up at [openrouter.ai](https://openrouter.ai/)
2. Add credits or enable auto-top-up
3. Get your API key from the dashboard

### 2. Configure Environment Variables

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### 3. Customize Models (Optional)

You can customize which models participate in the council using three methods (in priority order):

#### Option 1: Environment Variables (Recommended)

```bash
# Comma-separated list of council models
export LLM_COUNCIL_MODELS="openai/gpt-4,anthropic/claude-3-opus,google/gemini-pro"

# Chairman model (synthesizes final response)
export LLM_COUNCIL_CHAIRMAN="anthropic/claude-3-opus"
```

#### Option 2: Configuration File

Create `~/.config/llm-council/config.json`:

```json
{
  "council_models": [
    "openai/gpt-4-turbo",
    "anthropic/claude-3-opus",
    "google/gemini-pro",
    "meta-llama/llama-3-70b-instruct"
  ],
  "chairman_model": "anthropic/claude-3-opus"
}
```

#### Option 3: Use Defaults

If you don't configure anything, these defaults are used:
- Council: GPT-4, Gemini Pro, Claude 3 Sonnet, Grok Beta
- Chairman: Claude 3 Sonnet

**Finding Models**: Browse available models at [openrouter.ai/models](https://openrouter.ai/models)

## Usage

### With Claude Code

```bash
claude mcp add --transport stdio llm-council --scope user \
  --env OPENROUTER_API_KEY=your-key-here \
  -- llm-council
```

Then in Claude Code:
```
Consult the LLM council about best practices for error handling
```

### With Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "llm-council": {
      "command": "llm-council",
      "env": {
        "OPENROUTER_API_KEY": "sk-or-v1-..."
      }
    }
  }
}
```

### With Other MCP Clients

Any MCP client can use the server by running:
```bash
llm-council
```

## Available Tools

### `consult_council`

Ask the LLM council a question and get synthesized guidance.

**Arguments:**
- `query` (string, required): The question to ask the council
- `include_details` (boolean, optional): Include individual model responses and rankings (default: false)

**Example:**
```
Use consult_council to ask: "What are the trade-offs between microservices and monolithic architecture?"
```

## How It Works

The council uses a 3-stage process inspired by ensemble methods and peer review:

1. **Parallel Collection**: All models receive the question simultaneously and respond independently
2. **Anonymous Peer Review**: Each model ranks the other responses (anonymized to prevent model bias)
3. **Synthesis**: A chairman model reviews all responses and rankings to produce a comprehensive answer

This approach helps surface diverse perspectives, identify consensus, and produce more balanced, well-reasoned answers.

## Credits

Based on the original [llm-council](https://github.com/karpathy/llm-council) by Andrej Karpathy.

## License

MIT License - see [LICENSE](LICENSE) file for details.
