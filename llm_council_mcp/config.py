"""Configuration for the LLM Council."""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Default council members - list of OpenRouter model identifiers
DEFAULT_COUNCIL_MODELS = [
    "openai/gpt-4",
    "google/gemini-pro",
    "anthropic/claude-3-sonnet",
    "x-ai/grok-beta",
]

# Default chairman model - synthesizes final response
DEFAULT_CHAIRMAN_MODEL = "anthropic/claude-3-sonnet"


def _load_user_config():
    """Load user configuration from config file if it exists."""
    config_dir = Path.home() / ".config" / "llm-council"
    config_file = config_dir / "config.json"
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception:
            # If config file is invalid, return empty dict
            return {}
    return {}


def _get_models_from_env():
    """Get models from environment variable if set."""
    models_env = os.getenv("LLM_COUNCIL_MODELS")
    if models_env:
        # Comma-separated list of models
        return [m.strip() for m in models_env.split(",")]
    return None


# Load user configuration
_user_config = _load_user_config()

# Council models - priority: env var > config file > defaults
COUNCIL_MODELS = (
    _get_models_from_env() or 
    _user_config.get("council_models") or 
    DEFAULT_COUNCIL_MODELS
)

# Chairman model - priority: env var > config file > defaults
CHAIRMAN_MODEL = (
    os.getenv("LLM_COUNCIL_CHAIRMAN") or
    _user_config.get("chairman_model") or
    DEFAULT_CHAIRMAN_MODEL
)
