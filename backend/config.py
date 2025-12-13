"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv
# Note: keep this module import-safe (no network or argv processing).

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# OpenRouter attribution headers
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:5173/")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "llm-council-nk")

# Define model sets for premium and free councils
PREMIUM_COUNCIL_MODELS = [
    "openai/gpt-5.1",
    "anthropic/claude-sonnet-4.5",
    "x-ai/grok-4",
    "google/gemini-3-pro-preview"
]

FREE_COUNCIL_MODELS = [
    "tngtech/deepseek-r1t2-chimera:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "amazon/nova-2-lite-v1:free"
]


# By default use the premium council models. `main.py` will override
# `COUNCIL_MODELS` and `CHAIRMAN_MODEL` at startup based on CLI args.
COUNCIL_MODELS = PREMIUM_COUNCIL_MODELS

# Default chairman model
DEFAULT_CHAIRMAN = "google/gemini-3-pro-preview"

# Default chairman (can be overridden by `main.py` at startup)
CHAIRMAN_MODEL = DEFAULT_CHAIRMAN

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
