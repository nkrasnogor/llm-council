"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - list of OpenRouter model identifiers premium
#COUNCIL_MODELS = [
#    "openai/gpt-5.1",
#    "anthropic/claude-sonnet-4.5",
#    "x-ai/grok-4",
#    "google/gemini-3-pro-preview",
#]

# Council members - list of OpenRouter model identifiers free
COUNCIL_MODELS = [
   # "nvidia/nemotron-nano-12b-2-vl:free",
   # "mistralai/devstral-2-2512:free",
   # "google/gemini-2-0-flash-exp:free",
   # "deepseek/deepseek-r1-0528:free"
   "tngtech/deepseek-r1t2-chimera:free",
   "nvidia/nemotron-nano-12b-v2-vl:free",
   "amazon/nova-2-lite-v1:free"
]

# Chairman model - synthesizes final response premium
#CHAIRMAN_MODEL = "google/gemini-3-pro-preview"
# Chairman model - synthesizes final response free
CHAIRMAN_MODEL = "tngtech/deepseek-r1t2-chimera:free"

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
