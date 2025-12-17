"""
Manual test script for OpenRouter connectivity and model testing.

This script allows you to verify that your `OPENROUTER_API_KEY` is working
and to test the capabilities of different models before adding them to your
council. It tests both non-streaming and streaming responses.

Usage:
1. Make sure you have an `.env` file in the project root with your
   `OPENROUTER_API_KEY`.
2. Run the script from the project root directory:
   ```
   python -m backend.test_openrouter
   ```
"""
import asyncio
import httpx
import os
from typing import List, Dict, Any

# Ensure we can import from the 'backend' package
# This is a bit of a hack to make the script runnable directly
# while still using package-relative imports.
import sys
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_API_URL,
    OPENROUTER_HTTP_REFERER,
    OPENROUTER_APP_TITLE,
    PREMIUM_COUNCIL_MODELS,
    FREE_COUNCIL_MODELS,
    DEFAULT_CHAIRMAN,
)
from backend.openrouter import query_model

# --- Models to Test ---
# We collect all unique model names from the configuration to avoid duplication.
MODELS_TO_TEST = sorted(list(
    set(PREMIUM_COUNCIL_MODELS + FREE_COUNCIL_MODELS + [DEFAULT_CHAIRMAN])
))

# --- Test Messages ---
TEST_MESSAGES = [
    {"role": "user", "content": "What are the top 3 benefits of using Python for web development? Provide 3 very concise bullet points"}
]


async def test_streaming(model: str, messages: List[Dict[str, Any]]):
    """
    Tests a model's streaming response.
    """
    print(f"\n--- Testing STREAMING for model: {model} ---")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_HTTP_REFERER,
        "X-Title": OPENROUTER_APP_TITLE,
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", OPENROUTER_API_URL, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    print(f"Error: Received status code {response.status_code}")
                    async for byte in response.aiter_bytes():
                        print(byte.decode('utf-8'), end="")
                    return

                print("Streaming response:")
                full_response = ""
                async for chunk in response.aiter_text():
                    if chunk.startswith("data: "):
                        # We just print the content delta for this test
                        import json
                        try:
                            data = json.loads(chunk[len("data: "):])
                            if data['choices'][0]['delta']['content']:
                                delta = data['choices'][0]['delta']['content']
                                print(delta, end="", flush=True)
                                full_response += delta
                        except (json.JSONDecodeError, KeyError):
                            # Ignore non-json or non-content chunks
                            pass
                    elif "DONE" in chunk:
                        break
                print("\n--- End of Stream ---")

    except Exception as e:
        print(f"\nAn error occurred during streaming test for {model}: {e}")

async def test_non_streaming(model: str, messages: List[Dict[str, Any]]):
    """
    Tests a model's non-streaming (standard) response using the existing query_model function.
    """
    print(f"\n--- Testing NON-STREAMING for model: {model} ---")
    response = await query_model(model, messages)
    if response and response.get('content'):
        print("Response received:")
        print(response['content'])
    else:
        print("Failed to get a valid non-streaming response.")


async def main():
    """
    Main function to run the tests.
    """
    if not OPENROUTER_API_KEY:
        print("💥 Error: OPENROUTER_API_KEY environment variable not set.")
        print("Please create a .env file in the project root and add your key:")
        print("Example: OPENROUTER_API_KEY='sk-or-...'")
        return

    print("🔑 API key found. Starting tests...")

    for model in MODELS_TO_TEST:
        await test_non_streaming(model, TEST_MESSAGES)
        await test_streaming(model, TEST_MESSAGES)
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
