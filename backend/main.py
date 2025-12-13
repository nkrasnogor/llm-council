"""FastAPI backend for LLM Council."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
import uuid
import json
import asyncio
import sys

# Import config and perform CLI-based selection/validation BEFORE importing
# modules that depend on `COUNCIL_MODELS` / `CHAIRMAN_MODEL`.
from . import config

try:
    import httpx
except Exception:
    httpx = None


# CLI args usage: `python -m backend.main [0|1] [optional_chairman_model]`
# 0 -> free council, 1 -> premium council. Default: premium.
def _apply_cli_args_to_config(argv: list):
    # Select council mode
    try:
        if len(argv) >= 2:
            val = argv[1]
            if val == "0":
                config.COUNCIL_MODELS = config.FREE_COUNCIL_MODELS
            elif val == "1":
                config.COUNCIL_MODELS = config.PREMIUM_COUNCIL_MODELS
    except Exception:
        # keep defaults
        pass

    # Optional chairman override (validate against OpenRouter /models)
    try:
        if len(argv) >= 3:
            candidate = argv[2]
            if candidate:
                # If httpx and API key available, validate
                if httpx is not None and config.OPENROUTER_API_KEY:
                    models_url = "https://openrouter.ai/api/v1/models"
                    headers = {"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"}
                    try:
                        resp = httpx.get(models_url, headers=headers, timeout=10.0)
                        resp.raise_for_status()
                        data = resp.json()
                        models_list = []
                        if isinstance(data, list):
                            models_list = data
                        elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                            models_list = data["data"]

                        found = False
                        for m in models_list:
                            if not isinstance(m, dict):
                                continue
                            for key in ("id", "model", "name"):
                                if key in m and m[key] == candidate:
                                    found = True
                                    break
                            if found:
                                break

                        if found:
                            config.CHAIRMAN_MODEL = candidate
                        else:
                            # leave default
                            pass
                    except Exception:
                        # On any error keep default
                        pass
                else:
                    # No httpx or API key: trust the provided candidate
                    config.CHAIRMAN_MODEL = candidate
    except Exception:
        pass
        # Parse CLI flags for nicer UX
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--council", choices=["free", "premium"], help="Select council type (free or premium).")
        parser.add_argument("--chairman", type=str, help="Optional chairman model identifier to override default.")
        # Parse known args only to avoid interfering with uv/other tooling
        args, _ = parser.parse_known_args(argv[1:])

        # Apply council selection
        try:
            if args.council == "free":
                config.COUNCIL_MODELS = config.FREE_COUNCIL_MODELS
            elif args.council == "premium":
                config.COUNCIL_MODELS = config.PREMIUM_COUNCIL_MODELS
        except Exception:
            pass

        # Apply optional chairman override with validation when possible
        if getattr(args, "chairman", None):
            candidate = args.chairman
            if httpx is not None and config.OPENROUTER_API_KEY:
                try:
                    models_url = "https://openrouter.ai/api/v1/models"
                    headers = {"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"}
                    resp = httpx.get(models_url, headers=headers, timeout=10.0)
                    resp.raise_for_status()
                    data = resp.json()
                    models_list = []
                    if isinstance(data, list):
                        models_list = data
                    elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                        models_list = data["data"]

                    found = False
                    for m in models_list:
                        if not isinstance(m, dict):
                            continue
                        for key in ("id", "model", "name"):
                            if key in m and m[key] == candidate:
                                found = True
                                break
                        if found:
                            break

                    if found:
                        config.CHAIRMAN_MODEL = candidate
                    else:
                        print(f"Warning: chairman model '{candidate}' not found on OpenRouter; using default {config.CHAIRMAN_MODEL}")
                except Exception:
                    print("Warning: could not validate chairman model due to network/error; using default or provided value.")
            else:
                # No httpx or API key -> trust provided candidate
                config.CHAIRMAN_MODEL = candidate


# Apply CLI args immediately
_apply_cli_args_to_config(sys.argv)

# Import modules that depend on config after config is set
from . import storage
from .council import run_full_council, generate_conversation_title, stage1_collect_responses, stage2_collect_rankings, stage3_synthesize_final, calculate_aggregate_rankings

app = FastAPI(title="LLM Council API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    pass


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    content: str
    attachments: Optional[List["Attachment"]] = Field(default_factory=list)

    @field_validator("attachments", mode="before")
    @classmethod
    def validate_attachments(cls, value):
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("attachments must be a list")
        for att in value:
            if not isinstance(att, dict):
                continue
        return value


class Attachment(BaseModel):
    """Attachment metadata and content."""
    filename: str
    mime_type: str
    data_url: str

    @field_validator("data_url")
    @classmethod
    def validate_data_url(cls, v: str):
        if not v.startswith("data:"):
            raise ValueError("data_url must start with 'data:'")
        return v


SendMessageRequest.model_rebuild()


class ConversationMetadata(BaseModel):
    """Conversation metadata for list view."""
    id: str
    created_at: str
    title: str
    message_count: int


class Conversation(BaseModel):
    """Full conversation with all messages."""
    id: str
    created_at: str
    title: str
    messages: List[Dict[str, Any]]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "LLM Council API"}


@app.get("/api/conversations", response_model=List[ConversationMetadata])
async def list_conversations():
    """List all conversations (metadata only)."""
    return storage.list_conversations()


@app.post("/api/conversations", response_model=Conversation)
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    conversation = storage.create_conversation(conversation_id)
    return conversation


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get a specific conversation with all its messages."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and run the 3-stage council process.
    Returns the complete response with all stages.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    # Add user message
    storage.add_user_message(conversation_id, request.content, request.attachments or [])

    # If this is the first message, generate a title
    if is_first_message:
        title = await generate_conversation_title(request.content)
        storage.update_conversation_title(conversation_id, title)

    # Run the 3-stage council process
    stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
        request.content,
        request.attachments or []
    )

    # Add assistant message with all stages
    storage.add_assistant_message(
        conversation_id,
        stage1_results,
        stage2_results,
        stage3_result
    )

    # Return the complete response with metadata
    return {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage3": stage3_result,
        "metadata": metadata
    }


@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and stream the 3-stage council process.
    Returns Server-Sent Events as each stage completes.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    async def event_generator():
        try:
            # Add user message
            storage.add_user_message(conversation_id, request.content, request.attachments or [])

            # Start title generation in parallel (don't await yet)
            title_task = None
            if is_first_message:
                title_task = asyncio.create_task(generate_conversation_title(request.content))

            # Stage 1: Collect responses
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            stage1_results = await stage1_collect_responses(request.content, request.attachments or [])
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

            # Stage 2: Collect rankings
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
            stage2_results, label_to_model = await stage2_collect_rankings(request.content, stage1_results)
            aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
            yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate_rankings}})}\n\n"

            # Stage 3: Synthesize final answer
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
            stage3_result = await stage3_synthesize_final(request.content, stage1_results, stage2_results)
            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"

            # Wait for title generation if it was started
            if title_task:
                title = await title_task
                storage.update_conversation_title(conversation_id, title)
                yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"

            # Save complete assistant message
            storage.add_assistant_message(
                conversation_id,
                stage1_results,
                stage2_results,
                stage3_result
            )

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            # Send error event
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


if __name__ == "__main__":
    import uvicorn
    # Validate API key before starting server
    from .config import OPENROUTER_API_KEY, COUNCIL_MODELS, CHAIRMAN_MODEL

    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set or is empty. Please configure it before starting the server.")

    # Print startup summary
    print("Starting LLM Council backend")
    print(f"Selected council models ({'free' if COUNCIL_MODELS == getattr(config, 'FREE_COUNCIL_MODELS', []) else 'premium'}):")
    for m in COUNCIL_MODELS:
        print(f"  - {m}")
    print(f"Chairman model: {CHAIRMAN_MODEL}")

    uvicorn.run(app, host="0.0.0.0", port=8001)
