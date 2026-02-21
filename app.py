"""
FastAPI server exposing OllamaFreeAPI as a web service.
Provides both native and OpenAI-compatible API endpoints for Orcide IDE integration.
"""
import os
import json
import time
import uuid
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from ollamafreeapi import OllamaFreeAPI

app = FastAPI(
    title="OllamaFreeAPI - Orcest AI",
    description="Free API for open-source LLMs via Ollama. Part of the Orcest AI ecosystem.",
    version="1.0.0",
)

# CORS for Orcide IDE and all origins (public free API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize client at startup
api = OllamaFreeAPI()

# API Key for authentication (optional, from environment)
API_KEY = os.environ.get("OLLAMAFREEAPI_KEY", "")


def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verify API key if one is configured"""
    if not API_KEY:
        return True  # No key configured, allow all
    if not authorization:
        return False
    token = authorization.replace("Bearer ", "").strip()
    return token == API_KEY


# ============================================================================
# Native API Endpoints
# ============================================================================

class ChatRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    stream: bool = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    num_predict: Optional[int] = 128


@app.get("/")
async def root():
    """Health check and API info."""
    return {
        "message": "OllamaFreeAPI - Orcest AI Free LLM API",
        "docs": "/docs",
        "models": "/api/models",
        "openai_compatible": "/v1/chat/completions",
        "ecosystem": "https://orcest.ai",
    }


@app.get("/api/models")
async def list_models():
    """List all available models."""
    try:
        families = api.list_families()
        models = api.list_models()
        return {"families": families, "models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/{model_name}")
async def get_model_info(model_name: str):
    """Get info for a specific model."""
    try:
        info = api.get_model_info(model_name)
        return info
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat completion endpoint (native format)."""
    try:
        if request.stream:
            def stream():
                for chunk in api.stream_chat(
                    prompt=request.prompt,
                    model=request.model,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    num_predict=request.num_predict,
                ):
                    yield chunk

            return StreamingResponse(
                stream(),
                media_type="text/plain",
            )
        else:
            response = api.chat(
                prompt=request.prompt,
                model=request.model,
                temperature=request.temperature,
                top_p=request.top_p,
                num_predict=request.num_predict,
            )
            return {"response": response}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# OpenAI-Compatible API Endpoints (for Orcide IDE integration)
# ============================================================================

class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    n: Optional[int] = 1
    stop: Optional[List[str]] = None


@app.get("/v1/models")
async def openai_list_models(authorization: Optional[str] = Header(None)):
    """OpenAI-compatible models list endpoint."""
    try:
        models = api.list_models()
        model_list = []
        for model_name in models:
            model_list.append({
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollamafreeapi",
                "permission": [],
            })
        return {
            "object": "list",
            "data": model_list,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def openai_chat_completions(
    request: OpenAIChatRequest,
    authorization: Optional[str] = Header(None)
):
    """OpenAI-compatible chat completions endpoint for Orcide IDE."""
    # Build prompt from messages
    prompt_parts = []
    for msg in request.messages:
        if msg.role == "system":
            prompt_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")
    prompt_parts.append("Assistant:")
    prompt = "\n\n".join(prompt_parts)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model = request.model or "default"

    try:
        if request.stream:
            def stream_response():
                try:
                    for chunk in api.stream_chat(
                        prompt=prompt,
                        model=request.model,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        num_predict=request.max_tokens,
                    ):
                        chunk_data = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"

                    # Send final chunk
                    final_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    error_chunk = {
                        "error": {"message": str(e), "type": "server_error"}
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            response_text = api.chat(
                prompt=prompt,
                model=request.model,
                temperature=request.temperature,
                top_p=request.top_p,
                num_predict=request.max_tokens,
            )

            # Estimate token counts
            prompt_tokens = len(prompt.split()) * 2
            completion_tokens = len(response_text.split()) * 2

            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health & Utility
# ============================================================================

@app.get("/health")
async def health():
    """Health check for Render."""
    return {"status": "ok", "service": "ollamafreeapi.orcest.ai"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
