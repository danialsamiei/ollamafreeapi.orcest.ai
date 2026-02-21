"""
FastAPI server exposing OllamaFreeAPI as a web service.
"""
import os
import json
import time
import uuid as uuid_module
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from ollamafreeapi import OllamaFreeAPI

app = FastAPI(
    title="OllamaFreeAPI",
    description="Free API for open-source LLMs via Ollama",
    version="0.1.3",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://*.orcest.ai", "http://localhost:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting configuration
_request_counts = defaultdict(list)
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 30


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Skip rate limiting for health and info endpoints
    if request.url.path in ["/", "/health", "/docs", "/openapi.json"]:
        return await call_next(request)

    client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
    now = time.time()
    _request_counts[client_ip] = [t for t in _request_counts[client_ip] if now - t < RATE_LIMIT_WINDOW]

    if len(_request_counts[client_ip]) >= RATE_LIMIT_MAX:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Try again later."},
            headers={"Retry-After": str(RATE_LIMIT_WINDOW)},
        )
    _request_counts[client_ip].append(now)
    return await call_next(request)


# Initialize client at startup
api = OllamaFreeAPI()


class ChatRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    stream: bool = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    num_predict: Optional[int] = 128


class OpenAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[dict]
    stream: bool = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048


@app.get("/")
async def root():
    """Health check and API info."""
    return {
        "message": "OllamaFreeAPI - Free LLM API",
        "docs": "/docs",
        "models": "/api/models",
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
    """Chat completion endpoint."""
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


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIChatRequest, authorization: Optional[str] = Header(None)):
    """OpenAI-compatible chat completions endpoint."""
    try:
        # Extract the last user message as the prompt
        prompt = ""
        for msg in reversed(request.messages):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break

        if not prompt:
            raise HTTPException(status_code=400, detail="No user message found")

        model_name = request.model or "default"
        completion_id = f"chatcmpl-{uuid_module.uuid4().hex[:24]}"
        created = int(time.time())

        if request.stream:
            async def stream_openai():
                try:
                    for chunk in api.stream_chat(
                        prompt=prompt,
                        model=model_name if model_name != "default" else None,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        num_predict=request.max_tokens,
                    ):
                        data = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None,
                            }]
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                    # Send final chunk
                    final = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }]
                    }
                    yield f"data: {json.dumps(final)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    error_data = {"error": {"message": str(e), "type": "server_error"}}
                    yield f"data: {json.dumps(error_data)}\n\n"

            return StreamingResponse(
                stream_openai(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            response_text = api.chat(
                prompt=prompt,
                model=model_name if model_name != "default" else None,
                temperature=request.temperature,
                top_p=request.top_p,
                num_predict=request.max_tokens,
            )
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(prompt.split()) + len(response_text.split()),
                }
            }
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_openai_models():
    """OpenAI-compatible model list."""
    try:
        models = api.list_models()
        return {
            "object": "list",
            "data": [
                {
                    "id": model if isinstance(model, str) else model.get("name", "unknown"),
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ollamafreeapi",
                }
                for model in (models if isinstance(models, list) else [])
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check for Render."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
