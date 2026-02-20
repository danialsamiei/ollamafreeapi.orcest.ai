"""
FastAPI server exposing OllamaFreeAPI as a web service.

Provides both the native OllamaFreeAPI endpoints and OpenAI-compatible
endpoints (/v1/chat/completions, /v1/models) for integration with
LiteLLM, LangChain, and other OpenAI-compatible clients.
"""
import json
import os
import random
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from ollama import Client
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from ollamafreeapi import OllamaFreeAPI

app = FastAPI(
    title="OllamaFreeAPI",
    description="Free API for open-source LLMs via Ollama",
    version="0.1.3",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize client at startup
api = OllamaFreeAPI()


# ── Native OllamaFreeAPI models ──

class ChatRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    stream: bool = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    num_predict: Optional[int] = 128


# ── OpenAI-compatible models ──

class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OpenAIMessage]
    stream: bool = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    stop: Optional[List[str]] = None


@app.get("/")
async def root():
    """Health check and API info."""
    return {
        "message": "OllamaFreeAPI - Free LLM API",
        "docs": "/docs",
        "models": "/api/models",
        "openai_models": "/v1/models",
        "openai_chat": "/v1/chat/completions",
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


# ── OpenAI-compatible endpoints ──

def _select_server_and_model(model_name: Optional[str]):
    """Select a model and server for an OpenAI-compatible request."""
    if model_name is None or model_name == "":
        all_models = api.list_models()
        if not all_models:
            raise RuntimeError("No models available")
        model_name = random.choice(all_models)

    servers = api.get_model_servers(model_name)
    if not servers:
        raise RuntimeError(f"No servers available for model '{model_name}'")

    random.shuffle(servers)
    return model_name, servers


def _openai_chat_sync(model: str, servers: list, messages: list, **kwargs) -> dict:
    """Execute a chat completion against Ollama servers, returning OpenAI format."""
    ollama_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
    options = {}
    if kwargs.get("temperature") is not None:
        options["temperature"] = kwargs["temperature"]
    if kwargs.get("top_p") is not None:
        options["top_p"] = kwargs["top_p"]
    if kwargs.get("max_tokens") is not None:
        options["num_predict"] = kwargs["max_tokens"]
    if kwargs.get("stop") is not None:
        options["stop"] = kwargs["stop"]

    last_error = None
    for server in servers:
        try:
            client = Client(host=server["url"])
            response = client.chat(
                model=model,
                messages=ollama_messages,
                options=options,
            )
            content = response.get("message", {}).get("content", "")
            prompt_tokens = response.get("prompt_eval_count", 0) or 0
            completion_tokens = response.get("eval_count", 0) or 0
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"All servers failed for model '{model}'. Last error: {last_error!s}"
    )


def _openai_chat_stream(model: str, servers: list, messages: list, **kwargs):
    """Stream a chat completion in SSE format (OpenAI-compatible)."""
    ollama_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
    options = {}
    if kwargs.get("temperature") is not None:
        options["temperature"] = kwargs["temperature"]
    if kwargs.get("top_p") is not None:
        options["top_p"] = kwargs["top_p"]
    if kwargs.get("max_tokens") is not None:
        options["num_predict"] = kwargs["max_tokens"]
    if kwargs.get("stop") is not None:
        options["stop"] = kwargs["stop"]

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    last_error = None
    for server in servers:
        try:
            client = Client(host=server["url"])
            stream = client.chat(
                model=model,
                messages=ollama_messages,
                options=options,
                stream=True,
            )

            for chunk in stream:
                token = chunk.get("message", {}).get("content", "")
                done = chunk.get("done", False)

                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token} if token else {},
                            "finish_reason": "stop" if done else None,
                        }
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"

                if done:
                    break

            yield "data: [DONE]\n\n"
            return
        except Exception as e:
            last_error = e
            continue

    error_data = {
        "error": {
            "message": f"All servers failed for model '{model}': {last_error!s}",
            "type": "upstream_error",
        }
    }
    yield f"data: {json.dumps(error_data)}\n\n"


@app.get("/v1/models")
async def openai_list_models():
    """List available models in OpenAI format."""
    try:
        models = api.list_models()
        data = [
            {
                "id": m,
                "object": "model",
                "created": 0,
                "owned_by": "ollamafreeapi",
            }
            for m in set(models)
        ]
        return {"object": "list", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIChatRequest):
    """OpenAI-compatible chat completions endpoint."""
    try:
        model, servers = _select_server_and_model(request.model)
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        if request.stream:
            return StreamingResponse(
                _openai_chat_stream(
                    model=model,
                    servers=servers,
                    messages=messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    stop=request.stop,
                ),
                media_type="text/event-stream",
            )
        else:
            result = _openai_chat_sync(
                model=model,
                servers=servers,
                messages=messages,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop,
            )
            return JSONResponse(content=result)
    except RuntimeError as e:
        return JSONResponse(
            status_code=503,
            content={
                "error": {"message": str(e), "type": "upstream_error"}
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": {"message": str(e), "type": "server_error"}
            },
        )


@app.get("/health")
async def health():
    """Health check for Render."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
