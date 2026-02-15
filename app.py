"""
FastAPI server exposing OllamaFreeAPI as a web service.
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List

from ollamafreeapi import OllamaFreeAPI

app = FastAPI(
    title="OllamaFreeAPI",
    description="Free API for open-source LLMs via Ollama",
    version="0.1.3",
)

# Initialize client at startup
api = OllamaFreeAPI()


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


@app.get("/health")
async def health():
    """Health check for Render."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
