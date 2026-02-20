"""
OllamaFreeAPI Gateway - SSO-protected API gateway for the OllamaFreeAPI library.
All access requires OAuth2/OIDC authentication via login.orcest.ai.
"""

import os
import time
import logging
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Depends, Cookie, Header
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

from ollamafreeapi import OllamaFreeAPI

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
SSO_ISSUER = os.environ.get("SSO_ISSUER", "https://login.orcest.ai")
SSO_CLIENT_ID = os.environ.get("SSO_CLIENT_ID", "ollamafreeapi")
SSO_CLIENT_SECRET = os.environ.get("SSO_CLIENT_SECRET", "")
SSO_CALLBACK_URL = os.environ.get(
    "SSO_CALLBACK_URL", "https://ollamafreeapi.orcest.ai/auth/callback"
)

COOKIE_NAME = "ollamafreeapi_sso_token"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ollamafreeapi.gateway")

# ---------------------------------------------------------------------------
# Token verification cache  (token -> (user_info, expiry_ts))
# ---------------------------------------------------------------------------
_token_cache: Dict[str, tuple] = {}
TOKEN_CACHE_TTL = 300  # 5 minutes

# ---------------------------------------------------------------------------
# Per-user usage tracking  (sub -> stats)
# ---------------------------------------------------------------------------
_user_usage: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="OllamaFreeAPI Gateway",
    description="SSO-protected API gateway for the OllamaFreeAPI library",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# OllamaFreeAPI library instance (initialised once)
# ---------------------------------------------------------------------------
api = OllamaFreeAPI()

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    stream: bool = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    num_predict: Optional[int] = 128


# ---------------------------------------------------------------------------
# SSO helpers
# ---------------------------------------------------------------------------

async def _verify_token_remote(token: str) -> Optional[Dict[str, Any]]:
    """Verify an access token against the SSO issuer's token-verify endpoint."""
    verify_url = f"{SSO_ISSUER}/api/token/verify"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                verify_url,
                json={"token": token},
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                data = resp.json()
                return data  # expected to contain user info
            logger.warning("SSO verify returned %s: %s", resp.status_code, resp.text)
            return None
    except httpx.HTTPError as exc:
        logger.error("SSO verify request failed: %s", exc)
        return None


async def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Return cached user info or verify remotely, caching for 5 minutes."""
    now = time.time()

    # Evict stale entries lazily
    cached = _token_cache.get(token)
    if cached is not None:
        user_info, expiry = cached
        if now < expiry:
            return user_info
        else:
            del _token_cache[token]

    user_info = await _verify_token_remote(token)
    if user_info is not None:
        _token_cache[token] = (user_info, now + TOKEN_CACHE_TTL)
    return user_info


def _extract_token(request: Request) -> Optional[str]:
    """Extract bearer token from cookie or Authorization header."""
    # 1. Cookie
    token = request.cookies.get(COOKIE_NAME)
    if token:
        return token

    # 2. Authorization: Bearer <token>
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()

    return None


async def require_sso(request: Request) -> Dict[str, Any]:
    """
    Dependency that enforces SSO authentication.
    Returns the verified user info dict on success; raises 401 otherwise.
    """
    token = _extract_token(request)
    if token is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please log in via SSO.",
        )

    user_info = await verify_token(token)
    if user_info is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token. Please log in again.",
        )
    return user_info


def _track_usage(user_info: Dict[str, Any], action: str) -> None:
    """Record per-user usage statistics."""
    sub = user_info.get("sub") or user_info.get("email") or "unknown"
    if sub not in _user_usage:
        _user_usage[sub] = {"requests": 0, "chats": 0, "name": _get_display_name(user_info)}
    _user_usage[sub]["requests"] += 1
    if action == "chat":
        _user_usage[sub]["chats"] += 1
    logger.info("User %s performed %s (total requests: %d)", sub, action, _user_usage[sub]["requests"])


def _get_display_name(user_info: Dict[str, Any]) -> str:
    """Best-effort display name from user info."""
    return (
        user_info.get("name")
        or user_info.get("preferred_username")
        or user_info.get("email")
        or "Unknown"
    )


# ---------------------------------------------------------------------------
# Public endpoints (no SSO required)
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Public health check for uptime monitors and load balancers."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.get("/auth/callback")
async def auth_callback(code: Optional[str] = None, state: Optional[str] = None):
    """
    OAuth2 authorization-code callback.
    Exchanges the code for tokens and sets the session cookie.
    """
    if code is None:
        raise HTTPException(status_code=400, detail="Missing authorization code")

    token_url = f"{SSO_ISSUER}/api/token"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                token_url,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": SSO_CALLBACK_URL,
                    "client_id": SSO_CLIENT_ID,
                    "client_secret": SSO_CLIENT_SECRET,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
    except httpx.HTTPError as exc:
        logger.error("Token exchange failed: %s", exc)
        raise HTTPException(status_code=502, detail="SSO token exchange failed")

    if resp.status_code != 200:
        logger.warning("Token exchange returned %s: %s", resp.status_code, resp.text)
        raise HTTPException(status_code=401, detail="SSO token exchange rejected")

    tokens = resp.json()
    access_token = tokens.get("access_token")
    if not access_token:
        raise HTTPException(status_code=502, detail="No access_token in SSO response")

    # Redirect to root and set cookie
    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(
        key=COOKIE_NAME,
        value=access_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=86400,  # 24 hours
        path="/",
    )
    return response


@app.get("/auth/logout")
async def auth_logout():
    """Clear session cookie and redirect to root."""
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie(key=COOKIE_NAME, path="/")
    return response


# ---------------------------------------------------------------------------
# Protected API endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root(user_info: Dict[str, Any] = Depends(require_sso)):
    """API information page (requires SSO)."""
    display_name = _get_display_name(user_info)
    _track_usage(user_info, "root")
    return {
        "service": "OllamaFreeAPI Gateway",
        "version": "1.0.0",
        "user": display_name,
        "docs": "/docs",
        "endpoints": {
            "families": "/api/families",
            "models": "/api/models",
            "models_by_family": "/api/models/{family}",
            "model_info": "/api/model/{name}",
            "chat": "/api/chat",
            "me": "/api/me",
        },
        "auth": {
            "logout": "/auth/logout",
        },
    }


@app.get("/api/families")
async def list_families(user_info: Dict[str, Any] = Depends(require_sso)):
    """List all model families (requires SSO)."""
    _track_usage(user_info, "list_families")
    try:
        families = api.list_families()
        return {"families": families}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/models")
async def list_models(user_info: Dict[str, Any] = Depends(require_sso)):
    """List all available models (requires SSO)."""
    _track_usage(user_info, "list_models")
    try:
        families = api.list_families()
        models = api.list_models()
        return {"families": families, "models": models, "total": len(models)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/models/{family}")
async def list_models_by_family(
    family: str,
    user_info: Dict[str, Any] = Depends(require_sso),
):
    """List models belonging to a specific family (requires SSO)."""
    _track_usage(user_info, "list_models_by_family")
    try:
        models = api.list_models(family=family)
        if not models:
            raise HTTPException(status_code=404, detail=f"Family '{family}' not found")
        return {"family": family, "models": models, "total": len(models)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/model/{name:path}")
async def get_model_info(
    name: str,
    user_info: Dict[str, Any] = Depends(require_sso),
):
    """Get full metadata for a specific model (requires SSO)."""
    _track_usage(user_info, "get_model_info")
    try:
        info = api.get_model_info(name)
        return info
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/chat")
async def chat(
    request: ChatRequest,
    user_info: Dict[str, Any] = Depends(require_sso),
):
    """Chat completion endpoint with per-user tracking (requires SSO)."""
    _track_usage(user_info, "chat")
    display_name = _get_display_name(user_info)
    logger.info("Chat request from %s | model=%s", display_name, request.model)

    try:
        if request.stream:
            def stream_generator():
                for chunk in api.stream_chat(
                    prompt=request.prompt,
                    model=request.model,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    num_predict=request.num_predict,
                ):
                    yield chunk

            return StreamingResponse(stream_generator(), media_type="text/plain")

        response_text = api.chat(
            prompt=request.prompt,
            model=request.model,
            temperature=request.temperature,
            top_p=request.top_p,
            num_predict=request.num_predict,
        )
        return {"response": response_text, "user": display_name}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/me")
async def me(user_info: Dict[str, Any] = Depends(require_sso)):
    """Return the current user's profile from the SSO token."""
    _track_usage(user_info, "me")
    sub = user_info.get("sub") or user_info.get("email") or "unknown"
    usage = _user_usage.get(sub, {"requests": 0, "chats": 0})
    return {
        "user": user_info,
        "display_name": _get_display_name(user_info),
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# Entrypoint for local development
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
