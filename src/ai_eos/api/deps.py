"""FastAPI dependencies: container access, authentication and permission checks."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import Depends, HTTPException, Request, status

from ai_eos.container import Container
from ai_eos.security.auth import API_KEY_PREFIX, api_key_matches, has_permission

User = dict[str, Any]


def get_container(request: Request) -> Container:
    return request.app.state.container


async def _user_from_credentials(c: Container, token: str | None, api_key: str | None) -> User | None:
    if token:
        claims = c.tokens.verify(token)
        if claims:
            user = await c.users.get(claims["sub"])
            if user and user["is_active"]:
                return user
    if api_key and api_key.startswith(API_KEY_PREFIX):
        for key_id, user_id, key_hash in await c.users.find_api_keys(api_key[:12]):
            if api_key_matches(api_key, key_hash):
                user = await c.users.get(user_id)
                if user and user["is_active"]:
                    await c.users.touch_api_key(key_id)
                    return user
    return None


async def current_user(request: Request, c: Container = Depends(get_container)) -> User:
    auth = request.headers.get("authorization", "")
    token = auth[7:] if auth.lower().startswith("bearer ") else None
    if token and token.startswith(API_KEY_PREFIX):  # allow API keys as bearer tokens too
        token, api_key = None, auth[7:]
    else:
        api_key = request.headers.get("x-api-key")
    if request.url.path.endswith("/events") and not token and not api_key:
        token = request.query_params.get("token")  # EventSource cannot send headers
    user = await _user_from_credentials(c, token, api_key)
    if not user:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "authentication required", headers={"WWW-Authenticate": "Bearer"}
        )
    request.state.user_id = user["id"]
    return user


def require(permission: str) -> Callable[..., Coroutine[Any, Any, User]]:
    async def checker(user: User = Depends(current_user)) -> User:
        if not has_permission(user["role"], permission):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"requires permission '{permission}'")
        return user

    return checker


def client_ip(request: Request) -> str:
    return request.client.host if request.client else ""
