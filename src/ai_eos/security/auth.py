"""Password hashing, JWT access tokens, API keys and role-based permissions."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import timedelta
from typing import Any

import bcrypt
from jose import JWTError, jwt

from ai_eos.domain.models import Role, utcnow

# --- Permissions ------------------------------------------------------------------

PERMISSIONS: dict[str, set[Role]] = {
    "chat": {Role.ADMIN, Role.OPERATOR},
    "tasks:read": {Role.ADMIN, Role.OPERATOR, Role.VIEWER},
    "approvals:decide": {Role.ADMIN, Role.OPERATOR},
    "memory:read": {Role.ADMIN, Role.OPERATOR, Role.VIEWER},
    "memory:write": {Role.ADMIN, Role.OPERATOR},
    "knowledge:write": {Role.ADMIN, Role.OPERATOR},
    "agents:direct": {Role.ADMIN, Role.OPERATOR},
    "prompts:write": {Role.ADMIN},
    "secrets:write": {Role.ADMIN},
    "users:manage": {Role.ADMIN},
    "audit:read": {Role.ADMIN},
    "metrics:read": {Role.ADMIN, Role.OPERATOR, Role.VIEWER},
}


def has_permission(role: str, permission: str) -> bool:
    try:
        return Role(role) in PERMISSIONS.get(permission, set())
    except ValueError:
        return False


# --- Passwords --------------------------------------------------------------------


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode()[:72], bcrypt.gensalt(rounds=12)).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode()[:72], hashed.encode())
    except ValueError:
        return False


def password_problems(password: str) -> list[str]:
    problems = []
    if len(password) < 12:
        problems.append("at least 12 characters")
    if password.lower() == password or password.upper() == password:
        problems.append("upper and lower case letters")
    if not any(c.isdigit() for c in password):
        problems.append("a digit")
    return problems


# --- JWT --------------------------------------------------------------------------


class TokenService:
    def __init__(self, secret: str, algorithm: str = "HS256", minutes: int = 60) -> None:
        if not secret:
            secret = secrets.token_urlsafe(48)  # dev: tokens die on restart
        self.secret, self.algorithm, self.minutes = secret, algorithm, minutes

    def issue(self, user_id: str, role: str) -> tuple[str, int]:
        now = utcnow()
        payload = {
            "sub": user_id,
            "role": role,
            "iat": now,
            "exp": now + timedelta(minutes=self.minutes),
            "jti": secrets.token_hex(8),
        }
        return jwt.encode(payload, self.secret, algorithm=self.algorithm), self.minutes * 60

    def verify(self, token: str) -> dict[str, Any] | None:
        try:
            return jwt.decode(token, self.secret, algorithms=[self.algorithm])
        except JWTError:
            return None


# --- API keys ---------------------------------------------------------------------

API_KEY_PREFIX = "eos_"


def generate_api_key() -> tuple[str, str, str]:
    """Return (full_key, lookup_prefix, sha256_hash). Only the hash is stored."""
    raw = secrets.token_urlsafe(32)
    full = f"{API_KEY_PREFIX}{raw}"
    return full, full[:12], hash_api_key(full)


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def api_key_matches(key: str, stored_hash: str) -> bool:
    return hmac.compare_digest(hash_api_key(key), stored_hash)
