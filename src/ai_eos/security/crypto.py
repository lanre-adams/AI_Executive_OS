"""Symmetric encryption for data at rest (approval payloads, integration secrets)."""

from __future__ import annotations

import base64
import hashlib
import logging

from cryptography.fernet import Fernet, InvalidToken

log = logging.getLogger(__name__)


class Cipher:
    def __init__(self, key: str, fallback_seed: str = "") -> None:
        if key:
            self._f = Fernet(key.encode())
        else:
            # Development convenience only; production refuses to start without a key (config.validate_settings).
            log.warning("EOS_ENCRYPTION_KEY not set; deriving a development key. Do not use in production.")
            digest = hashlib.sha256(("eos-dev-key:" + fallback_seed).encode()).digest()
            self._f = Fernet(base64.urlsafe_b64encode(digest))

    def encrypt(self, plaintext: str) -> str:
        return self._f.encrypt(plaintext.encode()).decode()

    def decrypt(self, token: str) -> str:
        try:
            return self._f.decrypt(token.encode()).decode()
        except InvalidToken as exc:
            raise ValueError("cannot decrypt value; was EOS_ENCRYPTION_KEY changed?") from exc
