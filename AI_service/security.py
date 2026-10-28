import base64
import binascii
import hashlib
import hmac
import os
import secrets
import time

from AI_service import settings


PASSWORD_ITERATIONS = 310_000
_process_secret = secrets.token_bytes(32)


def _secret() -> bytes:
    configured = os.getenv("AUTH_SECRET", settings.AUTH_SECRET)
    return configured.encode("utf-8") if configured else _process_secret


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS
    )
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, rounds, salt_text, digest_text = encoded.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            _unb64(salt_text),
            int(rounds)
        )
        return hmac.compare_digest(digest, _unb64(digest_text))
    except (ValueError, TypeError):
        return False


def create_access_token(user_id: int) -> str:
    expires_at = int(time.time()) + settings.ACCESS_TOKEN_TTL_SECONDS
    payload = _b64(f"{user_id}:{expires_at}".encode("ascii"))
    signature = hmac.new(_secret(), payload.encode("ascii"), hashlib.sha256)
    return f"{payload}.{_b64(signature.digest())}"


def read_access_token(token: str) -> int | None:
    try:
        payload, signature_text = token.split(".", 1)
        expected = hmac.new(
            _secret(),
            payload.encode("ascii"),
            hashlib.sha256
        ).digest()
        if not hmac.compare_digest(expected, _unb64(signature_text)):
            return None
        user_id_text, expiry_text = _unb64(payload).decode("ascii").split(":")
        user_id = int(user_id_text)
        if user_id < 1 or int(expiry_text) <= int(time.time()):
            return None
        return user_id
    except (ValueError, UnicodeDecodeError, binascii.Error):
        return None


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
