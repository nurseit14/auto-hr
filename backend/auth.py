"""Small local-demo authentication without client-supplied roles or bearer data."""

import hashlib
import hmac
import secrets


AUTH_COOKIE = "autohr_auth"
AUTH_MAX_AGE = 30 * 24 * 60 * 60
PASSWORD_ITERATIONS = 240_000
BUILTIN_ACCOUNTS = (
    ("admin", "admin", "hr", "HR"),
    ("user", "user", "candidate", "Кандидат"),
)


def password_hash(password):
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password, encoded):
    try:
        algorithm, rounds, salt, expected = encoded.split("$")
        if algorithm != "pbkdf2_sha256" or not 1 <= int(rounds) <= 1_000_000:
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(rounds)).hex()
        return hmac.compare_digest(digest, expected)
    except (ValueError, TypeError, AttributeError):
        return False


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def public_user(account):
    return {key: account[key] for key in ("username", "role", "display_name")}
