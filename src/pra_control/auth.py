"""Provider-neutral browser authentication and signed Control Plane sessions."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from .config import ControlAuthConfig, IdentityProviderConfig
from .domain import CallerContext
from .rbac import Role, permissions_for_role


@dataclass(frozen=True)
class Identity:
    subject: str
    display_name: str
    email: str | None
    role: Role
    provider: str
    csrf_token: str

    def public(self) -> dict[str, Any]:
        return {
            "subject": self.subject, "display_name": self.display_name,
            "email": self.email, "role": self.role.value, "provider": self.provider,
        }

    def caller(
        self, *, transport: str, request_id: str | None = None,
        trace_id: str | None = None, metadata: dict[str, Any] | None = None,
    ) -> CallerContext:
        """Map browser/session identity into the canonical manager identity."""
        return CallerContext(
            subject=self.subject, roles=[self.role.value],
            permissions=set(permissions_for_role(self.role)), auth_source=self.provider,
            request_id=request_id, trace_id=trace_id, transport=transport,
            metadata=metadata or {},
        )


class SessionCodec:
    def __init__(self, secret: str, ttl_seconds: int) -> None:
        self.serializer = URLSafeTimedSerializer(secret, salt="pra-control-session-v1")
        self.ttl_seconds = ttl_seconds

    def encode(self, identity: Identity) -> str:
        return self.serializer.dumps({
            "sub": identity.subject, "name": identity.display_name,
            "email": identity.email, "role": identity.role.value,
            "provider": identity.provider, "csrf": identity.csrf_token,
        })

    def decode(self, token: str) -> Identity | None:
        try:
            value = self.serializer.loads(token, max_age=self.ttl_seconds)
            return Identity(
                subject=value["sub"], display_name=value["name"], email=value.get("email"),
                role=Role(value["role"]), provider=value["provider"], csrf_token=value["csrf"],
            )
        except (BadSignature, SignatureExpired, KeyError, ValueError):
            return None


class AuthService:
    """Authenticate local users and standards-based external providers."""

    def __init__(self, config: ControlAuthConfig, public_url: str, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        secret = config.cookie_secret()
        if not secret:
            raise ValueError(f"set {config.cookie_secret_env}")
        self.config = config
        self.public_url = public_url
        self.codec = SessionCodec(secret, config.session_ttl_seconds)
        self.transport = transport

    def providers(self) -> list[dict[str, str]]:
        return [{"name": row.name, "kind": row.kind} for row in self.config.providers if row.enabled]

    def provider(self, name: str) -> IdentityProviderConfig:
        for provider in self.config.providers:
            if provider.name == name and provider.enabled:
                return provider
        raise KeyError(name)

    def local_login(self, username: str, password: str) -> Identity | None:
        for user in self.config.local_users:
            expected = user.password()
            if user.username == username and expected and hmac.compare_digest(password, expected):
                return Identity(
                    subject=f"local:{username}", display_name=user.display_name or username,
                    email=None, role=user.role, provider="local", csrf_token=secrets.token_urlsafe(24),
                )
        return None

    def development_identity(self) -> Identity | None:
        providers = [provider for provider in self.config.providers if provider.enabled]
        if providers and all(provider.kind == "local" for provider in providers) and not self.config.local_users:
            return Identity("local:developer", "Local developer", None, Role.ADMINISTRATOR, "local", "local-dev-csrf")
        return None

    def begin(self, provider_name: str) -> tuple[str, str]:
        provider = self.provider(provider_name)
        if provider.kind not in {"github", "google", "oidc"}:
            raise ValueError("provider does not use OAuth/OIDC")
        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(48)
        challenge = _base64url(hashlib.sha256(verifier.encode()).digest())
        transaction = self.codec.serializer.dumps({"state": state, "nonce": nonce, "verifier": verifier}, salt=f"oauth:{provider.name}")
        authorization_url = provider.authorization_url or {
            "github": "https://github.com/login/oauth/authorize",
            "google": "https://accounts.google.com/o/oauth2/v2/auth",
        }.get(provider.kind)
        if not authorization_url:
            raise ValueError("authorization_url is required")
        query = urlencode({
            "client_id": provider.client_id, "redirect_uri": f"{self.public_url}/auth/callback/{provider.name}",
            "response_type": "code", "scope": " ".join(provider.scopes), "state": state,
            "nonce": nonce, "code_challenge": challenge, "code_challenge_method": "S256",
        })
        return f"{authorization_url}?{query}", transaction

    async def callback(self, provider_name: str, code: str, state: str, transaction: str) -> Identity:
        provider = self.provider(provider_name)
        try:
            expected = self.codec.serializer.loads(transaction, max_age=600, salt=f"oauth:{provider.name}")
        except (BadSignature, SignatureExpired) as error:
            raise ValueError("expired or invalid OAuth transaction") from error
        if not hmac.compare_digest(state, expected["state"]):
            raise ValueError("OAuth state mismatch")
        token_url = provider.token_url or {
            "github": "https://github.com/login/oauth/access_token",
            "google": "https://oauth2.googleapis.com/token",
        }.get(provider.kind)
        userinfo_url = provider.userinfo_url or {
            "github": "https://api.github.com/user", "google": "https://openidconnect.googleapis.com/v1/userinfo",
        }.get(provider.kind)
        if not token_url or not userinfo_url:
            raise ValueError("token_url and userinfo_url are required")
        async with httpx.AsyncClient(transport=self.transport, timeout=10) as client:
            token_response = await client.post(token_url, data={
                "grant_type": "authorization_code", "code": code,
                "redirect_uri": f"{self.public_url}/auth/callback/{provider.name}",
                "client_id": provider.client_id, "client_secret": provider.client_secret(),
                "code_verifier": expected["verifier"],
            }, headers={"Accept": "application/json"})
            token_response.raise_for_status()
            token_payload = token_response.json()
            access_token = token_payload["access_token"]
            oidc_claims: dict[str, Any] = {}
            if provider.kind in {"google", "oidc"}:
                oidc_claims = await self._validate_id_token(client, provider, token_payload.get("id_token"), expected["nonce"])
            user_response = await client.get(userinfo_url, headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"})
            user_response.raise_for_status()
            claims = {**oidc_claims, **user_response.json()}
        role_value = claims.get(provider.role_claim, provider.default_role.value)
        try:
            role = Role(role_value)
        except ValueError:
            role = provider.default_role
        subject = str(claims.get("sub") or claims.get("id") or claims.get("login"))
        return Identity(
            subject=f"{provider.name}:{subject}",
            display_name=str(claims.get("name") or claims.get("login") or claims.get("email") or subject),
            email=claims.get("email"), role=role, provider=provider.name,
            csrf_token=secrets.token_urlsafe(24),
        )

    async def _validate_id_token(
        self, client: httpx.AsyncClient, provider: IdentityProviderConfig,
        encoded: str | None, expected_nonce: str,
    ) -> dict[str, Any]:
        if not encoded:
            raise ValueError("OIDC token response has no id_token")
        metadata: dict[str, Any] = {}
        metadata_url = provider.metadata_url or (f"{provider.issuer.rstrip('/')}/.well-known/openid-configuration" if provider.issuer else None)
        if metadata_url:
            response = await client.get(metadata_url)
            response.raise_for_status()
            metadata = response.json()
        issuer = provider.issuer or metadata.get("issuer") or ("https://accounts.google.com" if provider.kind == "google" else None)
        jwks_url = provider.jwks_url or metadata.get("jwks_uri") or ("https://www.googleapis.com/oauth2/v3/certs" if provider.kind == "google" else None)
        if not issuer or not jwks_url:
            raise ValueError("OIDC issuer and JWKS URL are required")
        try:
            import jwt
        except ImportError as error:
            raise ValueError("install the control-plane authentication dependencies") from error
        response = await client.get(jwks_url)
        response.raise_for_status()
        header = jwt.get_unverified_header(encoded)
        keys = jwt.PyJWKSet.from_dict(response.json()).keys
        key = next((candidate for candidate in keys if candidate.key_id == header.get("kid")), None)
        if not key:
            raise ValueError("OIDC signing key is not trusted")
        claims = jwt.decode(encoded, key.key, algorithms=[header["alg"]], audience=provider.client_id, issuer=issuer)
        if not hmac.compare_digest(str(claims.get("nonce", "")), expected_nonce):
            raise ValueError("OIDC nonce mismatch")
        return dict(claims)


def _base64url(value: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")
