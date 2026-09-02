"""Optional SAML 2.0 adapter backed by the security-focused python3-saml library."""

from __future__ import annotations

from typing import Any, Mapping

from .auth import Identity
from .config import IdentityProviderConfig
from .rbac import Role


class SAMLUnavailable(RuntimeError):
    pass


class SAMLService:
    """Validate signed SAML responses without making SAML a base dependency."""

    def __init__(self, provider: IdentityProviderConfig, public_url: str) -> None:
        self.provider = provider
        self.public_url = public_url.rstrip("/")
        try:
            from onelogin.saml2.auth import OneLogin_Saml2_Auth
            from onelogin.saml2.idp_metadata_parser import OneLogin_Saml2_IdPMetadataParser
        except ImportError as error:
            raise SAMLUnavailable("install pra-hf[control-plane-saml] to enable SAML") from error
        self.auth_type = OneLogin_Saml2_Auth
        self.metadata_parser = OneLogin_Saml2_IdPMetadataParser

    def begin(self, request_data: Mapping[str, Any], return_to: str) -> str:
        return self.auth_type(dict(request_data), self._settings()).login(return_to=return_to)

    def callback(self, request_data: Mapping[str, Any]) -> Identity:
        auth = self.auth_type(dict(request_data), self._settings())
        auth.process_response()
        errors = auth.get_errors()
        if errors or not auth.is_authenticated():
            raise ValueError("invalid signed SAML response: " + ", ".join(errors))
        attributes = auth.get_attributes()
        role_value = _first(attributes.get(self.provider.role_claim)) or self.provider.default_role.value
        try:
            role = Role(role_value)
        except ValueError:
            role = self.provider.default_role
        subject = auth.get_nameid()
        if not subject:
            raise ValueError("SAML assertion has no subject")
        import secrets
        return Identity(
            subject=f"{self.provider.name}:{subject}",
            display_name=_first(attributes.get("displayName")) or subject,
            email=_first(attributes.get("email")), role=role,
            provider=self.provider.name, csrf_token=secrets.token_urlsafe(24),
        )

    def _settings(self) -> dict[str, Any]:
        if not self.provider.saml_metadata_url:
            raise ValueError("saml_metadata_url is required")
        # Metadata supplies the trusted IdP endpoints and signing certificates.
        settings = self.metadata_parser.parse_remote(self.provider.saml_metadata_url)
        settings.update({
            "strict": True,
            "sp": {
                "entityId": f"{self.public_url}/saml/metadata/{self.provider.name}",
                "assertionConsumerService": {
                    "url": f"{self.public_url}/auth/callback/{self.provider.name}",
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST",
                },
            },
            "security": {
                "wantAssertionsSigned": True, "wantMessagesSigned": True,
                "wantAttributeStatement": True, "rejectUnsolicitedResponsesWithInResponseTo": True,
            },
        })
        return settings


def _first(value: Any) -> str | None:
    if isinstance(value, list) and value:
        return str(value[0])
    return str(value) if value else None
