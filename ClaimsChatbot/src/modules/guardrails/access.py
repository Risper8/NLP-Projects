from __future__ import annotations
from dataclasses import dataclass

ADMIN_DOMAIN = "zep-re.com"


@dataclass(frozen=True)
class ClientScope:
    email: str
    domain: str
    is_admin: bool

    @property
    def filter_domain(self) -> str | None:
        """Domain to filter graph queries by, or None to skip filtering (admin)."""
        return None if self.is_admin else self.domain


def resolve_client_scope(email: str) -> ClientScope:
    if not email or "@" not in email:
        raise ValueError(f"Invalid email address: {email!r}")

    domain = email.strip().lower().split("@", 1)[1]

    return ClientScope(
        email=email,
        domain=domain,
        is_admin=domain == ADMIN_DOMAIN,
    )
