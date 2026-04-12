"""Static API-key authentication for quarry-server.

Keys are stored in PostgreSQL (api_keys table).
Only the sha256 hash of the token is persisted — the raw token is never stored.

TokenVerifier protocol (mcp.server.auth.provider) is implemented so FastMCP
picks it up automatically via token_verifier= constructor argument.
"""

from __future__ import annotations

import hashlib

from mcp.server.auth.provider import AccessToken

from quarry.store.pg import PGStore


class PgTokenVerifier:
    """Verify Bearer tokens against the api_keys table."""

    def __init__(self, pg_conninfo: str) -> None:
        self._conninfo = pg_conninfo
        self._db: PGStore | None = None

    def _get_db(self) -> PGStore:
        if self._db is None:
            self._db = PGStore(self._conninfo)
        return self._db

    async def verify_token(self, token: str) -> AccessToken | None:
        key_hash = hashlib.sha256(token.encode()).hexdigest()
        db = self._get_db()
        with db.conn.cursor() as cur:
            cur.execute(
                "SELECT client_id, scopes FROM api_keys "
                "WHERE key_hash = %s AND (expires_at IS NULL OR expires_at > now())",
                (key_hash,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return AccessToken(token=token, client_id=row[0], scopes=row[1] or [])
