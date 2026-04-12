"""quarry-server CLI — MCP server management and API key administration."""

from __future__ import annotations

import hashlib
import secrets
from typing import Optional

import typer

app = typer.Typer(
    name="quarry-server",
    help="Quarry MCP server management.",
    no_args_is_help=True,
)

keys_app = typer.Typer(help="API key management.", no_args_is_help=True)
app.add_typer(keys_app, name="keys")


@app.command()
def serve() -> None:
    """Start the MCP HTTP server."""
    from quarry.server.app import mcp
    from quarry.server.config import server_settings

    mcp.settings.host = server_settings.host
    mcp.settings.port = server_settings.port
    mcp.run(transport="streamable-http")


@keys_app.command("create")
def keys_create(
    name: str = typer.Argument(..., help="Client identifier, e.g. 'seungwon-claude'"),
    expires_days: Optional[int] = typer.Option(
        None, "--expires", "-e", help="Expiry in days. Omit for no expiry."
    ),
) -> None:
    """Issue a new API key and print the token once."""
    import psycopg

    from quarry.server.config import server_settings

    token = "qry_" + secrets.token_hex(32)
    key_hash = hashlib.sha256(token.encode()).hexdigest()

    with psycopg.connect(server_settings.pg_conninfo) as conn:
        with conn.cursor() as cur:
            if expires_days is not None:
                cur.execute(
                    "INSERT INTO api_keys (client_id, key_hash, expires_at)"
                    " VALUES (%s, %s, now() + %s * INTERVAL '1 day')",
                    (name, key_hash, expires_days),
                )
            else:
                cur.execute(
                    "INSERT INTO api_keys (client_id, key_hash) VALUES (%s, %s)",
                    (name, key_hash),
                )
        conn.commit()

    typer.echo(f"Created key for '{name}'")
    if expires_days:
        typer.echo(f"Expires in: {expires_days} days")
    typer.echo("Token (copy now — not shown again):")
    typer.echo(f"  {token}")


@keys_app.command("list")
def keys_list() -> None:
    """List all API keys (client_id, created_at, expires_at)."""
    import psycopg

    from quarry.server.config import server_settings

    with psycopg.connect(server_settings.pg_conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT client_id, created_at, expires_at FROM api_keys ORDER BY created_at"
            )
            rows = cur.fetchall()

    if not rows:
        typer.echo("No API keys found.")
        return

    typer.echo(f"{'CLIENT ID':<30} {'CREATED':<22} {'EXPIRES'}")
    typer.echo("-" * 70)
    for client_id, created_at, expires_at in rows:
        exp = str(expires_at.date()) if expires_at else "never"
        typer.echo(f"{client_id:<30} {str(created_at)[:19]:<22} {exp}")


@keys_app.command("revoke")
def keys_revoke(
    client_id: str = typer.Argument(..., help="Client ID to revoke"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Revoke an API key by client ID."""
    import psycopg

    from quarry.server.config import server_settings

    if not yes:
        typer.confirm(f"Revoke key for '{client_id}'?", abort=True)

    with psycopg.connect(server_settings.pg_conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM api_keys WHERE client_id = %s", (client_id,))
            deleted = cur.rowcount
        conn.commit()

    if deleted:
        typer.echo(f"Revoked key for '{client_id}'.")
    else:
        typer.echo(f"No key found for '{client_id}'.", err=True)
        raise typer.Exit(1)
