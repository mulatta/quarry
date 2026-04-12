{ inputs, ... }:
{
  perSystem =
    { pkgs, system, ... }:
    {
      # nixosTest runs in a VM — Linux only
      checks = pkgs.lib.optionalAttrs (system == "x86_64-linux" || system == "aarch64-linux") {
        nixos-quarry-server-smoke = pkgs.testers.nixosTest {
          name = "quarry-server-smoke";

          nodes.machine =
            { pkgs, ... }:
            {
              imports = [ (import ./nixos-module.nix { inherit inputs; }) ];

              services.postgresql = {
                enable = true;
                package = pkgs.postgresql_16;
                # Create DB + user + schema in one pass.
                # Tables are owned by postgres; grant quarry access.
                initialScript = pkgs.writeText "quarry-test-init" ''
                  CREATE USER quarry;
                  CREATE DATABASE quarry OWNER quarry;
                  \connect quarry
                  ${builtins.readFile ../sql/schema.sql}
                  GRANT ALL ON ALL TABLES IN SCHEMA public TO quarry;
                '';
              };

              services.quarry-server = {
                enable = true;
                requireAuth = true;
                # peer auth: quarry system user → quarry pg user, socket auth
                environmentFile = pkgs.writeText "quarry-server-test-env" ''
                  QUARRY_SERVER_PG_CONNINFO=host=/run/postgresql dbname=quarry user=quarry
                '';
              };
            };

          testScript = ''
            machine.start()
            machine.wait_for_unit("postgresql.service")
            machine.wait_for_unit("quarry-server.service")
            machine.wait_for_open_port(8000)

            with subtest("unauthenticated POST returns 401"):
                out = machine.succeed(
                    "curl -s -o /dev/null -w '%{http_code}' -X POST "
                    "http://127.0.0.1:8000/mcp"
                )
                assert out.strip() == "401", f"expected 401, got {out!r}"

            with subtest("wrong Bearer token returns 401"):
                out = machine.succeed(
                    "curl -s -o /dev/null -w '%{http_code}' -X POST "
                    "-H 'Authorization: Bearer wrongtoken' "
                    "http://127.0.0.1:8000/mcp"
                )
                assert out.strip() == "401", f"expected 401, got {out!r}"

            with subtest("valid Bearer token is accepted"):
                # Insert a test key: sha256("ci-test-token") stored as hex
                machine.succeed(
                    "sudo -u quarry psql quarry -c \""
                    "INSERT INTO api_keys (client_id, key_hash) VALUES "
                    "('ci-test', encode(sha256('ci-test-token'::bytea), 'hex'))\""
                )
                out = machine.succeed(
                    "curl -s -o /dev/null -w '%{http_code}' -X POST "
                    "-H 'Authorization: Bearer ci-test-token' "
                    "-H 'Content-Type: application/json' "
                    r"-d '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\","
                    r"\"params\":{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},"
                    r"\"clientInfo\":{\"name\":\"ci\",\"version\":\"0\"}}}' "
                    "http://127.0.0.1:8000/mcp"
                )
                assert out.strip() != "401", f"expected non-401 with valid token, got {out!r}"
          '';
        };
      };
    };
}
