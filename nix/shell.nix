{ inputs, ... }:
let
  pgSocketDir = "/tmp/quarry-pg";
in
{
  imports = [ inputs.process-compose-flake.flakeModule ];

  perSystem =
    { pkgs, ... }:
    {
      # Dev PostgreSQL via process-compose (user-local, no system PG interaction)
      process-compose.services = {
        imports = [ inputs.services-flake.processComposeModules.default ];

        services.postgres.quarry-pg = {
          enable = true;
          package = pkgs.postgresql_16;
          dataDir = "./.pg-data";
          socketDir = pgSocketDir;
          settings.listen_addresses = "";
          superuser = null;
          initialDatabases = [
            { name = "quarry"; }
          ];
        };
      };

      devShells.default = pkgs.mkShell {
        packages = with pkgs; [
          process-compose
          python313
          uv
          rustc
          cargo
          clippy
          maturin
          just
          postgresql_16
        ];

        env = {
          UV_PYTHON_DOWNLOADS = "never";
          LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
          # Single source of truth — consumed by psql, psycopg, and quarry config
          PGHOST = pgSocketDir;
          QUARRY_PG_CONNINFO = "host=${pgSocketDir} dbname=quarry";
        };

        shellHook = ''
          if [ -f pyproject.toml ]; then
            if [[ ! -f "uv.lock" ]] || [[ "pyproject.toml" -nt "uv.lock" ]]; then
              uv lock --quiet
            fi
            uv sync --all-extras --quiet
          fi

          export VIRTUAL_ENV="$PWD/.venv"
          export PATH="$VIRTUAL_ENV/bin:$PATH"
        '';
      };
    };
}
