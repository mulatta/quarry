{ inputs, ... }:
let
  pgSocketDir = "/tmp/quarry-pg";

  shellPackages =
    pkgs: with pkgs; [
      process-compose
      python313
      uv
      rustc
      cargo
      clippy
      maturin
      cmake
      just
      postgresql_16
      clickhouse
      awscli2
    ];

  shellEnv = {
    UV_PYTHON_DOWNLOADS = "never";
    # Single source of truth — consumed by psql, psycopg, and quarry config
    PGHOST = pgSocketDir;
    QUARRY_PG_CONNINFO = "host=${pgSocketDir} dbname=quarry";
  };

  activateVenv = ''
    export VIRTUAL_ENV="$PWD/.venv"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
  '';
in
{
  imports = [ inputs.process-compose-flake.flakeModule ];

  perSystem =
    { pkgs, self', ... }:
    {
      # Dev PostgreSQL + ClickHouse via process-compose (user-local)
      process-compose.services = {
        imports = [ inputs.services-flake.processComposeModules.default ];

        services.postgres.quarry-pg = {
          enable = true;
          package = pkgs.postgresql_16;
          dataDir = "./.pg-data";
          socketDir = pgSocketDir;
          settings = {
            listen_addresses = "";
            synchronous_commit = "off";
            wal_buffers = "64MB";
            max_wal_size = "4GB";
            shared_buffers = "2GB";
            work_mem = "256MB";
            maintenance_work_mem = "1GB";
            autovacuum = "off";
          };
          superuser = null;
          initialDatabases = [
            {
              name = "quarry";
              schemas = [ ../sql/schema.sql ];
            }
          ];
        };

        services.clickhouse."quarry-ch" = {
          enable = true;
          port = 9001;
          extraConfig = {
            http_port = 8124;
          };
          initialDatabases = [
            {
              name = "quarry";
              schemas = [ ../sql/ch_schema.sql ];
            }
          ];
        };
      };

      devShells.default = pkgs.mkShell {
        packages = (shellPackages pkgs) ++ [ self'.packages.quarry-parse ];
        env = shellEnv // {
          LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
        };
        shellHook = ''
          uv sync --all-extras --quiet
          ${activateVenv}
        '';
      };
    };
}
