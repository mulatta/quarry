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
      just
      postgresql_16
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
      # Dev PostgreSQL via process-compose (user-local, no system PG interaction)
      process-compose.services = {
        imports = [ inputs.services-flake.processComposeModules.default ];

        services.postgres.quarry-pg = {
          enable = true;
          package = pkgs.postgresql_16;
          dataDir = "./.pg-data";
          socketDir = pgSocketDir;
          settings = {
            listen_addresses = "";
            # Bulk-load tuning: reduce WAL contention for parallel COPY writers
            synchronous_commit = "off"; # safe: data is re-loadable from S3/XML
            wal_buffers = "64MB"; # reduce WALInsert lock contention (default ~4MB)
            max_wal_size = "4GB"; # fewer checkpoints during bulk load
            shared_buffers = "2GB"; # more buffer pool for BufferContent locks
            work_mem = "256MB";
            maintenance_work_mem = "1GB"; # faster CREATE INDEX / VACUUM
            autovacuum = "off"; # bulk load only; VACUUM ANALYZE runs after load
          };
          superuser = null;
          initialDatabases = [
            {
              name = "quarry";
              schemas = [ ../sql ];
            }
          ];
        };
      };

      # nix develop        — debug Rust build (fast iteration)
      devShells.default = pkgs.mkShell {
        packages = shellPackages pkgs;
        env = shellEnv // {
          LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
        };
        shellHook = ''
          uv sync --all-extras --quiet
          ${activateVenv}
          # quarry-ingest: prefer local debug build, fall back to cargo build
          if [ -x "$PWD/quarry-ingest/target/debug/quarry-ingest" ]; then
            export PATH="$PWD/quarry-ingest/target/debug:$PATH"
          elif [ -x "$PWD/quarry-ingest/target/release/quarry-ingest" ]; then
            export PATH="$PWD/quarry-ingest/target/release:$PATH"
          fi
        '';
      };

      # nix develop .#release — release Rust build (production)
      devShells.release = pkgs.mkShell {
        packages = (shellPackages pkgs) ++ [ self'.packages.quarry-ingest ];
        env = shellEnv // {
          LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
        };
        shellHook = ''
          uv sync --all-extras -C build-args='--profile=release' --quiet
          ${activateVenv}
        '';
      };
    };
}
