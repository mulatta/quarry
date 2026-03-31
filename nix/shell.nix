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
      duckdb
      awscli2
      cudaPackages.cuda_cudart
      cudaPackages.cuda_nvcc
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
            # Memory — tuned for 188GB system
            shared_buffers = "8GB";
            work_mem = "256MB";
            maintenance_work_mem = "4GB";
            effective_io_concurrency = 200; # NVMe
            autovacuum = "off";
            # Parallelism — tuned for 48-core system
            max_parallel_maintenance_workers = 8; # parallel index build
            max_parallel_workers_per_gather = 4;
            max_parallel_workers = 16;
            max_worker_processes = 20;
          };
          superuser = null;
          initialDatabases = [
            {
              name = "quarry";
              schemas = [ ../sql/schema.sql ];
            }
            { name = "dagster"; }
          ];
        };

        services.clickhouse."quarry-ch" = {
          enable = true;
          port = 9000;
          extraConfig = {
            http_port = 8124;
            # Performance — tuned for 48-core 188GB system
            max_threads = 48;
            max_insert_threads = 8;
            max_memory_usage = "64000000000"; # 64GB
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
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            "${pkgs.addDriverRunpath.driverLink}"
          ];
          TRITON_LIBCUDA_PATH = "${pkgs.addDriverRunpath.driverLink}/lib";
        };
        shellHook = ''
          uv sync --all-extras --quiet
          ${activateVenv}
          export DAGSTER_HOME="$(git rev-parse --show-toplevel)/.dg-home"
          export DAGSTER_PG_URL="postgresql:///dagster?host=${pgSocketDir}"
          mkdir -p "$DAGSTER_HOME"
        '';
      };
    };
}
