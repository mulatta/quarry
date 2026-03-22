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

      # nix develop        — debug Rust build (fast iteration)
      devShells.default = pkgs.mkShell {
        packages = shellPackages pkgs;
        env = shellEnv // {
          LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
        };
        shellHook = ''
          uv sync --all-extras --quiet
          ${activateVenv}
        '';
      };

      # nix develop .#release — release Rust build (production)
      devShells.release = pkgs.mkShell {
        packages = shellPackages pkgs;
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
