_:
let
  pgSocketDir = "/tmp/quarry-pg";
in
{
  perSystem =
    {
      pkgs,
      lib,
      ...
    }:
    let
      commonPackages = with pkgs; [
        python313
        uv
        cmake
      ];
    in
    {
      devShells.default = pkgs.mkShell {
        packages =
          commonPackages
          ++ (with pkgs; [
            rustc
            cargo
            clippy
            maturin
            just
            postgresql_16
            clickhouse
            duckdb
            awscli2
            cudaPackages.cuda_cudart
            cudaPackages.cuda_nvcc
          ]);

        env = {
          UV_PYTHON_DOWNLOADS = "never";
          PGHOST = pgSocketDir;
          QUARRY_PG_CONNINFO = "host=${pgSocketDir} dbname=quarry";
          LD_LIBRARY_PATH = lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            "${pkgs.addDriverRunpath.driverLink}"
          ];
          TRITON_LIBCUDA_PATH = "${pkgs.addDriverRunpath.driverLink}/lib";
        };

        shellHook = ''
          uv sync --all-extras --quiet
          source .venv/bin/activate
          export DAGSTER_HOME="$(git rev-parse --show-toplevel)/.dg-home"
          export DAGSTER_PG_URL="postgresql:///dagster?host=${pgSocketDir}"
          mkdir -p "$DAGSTER_HOME"
        '';
      };

      devShells.agents = pkgs.mkShell {
        packages = commonPackages;

        env = {
          UV_PYTHON_DOWNLOADS = "never";
          LD_LIBRARY_PATH = lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ];
        };

        shellHook = ''
          uv sync --directory agents --quiet
          source agents/.venv/bin/activate
        '';
      };
    };
}
