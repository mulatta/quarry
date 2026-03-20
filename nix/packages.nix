{
  perSystem =
    { pkgs, ... }:
    let
      python = pkgs.python313;
      inherit (pkgs) rustPlatform;

      mkMaturinPackage =
        {
          pname,
          src,
          lockFile,
        }:
        python.pkgs.buildPythonPackage {
          inherit pname src;
          version = "0.2.0";
          pyproject = true;

          cargoDeps = rustPlatform.importCargoLock { inherit lockFile; };

          nativeBuildInputs = with rustPlatform; [
            cargoSetupHook
            maturinBuildHook
          ];

          # Skip tests — no Python test suite, Rust tests run via cargo
          doCheck = false;
        };
    in
    {
      packages = {
        # Python extension module (cdylib) — parse + normalize functions
        quarry-rs = mkMaturinPackage {
          pname = "quarry-rs";
          src = ../quarry-rs;
          lockFile = ../quarry-rs/Cargo.lock;
        };

        quarry-graph = mkMaturinPackage {
          pname = "quarry-graph";
          src = ../quarry-graph;
          lockFile = ../quarry-graph/Cargo.lock;
        };

        # CLI binary — quarry-build (PubMed/OA → PG pipeline)
        quarry-build = rustPlatform.buildRustPackage {
          pname = "quarry-build";
          version = "0.2.0";
          src = ../quarry-rs;
          cargoLock.lockFile = ../quarry-rs/Cargo.lock;
          doCheck = false;
        };
      };
    };
}
