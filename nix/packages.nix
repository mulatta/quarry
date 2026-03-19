{
  perSystem =
    { pkgs, ... }:
    let
      python = pkgs.python313;
      inherit (pkgs) rustPlatform lib;

      mkMaturinPackage =
        {
          pname,
          src,
          lockFile,
        }:
        python.pkgs.buildPythonPackage {
          inherit pname src;
          version = "0.1.0";
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
        quarry-parse = mkMaturinPackage {
          pname = "quarry-parse";
          src = ../quarry-parse;
          lockFile = ../quarry-parse/Cargo.lock;
        };

        quarry-graph = mkMaturinPackage {
          pname = "quarry-graph";
          src = ../quarry-graph;
          lockFile = ../quarry-graph/Cargo.lock;
        };

        quarry-build = rustPlatform.buildRustPackage {
          pname = "quarry-build";
          version = "0.1.0";

          # Include both crates so path dependency resolves
          src = lib.fileset.toSource {
            root = ../.;
            fileset = lib.fileset.unions [
              ../quarry-build
              ../quarry-parse
            ];
          };

          cargoRoot = "quarry-build";
          cargoLock.lockFile = ../quarry-build/Cargo.lock;

          buildAndTestSubdir = "quarry-build";

          doCheck = false;
        };
      };
    };
}
