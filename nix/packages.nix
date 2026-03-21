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

      # Root of the quarry repo — needed so quarry-build can resolve
      # its `path = "../quarry-core"` dependency during Nix build.
      repoRoot = pkgs.lib.cleanSourceWith {
        src = ../.;
        filter =
          path: _type:
          let
            rel = pkgs.lib.removePrefix (toString ../.) path;
          in
          pkgs.lib.hasPrefix "/quarry-core" rel || pkgs.lib.hasPrefix "/quarry-build" rel;
      };
    in
    {
      packages = {
        # Python extension module (cdylib) — normalize + abstract_recon
        quarry-core = mkMaturinPackage {
          pname = "quarry-core";
          src = ../quarry-core;
          lockFile = ../quarry-core/Cargo.lock;
        };

        # Python extension module (cdylib) — build pipeline (PG, S3, XML)
        quarry-build = python.pkgs.buildPythonPackage {
          pname = "quarry-build";
          version = "0.1.0";
          pyproject = true;

          # Use repo root so Cargo can resolve quarry-core path dependency
          src = repoRoot;
          sourceRoot = "source/quarry-build";

          cargoDeps = rustPlatform.importCargoLock {
            lockFile = ../quarry-build/Cargo.lock;
          };

          nativeBuildInputs = with rustPlatform; [
            cargoSetupHook
            maturinBuildHook
          ];

          doCheck = false;
        };

        quarry-graph = mkMaturinPackage {
          pname = "quarry-graph";
          src = ../quarry-graph;
          lockFile = ../quarry-graph/Cargo.lock;
        };
      };
    };
}
