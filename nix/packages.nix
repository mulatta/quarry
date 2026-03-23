{ inputs, ... }:
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

      # crane — standalone Rust binary build (no nixpkgs input of its own)
      craneLib = inputs.crane.mkLib pkgs;

      # Source: repo root filtered to quarry-ingest + quarry-core (path dep) + sql/
      ingestSrc = pkgs.lib.cleanSourceWith {
        src = ../.;
        filter =
          path: type:
          let
            rel = pkgs.lib.removePrefix (toString ../.) path;
          in
          pkgs.lib.hasPrefix "/quarry-ingest" rel
          || pkgs.lib.hasPrefix "/quarry-core" rel
          || pkgs.lib.hasPrefix "/sql" rel
          || (craneLib.filterCargoSources path type);
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

        quarry-graph = mkMaturinPackage {
          pname = "quarry-graph";
          src = ../quarry-graph;
          lockFile = ../quarry-graph/Cargo.lock;
        };

        # Standalone Rust binary — data ingestion CLI
        quarry-ingest = craneLib.buildPackage {
          pname = "quarry-ingest";
          version = "0.1.0";
          src = ingestSrc;

          cargoLock = ../quarry-ingest/Cargo.lock;
          cargoExtraArgs = "--bin quarry-ingest";
          nativeBuildInputs = [ pkgs.cmake ];

          # Build from quarry-ingest subdirectory
          postUnpack = ''
            cd $sourceRoot/quarry-ingest
            export sourceRoot=$(pwd)
          '';

          doCheck = false;
        };
      };
    };
}
