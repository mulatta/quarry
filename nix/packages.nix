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

      # crane — standalone Rust binary build (no nixpkgs input of its own)
      craneLib = inputs.crane.mkLib pkgs;

      # Source: repo root filtered to crates/quarry-parse + crates/quarry-core (path dep)
      parseSrc = pkgs.lib.cleanSourceWith {
        src = ../.;
        filter =
          path: type:
          let
            rel = pkgs.lib.removePrefix (toString ../.) path;
          in
          pkgs.lib.hasPrefix "/crates/quarry-parse" rel
          || pkgs.lib.hasPrefix "/crates/quarry-core" rel
          || (craneLib.filterCargoSources path type);
      };
    in
    {
      packages = {
        # Python extension module (cdylib) — normalize + abstract_recon
        quarry-core = mkMaturinPackage {
          pname = "quarry-core";
          src = ../crates/quarry-core;
          lockFile = ../crates/quarry-core/Cargo.lock;
        };

        quarry-graph = mkMaturinPackage {
          pname = "quarry-graph";
          src = ../crates/quarry-graph;
          lockFile = ../crates/quarry-graph/Cargo.lock;
        };

        # Standalone Rust binary — parse CLI (local files → Parquet)
        quarry-parse = craneLib.buildPackage {
          pname = "quarry-parse";
          version = "0.1.0";
          src = parseSrc;

          cargoLock = ../crates/quarry-parse/Cargo.lock;
          cargoExtraArgs = "--bin quarry-parse";
          nativeBuildInputs = [ pkgs.cmake ];

          # Build from crates/quarry-parse subdirectory (crate lives here)
          postUnpack = ''
            cd $sourceRoot/crates/quarry-parse
            export sourceRoot=$(pwd)
          '';

          doCheck = false;
        };
      };
    };
}
