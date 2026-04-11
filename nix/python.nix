{ inputs, ... }:
{
  perSystem =
    { pkgs, ... }:
    let
      inherit (pkgs) lib;

      workspace = inputs.uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ../.; };

      overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };

      # quarry-core and quarry-graph are maturin (Rust/PyO3) packages.
      # uv2nix generates a pyproject-nix derivation for them (with dependencies metadata),
      # but it doesn't know about the Cargo workspace — we add cargoDeps + cargoSetupHook
      # via overrideAttrs so the pyproject-nix dependency graph stays intact.
      pyprojectOverrides =
        _final: prev:
        let
          addCargo =
            lockFile: pkg:
            pkg.overrideAttrs (old: {
              cargoDeps = pkgs.rustPlatform.importCargoLock { inherit lockFile; };
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                pkgs.rustPlatform.cargoSetupHook
                pkgs.cargo
                pkgs.rustc
              ];
              # Prevent maturin from importing puccinialin (its Rust-install fallback).
              # cargo is already in PATH via nativeBuildInputs above.
              env = (old.env or { }) // {
                MATURIN_NO_INSTALL_RUST = "1";
              };
            });
        in
        {
          quarry-core = addCargo ../crates/quarry-core/Cargo.lock prev.quarry-core;
          quarry-graph = addCargo ../crates/quarry-graph/Cargo.lock prev.quarry-graph;
          # lancedb: autoPatchelfHook applied automatically for manylinux wheels.
          # Uncomment and extend buildInputs if shared libs are missing at runtime:
          # lancedb = prev.lancedb.overrideAttrs (old: {
          #   buildInputs = (old.buildInputs or []) ++ [ pkgs.openssl pkgs.zlib ];
          # });
        };

      pythonSet =
        (pkgs.callPackage inputs.pyproject-nix.build.packages {
          python = pkgs.python313;
        }).overrideScope
          (
            lib.composeManyExtensions [
              inputs.pyproject-build-systems.overlays.wheel
              overlay
              pyprojectOverrides
            ]
          );

      inherit (pkgs.callPackages inputs.pyproject-nix.build.util { }) mkApplication;

      venv = pythonSet.mkVirtualEnv "quarry-server-venv" { quarry = [ "server" ]; };

      # quarry has two scripts: "quarry" (CLI) and "quarry-server" (MCP daemon).
      # mkApplication uses meta.mainProgram to select which binary `nix run` invokes.
      quarryServerPkg = pythonSet.quarry.overrideAttrs (old: {
        meta = old.meta // { mainProgram = "quarry-server"; };
      });
    in
    {
      packages.quarry-server = mkApplication {
        inherit venv;
        package = quarryServerPkg;
      };
    };
}
