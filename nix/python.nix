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
        final: prev:
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

          # Some legacy sdist packages use setuptools as build backend but omit it
          # from build-system.requires. Inject setuptools for each affected package.
          addSetuptools =
            pkg:
            pkg.overrideAttrs (old: {
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ final.setuptools ];
            });
        in
        # nvidia-*, torch, and torchvision wheels reference CUDA runtime libs
        # (libcudart, libcublas, libnvshmem, ...) that live in the host GPU
        # driver, not in the Nix store. Skip autoPatchelf dep checking for these;
        # torch finds its nvidia-* siblings at runtime via site-packages lookup.
        (lib.mapAttrs (
          name: pkg:
          if
            lib.hasPrefix "nvidia-" name
            || builtins.elem name [
              "torch"
              "torchvision"
            ]
          then
            pkg.overrideAttrs (_: {
              autoPatchelfIgnoreMissingDeps = true;
            })
          else
            pkg
        ) prev)
        // {
          quarry-core = addCargo ../crates/quarry-core/Cargo.lock prev.quarry-core;
          quarry-graph = addCargo ../crates/quarry-graph/Cargo.lock prev.quarry-graph;
          # lancedb: autoPatchelfHook applied automatically for manylinux wheels.
          # Uncomment and extend buildInputs if shared libs are missing at runtime:
          # lancedb = prev.lancedb.overrideAttrs (old: {
          #   buildInputs = (old.buildInputs or []) ++ [ pkgs.openssl pkgs.zlib ];
          # });

          # Packages that use setuptools as build backend without declaring it:
          "antlr4-python3-runtime" = addSetuptools prev."antlr4-python3-runtime";
          "pylatexenc" = addSetuptools prev."pylatexenc";
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
        meta = old.meta // {
          mainProgram = "quarry-server";
        };
      });
    in
    {
      packages.quarry-server = mkApplication {
        inherit venv;
        package = quarryServerPkg;
      };
    };
}
