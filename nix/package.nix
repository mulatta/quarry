{ inputs, ... }:
{
  perSystem =
    { pkgs, ... }:
    let
      craneLib = inputs.crane.mkLib pkgs;

      # ── Rust CLI source (existing) ──────────────────────────
      src = pkgs.lib.cleanSourceWith {
        src = craneLib.cleanCargoSource (craneLib.path ../.);
        filter = path: _type: !(pkgs.lib.hasInfix "/tests/" path);
      };

      commonArgs = {
        inherit src;
        strictDeps = true;
        nativeBuildInputs = with pkgs; [
          cmake
          pkg-config
        ];
        buildInputs = with pkgs; [
          openssl
        ];
      };

      cargoArtifacts = craneLib.buildDepsOnly commonArgs;

      # ── Python package source (Rust + Python files) ─────────
      pySrc = pkgs.lib.cleanSourceWith {
        src = craneLib.path ../.;
        filter =
          path: type:
          (craneLib.filterCargoSources path type)
          || (pkgs.lib.hasSuffix ".py" path)
          || (pkgs.lib.hasSuffix "pyproject.toml" path);
      };
    in
    {
      packages = rec {
        papeline = craneLib.buildPackage (
          commonArgs
          // {
            inherit cargoArtifacts;
            # Integration tests require network (manifest fetch); skip in sandbox
            cargoTestExtraArgs = "--bin papeline";
          }
        );

        papeline-py = pkgs.python3.pkgs.buildPythonPackage {
          pname = "papeline";
          version = "0.1.0";
          pyproject = true;

          src = pySrc;
          sourceRoot = "${pySrc.name}/py";

          postUnpack = ''
            chmod -R u+w ${pySrc.name}
            cp ${pySrc.name}/Cargo.lock ${pySrc.name}/py/
          '';

          cargoDeps = pkgs.rustPlatform.fetchCargoVendor {
            src = pySrc;
            hash = "sha256-+Rsupxedpg1/F/hH2MzcCEJ854g2ObCG2FJSdHimFkk=";
          };

          build-system = with pkgs.rustPlatform; [
            cargoSetupHook
            maturinBuildHook
          ];

          dontUseCmakeConfigure = true;

          nativeBuildInputs = with pkgs; [
            cmake
            pkg-config
          ];

          buildInputs = with pkgs; [
            openssl
          ];

          dependencies = with pkgs.python3.pkgs; [
            typer
            rich
          ];
        };

        default = papeline;
      };
    };
}
