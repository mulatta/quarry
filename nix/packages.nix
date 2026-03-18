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
    in
    {
      packages = {
        quarry-parse = mkMaturinPackage {
          pname = "quarry-parse";
          src = ../quarry-parse;
          lockFile = ../quarry-parse/Cargo.lock;
        };

        quarry-csr = mkMaturinPackage {
          pname = "quarry-csr";
          src = ../quarry-csr;
          lockFile = ../quarry-csr/Cargo.lock;
        };
      };
    };
}
