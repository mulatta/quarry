{
  perSystem =
    { pkgs, lib, ... }:
    let
      src = lib.cleanSourceWith {
        src = lib.cleanSource ../.;
        filter =
          path: type:
          let
            baseName = baseNameOf path;
          in
          type == "directory"
          || (lib.hasSuffix ".rs" path)
          || (lib.hasSuffix ".py" path)
          || baseName == "Cargo.toml"
          || baseName == "Cargo.lock"
          || baseName == "pyproject.toml";
      };
    in
    {
      packages =
        let
          quarryEtl = pkgs.python3.pkgs.buildPythonPackage {
            pname = "quarry-etl";
            version = "0.1.0";
            pyproject = true;

            inherit src;
            sourceRoot = "${src.name}/py";

            postUnpack = ''
              chmod -R u+w ${src.name}
              cp ${src.name}/Cargo.lock ${src.name}/py/
            '';

            cargoDeps = pkgs.rustPlatform.fetchCargoVendor {
              inherit src;
              hash = "sha256-mUMYAuElDlABhhbA2542v7bxf4QcuvnqYsjSDBkSFw8=";
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
        in
        {
          "quarry-etl" = quarryEtl;
          default = quarryEtl;
        };
    };
}
