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
          || baseName == "Cargo.toml"
          || baseName == "Cargo.lock";
      };
    in
    {
      packages =
        let
          quarryEtl = pkgs.rustPlatform.buildRustPackage {
            pname = "quarry-etl";
            version = "0.1.0";

            inherit src;

            cargoHash = "sha256-n1m03o6UYAl4Le0tZHApo0Htc82twBEoEZzxisddsl8=";

            dontUseCmakeConfigure = true;

            nativeBuildInputs = with pkgs; [
              cmake
              pkg-config
            ];

            buildInputs = with pkgs; [
              openssl
            ];
          };
        in
        {
          "quarry-etl" = quarryEtl;
          default = quarryEtl;
        };
    };
}
