{ inputs, ... }:
{
  perSystem =
    { pkgs, ... }:
    let
      craneLib = inputs.crane.mkLib pkgs;

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
        default = papeline;
      };
    };
}
