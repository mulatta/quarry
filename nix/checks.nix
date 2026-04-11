{
  perSystem =
    { config, ... }:
    {
      # Expose all built packages as checks so nix flake check / nix-fast-build
      # evaluates and builds them all in one pass.
      checks = {
        inherit (config.packages)
          quarry-core
          quarry-graph
          quarry-parse
          quarry-server
          ;
      };
    };
}
