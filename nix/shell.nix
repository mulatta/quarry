{
  perSystem =
    { pkgs, ... }:
    {
      devShells.default = pkgs.mkShell {
        buildInputs =
          with pkgs;
          [
            # Rust
            cargo
            rustc
            rust-analyzer
            clippy

            # Build deps
            cmake
            pkg-config
            openssl

            # Data inspection
            parquet-tools
            duckdb
            pqrs

            # Profiling (cross-platform)
            inferno
          ]
          ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [
            perf
          ];
      };
    };
}
