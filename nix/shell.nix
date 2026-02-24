{
  perSystem =
    { pkgs, ... }:
    {
      devShells.default = pkgs.mkShell {
        packages = with pkgs; [
          # Rust
          cargo
          rustc
          rust-analyzer
          clippy

          # Build deps
          cmake
          pkg-config
          openssl
        ];
      };
    };
}
