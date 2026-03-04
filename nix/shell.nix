{
  perSystem =
    { pkgs, ... }:
    {
      devShells.default = pkgs.mkShell {
        venvDir = "py/.venv";

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

            # Python bindings
            python3
            python3.pkgs.venvShellHook
            maturin

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

        postVenvCreation = ''
          pip install typer rich
          (cd py && maturin develop)
        '';
      };
    };
}
