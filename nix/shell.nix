{
  perSystem =
    { pkgs, ... }:
    {
      devShells.default = pkgs.mkShell {
        packages = with pkgs; [
          python3
          uv
          rustc
          cargo
          maturin
        ];

        env = {
          UV_PYTHON_DOWNLOADS = "never";
        };

        shellHook = ''
          if [ -f pyproject.toml ]; then
            if [[ ! -f "uv.lock" ]] || [[ "pyproject.toml" -nt "uv.lock" ]]; then
              uv lock --quiet
            fi
            uv sync --quiet
          fi
        '';
      };
    };
}
