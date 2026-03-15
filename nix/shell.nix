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
          LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
        };

        shellHook = ''
          if [ -f pyproject.toml ]; then
            if [[ ! -f "uv.lock" ]] || [[ "pyproject.toml" -nt "uv.lock" ]]; then
              uv lock --quiet
            fi
            uv sync --quiet
          fi

          export VIRTUAL_ENV="$PWD/.venv"
          export PATH="$VIRTUAL_ENV/bin:$PATH"
        '';
      };
    };
}
