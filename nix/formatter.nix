{ inputs, ... }:
{
  imports = [ inputs.treefmt-nix.flakeModule ];
  perSystem.treefmt = {
    projectRootFile = "flake.nix";
    # Exclude skill files — mdformat mangles YAML frontmatter (--- → ## heading)
    settings.global.excludes = [
      ".claude/skills/**"
      "docs/dogfood/**"
    ];
    programs = {
      nixfmt.enable = true;
      deadnix.enable = true;
      statix.enable = true;
      keep-sorted.enable = true;
      ruff-check.enable = true;
      ruff-format.enable = true;
      mdformat.enable = true;
    };
  };
}
