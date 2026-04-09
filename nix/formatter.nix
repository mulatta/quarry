{ inputs, ... }:
{
  imports = [ inputs.treefmt-nix.flakeModule ];
  perSystem.treefmt = {
    projectRootFile = "flake.nix";
    # Exclude files with YAML frontmatter — mdformat mangles --- delimiters
    settings.global.excludes = [
      ".claude/agents/**"
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
