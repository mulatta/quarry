{ inputs }:
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.quarry-server;
in
{
  options.services.quarry-server = {
    enable = lib.mkEnableOption "quarry MCP HTTP server";

    package = lib.mkOption {
      type = lib.types.package;
      default = inputs.self.packages.${pkgs.stdenv.hostPlatform.system}.quarry-server;
      defaultText = lib.literalExpression "quarry.packages.\${system}.quarry-server";
      description = "The quarry-server package to use.";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = ''
        Address to bind the MCP HTTP server (QUARRY_MCP_HOST).
        Set to 0.0.0.0 for nginx reverse proxy or Tailscale.
      '';
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8000;
      description = "Port for the MCP HTTP server (QUARRY_MCP_PORT).";
    };

    dataDir = lib.mkOption {
      type = lib.types.path;
      default = "/var/lib/quarry";
      description = ''
        Base data directory (QUARRY_DATA_DIR).
        All serving paths (csr/, lancedb/) are resolved relative to this.
      '';
    };

    environmentFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = ''
        Path to a file containing secret environment variables.
        Use for QUARRY_PG_CONNINFO and other credentials (e.g., managed by sops-nix).
      '';
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "quarry";
      description = "System user to run the quarry-server service.";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "quarry";
      description = "System group to run the quarry-server service.";
    };
  };

  config = lib.mkIf cfg.enable {
    users.users.${cfg.user} = {
      isSystemUser = true;
      inherit (cfg) group;
    };
    users.groups.${cfg.group} = { };

    systemd.services.quarry-server = {
      description = "quarry MCP HTTP server";
      wantedBy = [ "multi-user.target" ];
      after = [
        "network.target"
        "postgresql.service"
      ];

      environment = {
        QUARRY_MCP_HOST = cfg.host;
        QUARRY_MCP_PORT = toString cfg.port;
        QUARRY_DATA_DIR = cfg.dataDir;
      };

      serviceConfig = {
        ExecStart = "${cfg.package}/bin/quarry-server";
        EnvironmentFile = lib.mkIf (cfg.environmentFile != null) cfg.environmentFile;
        User = cfg.user;
        Group = cfg.group;
        Restart = "on-failure";
        RestartSec = "5s";
        # Hardening
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ cfg.dataDir ];
      };
    };

    systemd.tmpfiles.rules = [
      "d ${cfg.dataDir} 0750 ${cfg.user} ${cfg.group} -"
    ];
  };
}
