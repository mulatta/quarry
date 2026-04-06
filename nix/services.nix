{ inputs, ... }:
let
  pgSocketDir = "/tmp/quarry-pg";
in
{
  imports = [ inputs.process-compose-flake.flakeModule ];

  perSystem =
    { pkgs, ... }:
    {
      process-compose.services = {
        imports = [ inputs.services-flake.processComposeModules.default ];

        services.postgres.quarry-pg = {
          enable = true;
          package = pkgs.postgresql_16;
          dataDir = "./.pg-data";
          socketDir = pgSocketDir;
          settings = {
            listen_addresses = "";
            synchronous_commit = "off";
            shared_buffers = "48GB"; # ~25% of 188GB RAM
            effective_cache_size = "128GB"; # ~70% of 188GB RAM (includes OS page cache)
            work_mem = "256MB";
            maintenance_work_mem = "2GB";
            effective_io_concurrency = 200;
            autovacuum = "off";
            shared_preload_libraries = "pg_stat_statements";
          };
          superuser = null;
          initialDatabases = [
            {
              name = "quarry";
              schemas = [ ../sql/schema.sql ];
            }
            { name = "dagster"; }
          ];
        };

        services.clickhouse."quarry-ch" = {
          enable = true;
          port = 9000;
          extraConfig = {
            http_port = 8124;
          };
          initialDatabases = [
            {
              name = "quarry";
              schemas = [ ../sql/ch_schema.sql ];
            }
          ];
        };
      };
    };
}
