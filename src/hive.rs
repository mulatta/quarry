//! Hive-partitioned Parquet output via DataFusion.
//!
//! Repartitions raw shard parquets into `pub_year=YYYY/` Hive layout,
//! enabling efficient year-range queries with partition pruning.

use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use datafusion::config::TableParquetOptions;
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::logical_expr::JoinType;
use datafusion::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::{ResolvedHiveConfig, format_bytes};
use crate::oa::TABLES;

// ============================================================
// Idempotency state
// ============================================================

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct HiveState {
    pub source_manifest_hash: String,
    pub zstd_level: i32,
    pub completed_at: String,
    pub tables: Vec<String>,
}

impl HiveState {
    const FILENAME: &str = ".hive_state.json";

    fn load(hive_dir: &Path) -> Option<Self> {
        let path = hive_dir.join(Self::FILENAME);
        let text = fs::read_to_string(&path).ok()?;
        sonic_rs::from_str(&text).ok()
    }

    fn save(&self, hive_dir: &Path) -> Result<()> {
        let path = hive_dir.join(Self::FILENAME);
        let tmp = hive_dir.join(format!("{}.tmp", Self::FILENAME));
        let json = sonic_rs::to_string_pretty(self)?;
        fs::write(&tmp, &json)?;
        fs::rename(&tmp, &path)?;
        Ok(())
    }
}

// ============================================================
// Manifest hashing
// ============================================================

fn manifest_hash(raw_dir: &Path) -> Result<String> {
    let path = raw_dir.join(".manifest.json");
    let content = fs::read(&path).with_context(|| format!("Cannot read {}", path.display()))?;
    let hash = Sha256::digest(&content);
    Ok(format!("{hash:x}"))
}

// ============================================================
// DataFusion parquet writer options
// ============================================================

fn parquet_writer_props(config: &ResolvedHiveConfig) -> TableParquetOptions {
    let mut opts = TableParquetOptions::new();
    opts.global.compression = Some(format!("zstd({})", config.zstd_level));
    opts.global.max_row_group_size = config.row_group_size;
    opts
}

fn write_options() -> DataFrameWriteOptions {
    // No with_sort_by: sorting all partitions in parallel causes OOM on large tables
    // (each ExternalSorter consumes 6-8GB for works' 83 columns x millions of rows).
    // Partition pruning -- the primary goal -- works without intra-partition sorting.
    DataFrameWriteOptions::new().with_partition_by(vec!["pub_year".to_string()])
}

// ============================================================
// Per-table builders
// ============================================================

/// works table: pub_year derived directly from publication_year.
async fn build_works_hive(ctx: &SessionContext, config: &ResolvedHiveConfig) -> Result<()> {
    let raw_path = format!("{}/works/shard_*.parquet", config.raw_dir.display());
    let df = ctx
        .read_parquet(&raw_path, ParquetReadOptions::default())
        .await
        .context("Failed to read works parquet")?;

    let df = df.with_column("pub_year", coalesce(vec![col("publication_year"), lit(0)]))?;

    let staging = config.staging_dir.join("works");
    fs::create_dir_all(&staging)?;

    df.write_parquet(
        &staging.to_string_lossy(),
        write_options(),
        Some(parquet_writer_props(config)),
    )
    .await
    .context("Failed to write works hive parquet")?;

    Ok(())
}

/// Register works (work_id, publication_year) as a named table for child JOINs.
/// Avoids re-scanning works glob + parsing parquet footers for each child table.
async fn register_works_keys(ctx: &SessionContext, config: &ResolvedHiveConfig) -> Result<()> {
    let works_path = format!("{}/works/shard_*.parquet", config.raw_dir.display());
    ctx.register_parquet("_works_keys", &works_path, ParquetReadOptions::default())
        .await
        .context("Failed to register works for child JOINs")?;
    Ok(())
}

/// Child tables: JOIN with registered works keys to inject pub_year.
async fn build_child_hive(
    ctx: &SessionContext,
    table: &str,
    config: &ResolvedHiveConfig,
) -> Result<()> {
    // Use registered works table -- column pruning selects only 2 columns
    let works = ctx
        .table("_works_keys")
        .await
        .context("works keys table not registered")?
        .select(vec![col("work_id"), col("publication_year")])?;

    let child_path = format!("{}/{}/shard_*.parquet", config.raw_dir.display(), table);
    let child = ctx
        .read_parquet(&child_path, ParquetReadOptions::default())
        .await
        .with_context(|| format!("Failed to read {table} parquet"))?;

    // JOIN child with works on work_id
    let joined = child.join(works, JoinType::Inner, &["work_id"], &["work_id"], None)?;

    // Add pub_year, drop publication_year
    let joined = joined
        .with_column("pub_year", coalesce(vec![col("publication_year"), lit(0)]))?
        .drop_columns(&["publication_year"])?;

    let staging = config.staging_dir.join(table);
    fs::create_dir_all(&staging)?;

    joined
        .write_parquet(
            &staging.to_string_lossy(),
            write_options(),
            Some(parquet_writer_props(config)),
        )
        .await
        .with_context(|| format!("Failed to write {table} hive parquet"))?;

    Ok(())
}

// ============================================================
// Atomic swap
// ============================================================

/// Swap staging/{table} -> {hive_dir}/{table} atomically via rename.
fn atomic_swap_table(hive_dir: &Path, staging_dir: &Path, table: &str) -> Result<()> {
    let target = hive_dir.join(table);
    let staged = staging_dir.join(table);
    let old = hive_dir.join(format!(".old_{table}"));

    if target.exists() {
        fs::rename(&target, &old).with_context(|| format!("Failed to rename old {table}"))?;
    }
    fs::rename(&staged, &target).with_context(|| format!("Failed to swap staged {table}"))?;
    if old.exists()
        && let Err(e) = fs::remove_dir_all(&old)
    {
        tracing::warn!("Failed to remove old {table} dir: {e}");
    }
    Ok(())
}

/// Remove staging directory for a table on failure.
fn cleanup_staging(staging_dir: &Path, table: &str) {
    let staged = staging_dir.join(table);
    if staged.exists()
        && let Err(e) = fs::remove_dir_all(&staged)
    {
        tracing::warn!("Failed to clean staging for {table}: {e}");
    }
}

// ============================================================
// Public entry point
// ============================================================

/// Check if raw shard directory has data for a table.
fn has_raw_shards(raw_dir: &Path, table: &str) -> bool {
    let dir = raw_dir.join(table);
    dir.exists()
        && fs::read_dir(&dir)
            .map(|mut rd| {
                rd.any(|e| {
                    e.ok()
                        .is_some_and(|e| e.path().extension().is_some_and(|ext| ext == "parquet"))
                })
            })
            .unwrap_or(false)
}

fn fmt_elapsed(elapsed: std::time::Duration) -> String {
    let secs = elapsed.as_secs();
    if secs >= 60 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else {
        format!("{:.1}s", elapsed.as_secs_f64())
    }
}

pub fn run_hive(config: &ResolvedHiveConfig, force: bool, dry_run: bool) -> Result<()> {
    let total_start = Instant::now();

    // 1. Compute source manifest hash
    let hash = manifest_hash(&config.raw_dir)?;

    // 2. Check idempotency
    fs::create_dir_all(&config.hive_dir)?;
    if !force
        && let Some(state) = HiveState::load(&config.hive_dir)
        && state.source_manifest_hash == hash
        && state.zstd_level == config.zstd_level
    {
        println!("Up to date (manifest hash unchanged)");
        return Ok(());
    }

    // 3. Determine tables with raw data
    let tables: Vec<&str> = TABLES
        .iter()
        .copied()
        .filter(|t| has_raw_shards(&config.raw_dir, t))
        .collect();

    if tables.is_empty() {
        anyhow::bail!(
            "No raw parquet shards found in {}",
            config.raw_dir.display()
        );
    }

    // 4. Dry run
    if dry_run {
        println!("Hive partitioning plan:");
        println!("  Source:      {}", config.raw_dir.display());
        println!("  Destination: {}", config.hive_dir.display());
        println!("  ZSTD level:  {}", config.zstd_level);
        println!("  Row group:   {}", config.row_group_size);
        println!("  Threads:     {}", config.threads);
        println!("  Memory:      {}", format_bytes(config.memory_limit_bytes));
        println!("  Tables:      {}", tables.len());
        for t in &tables {
            println!("    - {t}");
        }
        return Ok(());
    }

    tracing::info!(
        "Hive config: threads={}, memory={}, zstd={}, row_group={}",
        config.threads,
        format_bytes(config.memory_limit_bytes),
        config.zstd_level,
        config.row_group_size,
    );

    // 5. Build tokio runtime + DataFusion session
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(config.threads)
        .enable_all()
        .build()
        .context("Failed to build tokio runtime")?;

    runtime.block_on(async {
        let session_config = SessionConfig::new()
            .with_target_partitions(config.threads)
            .with_batch_size(8192);

        let rt_env = RuntimeEnvBuilder::new()
            .with_memory_limit(config.memory_limit_bytes, 1.0)
            .build_arc()
            .context("Failed to build DataFusion runtime")?;

        let ctx = SessionContext::new_with_config_rt(session_config, rt_env);

        // 6. Create staging directory
        fs::create_dir_all(&config.staging_dir)?;

        // 7. Process works first (needed for child JOIN)
        let mut completed = 0usize;
        if tables.contains(&"works") {
            let t = Instant::now();
            println!("Processing: works (1/{})", tables.len());
            match build_works_hive(&ctx, config).await {
                Ok(()) => {
                    atomic_swap_table(&config.hive_dir, &config.staging_dir, "works")?;
                    completed += 1;
                    tracing::info!("works completed in {}", fmt_elapsed(t.elapsed()));
                }
                Err(e) => {
                    cleanup_staging(&config.staging_dir, "works");
                    return Err(e).context("Failed to build works hive");
                }
            }
        }

        // 8. Register works parquet once for child JOINs
        //    (avoids re-scanning works glob + re-parsing footers per child table)
        register_works_keys(&ctx, config).await?;

        // 9. Process remaining tables sequentially
        //    (DataFusion uses multi-threaded execution internally)
        for table in &tables {
            if *table == "works" {
                continue;
            }
            completed += 1;
            let t = Instant::now();
            println!("Processing: {table} ({completed}/{})", tables.len());
            match build_child_hive(&ctx, table, config).await {
                Ok(()) => {
                    atomic_swap_table(&config.hive_dir, &config.staging_dir, table)?;
                    tracing::info!("{table} completed in {}", fmt_elapsed(t.elapsed()));
                }
                Err(e) => {
                    cleanup_staging(&config.staging_dir, table);
                    return Err(e).with_context(|| format!("Failed to build {table} hive"));
                }
            }
        }

        // 10. Save hive state
        let state = HiveState {
            source_manifest_hash: hash,
            zstd_level: config.zstd_level,
            completed_at: chrono::Utc::now().to_rfc3339(),
            tables: tables.iter().map(|s| (*s).to_string()).collect(),
        };
        state.save(&config.hive_dir)?;

        // 11. Cleanup staging dir
        if config.staging_dir.exists()
            && let Err(e) = fs::remove_dir_all(&config.staging_dir)
        {
            tracing::warn!("Failed to remove staging dir: {e}");
        }

        println!(
            "Done: {} tables, {:.1}s",
            tables.len(),
            total_start.elapsed().as_secs_f64()
        );

        Ok(())
    })
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::parse_memory_limit;

    #[test]
    fn parse_memory_limit_gb() {
        assert_eq!(parse_memory_limit("32GB").unwrap(), 32 * 1024 * 1024 * 1024);
    }

    #[test]
    fn parse_memory_limit_mb() {
        assert_eq!(parse_memory_limit("512MB").unwrap(), 512 * 1024 * 1024);
    }

    #[test]
    fn parse_memory_limit_bytes() {
        assert_eq!(parse_memory_limit("1024").unwrap(), 1024);
    }

    #[test]
    fn parse_memory_limit_invalid() {
        assert!(parse_memory_limit("notanumber").is_err());
    }

    #[test]
    fn hive_state_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        let state = HiveState {
            source_manifest_hash: "abc123".to_string(),
            zstd_level: 8,
            completed_at: "2026-01-01T00:00:00Z".to_string(),
            tables: vec!["works".to_string(), "citations".to_string()],
        };
        state.save(dir.path()).unwrap();
        let loaded = HiveState::load(dir.path()).unwrap();
        assert_eq!(loaded.source_manifest_hash, "abc123");
        assert_eq!(loaded.zstd_level, 8);
        assert_eq!(loaded.tables.len(), 2);
    }

    #[test]
    fn hive_state_load_missing() {
        let dir = tempfile::TempDir::new().unwrap();
        assert!(HiveState::load(dir.path()).is_none());
    }

    #[test]
    fn manifest_hash_deterministic() {
        let dir = tempfile::TempDir::new().unwrap();
        fs::write(
            dir.path().join(".manifest.json"),
            r#"{"saved_at":"2026-01-01","entries":[]}"#,
        )
        .unwrap();
        let h1 = manifest_hash(dir.path()).unwrap();
        let h2 = manifest_hash(dir.path()).unwrap();
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // SHA-256 hex
    }

    #[test]
    fn manifest_hash_missing_file() {
        let dir = tempfile::TempDir::new().unwrap();
        assert!(manifest_hash(dir.path()).is_err());
    }

    #[test]
    fn has_raw_shards_empty_dir() {
        let dir = tempfile::TempDir::new().unwrap();
        assert!(!has_raw_shards(dir.path(), "works"));
    }

    #[test]
    fn has_raw_shards_with_parquet() {
        let dir = tempfile::TempDir::new().unwrap();
        let works_dir = dir.path().join("works");
        fs::create_dir_all(&works_dir).unwrap();
        fs::write(works_dir.join("shard_0000.parquet"), b"fake").unwrap();
        assert!(has_raw_shards(dir.path(), "works"));
    }

    #[test]
    fn has_raw_shards_no_parquet() {
        let dir = tempfile::TempDir::new().unwrap();
        let works_dir = dir.path().join("works");
        fs::create_dir_all(&works_dir).unwrap();
        fs::write(works_dir.join("readme.txt"), b"not parquet").unwrap();
        assert!(!has_raw_shards(dir.path(), "works"));
    }

    #[test]
    fn fmt_elapsed_seconds() {
        assert_eq!(fmt_elapsed(std::time::Duration::from_secs_f64(3.5)), "3.5s");
    }

    #[test]
    fn fmt_elapsed_minutes() {
        assert_eq!(fmt_elapsed(std::time::Duration::from_secs(125)), "2m05s");
    }
}
