//! Enrich pipeline: DataFusion JOIN.
//!
//! 1. works LEFT JOIN papers → enriched works with pm_ fields
//! 2. mesh_headings INNER JOIN id_crosswalk → work_mesh
//!
//! Uses DataFusion for spill-to-disk JOIN of 250M works × 36M papers.

use std::time::Instant;

use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::prelude::*;

use crate::config::BuildConfig;

/// Stats from an enrich run.
pub struct EnrichStats {
    pub enriched_works_files: usize,
    pub work_mesh_files: usize,
    pub elapsed_secs: f64,
}

/// Run the enrich pipeline (sync wrapper around async DataFusion).
pub fn enrich(config: &BuildConfig) -> Result<EnrichStats, Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(enrich_async(config))
}

const ENRICHED_WORKS_SQL: &str = r#"
SELECT
    w.work_id, w.work_id_int, w.tier, w.pmid, w.doi, w.title, w."abstract",
    w.pub_year, w.pub_date, w."type", w.cited_by_count,
    w.host_venue, w.oa_status, w.oa_url, w.is_retracted, w.updated_date,
    p.journal_abbr  AS pm_journal_abbr,
    p.country        AS pm_country,
    p.medline_status AS pm_medline_status,
    p.pub_type       AS pm_pub_type,
    p.created_date   AS pm_created_date,
    p.revised_date   AS pm_revised_date,
    p.indexed_date   AS pm_indexed_date
FROM works w
LEFT JOIN papers p ON w.pmid = p.pmid
"#;

const WORK_MESH_SQL: &str = r#"
SELECT
    c.work_id,
    m.descriptor_ui, m.descriptor_name,
    m.qualifier_ui, m.qualifier_name,
    m.is_major_topic
FROM mesh_headings m
INNER JOIN id_crosswalk c ON m.pmid = c.pmid
"#;

async fn enrich_async(config: &BuildConfig) -> Result<EnrichStats, Box<dyn std::error::Error>> {
    let t0 = Instant::now();

    let ctx = SessionContext::new();
    let opts = ParquetReadOptions::default();

    // Register input Parquet directories as tables
    let works_path = config.works_dir();
    let papers_path = config.papers_dir();
    let mesh_path = config.mesh_headings_dir();
    let crosswalk_path = config.id_crosswalk_dir();

    ctx.register_parquet("works", works_path.to_str().unwrap(), opts.clone())
        .await?;
    ctx.register_parquet("papers", papers_path.to_str().unwrap(), opts.clone())
        .await?;
    ctx.register_parquet(
        "mesh_headings",
        mesh_path.to_str().unwrap(),
        opts.clone(),
    )
    .await?;
    ctx.register_parquet(
        "id_crosswalk",
        crosswalk_path.to_str().unwrap(),
        opts,
    )
    .await?;

    // 1. Enriched works: LEFT JOIN papers for pm_ fields
    eprintln!("enrich: works LEFT JOIN papers → enriched_works");
    let enriched = ctx.sql(ENRICHED_WORKS_SQL).await?;

    let enriched_dir = config.enriched_works_dir();
    let results = enriched
        .write_parquet(
            enriched_dir.to_str().unwrap(),
            DataFrameWriteOptions::new(),
            None,
        )
        .await?;
    let enriched_files = results.len();
    eprintln!("enrich: wrote enriched_works ({enriched_files} part files)");

    // 2. work_mesh: mesh_headings JOIN id_crosswalk
    eprintln!("enrich: mesh_headings JOIN id_crosswalk → work_mesh");
    let work_mesh = ctx.sql(WORK_MESH_SQL).await?;

    let mesh_dir = config.work_mesh_dir();
    let results = work_mesh
        .write_parquet(
            mesh_dir.to_str().unwrap(),
            DataFrameWriteOptions::new(),
            None,
        )
        .await?;
    let mesh_files = results.len();
    eprintln!("enrich: wrote work_mesh ({mesh_files} part files)");

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!("enrich: done in {elapsed:.1}s");

    Ok(EnrichStats {
        enriched_works_files: enriched_files,
        work_mesh_files: mesh_files,
        elapsed_secs: elapsed,
    })
}
