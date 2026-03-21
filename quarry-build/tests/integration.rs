//! Integration tests for quarry-build pipeline.
//!
//! These tests require a running PostgreSQL instance.
//! Set QUARRY_TEST_PG_CONNINFO to enable (e.g., "postgresql:///quarry_test").
//! Tests are skipped if the env var is not set.

use std::path::Path;

use quarry_build::build::config::BuildConfig;

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn test_conninfo() -> Option<String> {
    std::env::var("QUARRY_TEST_PG_CONNINFO").ok()
}

fn init_test_db(conninfo: &str) {
    use postgres::{Client, NoTls};
    let mut client = Client::connect(conninfo, NoTls).unwrap();

    // Truncate all tables for a clean test
    for table in &[
        "cited_by_clin", "chemicals", "grants", "mesh_headings", "authors",
        "papers", "work_citations", "work_topics", "work_authors", "work_mesh",
        "id_crosswalk", "works", "_build_progress",
    ] {
        let _ = client.execute(&format!("TRUNCATE {table} CASCADE"), &[]);
    }
}

// ── PubMed build ──

#[test]
fn test_pubmed_build() {
    let Some(conninfo) = test_conninfo() else {
        eprintln!("SKIP: QUARRY_TEST_PG_CONNINFO not set");
        return;
    };
    init_test_db(&conninfo);

    let config = BuildConfig::default();
    let pubmed_fixtures = fixtures_dir().join("pubmed");

    let stats = quarry_build::build::pubmed::build_pubmed(
        &config,
        &pubmed_fixtures,
        None,
        &conninfo,
    )
    .unwrap();

    assert_eq!(stats.num_papers, 2, "fixture has 2 PubMed articles");
    assert!(stats.num_authors > 0);
    assert_eq!(stats.num_failed_files, 0);
}

#[test]
fn test_pubmed_progress_skip() {
    let Some(conninfo) = test_conninfo() else {
        eprintln!("SKIP: QUARRY_TEST_PG_CONNINFO not set");
        return;
    };
    init_test_db(&conninfo);

    let config = BuildConfig::default();
    let pubmed_fixtures = fixtures_dir().join("pubmed");

    // First build
    let stats1 = quarry_build::build::pubmed::build_pubmed(
        &config,
        &pubmed_fixtures,
        None,
        &conninfo,
    )
    .unwrap();
    assert_eq!(stats1.num_papers, 2);

    // Second build — files marked done in _build_progress → skip
    let stats2 = quarry_build::build::pubmed::build_pubmed(
        &config,
        &pubmed_fixtures,
        None,
        &conninfo,
    )
    .unwrap();
    // Papers count will be 0 because all files are skipped via progress table
    assert_eq!(stats2.num_papers, 0);
}

// ── OA local build ──

#[test]
fn test_oa_local_build() {
    let Some(conninfo) = test_conninfo() else {
        eprintln!("SKIP: QUARRY_TEST_PG_CONNINFO not set");
        return;
    };
    init_test_db(&conninfo);

    let config = BuildConfig::default();
    let oa_fixtures = fixtures_dir().join("oa");

    let stats = quarry_build::build::oa::build_oa_local(&config, &oa_fixtures, &conninfo).unwrap();
    assert_eq!(stats.num_works, 3, "expected 3 works (T1, T2, T3)");
    assert!(stats.num_citations > 0, "T1 has 2 referenced_works");
    assert!(stats.num_crosswalk > 0, "T1 has PMID → crosswalk");
    assert_eq!(stats.num_failed_lines, 0);
}
