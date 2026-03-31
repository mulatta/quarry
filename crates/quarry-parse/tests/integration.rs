//! Integration tests for quarry-parse pipeline.
//!
//! Tests parse local files → Parquet output (no DB required).

use std::path::Path;

use quarry_parse::build::config::ParseConfig;

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

// ── PubMed parse ──

#[test]
fn test_pubmed_parse() {
    let config = ParseConfig::default();
    let pubmed_fixtures = fixtures_dir().join("pubmed");
    let output_dir = tempfile::tempdir().unwrap();

    let stats = quarry_parse::build::pubmed::parse_pubmed(
        &config,
        &pubmed_fixtures,
        None,
        output_dir.path(),
    )
    .unwrap();

    assert_eq!(stats.num_papers, 2, "fixture has 2 PubMed articles");
    assert!(stats.num_authors > 0);
    assert_eq!(stats.num_failed_files, 0);

    // Verify Parquet files exist
    let papers_dir = output_dir.path().join("papers");
    assert!(papers_dir.exists(), "papers/ directory should exist");
}

#[test]
fn test_pubmed_idempotent_reparse() {
    let config = ParseConfig::default();
    let pubmed_fixtures = fixtures_dir().join("pubmed");
    let output_dir = tempfile::tempdir().unwrap();

    // First parse
    let stats1 = quarry_parse::build::pubmed::parse_pubmed(
        &config,
        &pubmed_fixtures,
        None,
        output_dir.path(),
    )
    .unwrap();
    assert_eq!(stats1.num_papers, 2);

    // Second parse — idempotent overwrite, same result
    let stats2 = quarry_parse::build::pubmed::parse_pubmed(
        &config,
        &pubmed_fixtures,
        None,
        output_dir.path(),
    )
    .unwrap();
    assert_eq!(stats2.num_papers, 2);
}

// ── OA parse ──

#[test]
fn test_oa_parse() {
    let config = ParseConfig::default();
    let oa_fixtures = fixtures_dir().join("oa");
    let output_dir = tempfile::tempdir().unwrap();

    let stats = quarry_parse::build::oa::parse_oa(
        &config,
        &oa_fixtures,
        output_dir.path(),
        None,
    )
    .unwrap();

    assert_eq!(stats.num_works, 3, "expected 3 works (T1, T2, T3)");
    assert!(stats.num_citations > 0, "T1 has 2 referenced_works");
    assert!(stats.num_crosswalk > 0, "T1 has PMID → crosswalk");
    assert_eq!(stats.num_failed_files, 0);
    assert_eq!(stats.num_failed_lines, 0);
}

#[test]
fn test_oa_parse_partition_filter() {
    let config = ParseConfig::default();
    let oa_fixtures = fixtures_dir().join("oa");
    let input_dir = tempfile::tempdir().unwrap();

    // Create two partitions from the same fixture
    let part1 = input_dir.path().join("updated_date=2025-01-01");
    let part2 = input_dir.path().join("updated_date=2025-01-02");
    std::fs::create_dir_all(&part1).unwrap();
    std::fs::create_dir_all(&part2).unwrap();
    std::fs::copy(
        oa_fixtures.join("oa_sample.jsonl.gz"),
        part1.join("part_000.gz"),
    )
    .unwrap();
    std::fs::copy(
        oa_fixtures.join("oa_sample.jsonl.gz"),
        part2.join("part_000.gz"),
    )
    .unwrap();

    let output_dir = tempfile::tempdir().unwrap();

    // Parse only partition 1
    let stats = quarry_parse::build::oa::parse_oa(
        &config,
        input_dir.path(),
        output_dir.path(),
        Some("updated_date=2025-01-01"),
    )
    .unwrap();

    assert_eq!(stats.num_files_total, 1, "should process only 1 .gz from partition 1");
    assert!(stats.num_works > 0);

    // Verify only partition 1 output exists
    let p1_output = output_dir.path().join("works/updated_date=2025-01-01");
    let p2_output = output_dir.path().join("works/updated_date=2025-01-02");
    assert!(p1_output.exists(), "partition 1 output should exist");
    assert!(!p2_output.exists(), "partition 2 output should not exist");
}
