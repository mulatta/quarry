//! Integration tests: manifest-diff based incremental sync.
//!
//! These tests require network access (one manifest fetch per dry-run, ~1.3s).
//! Run with: cargo test --test incremental
//! Skip in CI with: cargo test --lib  (unit tests only)

use std::path::Path;

use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;
use tempfile::TempDir;

fn run(dir: &Path, args: &[&str]) -> assert_cmd::assert::Assert {
    cargo_bin_cmd!("papeline")
        .arg("--output-dir")
        .arg(dir)
        .args(args)
        .assert()
}

fn count_parquet_files(dir: &Path) -> usize {
    walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "parquet"))
        .count()
}

/// Modify .manifest.json in place via serde_json.
fn patch_manifest(dir: &Path, f: impl FnOnce(&mut serde_json::Value)) {
    let path = dir.join(".manifest.json");
    let text = std::fs::read_to_string(&path).expect("read manifest");
    let mut val: serde_json::Value = serde_json::from_str(&text).expect("parse manifest");
    f(&mut val);
    std::fs::write(&path, serde_json::to_string(&val).expect("serialize")).expect("write manifest");
}

// ============================================================
// Tests
// ============================================================

#[test]
fn incremental_sync_lifecycle() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path();

    // === 1. Initial run: download 2 shards ===
    // tracing logs go to stdout when output is not a terminal
    run(dir, &["run", "--max-shards", "2", "--concurrency", "2"])
        .success()
        .stdout(predicate::str::contains("2 completed, 0 failed"));

    assert!(dir.join(".manifest.json").exists(), "manifest snapshot");
    assert!(dir.join(".state.json").exists(), "state file");
    assert!(dir.join(".sync_log.jsonl").exists(), "sync log");
    assert!(
        dir.join("works/shard_0000.parquet").exists(),
        "works shard 0"
    );
    assert!(
        dir.join("works/shard_0001.parquet").exists(),
        "works shard 1"
    );
    assert_eq!(count_parquet_files(dir), 24, "2 shards × 12 tables");

    // === 2. Incremental dry-run: completed shards skipped ===
    // Extract total shard count from dry-run output to avoid hardcoding
    // (remote manifest shard count changes over time)
    let output = cargo_bin_cmd!("papeline")
        .arg("--output-dir")
        .arg(dir)
        .args(["run", "--dry-run", "--max-shards", "5"])
        .output()
        .expect("failed to execute");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse total from "Remote manifest: N shards"
    let total: usize = {
        let re = regex::Regex::new(r"Remote manifest:\s+(\d+)\s+shards").unwrap();
        re.captures(&stdout).expect("should contain shard count")[1]
            .parse()
            .unwrap()
    };
    let expected_changed = total - 2; // 2 completed shards
    assert!(
        stdout.contains("Unchanged (ok):    2"),
        "expected 2 unchanged, got:\n{stdout}"
    );
    assert!(
        stdout.contains(&format!("Changed/new:       {expected_changed}")),
        "expected {expected_changed} changed, got:\n{stdout}"
    );
    assert!(stdout.contains("Removed:           0"));
    assert!(stdout.contains("shard_0002"));

    // === 3. Tampered content_length → shard re-detected ===
    let backup = std::fs::read_to_string(dir.join(".manifest.json")).unwrap();

    patch_manifest(dir, |v| {
        v["entries"][0]["content_length"] = serde_json::json!(999);
    });

    run(dir, &["run", "--dry-run", "--max-shards", "5"])
        .success()
        .stdout(predicate::str::contains(format!(
            "Changed/new:       {}",
            expected_changed + 1
        )))
        .stdout(predicate::str::is_match(r"Unchanged \(ok\):\s+1").unwrap())
        .stdout(predicate::str::contains("shard_0000"));

    std::fs::write(dir.join(".manifest.json"), &backup).unwrap();

    // === 4. Extra snapshot entry → detected as removed ===
    let backup = std::fs::read_to_string(dir.join(".manifest.json")).unwrap();

    patch_manifest(dir, |v| {
        let fake = serde_json::json!({
            "url": "https://fake.example.com/removed.gz",
            "shard_idx": 9999,
            "content_length": 100,
            "record_count": 10,
            "updated_date": "2020-01-01"
        });
        v["entries"].as_array_mut().unwrap().push(fake);
    });

    run(dir, &["run", "--dry-run", "--max-shards", "5"])
        .success()
        .stdout(predicate::str::is_match(r"Removed:\s+1").unwrap())
        .stdout(predicate::str::contains("shard_9999"));

    std::fs::write(dir.join(".manifest.json"), &backup).unwrap();

    // === 5. Removal run deletes parquet files ===
    let backup = std::fs::read_to_string(dir.join(".manifest.json")).unwrap();

    patch_manifest(dir, |v| {
        let fake = serde_json::json!({
            "url": "https://fake.example.com/will_be_removed.gz",
            "shard_idx": 8888,
            "content_length": 100,
            "record_count": 10,
            "updated_date": "2020-01-01"
        });
        v["entries"].as_array_mut().unwrap().push(fake);
    });

    // Create fake parquet files for the "removed" shard
    let tables = [
        "works",
        "works_keys",
        "citations",
        "work_authorships",
        "work_topics",
        "work_keywords",
        "work_mesh",
        "work_locations",
        "work_funders",
        "work_awards",
        "work_sdgs",
        "work_counts_by_year",
    ];
    for table in &tables {
        let table_dir = dir.join(table);
        std::fs::create_dir_all(&table_dir).unwrap();
        std::fs::write(table_dir.join("shard_8888.parquet"), b"fake").unwrap();
    }

    run(dir, &["run", "--max-shards", "0"]).success();

    for table in &["works", "citations", "work_topics"] {
        assert!(
            !dir.join(table).join("shard_8888.parquet").exists(),
            "{table}/shard_8888.parquet should be deleted"
        );
    }
    assert!(
        dir.join("works/shard_0000.parquet").exists(),
        "real files intact"
    );
    assert!(
        dir.join("works/shard_0001.parquet").exists(),
        "real files intact"
    );

    std::fs::write(dir.join(".manifest.json"), &backup).unwrap();

    // === 6. --force reprocesses all shards ===
    run(dir, &["run", "--dry-run", "--force", "--max-shards", "5"])
        .success()
        .stdout(predicate::str::contains(format!(
            "Changed/new:       {total}"
        )))
        .stdout(predicate::str::is_match(r"Unchanged \(ok\):\s+0").unwrap());
}
